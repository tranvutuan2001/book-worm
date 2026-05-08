"""
PDF Summarization Service — three-step document-to-PDF-JSON pipeline.

Step 1  Generate a rich text summary using the LLM and document retrieval tools.
         The LLM decides which tools to call — the full document is NOT sent.
Step 2  Split the summary into self-contained logical blocks (one per
          title/overview, chapter, key-themes section, conclusion, etc.)
          using a dedicated LLM call.  xgrammar constrained decoding enforces
          a valid JSON array of strings at the token level.
Step 3  For each block: convert it into a JSON sub-array that conforms to
         pdf-schema.json.  The LLM is given the full ``pdf-example.json``
         as the target to follow so the output mirrors its exact style.
         xgrammar constrained decoding enforces schema conformance at the
         token level.  All sub-arrays are concatenated into the final JSON.

After all blocks are merged the combined JSON is validated against
pdf-schema.json once as a sanity check.  No LLM repair is performed.
The final artefact is saved under ``PDF_DIR`` and the path is returned.
"""

import json
import logging
import time

from datetime import datetime
from pathlib import Path
from typing import Any
import re
import jsonschema
from langfuse import observe

from app.config.app_setting import app_setting
from app.util.exceptions import DocumentNotFoundError, DocumentProcessingError
from app.domain.entity.agent import Agent
from app.domain.entity.message import Message
from app.domain.enum.role import Role
from app.domain.value_object.chat_model_setting import ChatModelSettings
from app.infrastructure.llm_connector.llm_service import LLMService
from app.services.tools.document_retrieval_tool import (
    get_document_summary,
)
from app.services.commands.summarize_pdf_command import SummarizePDFCommand

# Sentinel used for the RunContext parameter (no dependency data needed)
_NO_CTX: None = None

logger = logging.getLogger("app.service.pdf_summarization")

JSON_ARRAY_OF_STRINGS_SCHEMA: str = json.dumps(
    {"type": "array", "items": {"type": "string"}}
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_STEP1_SYSTEM = """
You are a document analyst. You will be given a raw document summary.
Your job is to rewrite it into a clean, readable summary:
- Fix awkward phrasing and improve flow.
- Do NOT oversimplify or remove important details.
- Keep the summary comprehensive, informative, and under 3000 words.
- Output ONLY the final summary text. Do NOT add commentary or ask questions.
""".strip()

_STEP2_SPLIT_SYSTEM = """
Role: Expert Data Parser

Task: Segment the provided input text into a sequence of logical, self-contained strings formatted as a JSON array.

Segmentation Logic:
1. Thematic Integrity: Split the text at points of topical shifts, new arguments, or contextual changes. Each segment must be "semantically complete"—a reader should understand the segment without needing the previous one.
2. Zero-Loss Policy: You must perform a literal extraction. Every character, word, and punctuation mark from the original text must be present in the final segments, in the exact original order. Do not summarize, fix typos, or omit data.
3. Size Constraints: * Maximum: Each individual block MUST NOT exceed 300 words. * Minimum: Aim for blocks larger than 100 words whenever possible.
Note: If the entire input text is shorter than 100 words, provide it as a single-element array.
4. Granularity: Ensure blocks are large enough to retain their util meaning but strictly stay within the specified word-count range.

Output Constraints (Strict Compliance Required):

1. Format: Return a valid JSON array of strings: ["segment 1", "segment 2"].
2. No Wrappers: Do not use Markdown code blocks (e.g., no ```json). Do not include any preamble, headers, or "Here is the JSON" text.
3. First/Last Characters: The very first character of your response must be [ and the very last character must be ].
""".strip()

_STEP3_SYSTEM_TEMPLATE = """
Role: Expert Document Architect

Task: Transform the user-provided content into a structured JSON array for PDF rendering.

Structural Guidelines:
1. Schema Consistency: Use the exact node types and property structures found in the TARGET EXAMPLE.
2. Dynamic Layout: You are encouraged to be creative with the layout to ensure the report is visually engaging. Use the provided JSON schema to organize data into logical columns, grids, or blocks where appropriate.
3. Heading Logic: * If the content contains a document title or high-level overview, start with a level-1 heading (breakBefore: "always").
   Otherwise, begin directly with a level-2 heading for the section. Do not inject a level-1 heading if it isn't in the source text.

Output Constraints:

1. Format: Return ONLY a valid JSON array.
2. Cleanliness: No Markdown code fences (e.g., ` ` `json), no preamble, and no post-response commentary.

TARGET EXAMPLE:
{example}
""".strip()


class PDFSummarizationService:
    """Orchestrates the three-step PDF summarisation pipeline."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service
        self._schema: dict[str, Any] | None = None  # loaded lazily
        self._example: list[Any] | None = None  # loaded lazily

    @observe()
    async def summarize(
        self,
        command: SummarizePDFCommand,
    ) -> dict[str, Any]:
        """Run the full three-step summarisation pipeline.

        Args:
            command: SummarizePDFCommand containing document_name.

        Returns:
            ``{"output_file": "<path>", "content": [<pdf-json-nodes>]}``

        Raises:
            DocumentNotFoundError:     If the document folder is missing.
            DocumentProcessingError:   If any pipeline step fails.
        """
        document_name = command.document_name
        self._is_document_exist(document_name)

        schema = self._load_schema()
        schema_str = json.dumps(schema)

        logger.info("[summarize] Step 1 — generating text summary: %s", document_name)
        summary_text = await self._step1_generate_summary(document_name)
        logger.info("[summarize] Step 1 complete (%d words)", len(summary_text.split()))

        logger.info("[summarize] Step 2 — splitting summary into logical blocks")
        blocks = await self._step2_split_into_blocks(summary_text)
        logger.info("[summarize] %d block(s) to process", len(blocks))

        pdf_json: list[Any] = []
        for i, block in enumerate(blocks, start=1):
            logger.info(
                "[summarize] Step 3 — block %d/%d: generating JSON", i, len(blocks)
            )
            block_json = await self._step3_generate_json(
                block, schema_str=schema_str
            )
            logger.info(
                "[summarize] Step 3 block %d complete (%d nodes)", i, len(block_json)
            )
            pdf_json.extend(block_json)

        logger.info(
            "[summarize] All %d block(s) processed, %d total nodes",
            len(blocks),
            len(pdf_json),
        )

        logger.info("[summarize] Final validation against pdf-schema.json")
        error_msg = self._validate_once(pdf_json, schema)
        if error_msg is not None:
            raise DocumentProcessingError(
                f"Final PDF JSON failed schema validation: {error_msg}"
            )
        logger.info("[summarize] Final validation passed")

        output_path = self._save(document_name, pdf_json)
        logger.info("[summarize] Saved to %s", output_path)

        return {"output_file": str(output_path), "content": pdf_json}

    # ------------------------------------------------------------------
    # Step 1 — text summary via LLM + tools
    # ------------------------------------------------------------------

    async def _step1_generate_summary(self, document_name: str) -> str:
        """Fetch the raw document summary directly, then refine it with agent_complete_chat."""
        logger.info("[step1] Fetching raw document summary for '%s'", document_name)
        try:
            base_summary: str = get_document_summary(_NO_CTX, document_name=document_name)
        except Exception as exc:
            logger.error("[step1] get_document_summary failed: %s", exc, exc_info=True)
            raise DocumentProcessingError(
                f"Failed to fetch document summary: {exc}"
            ) from exc

        logger.info("[step1] Raw summary fetched (%d chars); refining with LLM", len(base_summary))
        request_message = Message(
            id="summarize_request",
            role=Role.USER,
            content=(
                f"Rewrite the following raw document summary into a clean, "
                f"comprehensive summary:\n\n{base_summary}"
            ),
            timestamp=int(time.time() * 1000),
        )
        try:
            step1_agent = Agent(
                system_prompt=_STEP1_SYSTEM,
                model_settings=ChatModelSettings(),
            )
            refined = await self._llm.agent_complete_chat(
                message_list=[request_message],
                domain_agent=step1_agent,
            )
            return refined
        except Exception as exc:
            logger.error("[step1] LLM refinement failed: %s", exc, exc_info=True)
            raise DocumentProcessingError(
                f"Failed to refine document summary: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Step 2 — split summary into logical blocks
    # ------------------------------------------------------------------

    async def _step2_split_into_blocks(
        self, summary_text: str, max_attempts: int = 5
    ) -> list[str]:
        """Ask the LLM to segment *summary_text* into self-contained logical blocks.

        The LLM is instructed to output a valid JSON array of strings.  If the
        output cannot be parsed, the request is retried up to *max_attempts*
        times.  Falls back to a single block only after all retries are
        exhausted.
        """
        user_content = (
            "Split the following document summary into logical blocks.\n"
            "Remember: output ONLY a JSON array of strings — start with `[` "
            "and end with `]`.\n\n"
            f"{summary_text}"
        )

        for attempt in range(1, max_attempts + 1):
            message = Message(
                id=f"split_request_{attempt}",
                role=Role.USER,
                content=user_content,
                timestamp=int(time.time() * 1000),
            )
            try:
                step2_agent = Agent(
                    system_prompt=_STEP2_SPLIT_SYSTEM,
                    model_settings=ChatModelSettings(json_schema=JSON_ARRAY_OF_STRINGS_SCHEMA),
                )
                raw_output = await self._llm.agent_complete_chat(
                    message_list=[message],
                    domain_agent=step2_agent,
                )
            except Exception as exc:
                logger.error(
                    "[split] LLM call failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                    exc_info=True,
                )
                raise DocumentProcessingError(
                    f"Failed to split summary into blocks: {exc}"
                ) from exc

            # Strip optional markdown fences
            cleaned = re.sub(
                r"^```[a-z]*\s*|\s*```$", "", raw_output.strip(), flags=re.MULTILINE
            ).strip()
            # Extract the outermost JSON array if the LLM added surrounding text
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

            try:
                blocks = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "[split] Attempt %d/%d — output is not valid JSON (%s); retrying.",
                    attempt,
                    max_attempts,
                    exc,
                )
                user_content = (
                    "Your previous response was not a valid JSON array.\n"
                    "Output ONLY a JSON array of strings — starting with `[` "
                    "and ending with `]` — no other text.\n\n"
                    f"{summary_text}"
                )
                continue

            if not isinstance(blocks, list) or not all(
                isinstance(b, str) for b in blocks
            ):
                logger.warning(
                    "[split] Attempt %d/%d — output has unexpected shape (%s); retrying.",
                    attempt,
                    max_attempts,
                    type(blocks).__name__,
                )
                user_content = (
                    "Your previous response was not a JSON array of strings.\n"
                    "Output ONLY a JSON array where every element is a string — "
                    "starting with `[` and ending with `]`.\n\n"
                    f"{summary_text}"
                )
                continue

            if not blocks:
                logger.warning(
                    "[split] Attempt %d/%d — empty array returned; retrying.",
                    attempt,
                    max_attempts,
                )
                continue

            logger.info(
                "[split] Summary split into %d block(s) on attempt %d",
                len(blocks),
                attempt,
            )
            for idx, block in enumerate(blocks, start=1):
                word_count = len(block.split())
                logger.info(
                    "[split] Block %d/%d: %d word(s)", idx, len(blocks), word_count
                )
            return blocks

        logger.warning(
            "[split] All %d attempt(s) failed to produce a valid JSON array; "
            "falling back to single block.",
            max_attempts,
        )
        return [summary_text]

    # ------------------------------------------------------------------
    # Step 3 — JSON generation via targeted LLM call
    # ------------------------------------------------------------------

    async def _step3_generate_json(
        self,
        block_text: str,
        max_attempts: int = 5,
        *,
        schema_str: str | None = None,
    ) -> list[Any]:
        """Ask the LLM to format one content *block_text* into a JSON sub-array.

        The model is given the full ``pdf-example.json`` as the target to
        follow, so the output mirrors its structure and style exactly.
        If the output cannot be parsed as a JSON array, the request is retried
        up to *max_attempts* times with the error and bad output included.
        """
        example = self._load_example()
        example_str = json.dumps(example, indent=2, ensure_ascii=False)
        step3_system = _STEP3_SYSTEM_TEMPLATE.format(example=example_str)

        user_content = (
            "Convert the following content block into a PDF JSON array.\n\n"
            f"{block_text}"
        )

        for attempt in range(1, max_attempts + 1):
            user_message = Message(
                id=f"format_request_{attempt}",
                role=Role.USER,
                content=user_content,
                timestamp=int(time.time() * 1000),
            )
            try:
                step3_agent = Agent(
                    system_prompt=step3_system,
                    model_settings=ChatModelSettings(
                        json_schema=schema_str,
                    ),
                )
                raw_output = await self._llm.agent_complete_chat(
                    message_list=[user_message],
                    domain_agent=step3_agent,
                )
            except Exception as exc:
                logger.error(
                    "[step3] LLM call failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                    exc_info=True,
                )
                raise DocumentProcessingError(
                    f"Failed to generate PDF JSON: {exc}"
                ) from exc

            # Strip optional markdown code fences (```json … ``` or ``` … ```)
            cleaned = re.sub(
                r"^```[a-z]*\s*|\s*```$", "", raw_output.strip(), flags=re.MULTILINE
            ).strip()

            # Extract the outermost JSON array if the LLM added surrounding text
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "[step3] Attempt %d/%d — output is not valid JSON (%s); retrying.",
                    attempt,
                    max_attempts,
                    exc,
                )
                user_content = (
                    "Your previous response could not be parsed as JSON.\n\n"
                    f"PARSE ERROR:\n{exc}\n\n"
                    f"YOUR PREVIOUS OUTPUT:\n{raw_output}\n\n"
                    "Fix the output so it is a valid JSON array. "
                    "Output ONLY the JSON array — no markdown, no commentary."
                )
                continue

            if not isinstance(result, list):
                logger.warning(
                    "[step3] Attempt %d/%d — expected a JSON array, got %s; retrying.",
                    attempt,
                    max_attempts,
                    type(result).__name__,
                )
                user_content = (
                    "Your previous response was not a JSON array "
                    f"(got {type(result).__name__} instead).\n\n"
                    f"YOUR PREVIOUS OUTPUT:\n{raw_output}\n\n"
                    "Output ONLY a JSON array — no markdown, no commentary."
                )
                continue

            return result

        raise DocumentProcessingError(
            f"[Step 3] Failed to produce a valid JSON array after {max_attempts} "
            "attempt(s)."
        )

    # ------------------------------------------------------------------
    # Helpers — validation
    # ------------------------------------------------------------------

    def _validate_once(self, pdf_json: list[Any], schema: dict[str, Any]) -> str | None:
        """Validate *pdf_json* against the schema.

        Returns ``None`` when valid, or an error description string when not.
        """
        try:
            jsonschema.validate(instance=pdf_json, schema=schema)
            return None
        except jsonschema.ValidationError as exc:
            return f"{exc.message} " f"(path: {list(exc.absolute_path)})"
        except jsonschema.SchemaError as exc:
            raise DocumentProcessingError(
                f"The PDF schema itself is invalid: {exc.message}"
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_document_exist(self, document_name: str) -> None:
        if not document_name:
            raise DocumentNotFoundError("Document name is required")
        doc_path = app_setting.data_storage_path / document_name
        if not doc_path.exists():
            raise DocumentNotFoundError(
                f"Document '{document_name}' not found at {doc_path}"
            )

    def _load_schema(self) -> dict[str, Any]:
        """Load and cache pdf-schema.json."""
        if self._schema is None:
            if not app_setting.pdf_schema_path.exists():
                raise DocumentProcessingError(
                    f"PDF schema not found at {app_setting.pdf_schema_path}"
                )
            try:
                self._schema = json.loads(app_setting.pdf_schema_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise DocumentProcessingError(
                    f"Failed to load PDF schema: {exc}"
                ) from exc
        return self._schema  # type: ignore[return-value]

    def _load_example(self) -> list[Any]:
        """Load and cache pdf-example.json."""
        if self._example is None:
            if not app_setting.pdf_example_path.exists():
                raise DocumentProcessingError(
                    f"PDF example not found at {app_setting.pdf_example_path}"
                )
            try:
                self._example = json.loads(app_setting.pdf_example_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise DocumentProcessingError(
                    f"Failed to load PDF example: {exc}"
                ) from exc
        return self._example  # type: ignore[return-value]

    def _save(self, document_name: str, pdf_json: list[Any]) -> Path:
        """Persist the generated JSON to ``PDF_DIR`` and return the path."""
        app_setting.pdf_storage_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{document_name}_summary_{timestamp}.json"
        output_path = app_setting.pdf_storage_path / filename
        try:
            output_path.write_text(
                json.dumps(pdf_json, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to save PDF JSON to {output_path}: {exc}"
            ) from exc
        return output_path


import asyncio
import json
import logging
import time
from pathlib import Path
from langfuse import observe

import pdfplumber

from app.config.app_setting import app_setting
from app.core.exceptions import DocumentProcessingError
from app.core.utils import write_json_file
from app.domain.entity.agent_factory import AgentFactory
from app.domain.entity.message import Message
from app.domain.enum.role import Role
from app.domain.value_object.chat_model_setting import ChatModelSettings
from app.infrastructure.llm_connector import LLMService
from app.services.commands.analyze_document_command import AnalyzeDocumentCommand

logger = logging.getLogger("app.service")

# ---------------------------------------------------------------------------
# Text-splitting helper (replaces langchain RecursiveCharacterTextSplitter)
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", " ", ""]


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Split *text* into chunks of at most *chunk_size* characters.

    Tries to split on the first separator that produces pieces shorter than
    *chunk_size*; when none do, falls back to a hard character split.
    Adjacent chunks overlap by *chunk_overlap* characters.
    """
    if separators is None:
        separators = list(_SEPARATORS)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Pick the best separator
    sep = separators[-1]
    for s in separators:
        if s in text:
            sep = s
            break

    parts = text.split(sep) if sep else list(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for part in parts:
        piece = part
        piece_len = len(piece) + (len(sep) if current else 0)

        if current_len + piece_len > chunk_size and current:
            merged = sep.join(current)
            # If a single merged chunk is still too big, recurse with the
            # next separator in the list.
            if len(merged) > chunk_size and len(separators) > 1:
                chunks.extend(
                    _recursive_split(merged, chunk_size, chunk_overlap, separators[1:])
                )
            else:
                chunks.append(merged)

            # Overlap: keep trailing parts whose total length ≤ chunk_overlap
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) + len(sep) > chunk_overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_len += len(p) + len(sep)
            current = overlap_parts
            current_len = overlap_len

        current.append(piece)
        current_len += piece_len

    if current:
        merged = sep.join(current)
        if len(merged) > chunk_size and len(separators) > 1:
            chunks.extend(
                _recursive_split(merged, chunk_size, chunk_overlap, separators[1:])
            )
        else:
            chunks.append(merged)

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Summarisation prompts
# ---------------------------------------------------------------------------

_SECTION_SUMMARY_SYSTEM = """
Summarize the text below into a comprehensive, high-density bulleted list.
Extract ALL relevant facts — use as many bullets as needed. Do not limit yourself
to the number of bullets shown in the example; the example only demonstrates the
expected format and style.

Organize bullets using these categories where applicable:
- Core Objective: The primary purpose or main point of the section.
- Key Metrics & Data: Specific numbers, percentages, or financial figures.
- Entities: Important names, organizations, or locations.
- Critical Insights: Key takeaways, decisions, or milestones.

Format example (style reference only — real output should be much longer):
- Core Objective: Evaluation of Q3 fiscal performance and operational overhead.
- Financials: Revenue of $14.2B (exceeding targets by 4%); Operating Margin at 22%.
- Key Entities: Marcus Vane (CFO), NeuralLink Corp (target of $2.1B acquisition).
- Operational Insight: 15% growth in APAC region driven by enterprise cloud adoption.
- Future Outlook: Management identified supply chain volatility as a 2025 risk factor.
""".strip()

_CHAPTER_SUMMARY_SYSTEM = """
Consolidate the provided section summaries into a comprehensive chapter-level overview.
Preserve ALL key details from every section — use as many bullets as needed. Do not
limit yourself to the number of bullets shown in the example; the example only
demonstrates the expected format and style.

Organize bullets using these categories where applicable:
- Overarching Theme: The central narrative or focus of the chapter.
- Consolidated Performance: Synthesis of key results and metrics from all sections.
- Major Milestones: High-level achievements or structural changes.
- Strategic Implications: Impact on the broader roadmap or business objectives.

Format example (style reference only — real output should be much longer):
- Overarching Theme: 2024 Strategic Pivot and Global Infrastructure Expansion.
- Performance Summary: Total annual revenue reached $58B with 12% YoY growth; net income at $8.4B.
- Key Milestones: Successful deployment of 12 data centers; acquisition of NeuralLink Corp for AI integration.
- Executive Changes: Appointed Clara Oswald as CTO to lead 'AI-First' transformation.
- Strategic Conclusion: Transition from hardware-centric to service-oriented model is 80% complete.
""".strip()


class DocumentAnalysisService:
    """Orchestrates the full document pre-analysis pipeline."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def pre_analyze_document(self, command: AnalyzeDocumentCommand) -> None:
        """Run the full analysis pipeline for a document.

        Args:
            command: AnalyzeDocumentCommand containing pdf_path and document_name.

        Raises:
            DocumentProcessingError: If any stage of the pipeline fails.
        """

        logger.info("Pre-analysis started: %s", command.document_name)

        try:
            pages = self._extract_pages(command.pdf_path)
            logger.info("Extracted %d pages from '%s'", len(pages), command.document_name)

            chunks = _recursive_split(
                "".join(pages),
                app_setting.document_chunk_size,
                app_setting.document_chunk_overlap
            )
            logger.info("Split into %d chunks", len(chunks))

            out_dir = app_setting.data_storage_path / command.document_name

            await self._process_chunks(chunks, command.document_name, out_dir)
            section_summaries = await self._process_sections(
                chunks, command.document_name, out_dir
            )
            await self._process_chapters(
                chunks, command.document_name, out_dir, section_summaries
            )

            logger.info("Pre-analysis completed: %s", command.document_name)

        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(
                f"Unexpected error during pre-analysis of '{command.document_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    @observe()
    async def _process_chunks(
        self, chunks: list[str], doc_name: str, out_dir: Path
    ) -> None:
        logger.info("[chunks] Writing %d chunks…", len(chunks))
        write_json_file(chunks, str(out_dir / f"{doc_name}{app_setting.suffix_chunks}"))
        embeddings = await self._embed_texts(chunks, label="chunk")
        write_json_file(
            embeddings, str(out_dir / f"{doc_name}{app_setting.suffix_chunk_embeddings}")
        )
        logger.info("[chunks] Done")

    @observe()
    async def _process_sections(
        self, chunks: list[str], doc_name: str, out_dir: Path
    ) -> list[str]:
        logger.info("[sections] Building section summaries…")
        summaries = await self._build_section_summaries(chunks)
        write_json_file(
            summaries, str(out_dir / f"{doc_name}{app_setting.suffix_section_summaries}")
        )
        embeddings = await self._embed_texts(summaries, label="section summary")
        write_json_file(
            embeddings, str(out_dir / f"{doc_name}{app_setting.suffix_section_embeddings}")
        )
        logger.info("[sections] Done (%d summaries)", len(summaries))
        return summaries

    @observe()
    async def _process_chapters(
        self,
        chunks: list[str],
        doc_name: str,
        out_dir: Path,
        section_summaries: list[str] | None,
    ) -> None:
        logger.info("[chapters] Building chapter summaries…")
        if section_summaries is None:
            section_summaries = await self._load_or_build_sections(
                chunks, doc_name, out_dir
            )

        chapter_summaries = await self._build_chapter_summaries(section_summaries)
        write_json_file(
            chapter_summaries,
            str(out_dir / f"{doc_name}{app_setting.suffix_chapter_summaries}"),
        )
        embeddings = await self._embed_texts(chapter_summaries, label="chapter summary")
        write_json_file(
            embeddings, str(out_dir / f"{doc_name}{app_setting.suffix_chapter_embeddings}")
        )
        logger.info("[chapters] Done (%d summaries)", len(chapter_summaries))

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_pages(self, pdf_path: str) -> list[str]:
        pages: list[str] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            pages.append(text)
                    except Exception as exc:
                        logger.warning("Skipping page %d: %s", i + 1, exc)
        except FileNotFoundError:
            raise DocumentProcessingError(f"PDF not found: {pdf_path}")
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to read PDF '{pdf_path}': {exc}"
            ) from exc
        return pages

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    async def _build_section_summaries(self, chunks: list[str]) -> list[str]:
        summaries: list[str] = []
        for i in range(0, len(chunks), app_setting.chunks_per_section):
            batch = chunks[i : i + app_setting.chunks_per_section]
            end = min(i + app_setting.chunks_per_section, len(chunks))
            logger.info("Section summary %d–%d / %d", i + 1, end, len(chunks))
            user_msg = Message(
                id="user",
                content="\n".join(batch),
                role=Role.USER,
                timestamp=int(time.time()),
            )
            try:
                summary_agent = AgentFactory.summary(
                    system_prompt=_SECTION_SUMMARY_SYSTEM,
                    model_settings=ChatModelSettings(
                        temperature=0.2,
                        frequency_penalty=0.2,
                    ),
                )
                summary = await self._llm.agent_complete_chat(
                    message_list=[user_msg],
                    agent=summary_agent,
                )
                logger.info("Section summary %d–%d done", i + 1, end)
                logger.info("Summary:\n%s", summary)
                summaries.append(summary)
            except Exception as exc:
                raise DocumentProcessingError(
                    f"Section summary {i + 1}–{end} failed: {exc}"
                ) from exc
        return summaries

    async def _build_chapter_summaries(self, section_summaries: list[str]) -> list[str]:
        chapters: list[str] = []
        for i in range(0, len(section_summaries), app_setting.sections_per_chapter):
            batch = section_summaries[i : i + app_setting.sections_per_chapter]
            end = min(i + app_setting.sections_per_chapter, len(section_summaries))
            logger.info(
                "Chapter summary from sections %d–%d / %d", i + 1, end,
                len(section_summaries),
            )
            combined = "\n\n".join(batch)
            user_msg = Message(
                id="user",
                content=combined,
                role=Role.USER,
                timestamp=int(time.time()),
            )
            try:
                chapter_agent = AgentFactory.summary(
                    system_prompt=_CHAPTER_SUMMARY_SYSTEM,
                    model_settings=ChatModelSettings(
                        temperature=0.2,
                        frequency_penalty=0.2,
                    ),
                )
                chapter = await self._llm.agent_complete_chat(
                    message_list=[user_msg],
                    agent=chapter_agent,
                )
                chapters.append(chapter)
            except Exception as exc:
                raise DocumentProcessingError(
                    f"Chapter summary {i + 1}–{end} failed: {exc}"
                ) from exc
        return chapters

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    async def _embed_text(self, text: str) -> list[float]:
        last_exc: Exception | None = None
        for attempt in range(app_setting.max_embedding_retries):
            try:
                return await self._llm.embed_text(text=text)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Embedding attempt %d/%d failed: %s",
                    attempt + 1,
                    app_setting.max_embedding_retries,
                    exc,
                )
                if attempt < app_setting.max_embedding_retries - 1:
                    await asyncio.sleep(1)
        raise DocumentProcessingError(
            f"Embedding failed after {app_setting.max_embedding_retries} attempts: {last_exc}"
        ) from last_exc

    async def _embed_texts(self, texts: list[str], label: str = "text") -> list[list[float]]:
        result: list[list[float]] = []
        for idx, text in enumerate(texts):
            logger.info("Embedding %s %d/%d…", label, idx + 1, len(texts))
            result.append(await self._embed_text(text))
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_or_build_sections(
        self, chunks: list[str], doc_name: str, out_dir: Path
    ) -> list[str]:
        section_file = out_dir / f"{doc_name}{app_setting.suffix_section_summaries}"
        if section_file.exists():
            try:
                with section_file.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning("Could not load section summaries: %s", exc)
        logger.info("Section summaries not found — building them first…")
        return await self._process_sections(chunks, doc_name, out_dir)


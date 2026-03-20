"""
Document analysis service — extracts text, generates summaries and embeddings.

This is the computationally heavy pipeline that runs in a background thread
after a document is uploaded.  It has no FastAPI / HTTP dependency.
"""

import json
import logging
import time
from pathlib import Path
from typing import Annotated, ClassVar, List, Optional

import pdfplumber
from fastapi import Depends
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PER_SECTION,
    DATA_DIR,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MAX_EMBEDDING_RETRIES,
    SECTIONS_PER_CHAPTER,
    SUFFIX_CHAPTER_EMBEDDINGS,
    SUFFIX_CHAPTER_SUMMARIES,
    SUFFIX_CHUNK_EMBEDDINGS,
    SUFFIX_CHUNKS,
    SUFFIX_SECTION_EMBEDDINGS,
    SUFFIX_SECTION_SUMMARIES,
)
from src.core.exceptions import DocumentProcessingError
from src.core.utils import write_json_file
from src.domain.entity.message import Message
from src.domain.enums import Role
from src.infra.llm_connector.llm_service import LLMService, get_llm_service
from src.service.tools.document_retrieval_tool import word_count_tool

logger = logging.getLogger("app.service")

# ---------------------------------------------------------------------------
# Summarisation prompts
# ---------------------------------------------------------------------------

_SECTION_SUMMARY_SYSTEM = """
/no_think
You are a fact extractor. Output ONLY the structured summary below. Do not explain, reflect, or reason.

RULES:
- Max 250 words total.
- Telegraphic style: drop articles (a, an, the) and auxiliary verbs.
- No preamble, no meta-commentary.

FORMAT (use exactly these headers):
ANCHORS: <concept> -> <change or role>
EVIDENCE: <bullet list of names, dates, numbers, hard facts>
BRIDGE: Pivots from <prior topic> to <next topic> by <mechanism>.
""".strip()

_CHAPTER_SUMMARY_SYSTEM = """
/no_think
You are a synthesis engine. Output ONLY the structured chapter summary below. Do not restate section facts, explain your process, or reason aloud.

RULES:
- Max 1000 words total.
- Synthesise across sections; never copy sentences from them.
- No preamble, no meta-commentary.

FORMAT (use exactly these headers):
THEMATIC ARCH: <one paragraph — how sections connect and build on each other>
CRITICAL PATH: <single sentence — the pivotal finding or event of this chapter>
TRANSITION: <single sentence — what unresolved tension or question this chapter hands to the next>
""".strip()


class DocumentAnalysisService:
    """Orchestrates the full document pre-analysis pipeline."""

    _instance: ClassVar["DocumentAnalysisService | None"] = None

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def pre_analyze_document(
        self,
        pdf_path: str,
        document_name: str,
    ) -> None:
        """Run the full analysis pipeline for *pdf_path*.

        Args:
            pdf_path: Absolute path to the PDF file on disk.
            document_name: Name used for output file prefixes.
            process_levels: Subset of ``["chunks", "sections", "chapters"]``
                to process.  Defaults to all three levels.

        Raises:
            DocumentProcessingError: If any stage of the pipeline fails.
        """

        logger.info("Pre-analysis started: %s", document_name)
        self._prepare_models()

        try:
            pages = self._extract_pages(pdf_path)
            logger.info("Extracted %d pages from '%s'", len(pages), document_name)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_text("".join(pages))
            logger.info("Split into %d chunks", len(chunks))

            out_dir = DATA_DIR / document_name


            self._process_chunks(chunks, document_name, out_dir)
            section_summaries: Optional[List[str]] = None
            section_summaries = self._process_sections(chunks, document_name, out_dir)
            self._process_chapters(
                chunks, document_name, out_dir, section_summaries
            )

            logger.info("Pre-analysis completed: %s", document_name)

        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(
                f"Unexpected error during pre-analysis of '{document_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _process_chunks(
        self, chunks: List[str], doc_name: str, out_dir: Path
    ) -> None:
        logger.info("[chunks] Writing %d chunks…", len(chunks))
        write_json_file(chunks, str(out_dir / f"{doc_name}{SUFFIX_CHUNKS}"))
        embeddings = self._embed_texts(chunks, label="chunk")
        write_json_file(
            embeddings, str(out_dir / f"{doc_name}{SUFFIX_CHUNK_EMBEDDINGS}")
        )
        logger.info("[chunks] Done")

    def _process_sections(
        self, chunks: List[str], doc_name: str, out_dir: Path
    ) -> List[str]:
        logger.info("[sections] Building section summaries…")
        summaries = self._build_section_summaries(chunks)
        write_json_file(
            summaries, str(out_dir / f"{doc_name}{SUFFIX_SECTION_SUMMARIES}")
        )
        embeddings = self._embed_texts(summaries, label="section summary")
        write_json_file(
            embeddings, str(out_dir / f"{doc_name}{SUFFIX_SECTION_EMBEDDINGS}")
        )
        logger.info("[sections] Done (%d summaries)", len(summaries))
        return summaries

    def _process_chapters(
        self,
        chunks: List[str],
        doc_name: str,
        out_dir: Path,
        section_summaries: Optional[List[str]],
    ) -> None:
        logger.info("[chapters] Building chapter summaries…")
        if section_summaries is None:
            section_summaries = self._load_or_build_sections(
                chunks, doc_name, out_dir
            )

        chapter_summaries = self._build_chapter_summaries(section_summaries)
        write_json_file(
            chapter_summaries,
            str(out_dir / f"{doc_name}{SUFFIX_CHAPTER_SUMMARIES}"),
        )
        embeddings = self._embed_texts(chapter_summaries, label="chapter summary")
        write_json_file(
            embeddings, str(out_dir / f"{doc_name}{SUFFIX_CHAPTER_EMBEDDINGS}")
        )
        logger.info("[chapters] Done (%d summaries)", len(chapter_summaries))

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _prepare_models(self) -> None:
        """Unload all cached models and eagerly reload the ones required for analysis."""
        logger.info("Clearing model caches and loading analysis models…")
        self._llm.unload_all_models()
        self._llm.load_model(DEFAULT_CHAT_MODEL, "chat")
        self._llm.load_model(DEFAULT_EMBEDDING_MODEL, "embedding")
        logger.info("Models ready")

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_pages(self, pdf_path: str) -> List[str]:
        pages: List[str] = []
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

    def _build_section_summaries(self, chunks: List[str]) -> List[str]:
        summaries: List[str] = []
        for i in range(0, len(chunks), CHUNKS_PER_SECTION):
            batch = chunks[i : i + CHUNKS_PER_SECTION]
            end = min(i + CHUNKS_PER_SECTION, len(chunks))
            logger.info("Section summary %d–%d / %d", i + 1, end, len(chunks))
            user_msg = Message(
                id="user",
                content=(
                    "Analyse and summarise the following document section:\n\n"
                    + "\n".join(batch)
                ),
                role=Role.USER,
                timestamp=int(time.time()),
            )
            try:
                summary = self._llm.agent_complete_chat(
                    message_list=[user_msg],
                    tools=[word_count_tool],
                    system_prompt=_SECTION_SUMMARY_SYSTEM,
                    model_path=DEFAULT_CHAT_MODEL,
                    max_tokens=8000
                )
                logger.info("Section summary %d–%d done", i + 1, end)
                logger.info("Summary:\n%s", summary)
                summaries.append(summary)
            except Exception as exc:
                raise DocumentProcessingError(
                    f"Section summary {i + 1}–{end} failed: {exc}"
                ) from exc
        return summaries

    def _build_chapter_summaries(self, section_summaries: List[str]) -> List[str]:
        chapters: List[str] = []
        for i in range(0, len(section_summaries), SECTIONS_PER_CHAPTER):
            batch = section_summaries[i : i + SECTIONS_PER_CHAPTER]
            end = min(i + SECTIONS_PER_CHAPTER, len(section_summaries))
            logger.info(
                "Chapter summary from sections %d–%d / %d", i + 1, end,
                len(section_summaries),
            )
            combined = "\n\n".join(
                f"Section {j + 1}: {s}" for j, s in enumerate(batch)
            )
            user_msg = Message(
                id="user",
                content=(
                    "Create a comprehensive chapter summary from the following "
                    "section summaries:\n\n" + combined
                ),
                role=Role.USER,
                timestamp=int(time.time()),
            )
            try:
                chapter = self._llm.agent_complete_chat(
                    message_list=[user_msg],
                    tools=[word_count_tool],
                    system_prompt=_CHAPTER_SUMMARY_SYSTEM,
                    model_path=DEFAULT_CHAT_MODEL,
                    max_tokens=12000
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

    def _embed_text(self, text: str) -> List[float]:
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_EMBEDDING_RETRIES):
            try:
                return self._llm.embed_text(
                    model_path=DEFAULT_EMBEDDING_MODEL, text=text
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Embedding attempt %d/%d failed: %s",
                    attempt + 1,
                    MAX_EMBEDDING_RETRIES,
                    exc,
                )
                if attempt < MAX_EMBEDDING_RETRIES - 1:
                    time.sleep(1)
        raise DocumentProcessingError(
            f"Embedding failed after {MAX_EMBEDDING_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def _embed_texts(self, texts: List[str], label: str = "text") -> List[List[float]]:
        result: List[List[float]] = []
        for idx, text in enumerate(texts):
            logger.info("Embedding %s %d/%d…", label, idx + 1, len(texts))
            result.append(self._embed_text(text))
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_or_build_sections(
        self, chunks: List[str], doc_name: str, out_dir: Path
    ) -> List[str]:
        section_file = out_dir / f"{doc_name}{SUFFIX_SECTION_SUMMARIES}"
        if section_file.exists():
            try:
                with section_file.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning("Could not load section summaries: %s", exc)
        logger.info("Section summaries not found — building them first…")
        return self._process_sections(chunks, doc_name, out_dir)


# ---------------------------------------------------------------------------
# Singleton & dependency factory
# ---------------------------------------------------------------------------


def get_document_analysis_service(
    llm_service: LLMService= Depends(get_llm_service),
) -> DocumentAnalysisService:
    """FastAPI dependency that provides the :class:`DocumentAnalysisService` singleton."""
    if DocumentAnalysisService._instance is None:
        DocumentAnalysisService._instance = DocumentAnalysisService(llm_service=llm_service)
    return DocumentAnalysisService._instance

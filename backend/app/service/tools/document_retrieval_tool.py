"""
Pydantic AI tools for document retrieval used by the chat agent.

These are plain Python functions that the Pydantic AI ``Agent`` can call
when the model decides it needs more context from the document.  They read
from the pre-computed JSON artefacts stored under ``data_storage_path / <document_name>/``.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import List

import faiss
import numpy as np
from pydantic_ai import RunContext

from app.config.config import settings
from app.infra.llm_connector import LLMService

logger = logging.getLogger("app.service.tools")


def _get_llm_service() -> LLMService:
    """Lazily resolve the LLMService singleton from the application container."""
    from app.container import container  # late import to avoid circular deps
    return container.llm_service()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json(file_path: Path, description: str) -> object:
    """Load a JSON file, raising ``RuntimeError`` on failure."""
    if not file_path.exists():
        msg = f"{description} not found: {file_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {file_path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read {file_path}: {exc}") from exc


def _chunk_embeddings(document_name: str) -> List[List[float]]:
    path = settings.data_storage_path / document_name / f"{document_name}{settings.suffix_chunk_embeddings}"
    return _load_json(path, "Chunk embeddings")  # type: ignore[return-value]


def _all_chunks(document_name: str) -> List[str]:
    path = settings.data_storage_path / document_name / f"{document_name}{settings.suffix_chunks}"
    return _load_json(path, "Chunks")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tools (plain functions — registered with pydantic_ai.Agent)
# ---------------------------------------------------------------------------

async def get_the_most_relevant_chunks(ctx: RunContext[None], question: str, document_name: str) -> List[str]:
    """Retrieve the most relevant text passages from the document based on
    the input question. Returns a list of relevant text chunks."""
    try:
        chunk_embeddings = _chunk_embeddings(document_name)
        dimension = len(chunk_embeddings[0])
        vectors = np.array(chunk_embeddings, dtype="float32")

        index = faiss.IndexFlatL2(dimension)
        index.add(n=vectors.shape[0], x=vectors)

        query_vec = np.array(
            [await _get_llm_service().embed_text(text=question)], dtype="float32"
        )
        n_queries = query_vec.shape[0]
        distances = np.empty((n_queries, settings.top_k_chunks), dtype="float32")
        raw_indices = np.empty((n_queries, settings.top_k_chunks), dtype="int64")
        index.search(n=n_queries, x=query_vec, k=settings.top_k_chunks, distances=distances, labels=raw_indices)

        all_chunks = _all_chunks(document_name)
        result = [all_chunks[i] for i in raw_indices[0] if i < len(all_chunks)]
        logger.info("Returned %d relevant chunks for query.", len(result))
        return result

    except Exception as exc:
        logger.error(
            "get_the_most_relevant_chunks failed: %s\n%s", exc, traceback.format_exc()
        )
        raise


def get_document_summary(ctx: RunContext[None], document_name: str) -> str:
    """Return a high-level summary of the entire document."""
    try:
        path = settings.data_storage_path / document_name / f"{document_name}{settings.suffix_chapter_summaries}"

        if not path.exists():
            logger.warning("Chapter summaries not found for '%s'.", document_name)
            return (
                "Document summary is not available — chapter summaries have not "
                "been generated for this document yet."
            )

        chapters: List[str] = _load_json(path, "Chapter summaries")  # type: ignore[assignment]
        summary = "\n".join(chapters)
        logger.info("Document summary: %d chars", len(summary))
        return summary

    except Exception as exc:
        logger.error(
            "get_document_summary failed: %s\n%s", exc, traceback.format_exc()
        )
        raise

def word_count_tool(ctx: RunContext[None], text: str) -> int:
    """Utility function to count total number of words in a text."""
    return len(text.split())
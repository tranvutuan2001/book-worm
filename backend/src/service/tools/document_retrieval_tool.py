"""
LangChain tools for document retrieval used by the chat agent.

These tools are passed to the LLM agent and called automatically when the
model decides it needs more context from the document.  They read from the
pre-computed JSON artefacts stored under ``DATA_DIR / <document_name>/``.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain.tools import tool

from src.core.config import DATA_DIR, TOP_K_CHUNKS
from src.infra.llm_connector.mlx_embedding import MLXEmbeddingModel

logger = logging.getLogger("app.service.tools")


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
    path = DATA_DIR / document_name / f"{document_name}_chunk_embeddings.json"
    return _load_json(path, "Chunk embeddings")  # type: ignore[return-value]


def _all_chunks(document_name: str) -> List[str]:
    path = DATA_DIR / document_name / f"{document_name}_chunks.json"
    return _load_json(path, "Chunks")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def make_retrieval_tools(embedding_model: str) -> Tuple:
    """Create document retrieval tools bound to the given embedding model.

    Args:
        embedding_model: Path/name of the embedding model to use for semantic search.

    Returns:
        Tuple of ``(get_the_most_relevant_chunks, get_document_summary)`` LangChain tools.
    """

    @tool(
        description=(
            "Retrieve the most relevant text passages from the document based on "
            "the input question. Returns a list of relevant text chunks."
        )
    )
    def get_the_most_relevant_chunks(question: str, document_name: str) -> List[str]:
        """Semantic search over pre-computed chunk embeddings using FAISS."""
        try:
            if not embedding_model:
                raise RuntimeError("No embedding model configured.")

            chunk_embeddings = _chunk_embeddings(document_name)
            dimension = len(chunk_embeddings[0])
            vectors = np.array(chunk_embeddings, dtype="float32")

            index = faiss.IndexFlatL2(dimension)
            index.add(vectors)

            query_vec = np.array(
                [MLXEmbeddingModel(embedding_model).embed(question)], dtype="float32"
            )
            _, indices = index.search(query_vec, TOP_K_CHUNKS)

            all_chunks = _all_chunks(document_name)
            result = [all_chunks[i] for i in indices[0] if i < len(all_chunks)]
            logger.info("Returned %d relevant chunks for query.", len(result))
            return result

        except Exception as exc:
            logger.error(
                "get_the_most_relevant_chunks failed: %s\n%s", exc, traceback.format_exc()
            )
            raise

    @tool(description="Return a high-level summary of the entire document.")
    def get_document_summary(document_name: str) -> str:
        """Read chapter-level summaries and join them into a single summary."""
        try:
            path = DATA_DIR / document_name / f"{document_name}_chapter_summaries.json"

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

    return get_the_most_relevant_chunks, get_document_summary

import json
from pathlib import Path
from typing import List
import faiss
import numpy as np
from langchain.tools import tool
from app.infra.llm_connector.llm_client import embed_text
from app.infra.session_manager import session_manager

# Project root directory (where main.py is located)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_chunk_embedding(document_name: str) -> List[List[float]]:
    file_path = PROJECT_ROOT / "0_data" / document_name / f"{document_name}_chunk_embeddings.json"

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    try:
        with file_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to read/parse {file_path}: {exc}")


def get_all_chunk(document_name: str) -> List[str]:
    file_path = PROJECT_ROOT / "0_data" / document_name / f"{document_name}_chunks.json"

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to read/parse {file_path}: {exc}")


@tool(
    description="Get the most relevant texts from the document based on the input question. Return a list of relevant texts."
)
def get_the_most_relevant_chunks(question: str) -> List[str]:
    document_name = session_manager.get_current_document_name()
    embedding_model = "Qwen/Qwen3-Embedding-4B-GGUF" # TODO: make this configurable at session level if we want to support different embedding models in the future
    chunk_embedding = get_chunk_embedding(document_name)

    embedding_dimension = len(chunk_embedding[0])
    chunk_embeddings_np = np.array([ch for ch in chunk_embedding]).astype("float32")
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(chunk_embeddings_np)

    embedded_question = embed_text(question, model_name=embedding_model)
    query_vec = np.array([embedded_question], dtype="float32")

    k = 3  # number of chunks you want returned
    _, indices = index.search(query_vec, k)
    all_chunk = get_all_chunk(document_name)
    res = [all_chunk[i] for i in indices[0]]
    return res

@tool(
    description="Get a high level summary of the entire document."
)
def get_document_summary() -> str:
    document_name = session_manager.get_current_document_name()
    file_path = PROJECT_ROOT / "0_data" / document_name / f"{document_name}_chapter_summaries.json"
    all_chapters = None

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            all_chapters = json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {file_path}: {exc}")
    
    return "\n".join(all_chapters)

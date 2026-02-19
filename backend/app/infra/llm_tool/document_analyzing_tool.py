import json
import logging
import traceback
from pathlib import Path
from typing import List
import faiss
import numpy as np
from langchain.tools import tool
from app.infra.llm_connector.llm_client import embed_text
from app.infra.session_manager import session_manager

logger = logging.getLogger('app.llm_tool')

# Project root directory (where main.py is located)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_chunk_embedding(document_name: str) -> List[List[float]]:
    file_path = PROJECT_ROOT / "0_data" / document_name / f"{document_name}_chunk_embeddings.json"

    if not file_path.exists():
        error_msg = f"Chunk embeddings file not found: {file_path}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise FileNotFoundError(f"File {file_path} does not exist")

    try:
        logger.info(f"Reading chunk embeddings from: {file_path}")
        with file_path.open("r", encoding="utf-8") as fh:
            embeddings = json.load(fh)
        logger.info(f"Successfully loaded {len(embeddings)} chunk embeddings")
        return embeddings
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format in {file_path}: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise RuntimeError(f"Failed to parse {file_path}: {exc}")
    except Exception as exc:
        error_msg = f"Failed to read chunk embeddings from {file_path}: {str(exc)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to read/parse {file_path}: {exc}")


def get_all_chunk(document_name: str) -> List[str]:
    file_path = PROJECT_ROOT / "0_data" / document_name / f"{document_name}_chunks.json"

    if not file_path.exists():
        error_msg = f"Chunks file not found: {file_path}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    try:
        logger.info(f"Reading chunks from: {file_path}")
        with file_path.open("r", encoding="utf-8") as fh:
            chunks = json.load(fh)
        logger.info(f"Successfully loaded {len(chunks)} chunks")
        return chunks
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format in {file_path}: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise RuntimeError(f"Failed to parse {file_path}: {exc}")
    except Exception as exc:
        error_msg = f"Failed to read chunks from {file_path}: {str(exc)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to read/parse {file_path}: {exc}")


@tool(
    description="Get the most relevant texts from the document based on the input question. Return a list of relevant texts."
)
def get_the_most_relevant_chunks(question: str) -> List[str]:
    try:
        logger.info(f"Searching for relevant chunks for question: {question[:100]}...")
        
        try:
            document_name = session_manager.get_current_document_name()
            logger.info(f"Using document: {document_name}")
        except Exception as e:
            error_msg = f"Failed to get current document name: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise
        
        embedding_model = "Qwen/Qwen3-Embedding-4B-GGUF" # TODO: make this configurable at session level if we want to support different embedding models in the future
        
        try:
            chunk_embedding = get_chunk_embedding(document_name)
        except Exception as e:
            error_msg = f"Failed to load chunk embeddings: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise

        try:
            embedding_dimension = len(chunk_embedding[0])
            chunk_embeddings_np = np.array([ch for ch in chunk_embedding]).astype("float32")
            logger.info(f"Building FAISS index with {len(chunk_embedding)} vectors of dimension {embedding_dimension}")
            index = faiss.IndexFlatL2(embedding_dimension)
            index.add(chunk_embeddings_np)
        except Exception as e:
            error_msg = f"Failed to build FAISS index: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

        try:
            embedded_question = embed_text(question, model_name=embedding_model)
            query_vec = np.array([embedded_question], dtype="float32")
        except Exception as e:
            error_msg = f"Failed to embed question: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise

        try:
            k = 3  # number of chunks you want returned
            _, indices = index.search(query_vec, k)
            logger.info(f"Found {k} relevant chunks at indices: {indices[0].tolist()}")
        except Exception as e:
            error_msg = f"FAISS search failed: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise
        
        try:
            all_chunk = get_all_chunk(document_name)
            res = [all_chunk[i] for i in indices[0]]
            logger.info(f"Successfully retrieved {len(res)} relevant chunks")
            return res
        except Exception as e:
            error_msg = f"Failed to retrieve chunks: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise
    
    except Exception as e:
        error_msg = f"Unexpected error in get_the_most_relevant_chunks: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise

@tool(
    description="Get a high level summary of the entire document."
)
def get_document_summary() -> str:
    try:
        logger.info("Retrieving document summary")
        
        try:
            document_name = session_manager.get_current_document_name()
            logger.info(f"Using document: {document_name}")
        except Exception as e:
            error_msg = f"Failed to get current document name: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise
        
        file_path = PROJECT_ROOT / "0_data" / document_name / f"{document_name}_chapter_summaries.json"

        if not file_path.exists():
            error_msg = f"Chapter summaries file not found: {file_path}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        try:
            logger.info(f"Reading chapter summaries from: {file_path}")
            with file_path.open("r", encoding="utf-8") as fh:
                all_chapters = json.load(fh)
            logger.info(f"Successfully loaded {len(all_chapters)} chapter summaries")
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format in {file_path}: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise RuntimeError(f"Failed to parse {file_path}: {exc}")
        except Exception as exc:
            error_msg = f"Failed to read chapter summaries from {file_path}: {str(exc)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to read {file_path}: {exc}")
        
        summary = "\n".join(all_chapters)
        logger.info(f"Document summary generated: {len(summary)} characters")
        return summary
    
    except Exception as e:
        error_msg = f"Unexpected error in get_document_summary: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise

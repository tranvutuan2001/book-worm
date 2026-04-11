"""
Application-wide configuration and path constants.

All magic strings for paths, model names, and processing parameters live here
so that the rest of the codebase references named constants rather than bare
string literals.
"""

from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# Absolute path to the backend/ project root (where main.py lives)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "0_data"
PDF_DIR: Path = PROJECT_ROOT / "pdf"
MODELS_DIR: Path = PROJECT_ROOT / "models"
CHAT_MODELS_DIR: Path = MODELS_DIR / "chat"
EMBEDDING_MODELS_DIR: Path = MODELS_DIR / "embedding"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Location of JSON schema / example for PDF output
PDF_SCHEMA_PATH: Path = PROJECT_ROOT / "pdf-schema.json"
PDF_EXAMPLE_PATH: Path = PROJECT_ROOT / "pdf-example.json"

# ---------------------------------------------------------------------------
# Default model paths (relative to PROJECT_ROOT)
# ---------------------------------------------------------------------------

DEFAULT_CHAT_MODEL: str = "models/chat/mlx-community/Qwen3.5-9B-MLX-4bit"
DEFAULT_EMBEDDING_MODEL: str = "models/embedding/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

# ---------------------------------------------------------------------------
# Document analysis parameters
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 100
CHUNKS_PER_SECTION: int = 5
SECTIONS_PER_CHAPTER: int = 10
MAX_EMBEDDING_RETRIES: int = 3

# ---------------------------------------------------------------------------
# LLM inference parameters
# ---------------------------------------------------------------------------

CHAT_MAX_TOKENS: int = 2048
CHAT_TEMPERATURE: float = 0.1
DEFAULT_CHAT_TEMPLATE: str = "qwen"
TOP_K_CHUNKS: int = 3

# ---------------------------------------------------------------------------
# File-name suffixes used when persisting analysis artefacts
# ---------------------------------------------------------------------------

SUFFIX_CHUNKS = "_chunks.json"
SUFFIX_CHUNK_EMBEDDINGS = "_chunk_embeddings.json"
SUFFIX_SECTION_SUMMARIES = "_section_summaries.json"
SUFFIX_SECTION_EMBEDDINGS = "_section_summary_embeddings.json"
SUFFIX_CHAPTER_SUMMARIES = "_chapter_summaries.json"
SUFFIX_CHAPTER_EMBEDDINGS = "_chapter_summary_embeddings.json"

# ---------------------------------------------------------------------------
# LLM backend selection
# ---------------------------------------------------------------------------

# Which inference backend to use for chat completions and embeddings.
#   "local"     — MLX models loaded directly in-process (Apple Silicon only).
#   "lm_studio" — Remote LM Studio instance via OpenAI-compatible HTTP API.
LLM_BACKEND: Literal["local", "lm_studio"] = "lm_studio"

# ---------------------------------------------------------------------------
# LM Studio connection settings
# (only used when LLM_BACKEND = "lm_studio")
# ---------------------------------------------------------------------------

LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"

# LM Studio accepts any non-empty string as the API key.
LM_STUDIO_API_KEY: str = "lm-studio"

# Model identifiers exactly as they appear in LM Studio's model list.
# Leave empty to fall back to the model_path argument passed by the caller.
LM_STUDIO_DEFAULT_CHAT_MODEL: str = "qwen3.5-9b-mlx"
LM_STUDIO_DEFAULT_EMBEDDING_MODEL: str = "text-embedding-qwen3-embedding-0.6b"

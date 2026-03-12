"""
Application-wide configuration and path constants.

All magic strings for paths, model names, and processing parameters live here
so that the rest of the codebase references named constants rather than bare
string literals.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# Absolute path to the backend/ project root (where main.py lives)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "0_data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
CHAT_MODELS_DIR: Path = MODELS_DIR / "chat"
EMBEDDING_MODELS_DIR: Path = MODELS_DIR / "embedding"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Default model paths (relative to PROJECT_ROOT)
# ---------------------------------------------------------------------------

DEFAULT_CHAT_MODEL: str = "models/chat/mlx-community/Qwen3.5-35B-A3B-4bit"
DEFAULT_EMBEDDING_MODEL: str = "models/embedding/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

# ---------------------------------------------------------------------------
# Document analysis parameters
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 2000
CHUNK_OVERLAP: int = 100
CHUNKS_PER_SECTION: int = 10
SECTIONS_PER_CHAPTER: int = 5
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

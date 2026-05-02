from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSetting(BaseSettings):
    """
    Central configuration for the Book-Worm application.

    This class uses Pydantic Settings to manage configuration from environment
    variables and default values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allow uppercase env vars to map to lowercase fields
        extra="ignore",
    )

    # --- Storage & Paths ---
    project_root: Path = Path(__file__).resolve().parents[2]

    # We use Field(default_factory=...) to allow environment overrides while 
    # maintaining relative paths as defaults.
    data_storage_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "0_data"
    )
    pdf_storage_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "pdf"
    )
    logs_storage_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "logs"
    )

    # --- Document Processing ---
    document_chunk_size: int = 1000
    document_chunk_overlap: int = 100
    chunks_per_section: int = 10
    sections_per_chapter: int = 15
    max_embedding_retries: int = 3

    # --- LLM Inference Defaults ---
    chat_max_tokens: int | None = None
    chat_temperature: float = 0.2
    top_k_chunks: int = 3

    # --- Model Artifact Suffixes ---
    suffix_chunks: str = "_chunks.json"
    suffix_chunk_embeddings: str = "_chunk_embeddings.json"
    suffix_section_summaries: str = "_section_summaries.json"
    suffix_section_embeddings: str = "_section_summary_embeddings.json"
    suffix_chapter_summaries: str = "_chapter_summaries.json"
    suffix_chapter_embeddings: str = "_chapter_summary_embeddings.json"

    # --- Provider Configuration (llm-server only) ---
    llm_server_url: str = "http://localhost:8001/v1"

    # --- Computed Properties ---
    @property
    def pdf_schema_path(self) -> Path:
        return self.project_root / "pdf-schema.json"

    @property
    def pdf_example_path(self) -> Path:
        return self.project_root / "pdf-example.json"


# Export a singleton instance
app_setting = AppSetting()
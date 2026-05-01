from pydantic_settings import BaseSettings, SettingsConfigDict
from app.domain.value_objects.llm_backend import LLMBackend

class Settings(BaseSettings):
    LLM_BACKEND: LLMBackend

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str | None = None
    MLX_MODEL_PATH: str | None = None
    MLX_CHAT_MODEL_PATH: str | None = None
    MLX_EMBEDDING_MODEL_PATH: str | None = None
    
    LANGFUSE_PUBLIC_KEY: str | None
    LANGFUSE_SECRET_KEY: str | None
    LANGFUSE_HOST: str

    HOST: str
    PORT: int

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from app.domain.value_objects.llm_backend import LLMBackend

class Settings(BaseSettings):
    LLM_BACKEND: LLMBackend

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[str] = None
    MLX_MODEL_PATH: Optional[str] = None
    MLX_CHAT_MODEL_PATH: Optional[str] = None
    MLX_EMBEDDING_MODEL_PATH: Optional[str] = None
    
    LANGFUSE_PUBLIC_KEY: Optional[str]
    LANGFUSE_SECRET_KEY: Optional[str]
    LANGFUSE_HOST: str

    HOST: str
    PORT: int

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

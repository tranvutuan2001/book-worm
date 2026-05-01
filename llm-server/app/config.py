from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from app.domain.models import LLMBackend

class Settings(BaseSettings):
    LLM_BACKEND: LLMBackend

    OPENAI_API_KEY: Optional[str]
    OPENAI_MODEL: Optional[str]
    MLX_MODEL_PATH: str
    
    LANGFUSE_PUBLIC_KEY: Optional[str]
    LANGFUSE_SECRET_KEY: Optional[str]
    LANGFUSE_HOST: str

    HOST: str
    PORT: int

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

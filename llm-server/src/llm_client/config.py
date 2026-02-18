from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )
    
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    n_threads: int = 4
    temperature: float = 0.7
    max_tokens: int = 512
    verbose: bool = False


settings = Settings()

from enum import Enum

class LLMBackend(str, Enum):
    """Domain concept for the supported LLM backend providers."""
    OPENAI = "openai"
    MLX = "mlx"

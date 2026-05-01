import mlx_lm
from typing import List
from app.domain.models import Message
from app.domain.protocols import LLMProvider
from app.domain.exceptions import LLMGenerationException
from app.infrastructure.mlx_model import MLXModel

class MLXLLMProvider(LLMProvider):
    """MLX-based implementation of the LLMProvider."""
    
    def __init__(self, mlx_model: MLXModel):
        self.mlx_model = mlx_model

    async def generate(self, messages: List[Message], max_tokens: int) -> str:
        try:
            # Simple prompt construction for MLX
            prompt = "\n".join([f"{m.role.value}: {m.content}" for m in messages])
            prompt += "\nassistant: "
            
            # Running directly to avoid 'no Stream' errors with asyncio.to_thread
            response = mlx_lm.generate(
                model=self.mlx_model.model,
                tokenizer=self.mlx_model.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
            return response
        except Exception as e:
            raise LLMGenerationException(
                message=str(e),
                provider="mlx",
                original_error=e
            )

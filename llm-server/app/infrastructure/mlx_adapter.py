import mlx_lm
from typing import List
from app.domain.models import Message
from app.domain.protocols import LLMProvider
from app.domain.exceptions import LLMGenerationException
import asyncio

class MLXProvider(LLMProvider):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Model loading can be deferred or done here
        self.model, self.tokenizer = mlx_lm.load(model_path)

    async def generate(self, messages: List[Message], max_tokens: int) -> str:
        try:
            # Simple prompt construction for MLX
            prompt = "\n".join([f"{m.role.value}: {m.content}" for m in messages])
            prompt += "\nassistant: "
            
            # Running in executor to avoid blocking event loop
            response = await asyncio.to_thread(
                mlx_lm.generate,
                model=self.model,
                tokenizer=self.tokenizer,
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

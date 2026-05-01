import mlx_lm
import mlx.core as mx
from typing import List
from app.domain.models import Message
from app.domain.protocols import LLMProvider, EmbeddingProvider
from app.domain.exceptions import LLMGenerationException
import asyncio

class MLXModel:
    """Shared resource for MLX model and tokenizer."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model, self.tokenizer = mlx_lm.load(model_path)

class MLXLLMProvider(LLMProvider):
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

class MLXEmbeddingProvider(EmbeddingProvider):
    def __init__(self, mlx_model: MLXModel):
        self.mlx_model = mlx_model

    async def embed(self, text: str) -> List[float]:
        try:
            # Tokenize text
            tokens = self.mlx_model.tokenizer.encode(text)
            tokens_mx = mx.array([tokens])
            
            # Get embeddings from the model
            def _get_embedding():
                output = self.mlx_model.model(tokens_mx)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Take the mean of all token embeddings
                embedding = mx.mean(hidden_states, axis=1)
                return embedding.tolist()[0]

            embedding = _get_embedding()
            return embedding
        except Exception as e:
            raise LLMGenerationException(
                message=f"Failed to generate embedding: {str(e)}",
                provider="mlx",
                original_error=e
            )

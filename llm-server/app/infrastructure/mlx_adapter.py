import mlx_lm
import mlx.core as mx
from typing import List
from app.domain.models import Message
from app.domain.protocols import LLMProvider, EmbeddingProvider
from app.domain.exceptions import LLMGenerationException
import asyncio

class MLXProvider(LLMProvider, EmbeddingProvider):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Model loading can be deferred or done here
        self.model, self.tokenizer = mlx_lm.load(model_path)

    async def generate(self, messages: List[Message], max_tokens: int) -> str:
        try:
            # Simple prompt construction for MLX
            prompt = "\n".join([f"{m.role.value}: {m.content}" for m in messages])
            prompt += "\nassistant: "
            
            # Running directly to avoid 'no Stream' errors with asyncio.to_thread
            response = mlx_lm.generate(
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

    async def embed(self, text: str) -> List[float]:
        try:
            # Tokenize text
            tokens = self.tokenizer.encode(text)
            tokens_mx = mx.array([tokens])
            
            # Get embeddings from the model
            # Note: This is a simplified implementation that extracts the last hidden state
            # for the last token. This might vary by model architecture.
            def _get_embedding():
                output = self.model(tokens_mx)
                # output is typically a tuple or a tensor depending on the model
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Take the last token's hidden state and average across sequence or take last
                # For simplicity, we take the mean of all token embeddings
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

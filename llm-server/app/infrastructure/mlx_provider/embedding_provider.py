import mlx.core as mx
from app.domain.protocols.embedding_provider import EmbeddingProvider
from app.domain.exceptions.llm_exception import LLMGenerationException
from app.infrastructure.mlx_provider.model_loader import MLXModel

class MLXEmbeddingProvider(EmbeddingProvider):
    """MLX-based implementation of the EmbeddingProvider."""
    
    def __init__(self, mlx_model: MLXModel):
        self.mlx_model = mlx_model

    async def embed(self, text: str) -> list[float]:
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

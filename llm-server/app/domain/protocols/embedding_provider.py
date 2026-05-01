from typing import Protocol

class EmbeddingProvider(Protocol):
    """Protocol defining the contract for generating text embeddings."""
    async def embed(self, text: str) -> list[float]:
        """Generates a vector embedding for the given text."""
        ...

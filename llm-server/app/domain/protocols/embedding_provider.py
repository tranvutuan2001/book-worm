from typing import Protocol, List

class EmbeddingProvider(Protocol):
    """Protocol defining the contract for generating text embeddings."""
    async def embed(self, text: str) -> List[float]:
        """Generates a vector embedding for the given text."""
        ...

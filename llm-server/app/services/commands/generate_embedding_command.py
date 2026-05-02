from dataclasses import dataclass

@dataclass(frozen=True)
class GenerateEmbeddingCommand:
    """Command to generate embeddings for given texts."""
    texts: list[str]

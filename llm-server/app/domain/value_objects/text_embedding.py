from pydantic import BaseModel

class TextEmbedding(BaseModel):
    """Value Object representing the result of a text embedding operation."""
    embedding: list[float]
    model: str

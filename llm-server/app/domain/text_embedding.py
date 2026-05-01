from pydantic import BaseModel
from typing import List

class TextEmbedding(BaseModel):
    """Value Object representing the result of a text embedding operation."""
    embedding: List[float]
    model: str

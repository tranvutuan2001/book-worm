from pydantic import BaseModel
from typing import Optional

class EmbeddingContext(BaseModel):
    """Value Object representing the context for an embedding request."""
    input: str
    model_name: Optional[str] = None

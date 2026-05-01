from pydantic import BaseModel

class EmbeddingContext(BaseModel):
    """Value Object representing the context for an embedding request."""
    input: str
    model_name: str | None = None

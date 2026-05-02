from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    """Request schema for OpenAI-compatible embeddings."""
    input: str | list[str]
    model: str

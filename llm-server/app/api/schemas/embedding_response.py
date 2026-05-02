from pydantic import BaseModel, Field

class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]

class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0

class EmbeddingResponse(BaseModel):
    """Response schema for OpenAI-compatible embeddings."""
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage = Field(default_factory=Usage)

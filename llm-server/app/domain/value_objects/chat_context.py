from pydantic import BaseModel, Field
from app.domain.value_objects.message import Message

class ChatContext(BaseModel):
    """Value Object representing the input context for a chat generation request."""
    messages: list[Message]
    model: str | None = None
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    tools: list[dict[str, object]] | None = None

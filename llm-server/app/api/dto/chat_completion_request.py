from pydantic import BaseModel, Field
from app.domain.value_objects.message import Message

class ChatCompletionRequest(BaseModel):
    """Request schema for OpenAI-compatible chat completions."""
    model: str
    messages: list[Message]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    tools: list[dict[str, object]] | None = None
    stream: bool | None = False

import time
from pydantic import BaseModel, Field
from app.domain.value_objects.message import Message

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    """Response schema for OpenAI-compatible chat completions."""
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage = Field(default_factory=Usage)

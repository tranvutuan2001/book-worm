from pydantic import BaseModel, Field
from typing import List
from app.domain.message import Message

class ChatContext(BaseModel):
    """Value Object representing the input context for a chat generation request."""
    messages: List[Message]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

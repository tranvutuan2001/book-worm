from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: Role
    content: str

class CompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class EmbeddingRequest(BaseModel):
    input: str
    model: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str

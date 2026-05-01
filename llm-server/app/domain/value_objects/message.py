from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.domain.value_objects.message_role import MessageRole

class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction

class Message(BaseModel):
    """Value Object representing a single message in a conversation."""
    role: MessageRole
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

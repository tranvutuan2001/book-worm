from pydantic import BaseModel
from app.domain.value_objects.message_role import MessageRole

class Message(BaseModel):
    """Value Object representing a single message in a conversation."""
    role: MessageRole
    content: str

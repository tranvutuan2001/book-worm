from pydantic import BaseModel, Field
from typing import List
from uuid import UUID, uuid4
from app.domain.value_objects.message import Message

class ChatSession(BaseModel):
    """Entity representing a persistent chat session with a unique identity."""
    id: UUID = Field(default_factory=uuid4)
    messages: List[Message] = Field(default_factory=list)
    
    def append_message(self, message: Message) -> None:
        """Adds a new message to the session."""
        self.messages.append(message)

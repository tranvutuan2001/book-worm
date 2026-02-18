from pydantic import BaseModel, Field
from app.constant import Role


class Message(BaseModel):
    id: str = Field(description="Unique identifier for the message", example="msg_123")
    content: str = Field(description="The content/text of the message", example="What is the main theme of this document?")
    role: Role = Field(description="The role of the message sender")
    timestamp: int = Field(description="Unix timestamp when message was created", example=1674567890)

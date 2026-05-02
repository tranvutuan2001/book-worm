from pydantic import BaseModel, Field
from app.domain.entity.message import Message

class AskRequest(BaseModel):
    """Payload for the /ask endpoint."""
    id: str = Field(description="Unique identifier for the conversation", example="conv_123")
    message_list: list[Message] = Field(description="List of messages in the conversation")
    timestamp: int = Field(description="Unix timestamp when conversation was created", example=1674567890)
    document_name: str | None = Field(
        default=None, 
        description="Name of the document to query", 
        example="sample_document"
    )

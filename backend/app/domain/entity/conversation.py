from pydantic import BaseModel, Field
from .message import Message
from typing import List, Optional


class Conversation(BaseModel):
    id: str = Field(description="Unique identifier for the conversation", example="conv_123")
    message_list: List[Message] = Field(description="List of messages in the conversation")
    timestamp: int = Field(description="Unix timestamp when conversation was created", example=1674567890)
    document_name: Optional[str] = Field(
        default=None, 
        description="Name of the document to query (required for /ask endpoint)", 
        example="sample_document_20240120_143022"
    )
    chat_model: str = Field(
        description="Name of the chat model to use for generating responses",
        example="qwen/qwen3-30b-a3b-2507"
    )
    embedding_model: str = Field(
        description="Name of the embedding model to use for text embeddings",
        example="text-embedding-3-large"
    )

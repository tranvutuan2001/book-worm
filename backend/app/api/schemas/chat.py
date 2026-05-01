"""Request / response schemas for the chat (document Q&A) endpoints."""

from pydantic import BaseModel, Field


class AskResponse(BaseModel):
    """Successful response returned by the ``POST /ask`` endpoint."""

    message: str = Field(
        description="AI-generated answer based on document content",
        example="The main theme of this document is...",
    )
    conversation_id: str = Field(
        description="Conversation identifier",
        example="a27z",
    )
    timestamp: int = Field(
        description="Response timestamp (Unix ms)",
        example=1710000000000,
    )


class ErrorResponse(BaseModel):
    """Generic error envelope."""

    detail: str = Field(
        description="Human-readable error message",
        example="Document not found",
    )

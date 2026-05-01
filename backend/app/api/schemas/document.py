"""Request / response schemas for the document management endpoints."""

from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    READY = "ready"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    ERROR = "error"


class UploadResponse(BaseModel):
    """Returned after a successful document upload."""

    message: str = Field(
        description="Success message",
        example="Document uploaded successfully and analysis started",
    )
    document_name: str = Field(
        description="Generated document name with timestamp",
        example="sample_document_20240120_143022",
    )
    status: DocumentStatus = Field(
        description="Current processing status of the document",
    )


class DocumentInfo(BaseModel):
    """Metadata for a single stored document."""

    name: str = Field(
        description="Document name with timestamp",
        example="sample_document_20240120_143022",
    )
    status: DocumentStatus = Field(
        description="Current processing status",
    )
    path: str = Field(
        description="Filesystem path to document folder",
        example="/path/to/0_data/sample_document_20240120_143022",
    )


class DocumentsResponse(BaseModel):
    """Returned by the ``GET /documents`` endpoint."""

    documents: list[DocumentInfo] = Field(
        description="List of all available documents",
    )

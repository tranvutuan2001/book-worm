from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class DocumentStatus(str, Enum):
    READY = "ready"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    ERROR = "error"


class UploadResponse(BaseModel):
    message: str = Field(description="Success message", example="Document uploaded successfully and analysis started")
    document_name: str = Field(description="Generated document name with timestamp", example="sample_document_20240120_143022")
    status: DocumentStatus = Field(description="Current processing status of the document")
    

class DocumentInfo(BaseModel):
    name: str = Field(description="Document name with timestamp", example="sample_document_20240120_143022")
    status: DocumentStatus = Field(description="Current processing status")
    path: str = Field(description="File system path to document folder", example="/path/to/0_data/sample_document_20240120_143022")


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo] = Field(description="List of all available documents")


class AskResponse(BaseModel):
    message: str = Field(description="AI-generated answer based on document content", example="The main theme of this document is...")
    conversation_id: str = Field(description="Conversation identifier", example="a27z")
    timestamp: int = Field(description="Response timestamp", example=123456789)


class ErrorResponse(BaseModel):
    detail: str = Field(description="Error message", example="Document not found")
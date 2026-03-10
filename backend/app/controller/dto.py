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


# ---------------------------------------------------------------------------
# Model Management DTOs
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    name: str = Field(description="Model name (directory name)")
    path: str = Field(description="Relative path from models directory")
    size: str = Field(description="Model size in human-readable format")
    status: str = Field(description="Model status: 'ready_to_use' or 'downloading'")


class DownloadableModelInfo(BaseModel):
    name: str = Field(description="Model display name")
    repository: str = Field(description="Hugging Face repository ID")
    filename: str = Field(description="Model directory / filename identifier")


class LoadedModelInfo(BaseModel):
    model_name: str = Field(description="Model name")
    model_path: str = Field(description="Full path to model directory")
    model_type: str = Field(description="Type: 'chat' or 'embedding'")
    loaded: bool = Field(description="Whether model is currently loaded in memory")


class ModelDownloadRequest(BaseModel):
    repository: str = Field(description="Hugging Face repository ID (e.g. 'mlx-community/Qwen3-4B-Instruct-4bit')")


class ModelDownloadResponse(BaseModel):
    repository: str = Field(description="Hugging Face repository ID")
    status: str = Field(description="Download status")
    path: str = Field(description="Local path where model will be stored")
    message: str = Field(description="Status message")


class ModelLoadRequest(BaseModel):
    model_path: str = Field(description="Relative path to the model directory within its type subdirectory (e.g. 'mlx-community/Qwen3-4B-Instruct-4bit')")
    model_type: str = Field(description="Type of model: 'chat' or 'embedding'")


class ModelLoadResponse(BaseModel):
    model: str = Field(description="Model name")
    model_type: str = Field(description="Type: 'chat' or 'embedding'")
    status: str = Field(description="Status: 'loaded' or 'already_loaded'")
    message: str = Field(description="Status message")
    model_path: str = Field(description="Full path to model directory")


class ModelUnloadRequest(BaseModel):
    model_path: str = Field(description="Relative path to the model directory within its type subdirectory (e.g. 'mlx-community/Qwen3-4B-Instruct-4bit')")
    model_type: str = Field(description="Type of model: 'chat' or 'embedding'")


class ModelUnloadResponse(BaseModel):
    model_path: str = Field(description="Relative path to the model directory")
    model_type: str = Field(description="Type: 'chat' or 'embedding'")
    status: str = Field(description="Status: 'unloaded' or 'not_loaded'")
    message: str = Field(description="Status message")
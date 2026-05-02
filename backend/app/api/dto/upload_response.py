from pydantic import BaseModel, Field
from app.api.dto.document_status import DocumentStatus

class UploadResponse(BaseModel):
    """Returned after a successful document upload."""
    message: str = Field(description="Success message")
    document_name: str = Field(description="Generated document name")
    status: DocumentStatus = Field(description="Current status")

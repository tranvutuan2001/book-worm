from pydantic import BaseModel, Field
from app.api.dto.document_status import DocumentStatus

class DocumentInfo(BaseModel):
    """Metadata for a single stored document."""
    name: str = Field(description="Document name")
    status: DocumentStatus = Field(description="Current status")
    path: str = Field(description="Filesystem path")

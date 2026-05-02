from pydantic import BaseModel, Field
from app.api.dto.document_info import DocumentInfo

class DocumentsResponse(BaseModel):
    """Returned by the GET /documents endpoint."""
    documents: list[DocumentInfo] = Field(description="List of all available documents")

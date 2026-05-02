from dataclasses import dataclass
from app.domain.enum.document_status import DocumentStatus

@dataclass(frozen=True)
class DocumentUploadResult:
    """Result of a document upload operation."""
    document_name: str
    status: DocumentStatus

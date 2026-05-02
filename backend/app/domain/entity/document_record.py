from dataclasses import dataclass
from app.domain.enum.document_status import DocumentStatus

@dataclass(frozen=True)
class DocumentRecord:
    """Identity of a document stored in the system."""
    name: str
    status: DocumentStatus
    path: str

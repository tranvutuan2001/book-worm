from dataclasses import dataclass, field
from app.domain.entity.document_record import DocumentRecord

@dataclass(frozen=True)
class DocumentListResult:
    """List of all documents in the system."""
    documents: list[DocumentRecord] = field(default_factory=list)

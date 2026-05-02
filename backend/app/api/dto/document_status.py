from enum import Enum

class DocumentStatus(str, Enum):
    """API representation of document status."""
    READY = "ready"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    ERROR = "error"

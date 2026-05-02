from enum import Enum

class DocumentStatus(str, Enum):
    """Status of a document in the processing pipeline."""
    READY = "ready"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    ERROR = "error"

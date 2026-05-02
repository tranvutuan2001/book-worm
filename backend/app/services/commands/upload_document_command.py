from dataclasses import dataclass

@dataclass(frozen=True)
class UploadDocumentCommand:
    """Command to upload a document."""
    filename: str
    content: bytes

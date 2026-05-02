from dataclasses import dataclass

@dataclass(frozen=True)
class AnalyzeDocumentCommand:
    """Command to pre-analyze a document."""
    pdf_path: str
    document_name: str

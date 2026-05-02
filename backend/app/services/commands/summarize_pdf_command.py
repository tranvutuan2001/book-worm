from dataclasses import dataclass

@dataclass(frozen=True)
class SummarizePDFCommand:
    """Command to summarize a document into a structured PDF JSON."""
    document_name: str

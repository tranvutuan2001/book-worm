from pydantic import BaseModel, Field
from typing import Any

class SummarizeResponse(BaseModel):
    """Successful response for document summarization."""
    document_name: str = Field(description="Name of the document")
    output_file: str = Field(description="Path to the generated JSON summary")
    content: list[Any] = Field(description="The structured PDF JSON content")

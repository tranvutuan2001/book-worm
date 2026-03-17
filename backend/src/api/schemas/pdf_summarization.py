"""Request / response schemas for the PDF summarization endpoint."""

from typing import Any

from pydantic import BaseModel, Field

class SummarizeResponse(BaseModel):
    """Returned after a successful summarization run."""

    document_name: str = Field(
        description="Name of the source document",
        example="my_book_20240120_143022",
    )
    output_file: str = Field(
        description="Absolute path to the generated PDF JSON file in the pdf/ folder",
        example="/path/to/pdf/my_book_20240120_143022_summary_20260315_120000.json",
    )
    content: list[Any] = Field(
        description=(
            "The generated PDF JSON array conforming to pdf-schema.json. "
            "Each element is a heading, paragraph, list, table, or image node."
        ),
    )

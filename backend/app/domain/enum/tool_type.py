from enum import Enum


class ToolType(str, Enum):
    """Available tool types that can be bound to a document."""
    DOCUMENT_SEARCH = "document_search"
    DOCUMENT_SUMMARY = "document_summary"

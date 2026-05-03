"""Factory for creating document-bound tool callables.

Exports :class:`ToolFactory` which maps a :class:`ToolType` and a document
name to a ready-to-use callable whose ``document_name`` parameter has been
pre-filled.
"""

from typing import Any, Callable

from app.domain.enum.tool_type import ToolType
from app.services.tools.tool_binder import bind_tool_to_document
from app.services.tools.document_retrieval_tool import (
    get_the_most_relevant_chunks,
    get_document_summary,
)

_TOOL_REGISTRY: dict[ToolType, Callable[..., Any]] = {
    ToolType.DOCUMENT_SEARCH: get_the_most_relevant_chunks,
    ToolType.DOCUMENT_SUMMARY: get_document_summary,
}


class ToolFactory:
    """Creates document-bound tool callables from a :class:`ToolType`."""

    @staticmethod
    def create(tool_type: ToolType, document_name: str) -> Callable[..., Any]:
        """Return a bound tool for the given *tool_type* and *document_name*.

        Raises:
            ValueError: If *tool_type* is not registered.
        """
        raw_tool = _TOOL_REGISTRY.get(tool_type)
        if raw_tool is None:
            raise ValueError(f"Unknown tool type: {tool_type}")
        return bind_tool_to_document(raw_tool, document_name)

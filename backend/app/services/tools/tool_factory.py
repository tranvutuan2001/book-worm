import functools
import inspect
from typing import Any, Callable

from app.domain.enum.tool_type import ToolType
from app.services.tools.document_retrieval_tool import (
    get_the_most_relevant_chunks,
    get_document_summary,
    get_document_title,
)

_TOOL_REGISTRY: dict[ToolType, Callable[..., Any]] = {
    ToolType.DOCUMENT_SEARCH: get_the_most_relevant_chunks,
    ToolType.DOCUMENT_SUMMARY: get_document_summary,
    ToolType.DOCUMENT_TITLE: get_document_title,
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
        
        return ToolFactory._bind_tool_to_document(raw_tool, document_name)

    @staticmethod
    def _bind_tool_to_document(tool: Callable[..., Any], document_name: str) -> Callable[..., Any]:
        """
        Bind a tool function to a specific document name by injecting it 
        as a keyword argument if the tool expects it.
        """
        # Capture the tool's signature once
        sig = inspect.signature(tool)
        has_doc_name = "document_name" in sig.parameters

        @functools.wraps(tool)
        async def wrapper(ctx: Any, **kwargs: Any) -> Any:
            # Inject document_name if the tool expects it and it wasn't provided
            if has_doc_name and "document_name" not in kwargs:
                kwargs["document_name"] = document_name
            
            if inspect.iscoroutinefunction(tool):
                return await tool(ctx, **kwargs)
            else:
                return tool(ctx, **kwargs)

        # Update signature to hide document_name from the LLM if it's there
        if has_doc_name:
            try:
                new_params = [
                    p for p in sig.parameters.values() 
                    if p.name != "document_name"
                ]
                wrapper.__signature__ = sig.replace(parameters=new_params) # type: ignore
            except Exception:
                pass

        return wrapper

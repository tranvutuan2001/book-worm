"""Unit tests for :class:`ToolFactory`."""

import inspect
import pytest

from app.domain.enum.tool_type import ToolType
from app.services.tools.tool_factory import ToolFactory


class TestToolFactory:

    def test_create_document_search_returns_callable(self):
        tool = ToolFactory.create(ToolType.DOCUMENT_SEARCH, "my_doc")

        assert callable(tool)

    def test_create_document_summary_returns_callable(self):
        tool = ToolFactory.create(ToolType.DOCUMENT_SUMMARY, "my_doc")

        assert callable(tool)

    def test_document_search_signature_excludes_document_name(self):
        tool = ToolFactory.create(ToolType.DOCUMENT_SEARCH, "my_doc")
        sig = inspect.signature(tool)

        assert "document_name" not in sig.parameters

    def test_document_summary_signature_excludes_document_name(self):
        tool = ToolFactory.create(ToolType.DOCUMENT_SUMMARY, "my_doc")
        sig = inspect.signature(tool)

        assert "document_name" not in sig.parameters

    def test_document_search_preserves_original_name(self):
        tool = ToolFactory.create(ToolType.DOCUMENT_SEARCH, "my_doc")

        assert tool.__name__ == "get_the_most_relevant_chunks"

    def test_document_summary_preserves_original_name(self):
        tool = ToolFactory.create(ToolType.DOCUMENT_SUMMARY, "my_doc")

        assert tool.__name__ == "get_document_summary"

    def test_document_search_is_async(self):
        tool = ToolFactory.create(ToolType.DOCUMENT_SEARCH, "my_doc")

        assert inspect.iscoroutinefunction(tool)

    def test_document_summary_is_sync(self):
        tool = ToolFactory.create(ToolType.DOCUMENT_SUMMARY, "my_doc")

        assert not inspect.iscoroutinefunction(tool)

    def test_unknown_tool_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown tool type"):
            ToolFactory.create("nonexistent", "my_doc")

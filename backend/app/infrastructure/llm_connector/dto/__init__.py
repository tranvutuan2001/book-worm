"""
dto — Data Transfer Objects for the LLM connector.

These Pydantic models represent the OpenAI-compatible wire format
used to communicate with the remote LLM server.
"""

from app.infrastructure.llm_connector.dto.chat_message import ChatMessage
from app.infrastructure.llm_connector.dto.completion_request import CompletionRequest
from app.infrastructure.llm_connector.dto.tool_call import ToolCall
from app.infrastructure.llm_connector.dto.tool_call_function import ToolCallFunction
from app.infrastructure.llm_connector.dto.tool_definition import ToolDefinition
from app.infrastructure.llm_connector.dto.tool_function_schema import ToolFunctionSchema

__all__ = [
    "ChatMessage",
    "CompletionRequest",
    "ToolCall",
    "ToolCallFunction",
    "ToolDefinition",
    "ToolFunctionSchema",
]

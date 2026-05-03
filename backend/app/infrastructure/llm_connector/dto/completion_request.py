from typing import Any
from pydantic import BaseModel

from app.infrastructure.llm_connector.dto.chat_message import ChatMessage
from app.infrastructure.llm_connector.dto.tool_definition import ToolDefinition


class CompletionRequest(BaseModel):
    """
    Chat completion request sent to an OpenAI-compatible LLM server.

    Includes standard OpenAI fields plus extension fields (name, metadata)
    used by the Multi-Provider LLM Server and Langfuse.
    """
    model: str = ""
    messages: list[ChatMessage]
    temperature: float | None = None
    tools: list[ToolDefinition] | None = None
    name: str | None = None
    metadata: dict[str, Any] | None = None

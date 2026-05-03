from pydantic import BaseModel
from app.infrastructure.llm_connector.dto.tool_call import ToolCall


class ChatMessage(BaseModel):
    """
    A single message in an OpenAI-compatible chat conversation.

    Supports system, user, assistant, and tool roles.
    Includes support for tool calls and tool call results.
    """
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

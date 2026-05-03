from pydantic import BaseModel
from app.infrastructure.llm_connector.dto.tool_call_function import ToolCallFunction


class ToolCall(BaseModel):
    """A tool invocation requested by the model, containing an ID and function details."""
    id: str
    type: str = "function"
    function: ToolCallFunction

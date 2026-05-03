from pydantic import BaseModel
from app.infrastructure.llm_connector.dto.tool_function_schema import ToolFunctionSchema


class ToolDefinition(BaseModel):
    """
    A tool available to the model during chat completion.

    Wraps a :class:`ToolFunctionSchema` with a type discriminator.
    """
    type: str = "function"
    function: ToolFunctionSchema

from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    """Function invocation details within a tool call (name and JSON arguments)."""
    name: str
    arguments: str

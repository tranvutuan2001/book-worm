from typing import Any
from pydantic import BaseModel


class ToolFunctionSchema(BaseModel):
    """Schema describing a tool's function: its name, description, and parameter spec."""
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None

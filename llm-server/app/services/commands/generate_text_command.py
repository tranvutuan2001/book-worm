from dataclasses import dataclass
from typing import Any
from app.domain.value_objects.message import Message

@dataclass(frozen=True)
class GenerateTextCommand:
    """Command to generate text using an LLM."""
    messages: list[Message]
    max_completion_tokens: int
    frequency_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None

from dataclasses import dataclass
from app.domain.value_objects.message import Message

@dataclass(frozen=True)
class GenerateTextCommand:
    """Command to generate text using an LLM."""
    messages: list[Message]
    max_completion_tokens: int
    tools: list[dict[str, object]] | None = None

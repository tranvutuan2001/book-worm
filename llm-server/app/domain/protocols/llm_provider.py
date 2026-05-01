from typing import Protocol
from app.domain.value_objects.message import Message

class LLMProvider(Protocol):
    """Protocol defining the contract for LLM text generation."""
    async def generate(self, messages: list[Message], max_tokens: int, tools: list[dict[str, object]] | None = None) -> Message:
        """Generates a text response based on the provided messages."""
        ...

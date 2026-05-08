from typing import Protocol, Any
from app.domain.value_objects.message import Message

class LLMProvider(Protocol):
    """Protocol defining the contract for LLM text generation."""
    async def generate(
        self, 
        messages: list[Message], 
        max_completion_tokens: int, 
        frequency_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None
    ) -> Message:
        """Generates a text response based on the provided messages."""
        ...

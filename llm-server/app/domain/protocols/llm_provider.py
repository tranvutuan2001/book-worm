from typing import Protocol, List, Optional, Dict, Any
from app.domain.value_objects.message import Message

class LLMProvider(Protocol):
    """Protocol defining the contract for LLM text generation."""
    async def generate(self, messages: List[Message], max_tokens: int, tools: Optional[List[Dict[str, Any]]] = None) -> Message:
        """Generates a text response based on the provided messages."""
        ...

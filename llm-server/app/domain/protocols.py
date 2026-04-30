from typing import Protocol, List
from app.domain.models import Message

class LLMProvider(Protocol):
    async def generate(self, messages: List[Message], max_tokens: int) -> str:
        ...

class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> List[float]:
        ...

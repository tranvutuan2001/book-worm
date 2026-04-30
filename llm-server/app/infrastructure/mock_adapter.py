from typing import List
from app.domain.models import Message
from app.domain.protocols import LLMProvider, EmbeddingProvider

class MockProvider(LLMProvider, EmbeddingProvider):
    """
    A concrete implementation of LLMProvider and EmbeddingProvider for testing.
    This is NOT a mock object created by a test framework, but a real class
    in the infrastructure layer.
    """
    async def generate(self, messages: List[Message], max_tokens: int) -> str:
        return "Mocked LLM Response from MockProvider"

    async def embed(self, text: str) -> List[float]:
        # Return a deterministic embedding for testing
        return [0.1, 0.2, 0.3]

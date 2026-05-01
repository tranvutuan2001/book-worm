from typing import List
from app.domain.embedding_provider import EmbeddingProvider
from langfuse import observe

class EmbeddingService:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    @observe()
    async def execute(self, text: str) -> List[float]:
        return await self.embedding_provider.embed(text)

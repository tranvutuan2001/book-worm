from app.domain.protocols.embedding_provider import EmbeddingProvider

class EmbeddingService:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    async def execute(self, text: str) -> list[float]:
        return await self.embedding_provider.embed(text)

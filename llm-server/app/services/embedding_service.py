from app.domain.protocols.embedding_provider import EmbeddingProvider
from app.services.commands.generate_embedding_command import GenerateEmbeddingCommand

class EmbeddingService:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self._embedding_provider = embedding_provider

    async def generate_embeddings(self, command: GenerateEmbeddingCommand) -> list[list[float]]:
        results = []
        for text in command.texts:
            embedding = await self._embedding_provider.embed(text)
            results.append(embedding)
        return results

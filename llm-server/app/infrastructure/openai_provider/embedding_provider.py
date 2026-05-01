from openai import AsyncOpenAI, OpenAIError
from typing import List
from app.domain.protocols.embedding_provider import EmbeddingProvider
from app.domain.exceptions.llm_exception import LLMGenerationException

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-based implementation of the EmbeddingProvider."""
    
    def __init__(self, client: AsyncOpenAI, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    async def embed(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except OpenAIError as e:
            raise LLMGenerationException(
                message=str(e),
                provider="openai",
                original_error=e
            )

from openai import AsyncOpenAI, OpenAIError
from typing import List
from app.domain.models import Message
from app.domain.protocols import LLMProvider, EmbeddingProvider
from app.domain.exceptions import LLMGenerationException

class OpenAILLMProvider(LLMProvider):
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    async def generate(self, messages: List[Message], max_tokens: int) -> str:
        try:
            formatted_messages = [
                {"role": m.role.value, "content": m.content} 
                for m in messages
            ]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
        except OpenAIError as e:
            raise LLMGenerationException(
                message=str(e),
                provider="openai",
                original_error=e
            )

class OpenAIEmbeddingProvider(EmbeddingProvider):
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

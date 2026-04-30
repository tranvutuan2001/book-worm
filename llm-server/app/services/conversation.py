from langfuse import observe
from app.domain.protocols import LLMProvider
from app.domain.models import Message
from typing import List

class ConversationService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    @observe()
    async def execute(self, messages: List[Message], max_tokens: int) -> str:
        return await self.llm_provider.generate(messages, max_tokens)

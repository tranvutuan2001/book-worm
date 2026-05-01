from langfuse import observe
from app.domain.protocols.llm_provider import LLMProvider
from app.domain.value_objects.message import Message
from typing import List

class ConversationService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    @observe()
    async def execute(self, messages: List[Message], max_tokens: int) -> str:
        return await self.llm_provider.generate(messages, max_tokens)

from app.domain.protocols.llm_provider import LLMProvider
from app.domain.value_objects.message import Message

class TextGenerationService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def execute(self, messages: list[Message], max_tokens: int, tools: list[dict[str, object]] | None = None) -> Message:
        return await self.llm_provider.generate(messages, max_tokens, tools)

from anthropic import AsyncAnthropic, AnthropicError
from typing import List
from app.domain.models import Message, Role
from app.domain.protocols import LLMProvider
from app.domain.exceptions import LLMGenerationException

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate(self, messages: List[Message], max_tokens: int) -> str:
        try:
            # Separate system message for Anthropic
            system_message = next((m.content for m in messages if m.role == Role.SYSTEM), "")
            other_messages = [
                {"role": m.role.value, "content": m.content} 
                for m in messages if m.role != Role.SYSTEM
            ]
            
            response = await self.client.messages.create(
                model=self.model,
                system=system_message,
                messages=other_messages,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except AnthropicError as e:
            raise LLMGenerationException(
                message=str(e),
                provider="anthropic",
                original_error=e
            )

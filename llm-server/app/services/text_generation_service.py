from app.domain.protocols.llm_provider import LLMProvider
from app.domain.value_objects.message import Message
from app.services.commands.generate_text_command import GenerateTextCommand

class TextGenerationService:
    def __init__(self, llm_provider: LLMProvider):
        self._llm_provider = llm_provider

    async def generate_text(self, command: GenerateTextCommand) -> Message:
        return await self._llm_provider.generate(
            messages=command.messages,
            max_completion_tokens=command.max_completion_tokens,
            frequency_penalty=command.frequency_penalty,
            response_format=command.response_format,
            tools=command.tools
        )

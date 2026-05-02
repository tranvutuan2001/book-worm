from app.api.dto.chat_completion_request import ChatCompletionRequest
from app.services.commands.generate_text_command import GenerateTextCommand

class ChatMapper:
    """Mapper to convert Chat DTOs to Service Commands."""
    
    @staticmethod
    def to_generate_text_command(request: ChatCompletionRequest) -> GenerateTextCommand:
        return GenerateTextCommand(
            messages=request.messages,
            max_tokens=request.max_tokens or 1024,
            tools=request.tools
        )

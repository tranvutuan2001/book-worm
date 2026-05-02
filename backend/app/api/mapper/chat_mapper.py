from app.services.commands.ask_question_command import AskQuestionCommand
from app.api.dto.ask_request import AskRequest

class ChatMapper:
    """Maps chat DTOs to service commands."""
    
    @staticmethod
    def map_to_ask_command(payload: AskRequest) -> AskQuestionCommand:
        return AskQuestionCommand(
            document_name=payload.document_name or "",
            messages=payload.message_list,
            conversation_id=payload.id,
            timestamp=payload.timestamp
        )

from dataclasses import dataclass
from app.domain.entity.message import Message

@dataclass(frozen=True)
class AskQuestionCommand:
    """Command to ask a question about a document."""
    document_name: str
    messages: list[Message]
    conversation_id: str
    timestamp: int

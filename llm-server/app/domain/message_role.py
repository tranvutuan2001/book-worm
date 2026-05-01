from enum import Enum

class MessageRole(str, Enum):
    """Business concept representing the role of a participant in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

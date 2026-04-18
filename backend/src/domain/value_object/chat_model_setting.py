from dataclasses import dataclass, field
from src.config.config import (
    CHAT_MAX_TOKENS,
    CHAT_TEMPERATURE,
)

@dataclass
class ChatModelSettings:
    """Settings controlling a single chat-completion request."""
    max_tokens: int = field(default_factory=lambda: CHAT_MAX_TOKENS)
    temperature: float = field(default_factory=lambda: CHAT_TEMPERATURE)
    frequency_penalty: float = 0.0
    json_schema: str | None = None
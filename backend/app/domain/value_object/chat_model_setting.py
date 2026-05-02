from dataclasses import dataclass, field
from app.config.config import settings

@dataclass
class ChatModelSettings:
    """Settings controlling a single chat-completion request."""
    max_tokens: int | None = field(default_factory=lambda: settings.chat_max_tokens)
    temperature: float = field(default_factory=lambda: settings.chat_temperature)
    frequency_penalty: float = 0.0
    json_schema: str | None = None
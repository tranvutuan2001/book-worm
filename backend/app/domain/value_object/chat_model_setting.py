from dataclasses import dataclass, field
from app.config.app_setting import app_setting

@dataclass
class ChatModelSettings:
    """Settings controlling a single chat-completion request."""
    temperature: float = field(default_factory=lambda: app_setting.chat_temperature)
    frequency_penalty: float = 0.0
    json_schema: str | None = None
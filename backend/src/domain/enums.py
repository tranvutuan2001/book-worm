"""Domain-level enumerations."""

from enum import Enum


class Role(str, Enum):
    """Speaker role in a conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

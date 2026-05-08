"""
Agent domain entity.

:class:`Agent` is a pure data-holder that carries everything :class:`LLMService`
needs to run an LLM job: agent type, system prompt, tools, generation settings,
and retry count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.domain.value_object.chat_model_setting import ChatModelSettings

# ---------------------------------------------------------------------------
# Default system prompts
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    """A configured LLM agent ready to be executed by :class:`LLMService`."""

    system_prompt: str
    tools: list[Callable[..., Any]] = field(default_factory=list)
    model_settings: ChatModelSettings = field(default_factory=ChatModelSettings)
    max_retries: int = 3

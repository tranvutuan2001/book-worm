"""
Domain-level value objects for chat inference results.

These replace LangChain's ``AIMessage`` / ``ToolCall`` as the internal
representation returned by parsing services and consumed by the model adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation extracted from the model's raw output."""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Parsed result of a single chat-model generation.

    Attributes:
        content:    The assistant's textual reply (may be empty when only
                    tool-calls are present).
        tool_calls: Zero or more tool invocations requested by the model.
        thinking:   Optional chain-of-thought text extracted from models
                    that emit ``<think>`` blocks (Qwen-3, etc.).
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None

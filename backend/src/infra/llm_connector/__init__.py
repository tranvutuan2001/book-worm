"""
llm_connector — public package interface
=========================================

All code **outside** this package must interact with LLM models exclusively
through :class:`LLMManager` (lifecycle) and :class:`LLMService` (inference).
Direct imports of the internal MLX model classes are an implementation detail
and must not be used by callers.

Public API
----------
* :class:`LLMManager`        — loading, caching, unloading, and getting models.
* :class:`LLMService`        — inference entry-point: agent_complete_chat and embed_text.
* :class:`ChatModelSettings` — dataclass controlling per-request inference parameters.
* :class:`LoadedModelRecord` — TypedDict returned by ``LLMManager.list_loaded_models``.
* :class:`ModelType`         — ``Literal["chat", "embedding"]`` type alias.
"""

from src.infra.llm_connector.llm_manager import (
    LLMManager,
    LoadedModelRecord,
    ModelType,
)
from src.infra.llm_connector.llm_service import LLMService

__all__ = [
    "LLMManager",
    "LLMService",
    "LoadedModelRecord",
    "ModelType",
]

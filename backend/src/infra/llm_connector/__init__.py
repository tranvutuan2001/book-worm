"""
llm_connector — public package interface
=========================================

All code **outside** this package must interact with LLM models exclusively
through :class:`LLMManager` (lifecycle) and :class:`LLMService` (inference).
Direct imports of the internal MLX model classes are an implementation detail
and must not be used by callers.

Public API
----------
* :class:`LLMManager`           — loading, caching, unloading, and getting models.
* :func:`get_llm_manager`       — FastAPI dependency factory for ``LLMManager``.
* :class:`LLMService`           — inference entry-point: complete_chat and embed_text.
* :func:`get_llm_service`       — FastAPI dependency factory for ``LLMService``.
* :class:`LoadedModelRecord`    — TypedDict returned by ``LLMManager.list_loaded_models``.
* :class:`ModelType`            — ``Literal["chat", "embedding"]`` type alias.
* :class:`ParsingService`       — response-parsing service, injected into ``LLMManager``.
* :func:`get_parsing_service`   — FastAPI dependency factory for ``ParsingService``.
"""

from src.infra.llm_connector.llm_manager import (
    LLMManager,
    LoadedModelRecord,
    ModelType,
    get_llm_manager,
)
from src.infra.llm_connector.llm_service import (
    LLMService,
    get_llm_service,
)

__all__ = [
    "LLMManager",
    "LLMService",
    "LoadedModelRecord",
    "ModelType",
    "get_llm_manager",
    "get_llm_service",
]

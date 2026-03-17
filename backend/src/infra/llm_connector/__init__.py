"""
llm_connector — public package interface
=========================================

All code **outside** this package must interact with LLM models exclusively
through :class:`LLMService`.  Direct imports of the internal MLX model
classes (``MLXChatModel``, ``MLXEmbeddingModel``, ``MLXModelBase``) are an
implementation detail and must not be used by callers.

Public API
----------
* :class:`LLMService`           — single entry-point for inference and model lifecycle.
* :func:`get_llm_service`       — FastAPI dependency factory for ``LLMService``.
* :class:`LoadedModelRecord`    — TypedDict returned by ``LLMService.list_loaded_models``.
* :class:`ParsingService`       — response-parsing service, injected into ``LLMService``.
* :func:`get_parsing_service`   — FastAPI dependency factory for ``ParsingService``.
"""

from src.infra.llm_connector.llm_service import (
    LLMService,
    LoadedModelRecord,
    ModelType,
    get_llm_service,
)
from src.infra.llm_connector.parsing_service import (
    ParsingService,
    get_parsing_service,
)

__all__ = [
    "LLMService",
    "LoadedModelRecord",
    "ModelType",
    "get_llm_service",
    "ParsingService",
    "get_parsing_service",
]

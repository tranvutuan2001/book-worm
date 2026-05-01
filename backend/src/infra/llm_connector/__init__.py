"""
llm_connector — public package interface
=========================================

All code **outside** this package must interact with LLM models exclusively
through :class:`LLMService` (inference).
"""

from src.infra.llm_connector.llm_service import LLMService

__all__ = [
    "LLMService",
]

"""
FastAPI dependency providers.

Each function here is a ``Depends``-compatible factory that resolves a
service instance.  Centralising them in one module keeps route handlers clean
and makes testing easier (just override the dependency in the test client).
"""

from src.service.chat_service import ChatService, get_chat_service as _get_chat
from src.service.document_service import (
    DocumentService,
    get_document_service as _get_doc,
)
from src.service.model_service import ModelService, get_model_service as _get_model


def get_chat_service() -> ChatService:
    return _get_chat()


def get_document_service() -> DocumentService:
    return _get_doc()


def get_model_service() -> ModelService:
    return _get_model()

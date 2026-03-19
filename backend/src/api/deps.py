"""
FastAPI dependency providers.

Each function here is a ``Depends``-compatible factory that resolves a
service instance.  Centralising them in one module keeps route handlers clean
and makes testing easier (just override the dependency in the test client).
"""

from src.service.chat_service import ChatService, get_chat_service as _get_chat
from src.service.document_analysis_service import DocumentAnalysisService, get_document_analysis_service
from src.service.document_service import (
    DocumentService,
    get_document_service as _get_doc,
)
from src.service.model_service import ModelService, get_model_service as _get_model
from src.service.pdf_summarization_service import (
    PDFSummarizationService,
    get_pdf_summarization_service as _get_pdf_summarization,
)
from src.infra.llm_connector.llm_service import LLMService, get_llm_service
from fastapi import Depends


def get_chat_service() -> ChatService:
    return _get_chat()


def get_document_service(
    analysis_service: DocumentAnalysisService = Depends(get_document_analysis_service),
) -> DocumentService:
    return _get_doc(analysis_service)


def get_model_service(
    llm_service: LLMService = Depends(get_llm_service),
) -> ModelService:
    return _get_model(llm_service=llm_service)


def get_pdf_summarization_service(
    llm_service: LLMService = Depends(get_llm_service),
) -> PDFSummarizationService:
    return _get_pdf_summarization(llm_service=llm_service)

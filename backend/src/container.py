"""
Dependency-injection container (dependency_injector).

All application-wide singletons are declared here.  The container is
instantiated once in ``main.py`` and wired to the API route modules so that
FastAPI can resolve ``Depends(Provide[Container.<provider>])`` at startup.

Dependency graph
----------------
LLMService (Remote)
  ├─ DocumentAnalysisService
  │    └─ DocumentService
  ├─ ChatService
  └─ PDFSummarizationService
"""

from dependency_injector import containers, providers

from src.infra.llm_connector.llm_service import LLMService
from src.service.chat_service import ChatService
from src.service.document_analysis_service import DocumentAnalysisService
from src.service.document_service import DocumentService
from src.service.pdf_summarization_service import PDFSummarizationService
from src.config import config


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "src.api.routes.chat",
            "src.api.routes.document",
        ]
    )

    llm_service = providers.Singleton(
        LLMService,
        base_url=config.LLM_SERVER_URL,
    )

    document_analysis_service = providers.Singleton(
        DocumentAnalysisService,
        llm_service=llm_service,
    )

    document_service = providers.Singleton(
        DocumentService,
        analysis_service=document_analysis_service,
    )

    chat_service = providers.Singleton(
        ChatService,
        llm_service=llm_service,
    )

    pdf_summarization_service = providers.Singleton(
        PDFSummarizationService,
        llm_service=llm_service,
    )


# Module-level container instance — use this everywhere outside of FastAPI routes.
container = Container()

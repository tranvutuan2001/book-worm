"""
Dependency-injection container (dependency_injector).

All application-wide singletons are declared here.  The container is
instantiated once in ``main.py`` and wired to the API route modules so that
FastAPI can resolve ``Depends(Provide[Container.<provider>])`` at startup.

Dependency graph
----------------
ParsingService
  └─ LLMManager
       ├─ LLMService
       │    ├─ DocumentAnalysisService
       │    │    └─ DocumentService
       │    ├─ ChatService
       │    └─ PDFSummarizationService
       └─ ModelService
"""

from dependency_injector import containers, providers

from src.infra.llm_connector.local_llm.parsing_service import ParsingService
from src.infra.llm_connector.llm_manager import LLMManager
from src.infra.llm_connector.llm_service import LLMService
from src.service.chat_service import ChatService
from src.service.document_analysis_service import DocumentAnalysisService
from src.service.document_service import DocumentService
from src.service.model_service import ModelService
from src.service.pdf_summarization_service import PDFSummarizationService


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "src.api.routes.chat",
            "src.api.routes.document",
            "src.api.routes.model",
        ]
    )

    parsing_service = providers.Singleton(ParsingService)

    llm_manager = providers.Singleton(
        LLMManager,
        parsing_service=parsing_service,
    )

    llm_service = providers.Singleton(
        LLMService,
        llm_manager=llm_manager,
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

    model_service = providers.Singleton(
        ModelService,
        llm_manager=llm_manager,
    )


# Module-level container instance — use this everywhere outside of FastAPI routes.
container = Container()

"""
Document service — handles upload and listing of documents.

No FastAPI or HTTP concerns live here; exceptions are domain-level and
translated to HTTP responses by the API route layer.
"""

import logging
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from fastapi import UploadFile

from src.core.config import DATA_DIR
from src.core.exceptions import DocumentProcessingError, InvalidDocumentError
from src.service.document_analysis_service import (
    DocumentAnalysisService,
    _document_analysis_service,
)

logger = logging.getLogger("app.service")


# ---------------------------------------------------------------------------
# Service-level data containers (no FastAPI / schema dependency)
# ---------------------------------------------------------------------------

class DocumentStatus(str, Enum):
    READY = "ready"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    ERROR = "error"


@dataclass
class DocumentUploadResult:
    document_name: str
    status: DocumentStatus


@dataclass
class DocumentRecord:
    name: str
    status: DocumentStatus
    path: str


@dataclass
class DocumentListResult:
    documents: list[DocumentRecord] = field(default_factory=list)




class DocumentService:
    def __init__(self, analysis_service: DocumentAnalysisService) -> None:
        self._analysis_service = analysis_service

    async def upload_document(self, file: UploadFile) -> DocumentUploadResult:
        """Upload a PDF document and trigger background pre-analysis.

        Raises:
            InvalidDocumentError: For unsupported file type or missing filename.
            DocumentProcessingError: If saving the file or starting analysis fails.
        """
        if not file.filename:
            raise InvalidDocumentError("No filename provided")

        if not file.filename.lower().endswith(".pdf"):
            raise InvalidDocumentError(
                f"Unsupported file type: '{file.filename}'. Only PDF files are accepted."
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = f"{file.filename.replace('.pdf', '')}_{timestamp}"
        doc_folder = DATA_DIR / doc_name

        logger.info("Starting upload for document: %s", doc_name)

        try:
            doc_folder.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to create document folder: {exc}"
            ) from exc

        pdf_path = doc_folder / f"{doc_name}.pdf"
        try:
            content = await file.read()
            pdf_path.write_bytes(content)
            logger.info("Saved PDF: %s", pdf_path)
        except Exception as exc:
            raise DocumentProcessingError(f"Failed to save PDF: {exc}") from exc

        def _run_analysis() -> None:
            try:
                logger.info("Background analysis started: %s", doc_name)
                self._analysis_service.pre_analyze_document(str(pdf_path), doc_name)
                logger.info("Background analysis done: %s", doc_name)
            except Exception as exc:
                logger.error(
                    "Background analysis failed for %s: %s\n%s",
                    doc_name,
                    exc,
                    traceback.format_exc(),
                )

        thread = threading.Thread(target=_run_analysis, daemon=True)
        thread.start()
        logger.info("Analysis thread started for: %s", doc_name)

        return DocumentUploadResult(
            document_name=doc_name,
            status=DocumentStatus.ANALYZING,
        )

    async def list_documents(self) -> DocumentListResult:
        """Return metadata for all documents stored in the data directory."""
        if not DATA_DIR.exists():
            return DocumentListResult()

        documents: list[DocumentRecord] = []
        for item in DATA_DIR.iterdir():
            if not item.is_dir():
                continue
            doc_name = item.name
            chunks_file = item / f"{doc_name}_chunks.json"
            embeddings_file = item / f"{doc_name}_chunk_embeddings.json"
            status = (
                DocumentStatus.READY
                if chunks_file.exists() and embeddings_file.exists()
                else DocumentStatus.PROCESSING
            )
            documents.append(DocumentRecord(name=doc_name, status=status, path=str(item)))

        logger.info("Found %d documents", len(documents))
        return DocumentListResult(documents=documents)


# ---------------------------------------------------------------------------
# Singleton & dependency factory
# ---------------------------------------------------------------------------

_document_service: DocumentService = DocumentService(
    analysis_service=_document_analysis_service
)


def get_document_service() -> DocumentService:
    """FastAPI dependency that provides the shared ``DocumentService`` instance."""
    return _document_service

def get_document_service() -> DocumentService:
    """
    FastAPI dependency factory for ``DocumentService``.

    Inject this into route handlers with::

        from fastapi import Depends
        from app.service.document_service import DocumentService, get_document_service

        async def my_route(
            service: DocumentService = Depends(get_document_service)
        ):
            ...
    """
    return _document_service

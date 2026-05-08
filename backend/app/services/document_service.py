"""
Document service — handles upload and listing of documents.

No FastAPI or HTTP concerns live here; exceptions are domain-level and
translated to HTTP responses by the API route layer.
"""

import logging
import asyncio

from datetime import datetime

from app.config.app_setting import app_setting
from app.util.exceptions import DocumentProcessingError, InvalidDocumentError
from app.domain.enum.document_status import DocumentStatus
from app.domain.entity.document_record import DocumentRecord
from app.domain.value_object.document_upload_result import DocumentUploadResult
from app.domain.value_object.document_list_result import DocumentListResult
from app.services.document_analysis_service import DocumentAnalysisService
from app.services.commands.upload_document_command import UploadDocumentCommand
from app.services.commands.list_documents_command import ListDocumentsCommand
from app.services.commands.analyze_document_command import AnalyzeDocumentCommand

logger = logging.getLogger("app.service")


class DocumentService:
    def __init__(self, analysis_service: DocumentAnalysisService) -> None:
        self._analysis_service = analysis_service

    async def upload_document(self, command: UploadDocumentCommand) -> DocumentUploadResult:
        """Upload a PDF document and trigger background pre-analysis.

        Raises:
            InvalidDocumentError: For unsupported file type or missing filename.
            DocumentProcessingError: If saving the file or starting analysis fails.
        """
        if not command.filename:
            raise InvalidDocumentError("No filename provided")

        if not command.filename.lower().endswith(".pdf"):
            raise InvalidDocumentError(
                f"Unsupported file type: '{command.filename}'. Only PDF files are accepted."
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = f"{command.filename.replace('.pdf', '')}_{timestamp}"
        doc_folder = app_setting.data_storage_path / doc_name

        logger.info("Starting upload for document: %s", doc_name)

        try:
            doc_folder.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to create document folder: {exc}"
            ) from exc

        pdf_path = doc_folder / f"{doc_name}.pdf"
        try:
            pdf_path.write_bytes(command.content)
            logger.info("Saved PDF: %s", pdf_path)
        except Exception as exc:
            raise DocumentProcessingError(f"Failed to save PDF: {exc}") from exc

        async def _run_analysis() -> None:
            try:
                logger.info("Background analysis started: %s", doc_name)
                analyze_command = AnalyzeDocumentCommand(
                    pdf_path=str(pdf_path),
                    document_name=doc_name
                )
                await self._analysis_service.pre_analyze_document(analyze_command)
                logger.info("Background analysis done: %s", doc_name)
            except Exception as exc:
                logger.error(
                    "Background analysis failed for %s: %s",
                    doc_name,
                    exc,
                    exc_info=True,
                )

        asyncio.create_task(_run_analysis())
        logger.info("Analysis task started for: %s", doc_name)

        return DocumentUploadResult(
            document_name=doc_name,
            status=DocumentStatus.ANALYZING,
        )

    async def list_documents(self, command: ListDocumentsCommand) -> DocumentListResult:
        """Return metadata for all documents stored in the data directory."""
        if not app_setting.data_storage_path.exists():
            return DocumentListResult()

        documents: list[DocumentRecord] = []
        for item in app_setting.data_storage_path.iterdir():
            if not item.is_dir():
                continue
            doc_name = item.name
            chunks_file = item / f"{doc_name}{app_setting.suffix_chunks}"
            embeddings_file = item / f"{doc_name}{app_setting.suffix_chunk_embeddings}"
            status = (
                DocumentStatus.READY
                if chunks_file.exists() and embeddings_file.exists()
                else DocumentStatus.PROCESSING
            )
            documents.append(DocumentRecord(name=doc_name, status=status, path=str(item)))

        logger.info("Found %d documents", len(documents))
        return DocumentListResult(documents=documents)

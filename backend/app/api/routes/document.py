"""Document management routes (upload, listing, and PDF summarization)."""

import logging
import traceback

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from dependency_injector.wiring import Provide, inject

from app.api.schemas.document import (
    DocumentInfo,
    DocumentStatus,
    DocumentsResponse,
    UploadResponse,
)
from app.api.schemas.pdf_summarization import SummarizeResponse
from app.container import Container
from app.core.exceptions import DocumentNotFoundError, DocumentProcessingError, InvalidDocumentError
from app.service.document_service import DocumentService
from app.service.pdf_summarization_service import PDFSummarizationService

logger = logging.getLogger("app.api")

router = APIRouter(tags=["Documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a PDF document",
    description=(
        "Upload a PDF file for analysis.  The document is saved to disk and "
        "pre-analysis (chunking, summaries, embeddings) runs in the background.  "
        "Poll ``GET /documents`` to check when status changes to ``ready``."
    ),
)
@inject
async def upload_document(
    file: UploadFile = File(description="PDF file to upload"),
    service: DocumentService = Depends(Provide[Container.document_service]),
) -> UploadResponse:
    logger.info("POST /upload — file: %s", file.filename)
    try:
        result = await service.upload_document(file)
        return UploadResponse(
            message="Document uploaded successfully and analysis started",
            document_name=result.document_name,
            status=DocumentStatus(result.status.value),
        )
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled error in /upload: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/documents/{document_name}/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a document as a PDF JSON",
    description=(
        "Three-step pipeline: "
        "(1) generate a rich text summary via LLM + document retrieval tools; "
        "(2) convert the text into a structured JSON array conforming to pdf-schema.json "
        "using outlines constrained generation; "
        "(3) validate the JSON against the schema.  "
        "The result is stored under ``pdf/`` and also returned in the response body."
    ),
)
@inject
async def summarize_document(
    document_name: str,
    service: PDFSummarizationService = Depends(Provide[Container.pdf_summarization_service]),
) -> SummarizeResponse:
    logger.info("POST /documents/%s/summarize", document_name)
    try:
        result = await service.summarize(document_name=document_name)
        return SummarizeResponse(
            document_name=document_name,
            output_file=result["output_file"],
            content=result["content"],
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unhandled error in /summarize: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="List uploaded documents",
    description=(
        "Return all documents currently stored on the server along with their "
        "processing status (``processing``, ``analyzing``, or ``ready``)."
    ),
)
@inject
async def list_documents(
    service: DocumentService = Depends(Provide[Container.document_service]),
) -> DocumentsResponse:
    logger.info("GET /documents")
    try:
        result = await service.list_documents()
        return DocumentsResponse(
            documents=[
                DocumentInfo(
                    name=doc.name,
                    status=DocumentStatus(doc.status.value),
                    path=doc.path,
                )
                for doc in result.documents
            ]
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unhandled error in /documents: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(status_code=500, detail="Internal server error")

"""Document management routes (upload & listing)."""

import logging
import traceback

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.api.deps import get_document_service
from src.api.schemas.document import (
    DocumentInfo,
    DocumentStatus,
    DocumentsResponse,
    UploadResponse,
)
from src.core.exceptions import DocumentProcessingError, InvalidDocumentError
from src.service.document_service import DocumentService

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
async def upload_document(
    file: UploadFile = File(description="PDF file to upload"),
    service: DocumentService = Depends(get_document_service),
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


@router.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="List uploaded documents",
    description=(
        "Return all documents currently stored on the server along with their "
        "processing status (``processing``, ``analyzing``, or ``ready``)."
    ),
)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
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

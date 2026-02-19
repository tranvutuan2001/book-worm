from .dto import UploadResponse, DocumentsResponse
from fastapi import UploadFile, File, APIRouter, HTTPException
from app.service.document_service import document_service
import logging
import traceback

logger = logging.getLogger('app.controller')

router = APIRouter()

@router.post(
    "/upload", 
    response_model=UploadResponse, 
    tags=["Document Management"],
    summary="Upload a PDF document",
    description="Upload a PDF file for analysis. The document will be processed in the background to extract chunks, summaries, and embeddings. Returns the document name with timestamp and processing status."
)
async def upload_document(file: UploadFile = File(description="PDF file to upload")) -> UploadResponse:
    try:
        logger.info(f"Document upload endpoint called with file: {file.filename}")
        result = await document_service.upload_document(file)
        logger.info(f"Document upload completed successfully: {result.document_name}")
        return result
    except HTTPException:
        # Re-raise HTTP exceptions as-is (already logged in service layer)
        raise
    except Exception as e:
        error_msg = f"Unexpected error in upload endpoint: {str(e)}"
        logger.error(error_msg)
        print(f"\u274c {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during file upload")


@router.get(
    "/documents", 
    response_model=DocumentsResponse, 
    tags=["Document Management"],
    summary="List all documents",
    description="Get a list of all uploaded documents with their processing status. Status can be 'ready' (fully processed) or 'processing' (still being analyzed)."
)
async def list_documents() -> DocumentsResponse:
    try:
        logger.info("List documents endpoint called")
        result = await document_service.list_documents()
        logger.info(f"List documents completed: {len(result.documents)} documents found")
        return result
    except HTTPException:
        # Re-raise HTTP exceptions as-is (already logged in service layer)
        raise
    except Exception as e:
        error_msg = f"Unexpected error in list documents endpoint: {str(e)}"
        logger.error(error_msg)
        print(f"\u274c {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error while listing documents")

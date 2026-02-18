from .dto import UploadResponse, DocumentsResponse
from fastapi import UploadFile, File, APIRouter
from app.service.document_service import document_service

router = APIRouter()

@router.post(
    "/upload", 
    response_model=UploadResponse, 
    tags=["Document Management"],
    summary="Upload a PDF document",
    description="Upload a PDF file for analysis. The document will be processed in the background to extract chunks, summaries, and embeddings. Returns the document name with timestamp and processing status."
)
async def upload_document(file: UploadFile = File(description="PDF file to upload")) -> UploadResponse:
    return await document_service.upload_document(file)


@router.get(
    "/documents", 
    response_model=DocumentsResponse, 
    tags=["Document Management"],
    summary="List all documents",
    description="Get a list of all uploaded documents with their processing status. Status can be 'ready' (fully processed) or 'processing' (still being analyzed)."
)
async def list_documents() -> DocumentsResponse:
    return await document_service.list_documents()

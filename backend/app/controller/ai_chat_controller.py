from fastapi import APIRouter, HTTPException
from app.domain.entity.conversation import Conversation
from app.controller.dto import AskResponse
from app.service.ai_chat_service import ai_service
import logging
import traceback

logger = logging.getLogger('app.controller')

router = APIRouter()

@router.post(
    "/ask", 
    response_model=AskResponse, 
    tags=["Document Analysis"],
    summary="Ask questions about a document",
    description="Submit a conversation with questions about a specific document. The document_name must be specified in the conversation payload. Returns AI-generated answers based on the document content."
)
async def ask(payload: Conversation) -> AskResponse:
    try:
        logger.info(f"Ask endpoint called for document: {payload.document_name}")
        result = await ai_service.ask(payload)
        logger.info(f"Ask endpoint completed successfully")
        return result
    except HTTPException:
        # Re-raise HTTP exceptions as-is (already logged in service layer)
        raise
    except Exception as e:
        error_msg = f"Unexpected error in ask endpoint: {str(e)}"
        logger.error(error_msg)
        print(f"\u274c {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during chat request")

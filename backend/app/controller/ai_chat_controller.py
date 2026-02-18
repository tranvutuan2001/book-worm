from fastapi import APIRouter
from app.domain.entity.conversation import Conversation
from app.controller.dto import AskResponse
from app.service.ai_chat_service import ai_service

router = APIRouter()

@router.post(
    "/ask", 
    response_model=AskResponse, 
    tags=["Document Analysis"],
    summary="Ask questions about a document",
    description="Submit a conversation with questions about a specific document. The document_name must be specified in the conversation payload. Returns AI-generated answers based on the document content."
)
async def ask(payload: Conversation) -> AskResponse:
    return await ai_service.ask(payload)

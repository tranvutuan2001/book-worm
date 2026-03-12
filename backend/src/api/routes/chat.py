"""Chat (document Q&A) route."""

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_chat_service
from src.api.schemas.chat import AskResponse
from src.core.exceptions import DocumentNotFoundError, LLMError
from src.domain.entity.conversation import Conversation
from src.service.chat_service import ChatService

logger = logging.getLogger("app.api")

router = APIRouter(tags=["Document Analysis"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about a document",
    description=(
        "Submit a conversation containing the user's question and receive an "
        "AI-generated answer grounded in the specified document's content.  "
        "``document_name`` in the request body must match an already-uploaded "
        "and fully analysed document."
    ),
)
async def ask(
    payload: Conversation,
    service: ChatService = Depends(get_chat_service),
) -> AskResponse:
    logger.info("POST /ask — document: %s", payload.document_name)
    try:
        answer = await service.ask(payload)
        return AskResponse(
            message=answer,
            conversation_id=payload.id,
            timestamp=payload.timestamp,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled error in /ask: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")

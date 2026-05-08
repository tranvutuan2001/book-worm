"""Chat (document Q&A) route."""
import logging


from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import Provide, inject

from app.api.dto.ask_request import AskRequest
from app.api.dto.ask_response import AskResponse
from app.api.mapper.chat_mapper import ChatMapper
from app.container import Container
from app.util.exceptions import DocumentNotFoundError, LLMError
from app.services.chat_service import ChatService

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
@inject
async def ask(
    payload: AskRequest,
    service: ChatService = Depends(Provide[Container.chat_service]),
) -> AskResponse:
    logger.info("POST /ask — document: %s", payload.document_name)
    try:
        command = ChatMapper.map_to_ask_command(payload)
        answer = await service.ask(command)
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
        logger.error("Unhandled error in /ask: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

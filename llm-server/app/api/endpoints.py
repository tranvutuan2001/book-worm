from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide
from app.containers import Container
from app.domain.models import CompletionRequest
from app.domain.exceptions import LLMGenerationException
from app.services.conversation import ConversationService
from app.config import settings

router = APIRouter()

@router.post("/generate")
@inject
async def generate_completion(
    request: CompletionRequest,
    service: ConversationService = Depends(Provide[Container.conversation_service])
):
    try:
        content = await service.execute(request.messages, request.max_tokens)
        return {"content": content}
    except LLMGenerationException as e:
        # Standard HTTP 502/503 for vendor failures as per mandate
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/health")
async def health():
    return {"status": "ok", "backend": settings.LLM_BACKEND}

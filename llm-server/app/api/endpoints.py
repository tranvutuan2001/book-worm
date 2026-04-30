from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide
from app.containers import Container
from app.domain.models import CompletionRequest, EmbeddingRequest, EmbeddingResponse
from app.domain.exceptions import LLMGenerationException
from app.services.conversation import ConversationService
from app.services.embedding import EmbeddingService
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
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/embeddings", response_model=EmbeddingResponse)
@inject
async def embed_text(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(Provide[Container.embedding_service])
):
    try:
        embedding = await service.execute(request.input)
        return EmbeddingResponse(embedding=embedding, model=request.model or settings.LLM_BACKEND)
    except LLMGenerationException as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/health")
async def health():
    return {"status": "ok", "backend": settings.LLM_BACKEND}

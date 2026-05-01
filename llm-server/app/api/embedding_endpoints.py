from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide
from app.containers import Container
from app.domain.value_objects.embedding_context import EmbeddingContext
from app.domain.value_objects.text_embedding import TextEmbedding
from app.domain.exceptions.llm_exception import LLMGenerationException
from app.services.embedding import EmbeddingService
from app.config import settings

router = APIRouter()

@router.post("/embeddings", response_model=TextEmbedding)
@inject
async def embed_text(
    request: EmbeddingContext,
    service: EmbeddingService = Depends(Provide[Container.embedding_service])
):
    """Endpoint for generating text embeddings using the configured provider."""
    try:
        embedding = await service.execute(request.input_text)
        return TextEmbedding(embedding=embedding, model=request.model_name or settings.LLM_BACKEND)
    except LLMGenerationException as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

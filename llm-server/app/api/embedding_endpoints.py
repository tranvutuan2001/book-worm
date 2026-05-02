from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide
from app.container import Container
from app.domain.value_objects.embedding_context import EmbeddingContext
from app.domain.value_objects.text_embedding import TextEmbedding
from app.domain.exceptions.llm_exception import LLMGenerationException
from app.services.embedding_service import EmbeddingService
from app.settings import settings
from app.api.schemas.embedding_request import EmbeddingRequest
from app.api.schemas.embedding_response import EmbeddingResponse, EmbeddingData

router = APIRouter()

@router.post("/v1/embeddings", response_model=EmbeddingResponse)
@inject
async def openai_embeddings(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(Provide[Container.embedding_service])
) -> EmbeddingResponse:
    """OpenAI-compatible endpoint for generating text embeddings."""
    try:
        # Handle both single string and list of strings
        inputs = [request.input] if isinstance(request.input, str) else request.input
        
        embeddings_data = []
        for i, text in enumerate(inputs):
            embedding = await service.execute(text)
            embeddings_data.append(
                EmbeddingData(
                    index=i,
                    embedding=embedding
                )
            )
            
        return EmbeddingResponse(
            data=embeddings_data,
            model=request.model
        )
    except LLMGenerationException as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

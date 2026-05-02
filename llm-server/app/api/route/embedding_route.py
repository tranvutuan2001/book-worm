from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide

from app.container import Container
from app.domain.exceptions.llm_exception import LLMGenerationException
from app.services.embedding_service import EmbeddingService
from app.api.dto.embedding_request import EmbeddingRequest
from app.api.dto.embedding_response import EmbeddingResponse, EmbeddingData
from app.api.mapper.embedding_mapper import EmbeddingMapper

router = APIRouter()

@router.post("/v1/embeddings", response_model=EmbeddingResponse)
@inject
async def openai_embeddings(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(Provide[Container.embedding_service])
) -> EmbeddingResponse:
    """OpenAI-compatible endpoint for generating text embeddings."""
    try:
        command = EmbeddingMapper.to_generate_embedding_command(request)
        embeddings = await service.generate_embeddings(command)
        
        embeddings_data = [
            EmbeddingData(index=i, embedding=emb)
            for i, emb in enumerate(embeddings)
        ]
            
        return EmbeddingResponse(
            data=embeddings_data,
            model=request.model
        )
    except LLMGenerationException as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

"""
Embedding controller with FastAPI router
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from src.api.schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingData, EmbeddingUsage
from src.api.services.embedding_service import EmbeddingService
from src.api.services.model_service import ModelService, get_model_service
from src.llm_client.embedding_model import load_embedding_model

router = APIRouter(prefix="/v1", tags=["embeddings"])
service = EmbeddingService()


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    summary="Create embeddings",
    description="""Generate embeddings for text input using a local embedding model.
    
    This endpoint accepts OpenAI-compatible embedding requests and returns OpenAI-compatible responses.
    
    **Features:**
    - Single or batch text embedding
    - Float encoding format (base64 not yet supported)
    - Token usage tracking
    - Local model execution
    
    **Usage:**
    Send a single text string or a list of strings to generate embeddings.
    The response includes embedding vectors and token usage information.
    """,
    responses={
        200: {
            "description": "Successful embedding generation",
            "content": {
                "application/json": {
                    "examples": {
                        "single_text": {
                            "summary": "Single text embedding",
                            "value": {
                                "object": "list",
                                "data": [
                                    {
                                        "object": "embedding",
                                        "embedding": [0.123, -0.456, 0.789],
                                        "index": 0
                                    }
                                ],
                                "model": "embedding-model.gguf",
                                "usage": {
                                    "prompt_tokens": 8,
                                    "total_tokens": 8
                                }
                            }
                        },
                        "batch_text": {
                            "summary": "Batch text embeddings",
                            "value": {
                                "object": "list",
                                "data": [
                                    {
                                        "object": "embedding",
                                        "embedding": [0.123, -0.456, 0.789],
                                        "index": 0
                                    },
                                    {
                                        "object": "embedding",
                                        "embedding": [0.321, -0.654, 0.987],
                                        "index": 1
                                    }
                                ],
                                "model": "embedding-model.gguf",
                                "usage": {
                                    "prompt_tokens": 16,
                                    "total_tokens": 16
                                }
                            }
                        }
                    }
                }
            }
        },
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
async def create_embeddings(
    request: EmbeddingRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Generate embeddings for input text(s).
    
    Args:
        request: Embedding request containing text input and model name
    
    Returns:
        EmbeddingResponse with embedding vectors and usage info
    """
    try:
        # Check encoding format
        if request.encoding_format == "base64":
            raise HTTPException(
                status_code=400,
                detail="base64 encoding format is not yet supported. Use 'float' instead."
            )
        
        # Validate that the model exists
        model_exists, model_path = model_service.embedding_model_exists(request.model)
        if not model_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Embedding model '{request.model}' not found in models directory. "
                       f"Please use /v1/models/embeddings to list available models."
            )
        
        # Load the requested model
        model = load_embedding_model(model_path)
        
        # Generate embeddings
        embeddings = service.generate_embeddings(model, request.input)
        
        # Count tokens
        prompt_tokens = service.count_tokens(model, request.input)
        
        # Build response data
        embedding_data: List[EmbeddingData] = []
        for idx, embedding in enumerate(embeddings):
            embedding_data.append(
                EmbeddingData(
                    object="embedding",
                    embedding=embedding,
                    index=idx
                )
            )
        
        # Use the requested model name in response
        model_name = request.model
        
        return EmbeddingResponse(
            object="list",
            data=embedding_data,
            model=model_name,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens
            )
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions without wrapping
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

"""
Model management controller with FastAPI router
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List

from src.api.schemas import (
    ModelInfo,
    DownloadableModelInfo,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelLoadRequest,
    ModelUnloadRequest,
    ModelLoadResponse,
    ModelUnloadResponse,
    LoadedModelInfo
)
from src.api.services.model_service import ModelService
from src.llm_client.chat_model import LLMModelManager, load_chat_model
from src.llm_client.embedding_model import EmbeddingModelManager, load_embedding_model

router = APIRouter(prefix="/v1/models", tags=["models"])
service = ModelService()
chat_manager = LLMModelManager()
embedding_manager = EmbeddingModelManager()


@router.get(
    "/chat",
    response_model=List[ModelInfo],
    summary="List available chat models",
    description="""List all chat models currently available in the models directory.
    
    Returns information about each model including:
    - Model name
    - Relative path from models directory
    - File size in human-readable format
    """,
)
async def list_chat_models():
    """Get list of available chat models"""
    try:
        models = service.list_available_chat_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list chat models: {str(e)}")


@router.get(
    "/embeddings",
    response_model=List[ModelInfo],
    summary="List available embedding models",
    description="""List all embedding models currently available in the models directory.
    
    Returns information about each model including:
    - Model name
    - Relative path from models directory
    - File size in human-readable format
    """,
)
async def list_embedding_models():
    """Get list of available embedding models"""
    try:
        models = service.list_available_embedding_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list embedding models: {str(e)}")


@router.get(
    "/chat/downloadable",
    response_model=List[DownloadableModelInfo],
    summary="List downloadable chat models",
    description="""List all chat models available for download from Hugging Face.
    
    This endpoint returns a curated list of supported chat models that can be
    downloaded using the POST /v1/models/download endpoint.
    """,
)
async def list_downloadable_chat_models():
    """Get list of downloadable chat models"""
    try:
        models = service.list_downloadable_chat_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list downloadable chat models: {str(e)}")


@router.get(
    "/embeddings/downloadable",
    response_model=List[DownloadableModelInfo],
    summary="List downloadable embedding models",
    description="""List all embedding models available for download from Hugging Face.
    
    This endpoint returns a curated list of supported embedding models that can be
    downloaded using the POST /v1/models/download endpoint.
    """,
)
async def list_downloadable_embedding_models():
    """Get list of downloadable embedding models"""
    try:
        models = service.list_downloadable_embedding_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list downloadable embedding models: {str(e)}")


@router.post(
    "/download",
    response_model=ModelDownloadResponse,
    status_code=202,
    summary="Download a model from Hugging Face",
    description="""Download a specific GGUF model file from Hugging Face and store it in the models directory.
    
    **Process:**
    1. Validates the repository is in the allowed list
    2. Downloads the specific quantized GGUF file (e.g., Q4_K_M for 4-bit quantization)
    3. Returns 202 Accepted immediately
    4. Model file is downloaded asynchronously to the models directory
    
    **Note:** This operation runs in the background. The endpoint returns immediately with a 202 status.
    The actual download may take several minutes depending on model size and network speed.
    
    **Allowed repositories:**
    - unsloth/Qwen3-4B-Instruct-2507-GGUF (downloads Q4_K_M quantized version)
    - Qwen/Qwen3-Embedding-4B-GGUF (downloads Q4_K_M quantized version)
    """,
    responses={
        202: {
            "description": "Download started successfully",
            "content": {
                "application/json": {
                    "example": {
                        "repository": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                        "status": "downloading",
                        "path": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                        "message": "Model download started in background"
                    }
                }
            }
        },
        400: {
            "description": "Invalid repository (not in allowed list)"
        }
    }
)
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Download a model from Hugging Face in the background"""
    try:
        # Validate repository first
        service.validate_repository(request.repository)
        
        # Get the filename that will be downloaded
        all_models = {**service.DOWNLOADABLE_CHAT_MODELS, **service.DOWNLOADABLE_EMBEDDING_MODELS}
        filename = all_models.get(request.repository, "model.gguf")
        
        # Start download in background
        background_tasks.add_task(service.download_model_async, request.repository)
        
        # Return immediately with 202
        return {
            "repository": request.repository,
            "status": "downloading",
            "path": filename,
            "message": "Model download started in background"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/load",
    response_model=ModelLoadResponse,
    summary="Load a model into memory",
    description="""Explicitly load a model into memory.
    
    This endpoint loads a model from the `/models` directory into RAM, making it ready for use.
    Models are cached, so subsequent calls to load the same model will be instant.
    
    **Use cases:**
    - Preload models during startup
    - Load models before making multiple requests
    - Warm up models in advance
    
    **Note:** Models are also automatically loaded on-demand when you make a request to
    `/v1/chat/completions` or `/v1/embeddings`, so this endpoint is optional.
    """,
    responses={
        200: {
            "description": "Model loaded successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "loaded": {
                            "summary": "Model loaded successfully",
                            "value": {
                                "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
                                "model_type": "chat",
                                "status": "loaded",
                                "message": "Model loaded successfully into memory",
                                "model_path": "/path/to/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
                            }
                        },
                        "already_loaded": {
                            "summary": "Model already in memory",
                            "value": {
                                "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
                                "model_type": "chat",
                                "status": "already_loaded",
                                "message": "Model is already loaded in memory",
                                "model_path": "/path/to/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
                            }
                        }
                    }
                }
            }
        },
        404: {"description": "Model not found"},
        500: {"description": "Error loading model"}
    }
)
async def load_model(request: ModelLoadRequest):
    """Load a model into memory"""
    try:
        # Validate model exists
        model_exists, model_path = service.model_exists(request.model, request.model_type)
        if not model_exists:
            raise HTTPException(
                status_code=404,
                detail=f"{request.model_type.capitalize()} model '{request.model}' not found in models directory. "
                       f"Please use /v1/models/{request.model_type if request.model_type == 'chat' else 'embeddings'} to list available models."
            )
        
        # Check if already loaded
        if request.model_type == "chat":
            is_loaded = chat_manager.is_loaded(model_path)
            if is_loaded:
                return ModelLoadResponse(
                    model=request.model,
                    model_type=request.model_type,
                    status="already_loaded",
                    message="Model is already loaded in memory",
                    model_path=model_path
                )
            # Load the model
            load_chat_model(model_path)
        else:  # embedding
            is_loaded = embedding_manager.is_loaded(model_path)
            if is_loaded:
                return ModelLoadResponse(
                    model=request.model,
                    model_type=request.model_type,
                    status="already_loaded",
                    message="Model is already loaded in memory",
                    model_path=model_path
                )
            # Load the model
            load_embedding_model(model_path)
        
        return ModelLoadResponse(
            model=request.model,
            model_type=request.model_type,
            status="loaded",
            message="Model loaded successfully into memory",
            model_path=model_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@router.post(
    "/unload",
    response_model=ModelUnloadResponse,
    summary="Unload a model from memory",
    description="""Explicitly unload a model from memory and free RAM.
    
    This endpoint removes a loaded model from memory, completely freeing the RAM it was using.
    Use this to free up memory when you no longer need a specific model.
    
    **Use cases:**
    - Free memory when switching to a different model
    - Clean up models during maintenance
    - Manage memory usage in production
    
    **Important:** After unloading, the model will need to be loaded again before use,
    which takes time. Only unload models you won't need for a while.
    """,
    responses={
        200: {
            "description": "Model unloaded successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "unloaded": {
                            "summary": "Model unloaded successfully",
                            "value": {
                                "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
                                "model_type": "chat",
                                "status": "unloaded",
                                "message": "Model unloaded from memory and RAM freed"
                            }
                        },
                        "not_loaded": {
                            "summary": "Model was not in memory",
                            "value": {
                                "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
                                "model_type": "chat",
                                "status": "not_loaded",
                                "message": "Model was not loaded in memory"
                            }
                        }
                    }
                }
            }
        },
        404: {"description": "Model not found"},
        500: {"description": "Error unloading model"}
    }
)
async def unload_model(request: ModelUnloadRequest):
    """Unload a model from memory"""
    try:
        # Validate model exists
        model_exists, model_path = service.model_exists(request.model, request.model_type)
        if not model_exists:
            raise HTTPException(
                status_code=404,
                detail=f"{request.model_type.capitalize()} model '{request.model}' not found in models directory."
            )
        
        # Unload the model
        if request.model_type == "chat":
            unloaded = chat_manager.unload_model(model_path)
        else:  # embedding
            unloaded = embedding_manager.unload_model(model_path)
        
        if unloaded:
            return ModelUnloadResponse(
                model=request.model,
                model_type=request.model_type,
                status="unloaded",
                message="Model unloaded from memory and RAM freed"
            )
        else:
            return ModelUnloadResponse(
                model=request.model,
                model_type=request.model_type,
                status="not_loaded",
                message="Model was not loaded in memory"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error unloading model: {str(e)}")


@router.get(
    "/loaded",
    response_model=List[LoadedModelInfo],
    summary="List currently loaded models",
    description="""List all models currently loaded in memory.
    
    This endpoint shows which models are currently loaded and ready to use.
    Loaded models consume RAM but respond instantly to requests.
    """,
    responses={
        200: {
            "description": "List of loaded models",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "model_name": "Qwen3-4B-Instruct-2507-Q4_K_M",
                            "model_path": "/path/to/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                            "model_type": "chat",
                            "loaded": True
                        },
                        {
                            "model_name": "Qwen3-Embedding-4B-Q4_K_M",
                            "model_path": "/path/to/models/Qwen3-Embedding-4B-Q4_K_M.gguf",
                            "model_type": "embedding",
                            "loaded": True
                        }
                    ]
                }
            }
        }
    }
)
async def list_loaded_models():
    """Get list of currently loaded models"""
    try:
        loaded_models = []
        
        # Get loaded chat models
        for model_path in chat_manager.list_loaded_models():
            model_name = model_path.split("/")[-1].replace(".gguf", "")
            loaded_models.append(
                LoadedModelInfo(
                    model_name=model_name,
                    model_path=model_path,
                    model_type="chat",
                    loaded=True
                )
            )
        
        # Get loaded embedding models
        for model_path in embedding_manager.list_loaded_models():
            model_name = model_path.split("/")[-1].replace(".gguf", "")
            loaded_models.append(
                LoadedModelInfo(
                    model_name=model_name,
                    model_path=model_path,
                    model_type="embedding",
                    loaded=True
                )
            )
        
        return loaded_models
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing loaded models: {str(e)}")


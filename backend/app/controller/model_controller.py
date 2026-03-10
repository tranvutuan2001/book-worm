import logging
import traceback
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.controller.dto import (
    DownloadableModelInfo,
    LoadedModelInfo,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelInfo,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelUnloadRequest,
    ModelUnloadResponse,
)
from app.service.model_service import ModelService, get_model_service

logger = logging.getLogger("app.controller")

router = APIRouter(prefix="/v1/models", tags=["Model Management"])


@router.get(
    "/chat",
    response_model=List[ModelInfo],
    summary="List available chat models",
    description="List all chat models currently available in the models directory.",
)
async def list_chat_models(
    service: ModelService = Depends(get_model_service),
) -> List[ModelInfo]:
    return service.list_chat_models()


@router.get(
    "/embeddings",
    response_model=List[ModelInfo],
    summary="List available embedding models",
    description="List all embedding models currently available in the models directory.",
)
async def list_embedding_models(
    service: ModelService = Depends(get_model_service),
) -> List[ModelInfo]:
    return service.list_embedding_models()


@router.get(
    "/chat/downloadable",
    response_model=List[DownloadableModelInfo],
    summary="List downloadable chat models",
    description="List all chat models available for download from Hugging Face.",
)
async def list_downloadable_chat_models(
    service: ModelService = Depends(get_model_service),
) -> List[DownloadableModelInfo]:
    return service.list_downloadable_chat_models()


@router.get(
    "/embeddings/downloadable",
    response_model=List[DownloadableModelInfo],
    summary="List downloadable embedding models",
    description="List all embedding models available for download from Hugging Face.",
)
async def list_downloadable_embedding_models(
    service: ModelService = Depends(get_model_service),
) -> List[DownloadableModelInfo]:
    return service.list_downloadable_embedding_models()


@router.post(
    "/download",
    response_model=ModelDownloadResponse,
    status_code=202,
    summary="Download a model from Hugging Face",
    description=(
        "Download a model from Hugging Face and store it in the models directory. "
        "The download runs in the background; this endpoint returns 202 immediately."
    ),
)
async def download_model(
    request: ModelDownloadRequest,
    service: ModelService = Depends(get_model_service),
) -> ModelDownloadResponse:
    try:
        return await service.download_model(request.repository)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in download_model: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during model download")


@router.post(
    "/load",
    response_model=ModelLoadResponse,
    summary="Load a model into memory",
    description="Explicitly load a model into RAM, making it ready for inference.",
)
async def load_model(
    request: ModelLoadRequest,
    service: ModelService = Depends(get_model_service),
) -> ModelLoadResponse:
    try:
        return service.load_model(request.model_path, request.model_type)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in load_model: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error while loading model")


@router.post(
    "/unload",
    response_model=ModelUnloadResponse,
    summary="Unload a model from memory",
    description="Remove a loaded model from memory and free RAM.",
)
async def unload_model(
    request: ModelUnloadRequest,
    service: ModelService = Depends(get_model_service),
) -> ModelUnloadResponse:
    try:
        return service.unload_model(request.model_path, request.model_type)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in unload_model: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error while unloading model")


@router.get(
    "/loaded",
    response_model=List[LoadedModelInfo],
    summary="List currently loaded models",
    description="List all models currently loaded in memory and ready to use.",
)
async def list_loaded_models(
    service: ModelService = Depends(get_model_service),
) -> List[LoadedModelInfo]:
    return service.list_loaded_models()

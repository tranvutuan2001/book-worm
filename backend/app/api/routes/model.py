"""Model management routes (list, download, load, unload)."""

import logging
import traceback
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_model_service
from app.api.schemas.model import (
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
from app.service.model_service import ModelService

logger = logging.getLogger("app.api")

router = APIRouter(prefix="/v1/models", tags=["Models"])


@router.get(
    "/chat",
    response_model=List[ModelInfo],
    summary="List available chat models",
    description="List all chat models present in the local ``models/chat/`` directory.",
)
async def list_chat_models(
    service: ModelService = Depends(get_model_service),
) -> List[ModelInfo]:
    return service.list_chat_models()


@router.get(
    "/embeddings",
    response_model=List[ModelInfo],
    summary="List available embedding models",
    description="List all embedding models present in the local ``models/embedding/`` directory.",
)
async def list_embedding_models(
    service: ModelService = Depends(get_model_service),
) -> List[ModelInfo]:
    return service.list_embedding_models()


@router.get(
    "/chat/downloadable",
    response_model=List[DownloadableModelInfo],
    summary="List downloadable chat models",
    description="List chat models available for download from Hugging Face.",
)
async def list_downloadable_chat_models(
    service: ModelService = Depends(get_model_service),
) -> List[DownloadableModelInfo]:
    return service.list_downloadable_chat_models()


@router.get(
    "/embeddings/downloadable",
    response_model=List[DownloadableModelInfo],
    summary="List downloadable embedding models",
    description="List embedding models available for download from Hugging Face.",
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
        "Start an asynchronous background download of a Hugging Face model.  "
        "Returns 202 immediately; poll ``GET /v1/models/chat`` or "
        "``GET /v1/models/embeddings`` to track completion."
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
        logger.error(
            "Unhandled error in /v1/models/download: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/load",
    response_model=ModelLoadResponse,
    summary="Load a model into memory",
    description="Explicitly load a model into RAM so inference can begin immediately.",
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
        logger.error(
            "Unhandled error in /v1/models/load: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/unload",
    response_model=ModelUnloadResponse,
    summary="Unload a model from memory",
    description="Remove a loaded model from RAM to free memory.",
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
        logger.error(
            "Unhandled error in /v1/models/unload: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/loaded",
    response_model=List[LoadedModelInfo],
    summary="List models currently in memory",
    description="List all models that are currently loaded in RAM and ready for inference.",
)
async def list_loaded_models(
    service: ModelService = Depends(get_model_service),
) -> List[LoadedModelInfo]:
    return service.list_loaded_models()

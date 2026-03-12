"""API schema package — request/response models for the presentation layer."""

from src.api.schemas.chat import AskResponse
from src.api.schemas.document import (
    DocumentInfo,
    DocumentStatus,
    DocumentsResponse,
    UploadResponse,
)
from src.api.schemas.model import (
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

__all__ = [
    "AskResponse",
    "DocumentInfo",
    "DocumentStatus",
    "DocumentsResponse",
    "UploadResponse",
    "DownloadableModelInfo",
    "LoadedModelInfo",
    "ModelDownloadRequest",
    "ModelDownloadResponse",
    "ModelInfo",
    "ModelLoadRequest",
    "ModelLoadResponse",
    "ModelUnloadRequest",
    "ModelUnloadResponse",
]

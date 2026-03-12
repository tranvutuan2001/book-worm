"""API schema package — request/response models for the presentation layer."""

from app.api.schemas.chat import AskResponse
from app.api.schemas.document import (
    DocumentInfo,
    DocumentStatus,
    DocumentsResponse,
    UploadResponse,
)
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

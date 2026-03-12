"""Request / response schemas for the model management endpoints."""

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str = Field(description="Model name (directory name)")
    path: str = Field(description="Relative path from models directory")
    size: str = Field(description="Model size in human-readable format")
    status: str = Field(description="'ready_to_use' or 'downloading'")


class DownloadableModelInfo(BaseModel):
    name: str = Field(description="Model display name")
    repository: str = Field(description="Hugging Face repository ID")
    filename: str = Field(description="Model directory / filename identifier")


class LoadedModelInfo(BaseModel):
    model_name: str = Field(description="Model name")
    model_path: str = Field(description="Full path to model directory")
    model_type: str = Field(description="'chat' or 'embedding'")
    loaded: bool = Field(description="Whether model is currently in memory")


class ModelDownloadRequest(BaseModel):
    repository: str = Field(
        description="Hugging Face repository ID",
        example="mlx-community/Qwen3-4B-Instruct-4bit",
    )


class ModelDownloadResponse(BaseModel):
    repository: str = Field(description="Hugging Face repository ID")
    status: str = Field(description="Download status")
    path: str = Field(description="Local path where model will be stored")
    message: str = Field(description="Status message")


class ModelLoadRequest(BaseModel):
    model_path: str = Field(
        description="Relative path within its type subdirectory",
        example="mlx-community/Qwen3-4B-Instruct-4bit",
    )
    model_type: str = Field(description="'chat' or 'embedding'")


class ModelLoadResponse(BaseModel):
    model: str = Field(description="Model name")
    model_type: str = Field(description="'chat' or 'embedding'")
    status: str = Field(description="'loaded' or 'already_loaded'")
    message: str = Field(description="Status message")
    model_path: str = Field(description="Full path to model directory")


class ModelUnloadRequest(BaseModel):
    model_path: str = Field(
        description="Relative path within its type subdirectory",
        example="mlx-community/Qwen3-4B-Instruct-4bit",
    )
    model_type: str = Field(description="'chat' or 'embedding'")


class ModelUnloadResponse(BaseModel):
    model_path: str = Field(description="Relative path to the model directory")
    model_type: str = Field(description="'chat' or 'embedding'")
    status: str = Field(description="'unloaded' or 'not_loaded'")
    message: str = Field(description="Status message")

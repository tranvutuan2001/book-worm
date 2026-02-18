"""
Model management schemas
"""
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about an available model"""
    name: str = Field(..., description="Model name (filename without extension)")
    path: str = Field(..., description="Relative path from models directory")
    size: str = Field(..., description="Model file size in human-readable format")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "qwen3-8b-instruct",
                    "path": "Qwen--Qwen3-8B/qwen3-8b-instruct.gguf",
                    "size": "8.53 GB"
                }
            ]
        }
    }


class DownloadableModelInfo(BaseModel):
    """Information about a downloadable model"""
    name: str = Field(..., description="Model name")
    repository: str = Field(..., description="Hugging Face repository ID")
    filename: str = Field(..., description="Specific GGUF file to download")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                    "repository": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                    "filename": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
                }
            ]
        }
    }


class ModelDownloadRequest(BaseModel):
    """Request to download a model"""
    repository: str = Field(
        ..., 
        description="Hugging Face repository ID (e.g., 'Qwen/Qwen3-8B')",
        examples=["Qwen/Qwen3-8B", "Qwen/Qwen3-4B-Instruct-2507"]
    )


class ModelDownloadResponse(BaseModel):
    """Response from model download"""
    repository: str = Field(..., description="Hugging Face repository ID")
    status: str = Field(..., description="Download status (e.g., 'downloading', 'success')")
    path: str = Field(..., description="Local path where model will be/was downloaded")
    message: str = Field(..., description="Status message")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "repository": "Qwen/Qwen3-8B",
                    "status": "downloading",
                    "path": "Qwen--Qwen3-8B",
                    "message": "Model download started in background"
                }
            ]
        }
    }


class ModelLoadRequest(BaseModel):
    """Request to load a model into memory"""
    model: str = Field(..., description="Model name (filename without .gguf extension)")
    model_type: str = Field(..., description="Type of model: 'chat' or 'embedding'")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
                    "model_type": "chat"
                },
                {
                    "model": "Qwen3-Embedding-4B-Q4_K_M",
                    "model_type": "embedding"
                }
            ]
        }
    }


class ModelUnloadRequest(BaseModel):
    """Request to unload a model from memory"""
    model: str = Field(..., description="Model name (filename without .gguf extension)")
    model_type: str = Field(..., description="Type of model: 'chat' or 'embedding'")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
                    "model_type": "chat"
                }
            ]
        }
    }


class LoadedModelInfo(BaseModel):
    """Information about a loaded model"""
    model_name: str = Field(..., description="Model name")
    model_path: str = Field(..., description="Full path to model file")
    model_type: str = Field(..., description="Type: 'chat' or 'embedding'")
    loaded: bool = Field(..., description="Whether model is currently loaded in memory")


class ModelLoadResponse(BaseModel):
    """Response from model load operation"""
    model: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Type: 'chat' or 'embedding'")
    status: str = Field(..., description="Status: 'loaded' or 'already_loaded'")
    message: str = Field(..., description="Status message")
    model_path: str = Field(..., description="Full path to model file")


class ModelUnloadResponse(BaseModel):
    """Response from model unload operation"""
    model: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Type: 'chat' or 'embedding'")
    status: str = Field(..., description="Status: 'unloaded' or 'not_loaded'")
    message: str = Field(..., description="Status message")

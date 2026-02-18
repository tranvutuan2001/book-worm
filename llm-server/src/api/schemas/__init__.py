"""
API Schemas - Centralized exports
"""
from src.api.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    Choice,
    Usage,
    Tool,
    ToolCall,
    FunctionDefinition
)
from src.api.schemas.embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage
)
from src.api.schemas.model import (
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
from src.api.schemas.llama import (
    LlamaChatCompletionResponse,
    LlamaChoice,
    LlamaMessage,
    LlamaToolCall,
    LlamaToolCallFunction,
    LlamaUsage
)

__all__ = [
    # Chat schemas
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Message",
    "Choice",
    "Usage",
    "Tool",
    "ToolCall",
    "FunctionDefinition",
    # Embedding schemas
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingData",
    "EmbeddingUsage",
    # Model schemas
    "ModelInfo",
    "DownloadableModelInfo",
    "ModelDownloadRequest",
    "ModelDownloadResponse",
    "ModelLoadRequest",
    "ModelUnloadRequest",
    "ModelLoadResponse",
    "ModelUnloadResponse",
    "LoadedModelInfo",
    # Llama internal schemas
    "LlamaChatCompletionResponse",
    "LlamaChoice",
    "LlamaMessage",
    "LlamaToolCall",
    "LlamaToolCallFunction",
    "LlamaUsage"
]

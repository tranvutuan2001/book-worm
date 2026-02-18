"""
Embedding schemas
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request for generating text embeddings"""
    input: str | List[str] = Field(..., description="Text or list of texts to embed")
    model: str = Field(..., description="Model name to use for embeddings (e.g., 'Qwen3-Embedding-4B-Q4_K_M')")
    encoding_format: Optional[Literal["float", "base64"]] = Field("float", description="Format for embeddings")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "The quick brown fox jumps over the lazy dog",
                    "model": "Qwen3-Embedding-4B-Q4_K_M"
                },
                {
                    "input": ["Hello world", "How are you?"],
                    "model": "Qwen3-Embedding-4B-Q4_K_M",
                    "encoding_format": "float"
                }
            ]
        }
    }


class EmbeddingData(BaseModel):
    """Single embedding vector data"""
    object: str = Field("embedding", description="Object type")
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index in the input list")


class EmbeddingUsage(BaseModel):
    """Token usage for embedding request"""
    prompt_tokens: int = Field(..., description="Number of tokens in the input")
    total_tokens: int = Field(..., description="Total tokens used")


class EmbeddingResponse(BaseModel):
    """Response containing text embeddings"""
    object: str = Field("list", description="Object type")
    data: List[EmbeddingData] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embeddings")
    usage: EmbeddingUsage = Field(..., description="Token usage information")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [0.123, -0.456, 0.789],
                            "index": 0
                        }
                    ],
                    "model": "/path/to/embedding-model.gguf",
                    "usage": {
                        "prompt_tokens": 8,
                        "total_tokens": 8
                    }
                }
            ]
        }
    }

"""
Internal llama-cpp-python response schemas
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class LlamaToolCallFunction(BaseModel):
    """Function details in a tool call from llama-cpp-python"""
    name: str = Field(..., description="Function name")
    arguments: str = Field(..., description="JSON string of function arguments")


class LlamaToolCall(BaseModel):
    """Tool call structure from llama-cpp-python response"""
    id: str = Field(..., description="Tool call identifier")
    type: Literal["function"] = Field(..., description="Type of tool call")
    function: LlamaToolCallFunction = Field(..., description="Function details")


class LlamaMessage(BaseModel):
    """Message structure from llama-cpp-python response"""
    role: str = Field(..., description="Message role")
    content: Optional[str] = Field(None, description="Message content")
    tool_calls: Optional[List[LlamaToolCall]] = Field(None, description="Tool calls if any")


class LlamaChoice(BaseModel):
    """Choice structure from llama-cpp-python response"""
    index: int = Field(..., description="Choice index")
    message: LlamaMessage = Field(..., description="Response message")
    finish_reason: str = Field(..., description="Reason for completion")


class LlamaUsage(BaseModel):
    """Token usage from llama-cpp-python response"""
    prompt_tokens: int = Field(..., description="Prompt tokens")
    completion_tokens: int = Field(..., description="Completion tokens")
    total_tokens: int = Field(..., description="Total tokens")


class LlamaChatCompletionResponse(BaseModel):
    """Complete response structure from llama-cpp-python"""
    id: str = Field(..., description="Response ID")
    object: str = Field(..., description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[LlamaChoice] = Field(..., description="Response choices")
    usage: LlamaUsage = Field(..., description="Token usage")

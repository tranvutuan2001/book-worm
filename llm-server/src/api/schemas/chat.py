"""
Chat completion schemas
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class FunctionDefinition(BaseModel):
    """Function definition for tool calling"""
    name: str = Field(..., description="Function name to be called")
    description: str = Field(..., description="Function description for the model")
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="JSON Schema object describing function parameters"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            ]
        }
    }


class Tool(BaseModel):
    """Tool definition for function calling"""
    type: Literal["function"] = Field("function", description="Tool type (currently only 'function' is supported)")
    function: FunctionDefinition = Field(..., description="Function definition")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }
            ]
        }
    }


class Message(BaseModel):
    """Chat message in the conversation"""
    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Name of the tool (for tool messages)")
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call (for tool messages)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "role": "user",
                    "content": "What's the weather in Paris?"
                },
                {
                    "role": "assistant",
                    "content": "I'll check the weather for you."
                }
            ]
        }
    }


class ChatCompletionRequest(BaseModel):
    """Chat completion request with optional tool calling"""
    model: str = Field(..., description="Model name to use for completion (e.g., 'Qwen3-4B-Instruct-2507-Q4_K_M')")
    messages: List[Message] = Field(..., description="List of messages in the conversation", min_length=1)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    tools: Optional[List[Tool]] = Field(None, description="List of available tools/functions")
    tool_choice: Optional[str] = Field("auto", description="How to use tools: 'auto', 'none', or specific function name")
    stream: bool = Field(False, description="Whether to stream responses (not yet supported)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
                    "messages": [
                        {"role": "user", "content": "What's the weather in Tokyo?"}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get current weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {"type": "string"}
                                    },
                                    "required": ["location"]
                                }
                            }
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            ]
        }
    }


class ToolCall(BaseModel):
    """Tool call in the response"""
    id: str = Field(..., description="Unique identifier for the tool call")
    type: Literal["function"] = Field("function", description="Type of tool call")
    function: Dict[str, Any] = Field(..., description="Function name and arguments")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"Tokyo, Japan\"}"
                    }
                }
            ]
        }
    }


class Choice(BaseModel):
    """Response choice from the model"""
    index: int = Field(..., description="Choice index (always 0 for single completions)")
    message: Message = Field(..., description="Generated message")
    finish_reason: str = Field(..., description="Reason for completion: 'stop', 'tool_calls', or 'length'")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls requested by the model")


class Usage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used (prompt + completion)")


class ChatCompletionResponse(BaseModel):
    """Chat completion response"""
    id: str = Field(..., description="Unique completion ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Choice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage information")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chatcmpl-abc123",
                    "object": "chat.completion",
                    "created": 1707782400,
                    "model": "/path/to/model.gguf",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! How can I help you?"
                            },
                            "finish_reason": "stop",
                            "tool_calls": None
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 15,
                        "completion_tokens": 10,
                        "total_tokens": 25
                    }
                }
            ]
        }
    }

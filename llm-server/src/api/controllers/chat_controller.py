"""
Chat completion controller with FastAPI router
"""
from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from src.api.services.chat_service import ChatCompletionService
from src.api.services.model_service import ModelService, get_model_service
from src.llm_client.chat_model import load_chat_model
from src.llm_client.config import settings

router = APIRouter(prefix="/v1", tags=["chat"])
service = ChatCompletionService()


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create chat completion",
    description="""Create a chat completion with optional tool/function calling support.
    
    This endpoint accepts OpenAI-format requests and returns OpenAI-compatible responses.
    Internally, it converts to Qwen3 format for optimal performance with Qwen models.
    
    **Flow:**
    1. Receive OpenAI format request with messages and optional tools
    2. Convert to Qwen3 prompt format (with `<|im_start|>` tags and `<tools>` section)
    3. Generate completion using llama-cpp-python
    4. Parse response and extract tool calls from `<tool_call>` tags
    5. Return OpenAI-compatible response
    
    **Tool Calling:**
    When tools are provided, the model may choose to call a function instead of responding directly.
    Tool calls are returned in the `tool_calls` field with structured arguments.
    """,
    responses={
        200: {
            "description": "Successful chat completion",
            "content": {
                "application/json": {
                    "examples": {
                        "basic_chat": {
                            "summary": "Basic chat without tools",
                            "value": {
                                "id": "chatcmpl-abc123",
                                "object": "chat.completion",
                                "created": 1707782400,
                                "model": "/path/to/model.gguf",
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Hello! How can I help you today?"
                                        },
                                        "finish_reason": "stop",
                                        "tool_calls": None
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 10,
                                    "total_tokens": 30
                                }
                            }
                        },
                        "tool_calling": {
                            "summary": "Chat with tool call",
                            "value": {
                                "id": "chatcmpl-xyz789",
                                "object": "chat.completion",
                                "created": 1707782400,
                                "model": "/path/to/model.gguf",
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": ""
                                        },
                                        "finish_reason": "tool_calls",
                                        "tool_calls": [
                                            {
                                                "id": "call_abc123",
                                                "type": "function",
                                                "function": {
                                                    "name": "get_weather",
                                                    "arguments": "{\"location\": \"San Francisco, CA\", \"unit\": \"celsius\"}"
                                                }
                                            }
                                        ]
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 85,
                                    "completion_tokens": 25,
                                    "total_tokens": 110
                                }
                            }
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Error generating completion: model not loaded"
                    }
                }
            }
        }
    }
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    model_service: ModelService = Depends(get_model_service)
) -> ChatCompletionResponse:
    """
    Create a chat completion with optional tool/function calling.
    
    Accepts OpenAI-format requests and uses llama-cpp-python's chat_format
    for automatic conversion to Qwen format. Returns OpenAI-compatible responses.
    """
    try:
        # Validate that the model exists
        model_exists, model_path = model_service.model_exists(request.model, "chat")
        if not model_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Chat model '{request.model}' not found in models directory. "
                       f"Please use /v1/models/chat to list available models."
            )
        
        # Load the requested model
        model = load_chat_model(model_path)
        
        # Get generation parameters
        temperature = request.temperature if request.temperature is not None else settings.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else settings.max_tokens
        
        # Generate completion using chat_format (automatic OpenAI to Qwen conversion)
        response = service.generate_completion(
            model=model,
            messages=request.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=request.tools
        )
        
        # Convert llama-cpp-python response to our schema
        return service.parse_model_response(response)
        
    except HTTPException:
        # Re-raise HTTPExceptions without wrapping
        raise
    except Exception as e:
        print(f"Error generating completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

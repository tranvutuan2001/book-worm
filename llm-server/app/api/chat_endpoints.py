import uuid
from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide
from app.container import Container
from app.domain.value_objects.chat_context import ChatContext
from app.domain.exceptions.llm_exception import LLMGenerationException
from app.services.text_generation_service import TextGenerationService
from app.api.schemas.chat_completion_request import ChatCompletionRequest
from app.api.schemas.chat_completion_response import ChatCompletionResponse, ChatChoice

router = APIRouter()

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@inject
async def chat_completions(
    request: ChatCompletionRequest,
    service: TextGenerationService = Depends(Provide[Container.text_generation_service])
) -> ChatCompletionResponse:
    """OpenAI-compatible endpoint for chat completions."""
    try:
        response_message = await service.execute(
            messages=request.messages,
            max_tokens=request.max_tokens or 1024,
            tools=request.tools
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=response_message,
                    finish_reason="stop"
                )
            ]
        )
    except LLMGenerationException as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

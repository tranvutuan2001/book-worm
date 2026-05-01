from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide
from app.containers import Container
from app.domain.chat_context import ChatContext
from app.domain.llm_exception import LLMGenerationException
from app.services.conversation import ConversationService

router = APIRouter()

@router.post("/generate")
@inject
async def generate_completion(
    request: ChatContext,
    service: ConversationService = Depends(Provide[Container.conversation_service])
):
    """Endpoint for generating text completions using the configured LLM provider."""
    try:
        content = await service.execute(request.messages, request.max_tokens)
        return {"content": content}
    except LLMGenerationException as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

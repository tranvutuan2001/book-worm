from fastapi import FastAPI, Depends, HTTPException
from dependency_injector.wiring import inject, Provide
from app.containers import Container
from app.domain.models import CompletionRequest
from app.domain.exceptions import LLMGenerationException
from app.services.conversation import ConversationService
from app.config import settings

def create_app() -> FastAPI:
    container = Container()
    # Map configuration to the container
    container.config.from_dict({
        "llm_backend": settings.LLM_BACKEND,
        "openai_api_key": settings.OPENAI_API_KEY,
        "openai_model": settings.OPENAI_MODEL,
        "anthropic_api_key": settings.ANTHROPIC_API_KEY,
        "anthropic_model": settings.ANTHROPIC_MODEL,
        "mlx_model_path": settings.MLX_MODEL_PATH,
    })
    
    container.wire(modules=[__name__])
    
    app = FastAPI(title="Multi-Provider LLM Server")
    app.container = container
    
    return app

app = create_app()

@app.post("/generate")
@inject
async def generate_completion(
    request: CompletionRequest,
    service: ConversationService = Depends(Provide[Container.conversation_service])
):
    try:
        content = await service.execute(request.messages, request.max_tokens)
        return {"content": content}
    except LLMGenerationException as e:
        # Standard HTTP 502/503 for vendor failures as per mandate
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
async def health():
    return {"status": "ok", "backend": settings.LLM_BACKEND}

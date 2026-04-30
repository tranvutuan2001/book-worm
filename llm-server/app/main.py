from fastapi import FastAPI
from app.containers import Container
from app.config import settings
from app.api.endpoints import router

def create_app() -> FastAPI:
    container = Container()
    # Map configuration to the container
    container.config.from_dict({
        "llm": {
            "backend": settings.LLM_BACKEND,
            "openai_key": settings.OPENAI_API_KEY,
            "openai_model": settings.OPENAI_MODEL,
            "anthropic_key": settings.ANTHROPIC_API_KEY,
            "anthropic_model": settings.ANTHROPIC_MODEL,
            "mlx_path": settings.MLX_MODEL_PATH,
        }
    })
    
    # Wire the container to the api endpoints module
    container.wire(modules=["app.api.endpoints"])
    
    app = FastAPI(title="Multi-Provider LLM Server")
    app.container = container
    app.include_router(router)
    
    return app

app = create_app()

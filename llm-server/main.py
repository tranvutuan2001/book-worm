from fastapi import FastAPI
import uvicorn
from app.container import Container
from app.settings import settings
from app.api.route.chat_route import router as chat_router
from app.api.route.embedding_route import router as embedding_router
from app.api.route.health_route import router as health_router

def create_app() -> FastAPI:
    container = Container()
    # Map configuration to the container
    container.config.from_dict({
        "llm": {
            "backend": settings.LLM_BACKEND,
            "openai_key": settings.OPENAI_API_KEY,
            "openai_model": settings.OPENAI_MODEL,
            "mlx_chat_path": settings.MLX_CHAT_MODEL_PATH,
            "mlx_embedding_path": settings.MLX_EMBEDDING_MODEL_PATH,
        }
    })
    
    # Wire the container to the api endpoints modules
    container.wire(modules=[
        "app.api.route.chat_route",
        "app.api.route.embedding_route",
        "app.api.route.health_route"
    ])
    
    app = FastAPI(title="Multi-Provider LLM Server")
    app.container = container
    
    app.include_router(chat_router)
    app.include_router(embedding_router)
    app.include_router(health_router)
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)

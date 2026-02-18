#!/usr/bin/env python
"""
Entry point for running the LLM server
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.controllers import health_controller, chat_controller, embedding_controller, model_controller


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application instance.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="LLM Server",
        description="""Local LLM server with llama-cpp-python and FastAPI.
        
        ## Features
        
        * **OpenAI-Compatible API** - Use the `/v1/chat/completions` endpoint with OpenAI SDK
        * **Dynamic Model Loading** - Load any model from the `/models` directory by specifying its name
        * **Tool/Function Calling** - Qwen3-style tool binding with automatic format conversion
        * **Model Management** - List, download, and manage multiple models
        * **Efficient Caching** - Models are cached after first load for fast subsequent requests
        * **CORS Enabled** - Ready for web applications
        * **Streaming Support** - Stream responses in real-time (coming soon)
        
        ## Model Format
        
        This server uses Qwen3 format internally with automatic conversion from OpenAI format.
        Tool calls are extracted from `<tool_call>` tags and returned in OpenAI-compatible format.
        
        ## Model Selection
        
        Specify the model name in each request. Use `/v1/models/chat` or `/v1/models/embeddings` 
        to list available models. Models are loaded on-demand and cached for performance.
        
        ## Authentication
        
        Currently, no authentication is required. Add authentication middleware for production use.
        """,
        version="1.0.0",
        contact={
            "name": "LLM Server Support",
            "url": "https://github.com/yourusername/llm-server",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=[
            {
                "name": "health",
                "description": "Health check and server status endpoints"
            },
            {
                "name": "chat",
                "description": "Chat completion endpoints with tool/function calling support"
            },
            {
                "name": "embeddings",
                "description": "Text embedding endpoints for semantic search and similarity"
            },
            {
                "name": "models",
                "description": "Model management endpoints for listing and downloading models"
            },
        ]
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_controller.router)
    app.include_router(chat_controller.router)
    app.include_router(embedding_controller.router)
    app.include_router(model_controller.router)

    @app.on_event("startup")
    async def startup_event():
        """Server startup event"""
        print("LLM Server started successfully")
        print("Models will be loaded on-demand when requested")
        print("Use /v1/models/chat or /v1/models/embeddings to list available models")

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

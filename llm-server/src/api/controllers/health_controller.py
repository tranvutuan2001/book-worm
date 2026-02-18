"""
Health check controller with FastAPI router
"""
from fastapi import APIRouter

from src.llm_client.config import settings

router = APIRouter(tags=["health"])


@router.get(
    "/",
    summary="Root endpoint",
    description="Get basic server information and status",
    responses={
        200: {
            "description": "Server is running successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "LLM Server is running",
                        "model": "/path/to/model.gguf",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def root():
    """Get basic server information including model path and version."""
    return {
        "message": "LLM Server is running",
        "version": "1.0.0",
    }


@router.get(
    "/health",
    summary="Health check",
    description="Check if the server is running and healthy",
    responses={
        200: {
            "description": "Server is healthy and ready to accept requests",
            "content": {
                "application/json": {
                    "example": {"status": "healthy"}
                }
            }
        }
    }
)
async def health():
    """Health check endpoint for monitoring and load balancers."""
    return {"status": "healthy"}

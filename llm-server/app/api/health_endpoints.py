from fastapi import APIRouter
from app.config import settings

router = APIRouter()

@router.get("/health")
async def health():
    """Health check endpoint to verify server and backend status."""
    return {"status": "ok", "backend": settings.LLM_BACKEND}

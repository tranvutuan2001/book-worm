from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.infra.logging_config import setup_logging
from src.api.routes.chat import router as chat_router
from src.api.routes.document import router as document_router
from src.api.routes.model import router as model_router

setup_logging()

app = FastAPI(
    title="Book Worm — Document Analysis API",
    description=(
        "Upload PDF documents, trigger AI-powered analysis, and ask questions "
        "about their content using locally-running LLM models."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# Note: tighten ``allow_origins`` in production.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(document_router)
app.include_router(model_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.infra.logging_config import setup_logging
from app.controller.ai_chat_controller import router as ai_router
from app.controller.document_controller import router as document_router

setup_logging()

app = FastAPI(
    title="Book Worm Document Analysis API",
    description="API for uploading PDF documents, analyzing them, and querying their content using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for the API. Change `allow_origins` to a more restrictive list in
# production if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router, tags=["AI"])
app.include_router(document_router, tags=["Documents"])

if __name__ == "__main__":
    import uvicorn   
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

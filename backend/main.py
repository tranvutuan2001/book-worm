import os
# Multiple packages (mlx, torch inside xgrammar) each bundle their own copy of
# libomp.dylib on macOS, causing OpenMP to complain at startup.
# KMP_DUPLICATE_LIB_OK silences the abort; it must be set before any import
# that loads OpenMP.


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.infrastructure.logging_config import setup_logging
from app.api.route.chat import router as chat_router
from app.api.route.document import router as document_router

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
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

if __name__ == "__main__":
    import uvicorn
    import argparse

    # 1. Setup the argument parser
    parser = argparse.ArgumentParser(description="Run the Book Worm API server.")
    
    # 2. Add the --watch flag
    # action="store_true" means if the flag is present, 'watch' becomes True.
    # If it's missing, 'watch' stays False.
    parser.add_argument(
        "--watch", 
        action="store_true", 
        help="Enable auto-reload on file changes"
    )
    
    # 3. Parse the arguments from the command line
    args = parser.parse_args()

    # 4. Pass the result to uvicorn
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=args.watch
    )


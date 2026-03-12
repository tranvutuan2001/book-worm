# Book Worm — Document Analysis API

A FastAPI backend that lets users upload PDF documents, automatically analyses them with a local LLM, and answers natural-language questions about their content — all without sending data to any external service.

---

## Features

- **PDF upload** — stores documents locally and triggers background analysis
- **AI-powered pre-analysis** — chunks the document, generates section and chapter summaries, and builds vector embeddings
- **Document Q&A** — answers user questions by retrieving the most relevant passages via FAISS semantic search and sending them to a local LLM
- **Answer verification** — runs a second LLM pass to fact-check the initial answer against the document
- **Model management** — list, download, load, and unload local MLX models at runtime
- **Fully local** — all inference uses [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon; no data leaves the machine

---

## Architecture

The project follows a layered clean-architecture style where each layer has a single responsibility and dependencies only point inward.

```
backend/
├── main.py                              # FastAPI application entry point
├── app/
│   ├── api/                             # Presentation layer
│   │   ├── deps.py                      # FastAPI dependency providers
│   │   ├── routes/                      # HTTP route handlers (thin — no business logic)
│   │   │   ├── chat.py                  # POST /ask
│   │   │   ├── document.py              # POST /upload, GET /documents
│   │   │   └── model.py                 # GET|POST /v1/models/*
│   │   └── schemas/                     # Pydantic request/response models (DTOs)
│   │       ├── chat.py
│   │       ├── document.py
│   │       └── model.py
│   ├── core/                            # Shared cross-cutting concerns
│   │   ├── config.py                    # Path constants, model defaults, algorithm params
│   │   ├── exceptions.py                # Domain exceptions (no FastAPI dependency)
│   │   └── utils.py                     # General helpers (JSON writing, etc.)
│   ├── domain/                          # Business domain — pure Python, no framework deps
│   │   ├── enums.py                     # Role enum (User / Assistant / System)
│   │   └── entity/
│   │       ├── conversation.py          # Conversation domain object
│   │       └── message.py               # Message domain object
│   ├── service/                         # Application use-case layer
│   │   ├── chat_service.py              # Document Q&A orchestration
│   │   ├── document_service.py          # Upload & listing logic
│   │   ├── document_analysis_service.py # PDF → chunks → summaries → embeddings
│   │   ├── model_service.py             # Model discovery, download, load/unload
│   │   └── tools/                       # LangChain tools exposed to the LLM agent
│   │       └── document_retrieval_tool.py
│   └── infra/                           # Technical infrastructure
│       ├── logging_config.py            # Rotating-file logging with request-ID tracking
│       ├── session_manager.py           # Thread-safe per-request document context
│       └── llm_connector/              # MLX + LangChain integration
│           ├── llm_client.py            # High-level LLM service (chat + embed)
│           ├── llm_logging_handler.py   # LangChain callback logger
│           ├── mlx_base.py              # Shared model-path resolution
│           ├── mlx_chat.py              # LangChain BaseChatModel backed by mlx-lm
│           ├── mlx_embedding.py         # Embedding model wrapper
│           └── parsing_service.py       # Model-output parsers (Qwen3, extensible)
```

### Layer dependency rules

```
api  →  service  →  domain
api  →  core
service  →  core
service  →  infra
infra  →  domain
```

The `api` layer is the only place that knows about FastAPI, HTTP status codes, and response schemas. Services raise domain exceptions (`core/exceptions.py`) and route handlers translate them to `HTTPException`.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| macOS with Apple Silicon | Inference uses the MLX framework |
| Python 3.11+ | Tested on 3.12 |
| Local MLX chat model | e.g. `mlx-community/Qwen3.5-35B-A3B-4bit` |
| Local MLX embedding model | e.g. `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` |

---

## Quick Start

### 1. Create and activate a virtual environment

```zsh
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```zsh
pip install -r requirements.txt
```

### 3. Download models

Models live under `models/chat/` and `models/embedding/`. You can download them via the API (see below) or with the Hugging Face CLI:

```zsh
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit \\
    --local-dir models/chat/mlx-community/Qwen3.5-35B-A3B-4bit

huggingface-cli download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \\
    --local-dir models/embedding/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
```

### 4. Start the server

```zsh
uvicorn main:app --reload
```

The API is now available at http://127.0.0.1:8000
Interactive docs: http://127.0.0.1:8000/docs

---

## API Reference

### Document Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload a PDF; triggers background analysis |
| `GET`  | `/documents` | List all documents and their status |

**Upload a document**

```zsh
curl -X POST http://127.0.0.1:8000/upload -F "file=@/path/to/your/book.pdf"
```

Response:
```json
{
  "message": "Document uploaded successfully and analysis started",
  "document_name": "book_20260312_143022",
  "status": "analyzing"
}
```

Document status transitions: `analyzing` → `processing` → `ready`.

---

### Document Q&A

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | Ask a question about an uploaded document |

**Ask a question**

```zsh
curl -X POST http://127.0.0.1:8000/ask \\
  -H "Content-Type: application/json" \\
  -d '{
    "id": "conv_001",
    "timestamp": 1710000000000,
    "document_name": "book_20260312_143022",
    "chat_model": "models/chat/mlx-community/Qwen3.5-35B-A3B-4bit",
    "embedding_model": "models/embedding/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "message_list": [
      {
        "id": "msg_001",
        "role": "user",
        "content": "What is the main argument of this book?",
        "timestamp": 1710000000000
      }
    ]
  }'
```

---

### Model Management

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/v1/models/chat` | List locally available chat models |
| `GET`  | `/v1/models/embeddings` | List locally available embedding models |
| `GET`  | `/v1/models/chat/downloadable` | List chat models available for download |
| `GET`  | `/v1/models/embeddings/downloadable` | List embedding models available for download |
| `POST` | `/v1/models/download` | Start a background Hugging Face download |
| `POST` | `/v1/models/load` | Load a model into memory |
| `POST` | `/v1/models/unload` | Unload a model and free RAM |
| `GET`  | `/v1/models/loaded` | List models currently in memory |

---

## Configuration

All tuneable constants live in `app/core/config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_CHAT_MODEL` | `models/chat/mlx-community/Qwen3.5-35B-A3B-4bit` | Chat model path |
| `DEFAULT_EMBEDDING_MODEL` | `models/embedding/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | Embedding model path |
| `CHUNK_SIZE` | `2000` | Characters per text chunk |
| `CHUNKS_PER_SECTION` | `10` | Chunks grouped into one section summary |
| `SECTIONS_PER_CHAPTER` | `5` | Sections grouped into one chapter summary |
| `TOP_K_CHUNKS` | `3` | Chunks returned per semantic search query |
| `CHAT_MAX_TOKENS` | `2048` | Max tokens per LLM response |

---

## How Document Analysis Works

1. **Upload** — PDF is saved to `0_data/<document_name>/` and analysis begins in a background thread.
2. **Chunking** — text is extracted page-by-page and split into overlapping chunks.
3. **Section summaries** — chunks are grouped in batches and summarised by the LLM.
4. **Chapter summaries** — section summaries are further grouped into chapter-level overviews.
5. **Embeddings** — all chunks and summaries are embedded and stored as JSON arrays for FAISS retrieval.
6. **Query time** — the user question is embedded, top-k similar chunks are retrieved via FAISS, and passed as context to the LLM together with the chapter summaries.
7. **Verification** — a second LLM call fact-checks the initial answer against the document.

---

## Logs

Logs are written to the `logs/` directory:

| File | Contents |
|------|----------|
| `logs/app.log` | General application events |
| `logs/llm.log` | LLM/tool calls and agent interactions |

Both files rotate at 10 MB, keeping the last 5 backups.

---

## Docker

A `Dockerfile` is included for containerised deployment. Build and run:

```zsh
docker build -t book-worm-backend .
docker run -p 8000:8000 \\
  -v $(pwd)/models:/app/models \\
  -v $(pwd)/0_data:/app/0_data \\
  book-worm-backend
```
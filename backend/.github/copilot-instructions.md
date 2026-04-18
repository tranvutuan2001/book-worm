# GitHub Copilot Instructions — Book Worm Backend

## Project Overview

Book Worm is a **FastAPI** backend that enables PDF document analysis and Q&A using locally-running LLM models (MLX-based, Apple Silicon). It uses dependency injection (`dependency_injector`), structured logging, and a clean layered architecture.

---

## Architecture & Layer Rules

```
api/routes/      → HTTP entry points only; no business logic
api/schemas/     → Pydantic request/response models
service/         → Business logic; no FastAPI imports
domain/          → Pure domain entities/enums; no I/O
infra/           → External concerns (LLM connectors, logging)
config/config.py → All magic strings/constants live here
container.py     → Dependency-injection wiring (singleton providers)
```

- **Services** (`src/service/`) must never import from `fastapi`. Raise domain exceptions (`src/core/exceptions.py`); let the route layer translate them to HTTP responses.
- **Routes** (`src/api/routes/`) must only call services via injected providers (`Depends(Provide[Container.*])`). No direct instantiation of services.
- **Config** (`src/config/config.py`) is the single source of truth for paths, model names, and tuning parameters. Never hardcode magic strings elsewhere.
- **Domain entities** (`src/domain/`) must be plain Python dataclasses or Pydantic models with no side effects.

---

## Coding Standards

- **Python 3.11+** — use modern typing (`list[str]` not `List[str]`, `X | None` not `Optional[X]`).
- All public modules must have a module-level docstring explaining their purpose.
- Use `logging.getLogger("app.<layer>")` (e.g. `app.service`, `app.route`) — never `print()`.
- Follow the request-logging helpers: `start_request_logging` / `end_request_logging` / `get_request_logger` from `src.infra.logging_config`.
- Constants belong in `src/config/config.py`; import them by name.
- File paths must be constructed via `pathlib.Path`; never `os.path.join`.

---

## Dependency Injection

- All singletons are declared in `src/container.py` using `dependency_injector`.
- When adding a new service, add a `providers.Singleton(...)` entry in `Container` and wire it via `wiring_config`.
- Injected dependencies in route functions use `Depends(Provide[Container.<name>])`.

---

## LLM / MLX Specifics

- Chat models live under `models/chat/`, embedding models under `models/embedding/`.
- Default model paths are constants in `config.py` (`DEFAULT_CHAT_MODEL`, `DEFAULT_EMBEDDING_MODEL`).
- `KMP_DUPLICATE_LIB_OK=TRUE` must be set before any MLX/PyTorch imports (already handled in `main.py`).
- Inference parameters (`CHAT_MAX_TOKENS`, `CHAT_TEMPERATURE`, `TOP_K_CHUNKS`) are in `config.py`; adjust there, not inline.

---

## API Design

- Route modules: `chat.py`, `document.py`, `model.py` under `src/api/routes/`.
- All request/response bodies are Pydantic schemas in `src/api/schemas/`.
- Return meaningful HTTP status codes; use `HTTPException` only in route layer.
- Document all endpoints with FastAPI `summary=` and `description=` parameters.

---

## Adding New Features — Checklist

1. Add any new constants/paths to `src/config/config.py`.
2. Create or extend domain entities in `src/domain/entity/` if needed.
3. Implement business logic in `src/service/` (no FastAPI imports).
4. Add the new service as a `providers.Singleton` in `src/container.py`.
5. Expose via a Pydantic schema in `src/api/schemas/` and a route in `src/api/routes/`.
6. Wire the new route module in `Container.wiring_config`.

---

## What NOT to Do

- Do not put business logic in route handlers.
- Do not hardcode file paths or model names outside `config.py`.
- Do not use `print()` for logging.
- Do not import `fastapi` inside `src/service/` or `src/domain/`.
- Do not create new singletons outside `container.py`.
- Do not use `os.path` — always use `pathlib.Path`.

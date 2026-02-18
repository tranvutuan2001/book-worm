# Project Structure

This project follows a clean architecture with separation of concerns using FastAPI routers:

```
llm-server/
├── main.py                          # Application entry point & app factory
├── src/
│   ├── api/
│   │   ├── schemas.py              # Pydantic models for request/response
│   │   ├── controllers/            # FastAPI routers for HTTP endpoints
│   │   │   ├── health_controller.py (exports router)
│   │   │   └── chat_controller.py   (exports router)
│   │   └── services/               # Business logic layer
│   │       └── chat_service.py
│   ├── core/
│   │   ├── config.py               # Configuration management
│   │   └── model.py                # Model loading and management
│   └── formatters/
│       └── prompt_formatters.py    # Prompt format conversions
└── tests/
```

## Architecture

### Layer Responsibilities

1. **Entry Point & App Factory (`main.py`)**
   - FastAPI application creation and configuration
   - Middleware setup (CORS)
   - Router registration
   - Startup/shutdown events
   - Server startup (host, port)

2. **Controllers (`api/controllers/`)**
   - Each controller exports a FastAPI `APIRouter`
   - Defines routes with OpenAPI documentation
   - HTTP request/response handling
   - Input validation (via Pydantic schemas)
   - Dependency injection
   - Error handling
   - Delegates business logic to services

3. **Services (`api/services/`)**
   - Business logic implementation
   - Model interaction
   - Response parsing
   - Token calculation

4. **Core (`core/`)**
   - Configuration management
   - Model loading (singleton pattern)
   - Shared utilities

5. **Formatters (`formatters/`)**
   - Prompt format conversions
   - Response parsing

## Running the Server

```bash
python3 main.py
```

The server will start on `http://0.0.0.0:8001`

## Benefits of This Structure

- **FastAPI Routers**: Each controller uses `APIRouter` for modular route organization
- **Separation of Concerns**: Each layer has a specific responsibility
- **Testability**: Services and routers can be tested independently
- **Maintainability**: Easy to locate and modify specific functionality
- **Scalability**: Easy to add new routers/services as features grow
- **Reusability**: Services can be used by multiple controllers
- **Clean main.py**: App initialization and configuration in one place

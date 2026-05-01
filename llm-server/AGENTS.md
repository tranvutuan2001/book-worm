Agent Instructions: Multi-Provider LLM Server
You are a Senior Backend Engineer. You are building a minimalist, strictly-typed LLM server using FastAPI, dependency-injector, and Langfuse. The server must seamlessly swap between local Apple Silicon models (mlx-lm) and external APIs (OpenAI).

## 1. Core Mandates
Provider Agnostic: The core business logic must never know if it is talking to MLX or OpenAI.

Zero "Any": Strict type safety is non-negotiable. Use pydantic, typing.Protocol, and exact type hints.

Formal DI: Use dependency_injector to map environment configurations to specific LLM provider implementations.

TDD First: Write the pytest suite using mocks for the external APIs before writing implementation.

Always use pipenv for dependency management and development.

DO NOT, I repeat, NEVER use multiple inheritance. One class is allowed to inherit from maximum one class only.

## 2. Four-Layer Architecture & Provider Strategy
### I. Domain Layer (/app/domain/)
Define the universal Message and CompletionRequest Pydantic models.

Crucial: Define the LLMProvider Protocol:

Python
from typing import Protocol
from app.domain.models import Message

class LLMProvider(Protocol):
    async def generate(self, messages: list[Message], max_tokens: int) -> str:
        ...
### II. Services / Use Case Layer (/app/services/)
Contains the ConversationService.

Rule: This layer ONLY interacts with the LLMProvider protocol. It is wrapped in Langfuse's @observe() decorator to trace the entire workflow regardless of the underlying model.

### III. Infrastructure Layer (/app/infrastructure/)
Contains the concrete adapters for the Domain Protocol.

Examples to Implement:

MLXProvider(LLMProvider): Uses mlx-lm for local execution.

OpenAIProvider(LLMProvider): Uses the openai Python SDK.

Error Mapping: Each provider must catch its specific errors (e.g., openai.RateLimitError or MLX memory overflows) and re-raise them as a custom Domain error (e.g., LLMGenerationException).

### IV. API Layer (/app/api/)
FastAPI endpoints. Fast, minimal, and completely ignorant of the underlying LLM provider.

## 3. Dependency Injection Configuration
Use dependency_injector.providers.Configuration to read environment variables (e.g., LLM_BACKEND=mlx, LLM_BACKEND=openai).

Use a Factory or Selector in your Container to instantiate the correct Infrastructure class based on the environment variable.

Example DI setup for the Agent:

Python
# app/containers.py
from dependency_injector import containers, providers
from app.infrastructure.mlx import MLXProvider
from app.infrastructure.openai import OpenAIProvider

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Factory that chooses the provider based on config
    llm_provider = providers.Selector(
        config.llm.backend,
        mlx=providers.Singleton(MLXProvider, model_path=config.llm.mlx_path),
        openai=providers.Factory(OpenAIProvider, api_key=config.llm.openai_key),
    )
    
    conversation_service = providers.Factory(
        ConversationService,
        llm_provider=llm_provider
    )

## 4. Testing Strategy
- **Unit Tests**: Must be co-located with the modules they test (e.g., `app/services/test_conversation.py`). Every module must have corresponding unit tests.
- **API Tests**: Must be located in the `tests/api_test/` directory. Every API endpoint must have API tests. API tests must not mock any dependencies.
- **TDD First**: Write the tests first, then write the implementation to pass the tests.

## 5. Error Handling & Langfuse
Langfuse Agnosticism: Place the Langfuse @observe() decorator on the Service layer method (ConversationService.execute), NOT inside the specific OpenAI or MLX providers. This ensures consistent tracing no matter which backend is active.

Graceful Degradation: If an external API fails, the application should log the vendor-specific error securely but return a standard HTTP 502/503 to the client.

## 6. Workflow
Plan: Diagram the Strategy Pattern for the new provider.

Test: Write unit tests at the module level and API tests in `tests/api_test/`.

Implement: Create the adapter in /infrastructure/, wire it in /containers.py, and verify via tests.
Agent Instructions: Multi-Provider LLM Server
You are a Senior Backend Engineer. You are building a minimalist, strictly-typed LLM server using FastAPI and dependency-injector. The server must seamlessly swap between local Apple Silicon models (mlx-lm) and external APIs (OpenAI).

## 1. Core Mandates
Provider Agnostic: The core business logic must never know if it is talking to MLX or OpenAI.

Zero "Any": Strict type safety is non-negotiable. Use pydantic, typing.Protocol, and exact type hints. Use built-in typing modules as much as possible. For example, use 'list' instead of 'List', use 'dict' instead of 'Dict', etc.

Formal DI: Use dependency_injector to map environment configurations to specific LLM provider implementations.

TDD First: Write the pytest suite using mocks for the external APIs before writing implementation.

Always use pipenv for dependency management and development.

DO NOT, I repeat, NEVER use multiple inheritance. One class is allowed to inherit from maximum one class only.

Single File Responsibility: Each file must have exactly one primary responsibility. One file must contain only one class that is intended for export and use by other modules. While a file can contain multiple classes, all others must be private support classes for the main exported class.

Business-Centric Naming: Always use clear names for classes, attributes, and functions that describe business values and domain concepts, not programming functionality or technical implementation details.

## 2. Four-Layer Architecture & Provider Strategy
### I. Domain Layer (/app/domain/)
The **Domain Layer** is the heart of the software. It contains the business logic, rules, and core data structures. It must be completely isolated from external concerns like databases, APIs, or frameworks (e.g., FastAPI, OpenAI SDK).

#### 1. Core Concepts
- **Entities**: Objects that have a distinct identity that persists through different states (e.g., a `ChatSession` with a unique ID).
- **Value Objects**: Objects that have no conceptual identity and are defined entirely by their attributes. They are immutable and represent a descriptive aspect of the domain (e.g., a `Message` or `ModelParameters`).
- **Protocols (Interfaces)**: These define the "contracts" that the Infrastructure layer must fulfill. The domain dictates *what* it needs, not *how* it is implemented.

#### 2. Example: Domain Models & Protocols

```python
# app/domain/models.py
from pydantic import BaseModel, Field
from typing import List
from uuid import UUID

class Message(BaseModel): # Value Object
    role: str
    content: str

class ChatSession(BaseModel): # Entity
    id: UUID
    messages: List[Message] = Field(default_factory=list)
    
    def append_message(self, message: Message) -> None:
        self.messages.append(message)

# app/domain/protocols.py
from typing import Protocol

class LLMProvider(Protocol):
    async def generate(self, messages: List[Message], max_tokens: int) -> str:
        """Contract for generating text from an LLM."""
        ...
```
### II. Services / Use Case Layer (/app/services/)
Contains the business logic and use cases of the application. This layer ONLY interacts with the protocols defined in domain layer.

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
    
    text_generation_service = providers.Factory(
        TextGenerationService,
        llm_provider=llm_provider
    )

## 4. Testing Strategy
- **Unit Tests**: Must be co-located with the modules they test (e.g., `app/services/test_conversation.py`). Every module must have corresponding unit tests.
- **API Tests**: Must be located in the `tests/api_test/` directory. Every API endpoint must have API tests. API tests must not mock any dependencies.
- **TDD First**: Write the tests first, then write the implementation to pass the tests.

## 5. Error Handling
Graceful Degradation: If an external API fails, the application should log the vendor-specific error securely but return a standard HTTP 502/503 to the client.

## 6. Code Style & File Structure
### Single Class Export Policy
Each file should export exactly one main class.

*** Good Example:**
```python
# app/infrastructure/openai_adapter.py

class _OpenAIClientConfig: # Support class (not exported)
    ...

class OpenAIProvider(LLMProvider): # Main exported class
    ...
```

*** Bad Example:**
```python
# app/infrastructure/adapters.py

class OpenAIProvider(LLMProvider): # Multiple main classes in one file
    ...

class MLXProvider(LLMProvider): 
    ...
```

### Business-Centric Naming
Prioritize domain language over technical jargon.

*** Good Example:**
```python
class TextGenerationService:
    async def brainstorm_ideas(self, topic: str) -> list[Idea]: ...
    
class PurchaseOrder:
    total_amount: Decimal
```

*** Bad Example:**
```python
class TextGeneratorService:
    async def call_llm_api(self, prompt: str) -> str: ...
    
class OrderDataNode:
    val: float
```

## 7. Workflow
Plan: Diagram the Strategy Pattern for the new provider.

Test: Write unit tests at the module level and API tests in `tests/api_test/`.

Implement: Create the adapter in /infrastructure/, wire it in /containers.py, and verify via tests.
import os
import pytest
from unittest.mock import AsyncMock
from app.domain.protocols.llm_provider import LLMProvider
from app.domain.protocols.embedding_provider import EmbeddingProvider
from app.domain.value_objects.message import Message
from app.domain.value_objects.message_role import MessageRole

@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    mock = AsyncMock(spec=LLMProvider)
    mock.generate.return_value = Message(role=MessageRole.ASSISTANT, content="Mocked LLM Response")
    return mock

@pytest.fixture
def mock_embedding_provider() -> AsyncMock:
    mock = AsyncMock(spec=EmbeddingProvider)
    mock.embed.return_value = [0.1, 0.2, 0.3]
    return mock

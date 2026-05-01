import os
import pytest
from unittest.mock import AsyncMock
from app.domain.llm_provider import LLMProvider
from app.domain.embedding_provider import EmbeddingProvider

@pytest.fixture
def mock_llm_provider():
    mock = AsyncMock(spec=LLMProvider)
    mock.generate.return_value = "Mocked LLM Response"
    return mock

@pytest.fixture
def mock_embedding_provider():
    mock = AsyncMock(spec=EmbeddingProvider)
    mock.embed.return_value = [0.1, 0.2, 0.3]
    return mock

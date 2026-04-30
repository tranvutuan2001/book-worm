import os
import pytest
from unittest.mock import AsyncMock
from app.domain.protocols import LLMProvider, EmbeddingProvider

# Force mock backend for all tests to ensure reproducibility and speed
os.environ["LLM_BACKEND"] = "mock"
os.environ["MLX_MODEL_PATH"] = "mock_path" 
os.environ["OPENAI_API_KEY"] = "mock_key"

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

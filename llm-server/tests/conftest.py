import pytest
from unittest.mock import AsyncMock
from app.domain.protocols import LLMProvider

@pytest.fixture
def mock_llm_provider():
    mock = AsyncMock(spec=LLMProvider)
    mock.generate.return_value = "Mocked LLM Response"
    return mock

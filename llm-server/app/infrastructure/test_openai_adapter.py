import pytest
from unittest.mock import AsyncMock, patch
from app.infrastructure.openai_adapter import OpenAIProvider
from app.domain.models import Message, Role
from app.domain.exceptions import LLMGenerationException

@pytest.mark.asyncio
async def test_openai_provider_generate():
    with patch("app.infrastructure.openai_adapter.AsyncOpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create = AsyncMock()
        mock_client.chat.completions.create.return_value.choices = [
            AsyncMock(message=AsyncMock(content="OpenAI Response"))
        ]
        
        provider = OpenAIProvider(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        
        response = await provider.generate(messages, max_tokens=10)
        
        assert response == "OpenAI Response"
        mock_client.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_openai_provider_embed():
    with patch("app.infrastructure.openai_adapter.AsyncOpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.embeddings.create = AsyncMock()
        mock_client.embeddings.create.return_value.data = [
            AsyncMock(embedding=[0.1, 0.2, 0.3])
        ]
        
        provider = OpenAIProvider(api_key="test-key")
        response = await provider.embed("Hello")
        
        assert response == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

@pytest.mark.asyncio
async def test_openai_provider_error():
    with patch("app.infrastructure.openai_adapter.AsyncOpenAI") as mock_openai:
        from openai import OpenAIError
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create = AsyncMock(side_effect=OpenAIError("API Error"))
        
        provider = OpenAIProvider(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        
        with pytest.raises(LLMGenerationException) as excinfo:
            await provider.generate(messages, max_tokens=10)
        
        assert "API Error" in str(excinfo.value)
        assert excinfo.value.provider == "openai"

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.infrastructure.openai_provider.openai_llm_provider import OpenAILLMProvider
from app.domain.value_objects.message import Message
from app.domain.value_objects.message_role import MessageRole
from app.domain.exceptions.llm_exception import LLMGenerationException
from openai import OpenAIError

@pytest.mark.asyncio
async def test_openai_llm_provider_generate():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="OpenAI Response", role="assistant", tool_calls=None))
    ]
    
    provider = OpenAILLMProvider(client=mock_client)
    messages = [Message(role=MessageRole.USER, content="Hello")]
    
    response = await provider.generate(messages, max_tokens=10)
    
    assert response.content == "OpenAI Response"
    assert response.role == MessageRole.ASSISTANT
    mock_client.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_openai_llm_provider_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=OpenAIError("API Error"))
    
    provider = OpenAILLMProvider(client=mock_client)
    messages = [Message(role=MessageRole.USER, content="Hello")]
    
    with pytest.raises(LLMGenerationException) as excinfo:
        await provider.generate(messages, max_tokens=10)
    
    assert "API Error" in str(excinfo.value)
    assert excinfo.value.provider == "openai"

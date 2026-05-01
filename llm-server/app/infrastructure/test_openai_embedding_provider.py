import pytest
from unittest.mock import AsyncMock, MagicMock
from app.infrastructure.openai_embedding_provider import OpenAIEmbeddingProvider
from app.domain.llm_exception import LLMGenerationException
from openai import OpenAIError

@pytest.mark.asyncio
async def test_openai_embedding_provider_embed():
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3])
    ]
    
    provider = OpenAIEmbeddingProvider(client=mock_client)
    response = await provider.embed("Hello")
    
    assert response == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once()

@pytest.mark.asyncio
async def test_openai_embedding_provider_error():
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=OpenAIError("API Error"))
    
    provider = OpenAIEmbeddingProvider(client=mock_client)
    
    with pytest.raises(LLMGenerationException) as excinfo:
        await provider.embed("Hello")
    
    assert "API Error" in str(excinfo.value)
    assert excinfo.value.provider == "openai"

import pytest
from app.services.embedding_service import EmbeddingService

@pytest.mark.asyncio
async def test_embedding_service_execution(mock_embedding_provider):
    service = EmbeddingService(embedding_provider=mock_embedding_provider)
    text = "Hello world"
    
    response = await service.execute(text)
    
    assert response == [0.1, 0.2, 0.3]
    mock_embedding_provider.embed.assert_called_once_with(text)

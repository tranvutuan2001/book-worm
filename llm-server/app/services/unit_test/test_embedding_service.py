import pytest
from app.services.embedding_service import EmbeddingService
from app.services.commands.generate_embedding_command import GenerateEmbeddingCommand

@pytest.mark.asyncio
async def test_embedding_service_generate_embeddings(mock_embedding_provider):
    service = EmbeddingService(embedding_provider=mock_embedding_provider)
    texts = ["Hello", "World"]
    command = GenerateEmbeddingCommand(texts=texts)
    
    # Setup mock to return different values for different calls
    mock_embedding_provider.embed.side_effect = [[0.1, 0.1], [0.2, 0.2]]
    
    response = await service.generate_embeddings(command)
    
    assert response == [[0.1, 0.1], [0.2, 0.2]]
    assert mock_embedding_provider.embed.call_count == 2
    mock_embedding_provider.embed.assert_any_call("Hello")
    mock_embedding_provider.embed.assert_any_call("World")

import pytest
from unittest.mock import MagicMock, patch
from app.infrastructure.mlx_provider.mlx_embedding_provider import MLXEmbeddingProvider
from app.infrastructure.mlx_provider.mlx_model import MLXModel
from app.domain.exceptions.llm_exception import LLMGenerationException

@pytest.fixture
def mock_mlx_model():
    with patch("app.infrastructure.mlx_provider.mlx_model.mlx_lm") as mock_mlx_lm:
        mock_mlx_lm.load.return_value = (MagicMock(), MagicMock())
        model = MLXModel(model_path="test-path")
        return model

@pytest.mark.asyncio
async def test_mlx_embedding_provider_embed(mock_mlx_model):
    mlx_model = mock_mlx_model
    
    # Mocking mlx.core.mean and other bits is complex, let's mock the internal _get_embedding
    provider = MLXEmbeddingProvider(mlx_model=mlx_model)
    
    with patch("app.infrastructure.mlx_provider.mlx_embedding_provider.mx") as mock_mx:
        mock_mx.array.return_value = MagicMock()
        mock_mx.mean.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock tokenizer encode
        mlx_model.tokenizer.encode.return_value = [1, 2, 3]
        
        embedding = await provider.embed("Hello world")
        
        assert embedding == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_mlx_embedding_provider_error(mock_mlx_model):
    mlx_model = mock_mlx_model
    provider = MLXEmbeddingProvider(mlx_model=mlx_model)
    
    # Force an error in tokenizer
    mlx_model.tokenizer.encode.side_effect = Exception("Embedding Error")
    
    with pytest.raises(LLMGenerationException) as excinfo:
        await provider.embed("Hello world")
    
    assert "Embedding Error" in str(excinfo.value)
    assert excinfo.value.provider == "mlx"

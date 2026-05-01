import pytest
from unittest.mock import MagicMock, patch
from app.infrastructure.mlx_provider.llm_provider import MLXLLMProvider
from app.infrastructure.mlx_provider.model_loader import MLXModel
from app.domain.value_objects.message import Message
from app.domain.value_objects.message_role import MessageRole
from app.domain.exceptions.llm_exception import LLMGenerationException

@pytest.fixture
def mock_mlx_model():
    with patch("app.infrastructure.mlx_provider.model_loader.mlx_lm") as mock_mlx_lm:
        mock_mlx_lm.load.return_value = (MagicMock(), MagicMock())
        model = MLXModel(model_path="test-path")
        return model, mock_mlx_lm

@pytest.mark.asyncio
async def test_mlx_llm_provider_generate(mock_mlx_model):
    mlx_model, _ = mock_mlx_model
    
    with patch("app.infrastructure.mlx_provider.llm_provider.mlx_lm") as mock_mlx_lm:
        mock_mlx_lm.generate.return_value = "MLX Response"
        
        provider = MLXLLMProvider(mlx_model=mlx_model)
        messages = [Message(role=MessageRole.USER, content="Hello")]
        
        response = await provider.generate(messages, max_tokens=10)
        
        assert response == "MLX Response"
        mock_mlx_lm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_mlx_llm_provider_error(mock_mlx_model):
    mlx_model, _ = mock_mlx_model
    
    with patch("app.infrastructure.mlx_provider.llm_provider.mlx_lm") as mock_mlx_lm:
        mock_mlx_lm.generate.side_effect = Exception("MLX Error")
        
        provider = MLXLLMProvider(mlx_model=mlx_model)
        messages = [Message(role=MessageRole.USER, content="Hello")]
        
        with pytest.raises(LLMGenerationException) as excinfo:
            await provider.generate(messages, max_tokens=10)
        
        assert "MLX Error" in str(excinfo.value)
        assert excinfo.value.provider == "mlx"

import pytest
from unittest.mock import MagicMock, patch
from app.infrastructure.mlx_adapter import MLXProvider
from app.domain.models import Message, Role
from app.domain.exceptions import LLMGenerationException

@pytest.fixture
def mock_mlx_lm():
    with patch("app.infrastructure.mlx_adapter.mlx_lm") as mock:
        mock.load.return_value = (MagicMock(), MagicMock())
        mock.generate.return_value = "MLX Response"
        yield mock

@pytest.mark.asyncio
async def test_mlx_provider_generate(mock_mlx_lm):
    provider = MLXProvider(model_path="test-path")
    messages = [Message(role=Role.USER, content="Hello")]
    
    response = await provider.generate(messages, max_tokens=10)
    
    assert response == "MLX Response"
    mock_mlx_lm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_mlx_provider_error(mock_mlx_lm):
    mock_mlx_lm.generate.side_effect = Exception("MLX Error")
    
    provider = MLXProvider(model_path="test-path")
    messages = [Message(role=Role.USER, content="Hello")]
    
    with pytest.raises(LLMGenerationException) as excinfo:
        await provider.generate(messages, max_tokens=10)
    
    assert "MLX Error" in str(excinfo.value)
    assert excinfo.value.provider == "mlx"

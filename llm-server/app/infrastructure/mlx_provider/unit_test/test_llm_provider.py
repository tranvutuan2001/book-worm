import pytest
from unittest.mock import MagicMock, patch
from app.infrastructure.mlx_provider.mlx_llm_provider import MLXLLMProvider
from app.infrastructure.mlx_provider.mlx_model import MLXModel
from app.domain.value_objects.message import Message
from app.domain.value_objects.message_role import MessageRole
from app.domain.exceptions.llm_exception import LLMGenerationException

@pytest.fixture
def mock_mlx_model():
    with patch("app.infrastructure.mlx_provider.mlx_model.mlx_lm") as mock_mlx_lm:
        mock_mlx_lm.load.return_value = (MagicMock(), MagicMock())
        model = MLXModel(model_path="qwen-test-path")
        return model, mock_mlx_lm

@pytest.mark.asyncio
async def test_mlx_llm_provider_generate(mock_mlx_model):
    mlx_model, _ = mock_mlx_model
    
    with patch("app.infrastructure.mlx_provider.mlx_llm_provider.mlx_lm") as mock_mlx_lm:
        mock_mlx_lm.generate.return_value = "MLX Response"
        
        provider = MLXLLMProvider(mlx_model=mlx_model)
        messages = [Message(role=MessageRole.USER, content="Hello")]
        
        response = await provider.generate(messages, max_completion_tokens=10)
        
        assert response.content == "MLX Response"
        assert response.role == MessageRole.ASSISTANT
        mock_mlx_lm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_mlx_llm_provider_error(mock_mlx_model):
    mlx_model, _ = mock_mlx_model
    
    with patch("app.infrastructure.mlx_provider.mlx_llm_provider.mlx_lm") as mock_mlx_lm:
        mock_mlx_lm.generate.side_effect = Exception("MLX Error")
        
        provider = MLXLLMProvider(mlx_model=mlx_model)
        messages = [Message(role=MessageRole.USER, content="Hello")]
        
        with pytest.raises(LLMGenerationException) as excinfo:
            await provider.generate(messages, max_completion_tokens=10)
        
        assert "MLX Error" in str(excinfo.value)
        assert excinfo.value.provider == "mlx"

@pytest.mark.asyncio
async def test_mlx_llm_provider_generate_with_frequency_penalty(mock_mlx_model):
    mlx_model, _ = mock_mlx_model
    
    with patch("app.infrastructure.mlx_provider.mlx_llm_provider.mlx_lm") as mock_mlx_lm, \
         patch("app.infrastructure.mlx_provider.mlx_llm_provider.make_logits_processors") as mock_make_processors:
        
        mock_mlx_lm.generate.return_value = "MLX Response"
        mock_make_processors.return_value = [MagicMock()]
        
        provider = MLXLLMProvider(mlx_model=mlx_model)
        messages = [Message(role=MessageRole.USER, content="Hello")]
        
        await provider.generate(
            messages, 
            max_completion_tokens=10, 
            frequency_penalty=0.5
        )
        
        mock_make_processors.assert_called_once_with(frequency_penalty=0.5)
        mock_mlx_lm.generate.assert_called_once()
        args, kwargs = mock_mlx_lm.generate.call_args
        assert kwargs["logits_processors"] == mock_make_processors.return_value

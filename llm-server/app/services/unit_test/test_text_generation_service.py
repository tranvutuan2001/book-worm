import pytest
from app.services.text_generation_service import TextGenerationService
from app.domain.value_objects.message import Message
from app.domain.value_objects.message_role import MessageRole

@pytest.mark.asyncio
async def test_text_generation_service_execution(mock_llm_provider):
    service = TextGenerationService(llm_provider=mock_llm_provider)
    messages = [Message(role=MessageRole.USER, content="Hello")]
    
    response = await service.execute(messages, max_tokens=10)
    
    assert response.content == "Mocked LLM Response"
    assert response.role == MessageRole.ASSISTANT
    mock_llm_provider.generate.assert_called_once_with(messages, 10, None)

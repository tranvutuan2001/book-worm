import pytest
from app.services.text_generation_service import TextGenerationService
from app.domain.value_objects.message import Message
from app.domain.value_objects.message_role import MessageRole

from app.services.commands.generate_text_command import GenerateTextCommand

@pytest.mark.asyncio
async def test_text_generation_service_execution(mock_llm_provider):
    service = TextGenerationService(llm_provider=mock_llm_provider)
    messages = [Message(role=MessageRole.USER, content="Hello")]
    command = GenerateTextCommand(
        messages=messages,
        max_completion_tokens=10,
        frequency_penalty=0.5,
        response_format={"type": "json_schema", "json_schema": {}},
        tools=None
    )
    
    response = await service.generate_text(command)
    
    assert response.content == "Mocked LLM Response"
    assert response.role == MessageRole.ASSISTANT
    mock_llm_provider.generate.assert_called_once_with(
        messages=messages, 
        max_completion_tokens=10, 
        frequency_penalty=0.5,
        response_format={"type": "json_schema", "json_schema": {}},
        tools=None
    )

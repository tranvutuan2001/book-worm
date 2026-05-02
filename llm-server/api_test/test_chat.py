import pytest
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def openai_client():
    """Fixture to provide an OpenAI client pointing to the local running server."""
    port = os.getenv("PORT", "8001")
    return OpenAI(
        api_key="required-but-not-used",
        base_url=f"http://localhost:{port}/v1"
    )

def test_chat_completions(openai_client):
    """Test standard chat completions via official OpenAI SDK."""
    model_name = "models/chat/mlx-community/gemma-4-26b-a4b-it-4bit"
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Say 'hello'"}],
        max_tokens=100,
        temperature=0.0
    )
    
    assert response.id.startswith("chatcmpl-")
    assert response.object == "chat.completion"
    assert response.model == model_name
    assert len(response.choices) > 0
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content is not None

def test_chat_completions_with_tools(openai_client):
    """Test chat completions with tool calls via official OpenAI SDK."""
    model_name = "models/chat/mlx-community/gemma-4-26b-a4b-it-4bit"
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What is the weather like in Paris?"}
        ],
        max_tokens=500,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}}
                    }
                }
            }
        ]
    )
    
    assert response.choices[0].message.role == "assistant"
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
    elif response.choices[0].message.content:
        assert isinstance(response.choices[0].message.content, str)

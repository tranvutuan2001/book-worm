import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_generate_endpoint(client):
    payload = {
        "messages": [{"role": "user", "content": "Say 'hello'"}],
        "max_tokens": 500
    }
    response = client.post("/generate", json=payload)
    
    # Asserting 200 (Success) or 502 (Bad Gateway/Provider Error) 
    # depending on environment configuration, but the goal is to not mock.
    assert response.status_code == 200
    data = response.json()
    assert "role" in data
    assert data["role"] == "assistant"
    assert "content" in data
    assert isinstance(data["content"], str)
    assert len(data["content"]) > 0
    assert "<|channel>thought" not in data["content"]
    assert "<think>" not in data["content"]

def test_generate_endpoint_with_tools(client):
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What is the weather like in Paris?"}
        ],
        "max_tokens": 500,
        "tools": [
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
    }
    response = client.post("/generate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "role" in data
    assert data["role"] == "assistant"
    
    # Verify thoughts are stripped
    if "content" in data and data["content"] is not None:
        assert "<|channel>thought" not in data["content"]
        assert "<think>" not in data["content"]
        
    assert "tool_calls" in data
    assert len(data["tool_calls"]) > 0
    assert data["tool_calls"][0]["type"] == "function"
    assert data["tool_calls"][0]["function"]["name"] == "get_weather"

def test_embedding_endpoint(client):
    payload = {
        "input": "Hello world"
    }
    response = client.post("/embeddings", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) > 0
    assert all(isinstance(x, float) for x in data["embedding"])

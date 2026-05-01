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
        "max_tokens": 10
    }
    response = client.post("/generate", json=payload)
    
    # Asserting 200 (Success) or 502 (Bad Gateway/Provider Error) 
    # depending on environment configuration, but the goal is to not mock.
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert isinstance(data["content"], str)
    assert len(data["content"]) > 0

def test_embedding_endpoint(client):
    payload = {
        "input_text": "Hello world"
    }
    response = client.post("/embeddings", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) > 0
    assert all(isinstance(x, float) for x in data["embedding"])

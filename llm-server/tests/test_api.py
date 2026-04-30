import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.containers import Container
from unittest.mock import AsyncMock

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_generate_endpoint(client, mock_llm_provider):
    # Override the container provider with the mock
    with app.container.llm_provider.override(mock_llm_provider):
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        response = client.post("/generate", json=payload)
        
        assert response.status_code == 200
        assert response.json() == {"content": "Mocked LLM Response"}
        mock_llm_provider.generate.assert_called_once()

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

def test_embeddings_single_input(openai_client):
    """Test generating a single embedding via official OpenAI SDK."""
    model_name = "models/embedding/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    response = openai_client.embeddings.create(
        model=model_name,
        input="Hello world"
    )
    
    assert response.object == "list"
    assert len(response.data) == 1
    assert response.data[0].index == 0
    assert isinstance(response.data[0].embedding, list)
    assert len(response.data[0].embedding) > 0
    assert response.model == model_name

def test_embeddings_batch_input(openai_client):
    """Test generating batch embeddings via official OpenAI SDK."""
    model_name = "models/embedding/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    inputs = ["First sentence", "Second sentence"]
    response = openai_client.embeddings.create(
        model=model_name,
        input=inputs
    )
    
    assert response.object == "list"
    assert len(response.data) == 2
    assert response.data[0].index == 0
    assert response.data[1].index == 1
    assert all(isinstance(d.embedding, list) for d in response.data)
    assert response.model == model_name

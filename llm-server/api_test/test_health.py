import pytest
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

def test_health_check():
    port = os.getenv("PORT", "8001")
    url = f"http://localhost:{port}/health"
    response = httpx.get(url)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

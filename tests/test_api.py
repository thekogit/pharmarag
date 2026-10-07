import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_query_endpoint():
    mock_response = {
        "answer": "Test answer",
        "docs": [{"text": "Doc 1", "meta": {"source": "test"}}]
    }
    with patch("src.api.rag_app.invoke") as mock_invoke:
        mock_invoke.return_value = mock_response
        
        response = client.post("/query", json={"question": "What is testing?"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert len(data["docs"]) == 1
        assert data["docs"][0]["text"] == "Doc 1"
        assert data["docs"][0]["meta"]["source"] == "test"
        mock_invoke.assert_called_once()

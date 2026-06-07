"""
tests/test_middleware_app.py
Smoke tests for FastAPI middleware skeleton (spec 1.5.1).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.middleware.app import app
from src.middleware.models import SourceCitation


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_import_app():
    from src.middleware.app import app as imported_app

    assert imported_app is not None


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "storage" in data
    assert "model_available" in data


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data
    assert data["health"] == "/health"


def test_search_empty(client):
    resp = client.post("/search", json={"query": "NVDA revenue"})
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "facts" in data
    assert isinstance(data["documents"], list)
    assert isinstance(data["facts"], list)


def test_query_returns_response(client):
    mock_citations = [SourceCitation(source_type="sec_10k", ticker="NVDA")]
    with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("Test answer about NVDA.", mock_citations)
        resp = client.post("/query", json={"question": "What is NVDA revenue?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer about NVDA."
    assert "citations" in data
    assert "latency_ms" in data
    assert data["detected_ticker"] == "NVDA"


def test_query_validation(client):
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422

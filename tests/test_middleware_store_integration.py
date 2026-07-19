"""
tests/test_middleware_store_integration.py
Phase 1.8.3 — Integration tests for the middleware -> store path.

These exercise the real hybrid store (SQLite + ChromaDB) behind the FastAPI
middleware, using the live embedding endpoint on :8087. The model call is
mocked so the tests assert retrieval/wiring, not model output.

Usage:
    pytest tests/test_middleware_store_integration.py -v -m integration
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.storage.store import Store

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _embeddings_up() -> bool:
    try:
        r = httpx.post(
            "http://127.0.0.1:8087/v1/embeddings",
            json={"model": "tracealchemy", "input": "ping"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark.append(
    pytest.mark.skipif(not _embeddings_up(),
                       reason="embedding endpoint (:8087) not available")
)


@pytest.fixture
def live_store(tmp_path):
    """Real Store with tmp SQLite + tmp ChromaDB and the live embedder."""
    store = Store(
        db_path=tmp_path / "mw_int.db",
        chroma_path=tmp_path / "chroma",
        embedding_endpoint="http://127.0.0.1:8087/v1/embeddings",
    )
    yield store
    import shutil
    shutil.rmtree(tmp_path / "chroma", ignore_errors=True)


@pytest.fixture
def client_with_store(live_store, monkeypatch):
    from src.middleware import app as middleware_app
    monkeypatch.setattr(middleware_app, "store", live_store)
    monkeypatch.setattr(middleware_app, "config", middleware_app.MiddlewareConfig())
    monkeypatch.setattr(middleware_app, "retriever", None)
    client = TestClient(middleware_app.app)
    try:
        yield client, live_store
    finally:
        client.close()


def test_query_with_fresh_data(client_with_store):
    """Query returns an answer when the store has fresh data."""
    client, store = client_with_store
    store.save_fundamental("NVDA", "total_revenue", 26.0, "usd", "2026-Q1")
    store.save_document("test/nvda/report", "NVIDIA reported strong revenue growth...",
                        ticker="NVDA", source="test")

    with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call, \
            patch("src.middleware.app._check_model_health", new_callable=AsyncMock,
                  return_value=True), \
            patch("src.middleware.app._refresh_ticker_sources", return_value=([], [])):
        mock_call.return_value = ("NVDA revenue was $26B.", [])
        resp = client.post("/query", json={"question": "What is NVDA's revenue?",
                                           "refresh": False})

    assert resp.status_code == 200
    data = resp.json()
    assert data["detected_ticker"] == "NVDA"
    assert data["facts_used"] >= 1


def test_search_returns_results(client_with_store):
    """Raw /search returns stored documents from the live store."""
    client, store = client_with_store
    store.save_document("test/amd/report", "AMD is a semiconductor company making CPUs and GPUs.",
                        ticker="AMD", source="test")

    resp = client.post("/search", json={"query": "AMD semiconductor", "n_results": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["documents"]) > 0


def test_degraded_mode_when_model_down(client_with_store, monkeypatch):
    """When the model is unavailable, /query degrades to raw retrieved data.

    Pinned to the legacy (non-adaptive) path: under the promoted 2.3.7.4
    defaults a fully-covered fact lookup answers deterministically without
    the model, so the degraded fallback would never be reached here.
    """
    client, store = client_with_store
    from src.middleware import app as middleware_app
    monkeypatch.setattr(middleware_app.config, "enable_adaptive_rag", False)
    monkeypatch.setattr(middleware_app.config, "enable_deterministic_tool_routing", False)
    monkeypatch.setattr(middleware_app.config, "enable_deterministic_answers", False)
    store.save_fundamental("NVDA", "total_revenue", 26.0, "usd", "2026-Q1")

    with patch("src.middleware.app._check_model_health", new_callable=AsyncMock,
               return_value=False), \
            patch("src.middleware.app._refresh_ticker_sources", return_value=([], [])):
        resp = client.post("/query", json={"question": "What is NVDA's revenue?",
                                           "refresh": False})

    assert resp.status_code == 200
    data = resp.json()
    assert data["model_available"] is False
    assert "Model unavailable" in data["answer"]

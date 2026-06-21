"""
tests/test_middleware_app.py
Smoke tests for FastAPI middleware skeleton (spec 1.5.1) plus
cache-freshness / staleness-aware querying tests (spec 1.7.4).
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.middleware.app import app
from src.middleware.models import SourceCitation
from src.storage.store import Store


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ── Freshness / staleness fixtures (1.7.4) ─────────────

@pytest.fixture
def temp_store(tmp_path):
    """A Store with real SQLite (tmp) and a mocked ChromaStore."""
    with patch("src.storage.store.ChromaStore") as mock_cls:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mock_cls.return_value = inst
        yield Store(db_path=tmp_path / "fresh.db", chroma_path=tmp_path / "chroma")


def _backdate(store: Store, ticker: str, cache_source: str, hours_ago: float):
    """Insert a fresh cache row then backdate last_updated to age it."""
    store.mark_source_fresh(ticker, cache_source, 24)
    old = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with store.sqlite._connect() as conn:
        conn.execute(
            "UPDATE cache_meta SET last_updated=? WHERE ticker=? AND source=?",
            (old, ticker, cache_source),
        )
        conn.commit()


@pytest.fixture
def fresh_client(temp_store, monkeypatch):
    """TestClient whose global store is the temp_store."""
    from src.middleware import app as middleware_app
    with TestClient(app) as test_client:
        monkeypatch.setattr(middleware_app, "store", temp_store)
        yield test_client, temp_store


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
    with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call, \
            patch("src.middleware.app._check_model_health", new_callable=AsyncMock, return_value=True), \
            patch("src.middleware.app._refresh_ticker_sources", return_value=([], [])):
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


# ── Freshness endpoint (1.7.4) ─────────────────────────

def test_freshness_endpoint(fresh_client):
    client, store = fresh_client
    # Fresh fundamentals, stale (aged-out) news.
    store.mark_source_fresh("NVDA", "yfinance_fundamentals", 24)
    _backdate(store, "NVDA", "yfinance_news", hours_ago=48)

    resp = client.get("/freshness/NVDA")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "NVDA"
    assert data["sources"]["yfinance_fundamentals"]["status"] == "fresh"
    assert data["sources"]["yfinance_news"]["status"] == "stale"
    assert "yfinance_news" in data["stale_sources"]
    assert data["overall"] in ("partial", "stale", "fresh")


def test_freshness_unknown_ticker(fresh_client):
    client, _store = fresh_client
    resp = client.get("/freshness/ZZZZ")
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall"] == "never_fetched"
    assert all(s["status"] == "never_fetched" for s in data["sources"].values())
    assert data["stale_sources"] == []


def test_refresh_endpoint(fresh_client):
    client, store = fresh_client
    _backdate(store, "NVDA", "yfinance_news", hours_ago=48)
    with patch("src.middleware.app._refresh_one_source") as mock_one:
        resp = client.post("/refresh/NVDA", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "NVDA"
    # All never_fetched + stale sources were attempted.
    assert "yfinance_news" in data["refreshed"]
    assert mock_one.called


def test_refresh_specific_sources(fresh_client):
    client, store = fresh_client
    with patch("src.middleware.app._refresh_one_source") as mock_one:
        resp = client.post("/refresh/NVDA", json={"sources": ["news"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["refreshed"] == ["yfinance_news"]
    # Only the requested logical source was refreshed.
    called_logicals = [c.args[1] for c in mock_one.call_args_list]
    assert called_logicals == ["yfinance_news"]
    assert "yfinance_fundamentals" in data["skipped"]


def test_query_with_refresh(fresh_client):
    client, store = fresh_client
    _backdate(store, "NVDA", "yfinance_news", hours_ago=48)
    with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call, \
            patch("src.middleware.app._check_model_health", new_callable=AsyncMock, return_value=True), \
            patch("src.middleware.app._refresh_one_source") as mock_one:
        mock_call.return_value = ("answer", [])
        resp = client.post("/query", json={"question": "What is NVDA revenue?", "refresh": True})
    assert resp.status_code == 200
    data = resp.json()
    assert "yfinance_news" in data["freshness"]["refreshed_during_query"]
    assert mock_one.called


def test_query_without_refresh(fresh_client):
    client, store = fresh_client
    _backdate(store, "NVDA", "yfinance_news", hours_ago=48)
    with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call, \
            patch("src.middleware.app._check_model_health", new_callable=AsyncMock, return_value=True), \
            patch("src.middleware.app._refresh_one_source") as mock_one:
        mock_call.return_value = ("answer", [])
        resp = client.post("/query", json={"question": "What is NVDA revenue?", "refresh": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "yfinance_news" in data["freshness"]["stale_sources_used"]
    assert data["freshness"]["warning"]
    mock_one.assert_not_called()


def test_freshness_after_refresh(fresh_client):
    client, store = fresh_client
    _backdate(store, "NVDA", "yfinance_news", hours_ago=48)
    before = client.get("/freshness/NVDA").json()
    assert before["sources"]["yfinance_news"]["status"] == "stale"

    # Simulate a successful refresh marking the source fresh.
    store.mark_source_fresh("NVDA", "yfinance_news", 6)
    after = client.get("/freshness/NVDA").json()
    assert after["sources"]["yfinance_news"]["status"] == "fresh"


def test_stale_tickers_list(temp_store):
    store = temp_store
    store.mark_source_fresh("NVDA", "yfinance_fundamentals", 24)      # fresh
    _backdate(store, "AMD", "yfinance_fundamentals", hours_ago=48)    # aged out
    store.mark_source_stale("MSFT", "yfinance_fundamentals", "boom")  # explicit stale

    stale = store.get_stale_tickers("yfinance_fundamentals")
    assert "AMD" in stale
    assert "MSFT" in stale
    assert "NVDA" not in stale

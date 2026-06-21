"""
tests/test_regression.py
Phase 1.8.3 — Regression suite verifying Phase 1.1-1.7 features still work
after the Phase 1.8 changes.

SQLite-only checks stub ChromaDB; embedding/network checks are marked
``network`` and skip cleanly when services are unavailable.

Usage:
    pytest tests/test_regression.py -v -m regression
"""

import sys
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.storage.store import Store

pytestmark = pytest.mark.regression


@pytest.fixture
def store(tmp_path):
    """Store with real SQLite (tmp) and a mocked ChromaStore."""
    with patch("src.storage.store.ChromaStore") as mock_cls:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mock_cls.return_value = inst
        yield Store(db_path=tmp_path / "regression.db", chroma_path=tmp_path / "chroma")


def _live_store(tmp_path):
    return Store(
        db_path=tmp_path / "regression_live.db",
        chroma_path=tmp_path / "chroma_live",
        embedding_endpoint="http://127.0.0.1:8087/v1/embeddings",
    )


def _embeddings_up() -> bool:
    try:
        r = httpx.post("http://127.0.0.1:8087/v1/embeddings",
                       json={"model": "tracealchemy", "input": "ping"}, timeout=10)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


# ── Phase 1.1 — Model serving ──────────────────────────

@pytest.mark.network
def test_llama_server_health():
    """Phase 1.1: Model endpoint is reachable (if running)."""
    try:
        r = httpx.get("http://127.0.0.1:8087/health", timeout=5)
        assert r.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.skip("llama-server not running")


# ── Phase 1.2 — Storage layer ──────────────────────────

def test_sqlite_upsert_and_read(store):
    """Phase 1.2: SQLite store upserts and retrieves fundamentals."""
    store.save_fundamental("TEST", "test_metric", 42.0, "units", "2026-Q1")
    result = store.get_fundamental("TEST", "test_metric")
    assert result is not None
    assert result["value"] == 42.0


@pytest.mark.network
def test_chromadb_save_and_search(tmp_path):
    """Phase 1.2: ChromaDB stores and retrieves documents."""
    if not _embeddings_up():
        pytest.skip("embedding endpoint not available")
    store = _live_store(tmp_path)
    doc_id = store.save_document("regression/test/doc", "Test document content here",
                                 ticker="TEST", source="regression")
    assert doc_id is not None
    results = store.search("test document", n_results=5)
    assert len(results["documents"]) > 0
    import shutil
    shutil.rmtree(tmp_path / "chroma_live", ignore_errors=True)


# ── Phase 1.3 — Yahoo Finance ingestion ────────────────

@pytest.mark.network
@pytest.mark.slow
def test_yfinance_ticker_fetch(tmp_path):
    """Phase 1.3: YFinanceIngestor fetches and stores ticker data."""
    store = _live_store(tmp_path) if _embeddings_up() else None
    if store is None:
        with patch("src.storage.store.ChromaStore") as mock_cls:
            inst = MagicMock()
            inst.heartbeat.return_value = True
            inst.count.return_value = 0
            mock_cls.return_value = inst
            store = Store(db_path=tmp_path / "yf.db", chroma_path=tmp_path / "c")
    from src.ingestion.yfinance_ingestor import YFinanceIngestor
    ing = YFinanceIngestor(store=store)
    try:
        t = ing._fetch_ticker("NVDA")
        if t is None:
            pytest.skip("Yahoo Finance unreachable")
        ing._ingest_ticker_fundamentals("NVDA", t)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Yahoo Finance unavailable: {e}")
    assert store.get_cache_status("NVDA", "yfinance_fundamentals") is not None


# ── Phase 1.4 — SEC filing pipeline ────────────────────

def test_sec_filing_registration(store):
    """Phase 1.4: SEC filing registration and indexing."""
    result = store.register_filing("TEST", "10-Q", "2026-01-15", "2026-Q1",
                                   "TEST-ACCESSION", "https://sec.gov/...")
    assert result is True


# ── Phase 1.5 — Middleware ─────────────────────────────

def test_middleware_health():
    """Phase 1.5: Middleware health endpoint returns valid response."""
    from fastapi.testclient import TestClient
    from src.middleware.app import app
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code in (200, 503)


def test_intent_parser_detects_ticker():
    """Phase 1.5: Intent parser detects tickers in questions."""
    from src.middleware.intent_parser import IntentParser
    parser = IntentParser()
    intent = parser.parse("What is NVDA's revenue?")
    assert intent["ticker"] == "NVDA"
    assert "total_revenue" in intent["metrics"]


# ── Phase 1.6 — Multi-source ingestion ─────────────────

def test_fred_indicator_storage(store):
    """Phase 1.6: FRED indicators are stored in SQLite."""
    store.save_fundamental("MACRO", "GDP", 28.5, "trillion_usd", "2026-Q1",
                           source_type="fred")
    gdp = store.get_fundamental("MACRO", "GDP")
    assert gdp is not None


# ── Phase 1.7 — Scheduler + Resilience ─────────────────

def test_scheduler_status_report():
    """Phase 1.7: Scheduler status report returns valid structure."""
    from src.scheduler import UnifiedScheduler
    with patch("src.storage.store.ChromaStore") as mock_cls:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mock_cls.return_value = inst
        sched = UnifiedScheduler()
    report = sched.status_report()
    assert "sources" in report
    assert "timestamp" in report


def test_circuit_breaker_opens_and_recovers():
    """Phase 1.7: Circuit breaker opens after failures and recovers."""
    from src.utils.resilience import CircuitBreaker, CircuitBreakerOpenError
    breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=0.1)
    for _ in range(3):
        try:
            with breaker:
                raise ConnectionError("test failure")
        except (ConnectionError, CircuitBreakerOpenError):
            pass
    assert breaker.state == "OPEN"
    time.sleep(0.2)
    with breaker:
        pass  # Should succeed in HALF_OPEN -> CLOSED
    assert breaker.state == "CLOSED"

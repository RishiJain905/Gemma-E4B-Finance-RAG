"""
tests/test_phase1_7_e2e.py
End-to-end integration test for Phase 1.7 — verifies the full
scheduler -> resilience -> freshness -> query pipeline works together.

These tests use mocked external APIs but real internal components
(UnifiedScheduler, Store, CircuitBreaker, DeadLetterQueue, IRIngestor,
and the FastAPI middleware).
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.scheduler import UnifiedScheduler
from src.storage.store import Store
from src.utils.resilience import CircuitBreaker, CircuitBreakerOpenError, DeadLetterQueue


@pytest.fixture
def store(tmp_path):
    with patch("src.storage.store.ChromaStore") as mock_cls:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mock_cls.return_value = inst
        yield Store(db_path=tmp_path / "e2e.db", chroma_path=tmp_path / "chroma")


def _backdate(store, ticker, source, hours_ago):
    store.mark_source_fresh(ticker, source, 24)
    old = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with store.sqlite._connect() as conn:
        conn.execute(
            "UPDATE cache_meta SET last_updated=? WHERE ticker=? AND source=?",
            (old, ticker, source),
        )
        conn.commit()


# ════════════════════════════════════════════════════════
# Scenario 1: Full Daily Run
# ════════════════════════════════════════════════════════

def test_scenario_1_full_daily_run(store):
    scheduler = UnifiedScheduler(store=store, inter_source_delay=0)
    scheduler._run_source = MagicMock(return_value={"ok": True})

    result = scheduler.run_daily()

    # Daily sources ran; hourly/weekly-only sources were not part of the run.
    assert set(result) == {
        "yfinance", "fred", "sec_filings", "sec_companyfacts", "ir_pages", "estimates",
    }
    assert all(r["status"] == "success" for r in result.values())
    assert "gdelt" not in result
    assert "earnings_transcripts" not in result

    # cache_meta entries were created under the unified:* convention.
    for name in ("yfinance", "fred", "sec_filings", "ir_pages", "estimates"):
        cache = store.get_cache_status("SCHEDULER", f"unified:{name}")
        assert cache is not None and cache["status"] == "fresh"

    # status_report reflects the run.
    report = scheduler.status_report()
    assert report["sources"]["yfinance"]["status"] == "fresh"
    assert report["sources"]["gdelt"]["status"] == "never_fetched"


# ════════════════════════════════════════════════════════
# Scenario 2: Staleness-Aware Query
# ════════════════════════════════════════════════════════

def test_scenario_2_staleness_aware_query(store, monkeypatch):
    from fastapi.testclient import TestClient
    from src.middleware import app as middleware_app
    from src.middleware.app import app

    # Seed stale data (aged-out news) for NVDA.
    _backdate(store, "NVDA", "yfinance_news", hours_ago=48)

    with TestClient(app) as client:
        monkeypatch.setattr(middleware_app, "store", store)
        with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call, \
                patch("src.middleware.app._check_model_health", new_callable=AsyncMock, return_value=True), \
                patch("src.middleware.app._refresh_one_source") as mock_one:
            mock_call.return_value = ("NVDA answer", [])
            resp = client.post(
                "/query", json={"question": "What is NVDA revenue?", "refresh": True},
            )

    assert resp.status_code == 200
    data = resp.json()
    # Stale source was refreshed before answering.
    assert "yfinance_news" in data["freshness"]["refreshed_during_query"]
    assert mock_one.called


# ════════════════════════════════════════════════════════
# Scenario 3: Circuit Breaker Recovery
# ════════════════════════════════════════════════════════

def test_scenario_3_circuit_breaker_recovery():
    breaker = CircuitBreaker(name="gdelt", failure_threshold=3, recovery_timeout=0.05)

    # Drive failures until the breaker opens.
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with breaker:
                raise RuntimeError("API down")
    assert breaker.state == "OPEN"

    # While OPEN, calls are rejected immediately.
    with pytest.raises(CircuitBreakerOpenError):
        with breaker:
            pass

    # After the recovery timeout, a successful call closes the breaker.
    time.sleep(0.06)
    with breaker:
        pass  # success
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


# ════════════════════════════════════════════════════════
# Scenario 4: Dead-Letter Queue
# ════════════════════════════════════════════════════════

def test_scenario_4_dead_letter_queue(store):
    scheduler = UnifiedScheduler(store=store, inter_source_delay=0)
    # Make every source fail persistently.
    scheduler._run_source = MagicMock(side_effect=RuntimeError("persistent failure"))

    scheduler.run_all_stale()

    dlq = DeadLetterQueue(store)
    pending = dlq.get_pending()
    pending_sources = {p["source"] for p in pending}
    assert "fred" in pending_sources
    assert "gdelt" in pending_sources
    assert len(pending) == len(scheduler.SOURCES)

    # "Fix" the source and retry from the DLQ.
    assert dlq.retry("fred", "scheduler") is True
    remaining = {p["source"] for p in dlq.get_pending()}
    assert "fred" not in remaining


# ════════════════════════════════════════════════════════
# Scenario 5: IR Pages Integration
# ════════════════════════════════════════════════════════

def test_scenario_5_ir_pages_integration(store):
    from src.macros.ir_ingestor import IRIngestor

    fake_feed = {
        "entries": [
            {
                "title": "NVIDIA Announces New GPU Launch",
                "link": "https://investor.nvidia.com/news/1",
                "published": "2026-06-01",
                "summary": "NVIDIA today announced a new product launch.",
            },
            {
                "title": "NVIDIA Reports Q1 Financial Results",
                "link": "https://investor.nvidia.com/news/2",
                "published": "2026-06-02",
                "summary": "Quarterly financial results and non-GAAP reconciliations.",
            },
        ]
    }

    ingestor = IRIngestor(store=store)
    with patch("src.macros.ir_ingestor.feedparser.parse", return_value=fake_feed):
        result = ingestor.fetch_for_ticker("NVDA")

    assert result["status"] == "success"
    assert result["items_found"] == 2
    assert result["items_stored"] == 2

    # Documents were stored in ChromaDB with IR metadata + classification.
    assert store.chroma.add_document.called
    stored_meta = [c.kwargs["metadata"] for c in store.chroma.add_document.call_args_list]
    doc_types = {m["doc_type"] for m in stored_meta}
    assert "press_release" in doc_types          # "Announces ... Launch"
    assert "earnings_material" in doc_types       # "Earnings Supplemental"
    assert all(m["source"] == "ir" for m in stored_meta)

    # Freshness now shows IR data as fresh for NVDA.
    report = store.get_freshness_report("NVDA")
    assert report["sources"]["ir_pages"]["status"] == "fresh"

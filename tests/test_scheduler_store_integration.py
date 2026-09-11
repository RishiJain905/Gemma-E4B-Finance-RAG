"""
tests/test_scheduler_store_integration.py
Phase 1.8.3 — Integration tests for the scheduler -> store contract.

These verify that scheduler run modes write the correct cache_meta entries
to the store (so freshness/staleness tracking stays consistent) without
performing real network ingestion. The per-source ingestion itself is stubbed;
the focus is the orchestration + store bookkeeping.

Usage:
    pytest tests/test_scheduler_store_integration.py -v -m integration
"""

import sys
from unittest.mock import MagicMock, patch

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

pytestmark = pytest.mark.integration


@pytest.fixture
def store(tmp_path):
    with patch("src.storage.store.ChromaStore") as mock_cls:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mock_cls.return_value = inst
        yield Store(db_path=tmp_path / "sched_int.db", chroma_path=tmp_path / "chroma")


def test_scheduler_daily_updates_store(store):
    """After run_daily(force=True), daily sources are fresh in the store."""
    sched = UnifiedScheduler(store=store, inter_source_delay=0)
    sched.reset_schedule()
    sched._run_source = MagicMock(return_value={"ok": True})

    sched.run_daily(force=True)
    report = sched.status_report()
    for source in ["yfinance", "fred", "sec_filings", "ir_pages"]:
        assert report["sources"][source]["status"] in ("fresh", "success")


def test_scheduler_hourly_only_massive_news(store, monkeypatch):
    """run_hourly() runs Massive news; GDELT is daily-only now."""
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    sched = UnifiedScheduler(store=store, inter_source_delay=0)
    sched.reset_schedule()
    sched._run_source = MagicMock(return_value={"ok": True})

    results = sched.run_hourly(force=True)
    assert set(results) == {"massive_news"}
    sched._run_source.assert_called_once()
    assert sched._run_source.call_args.args[0] == "massive_news"
    assert "yfinance" not in results
    assert "fred" not in results
    assert "gdelt" not in results


def test_scheduler_weekly_marks_store(store):
    """run_weekly() selects legacy deep sources plus weekly registry feeds."""
    sched = UnifiedScheduler(store=store, inter_source_delay=0)
    sched.reset_schedule()
    sched._run_source = MagicMock(return_value={"ok": True})

    results = sched.run_weekly(force=True)
    assert set(results) == {"earnings_transcripts", "sec_filings", "cftc"}
    report = sched.status_report()
    assert report["sources"]["earnings_transcripts"]["status"] in ("fresh", "success")


def test_scheduler_error_marks_source_stale(store):
    """A failing source is recorded as stale in the store cache_meta."""
    sched = UnifiedScheduler(store=store, inter_source_delay=0)
    sched._run_source = MagicMock(side_effect=RuntimeError("kaboom"))

    sched.run_hourly(force=True)
    cache = store.get_cache_status("SCHEDULER", "unified:massive_news")
    assert cache is not None
    assert cache["status"] == "stale"


def test_scheduler_freshness_does_not_create_or_advance_a_cursor(store):
    """Successful cadence bookkeeping remains separate from incremental state."""
    sched = UnifiedScheduler(store=store, inter_source_delay=0)
    sched._run_source = MagicMock(return_value={"ok": True})

    sched.run_daily(force=True, source="fred")

    assert store.get_cache_status("SCHEDULER", "unified:fred")["status"] == "fresh"
    assert store.get_source_cursor("fred", "global") is None

"""
tests/test_filing_scheduler.py
Test suite for FilingScheduler — SEC filing discovery cadence and orchestration.

Two layers:
  1. Deterministic unit tests (no network) using tmp_path Store + mocked
     FilingProcessor to exercise cache TTL skip/stale/force, error handling,
     run_full_pipeline composition, status_report, and reset_discovery_cache.
  2. One LIVE incremental discovery test (EDGAR + llama-server :8087 skip-guarded)
     proving cache-aware scheduling without requiring full pipeline processing.
"""

import socket
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import httpx
import pytest

from src.sec import FilingScheduler
from src.storage.store import Store


CORE_TICKERS = ["T1", "T2", "T3"]
DISCOVERY_SOURCE = FilingScheduler.DISCOVERY_SOURCE

EDGAR_HOST = "www.sec.gov"
MODEL_ENDPOINT = "http://127.0.0.1:8087/v1/embeddings"


# ── Helpers / fixtures ──────────────────────────────

def _make_scheduler(tmp_path, processor=None):
    """Build a FilingScheduler on an isolated temp Store."""
    store = Store(
        db_path=tmp_path / "finance.db",
        chroma_path=tmp_path / "chroma",
    )
    proc = processor or MagicMock()
    scheduler = FilingScheduler(store=store, processor=proc)
    return scheduler, store, proc


def _patch_core_tickers(tickers=None):
    """Patch YFinanceIngestor to return a fixed core ticker list."""
    tickers = tickers or CORE_TICKERS
    mock_ingestor = MagicMock()
    mock_ingestor.core_tickers = tickers
    return patch(
        "src.ingestion.yfinance_ingestor.YFinanceIngestor",
        return_value=mock_ingestor,
    )


def _endpoint_reachable(url: str, timeout: float = 3.0) -> bool:
    """TCP-connect check for a host:port derived from a URL."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _model_alive(timeout: float = 5.0) -> bool:
    """Confirm the llama-server embeddings endpoint actually responds."""
    try:
        resp = httpx.post(
            MODEL_ENDPOINT,
            json={"input": "ping", "model": "tracealchemy"},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ── 1. run_discovery cache TTL skip ─────────────────

def test_run_discovery_skips_fresh_cache(tmp_path):
    """Tickers with fresh cache within TTL are skipped."""
    scheduler, store, proc = _make_scheduler(tmp_path)
    store.mark_cache_fresh("T1", DISCOVERY_SOURCE, 12)
    store.mark_cache_fresh("T2", DISCOVERY_SOURCE, 12)

    with _patch_core_tickers(["T1", "T2"]):
        result = scheduler.run_discovery(force=False)

    assert result["skipped"] == 2
    assert result["checked"] == 0
    assert result["new_filings"] == 0
    proc.discover_new_filings.assert_not_called()


# ── 2. run_discovery checks stale/missing cache ─────

def test_run_discovery_checks_stale_cache(tmp_path):
    """Stale or missing cache triggers discovery and marks cache fresh."""
    scheduler, store, proc = _make_scheduler(tmp_path)
    proc.discover_new_filings.side_effect = [2, 0]

    store.mark_cache_fresh("T1", DISCOVERY_SOURCE, 12)
    stale_time = datetime.now(timezone.utc) - timedelta(hours=13)
    with store.sqlite._connect() as conn:
        conn.execute(
            "UPDATE cache_meta SET last_updated = ? WHERE ticker = ? AND source = ?",
            (stale_time.strftime("%Y-%m-%d %H:%M:%S"), "T1", DISCOVERY_SOURCE),
        )
        conn.commit()

    with _patch_core_tickers(["T1", "T2"]):
        result = scheduler.run_discovery(force=False)

    assert result["checked"] == 2
    assert result["skipped"] == 0
    assert result["new_filings"] == 2
    assert result["details"] == {"T1": 2, "T2": 0}
    assert proc.discover_new_filings.call_count == 2

    assert store.get_cache_status("T1", DISCOVERY_SOURCE)["status"] == "fresh"
    assert store.get_cache_status("T2", DISCOVERY_SOURCE)["status"] == "fresh"


# ── 3. force=True bypasses freshness skip ───────────

def test_run_discovery_force_bypasses_cache(tmp_path):
    """force=True discovers all tickers even when cache is fresh."""
    scheduler, store, proc = _make_scheduler(tmp_path)
    proc.discover_new_filings.return_value = 1
    store.mark_cache_fresh("T1", DISCOVERY_SOURCE, 12)

    with _patch_core_tickers(["T1"]):
        result = scheduler.run_discovery(force=True)

    assert result["checked"] == 1
    assert result["skipped"] == 0
    proc.discover_new_filings.assert_called_once_with("T1")


# ── 4. discovery exception marks stale + details=-1 ─

def test_run_discovery_error_marks_stale(tmp_path):
    """Discovery exception marks cache stale and records details[ticker]=-1."""
    scheduler, store, proc = _make_scheduler(tmp_path)
    proc.discover_new_filings.side_effect = RuntimeError("EDGAR timeout")
    store.mark_cache_fresh("T1", DISCOVERY_SOURCE, 12)

    with _patch_core_tickers(["T1"]):
        result = scheduler.run_discovery(force=True)

    assert result["details"]["T1"] == -1
    assert result["checked"] == 0
    assert result["new_filings"] == 0
    cache = store.get_cache_status("T1", DISCOVERY_SOURCE)
    assert cache is not None
    assert cache["status"] == "stale"


# ── 5. run_full_pipeline composition ────────────────

def test_run_full_pipeline_composition(tmp_path):
    """run_full_pipeline composes discovery + process_pending_filings(limit=50)."""
    scheduler, store, proc = _make_scheduler(tmp_path)
    proc.discover_new_filings.return_value = 1
    proc.process_pending_filings.return_value = {
        "processed": 1,
        "failed": 0,
        "errors": [],
    }

    with _patch_core_tickers(["T1"]):
        report = scheduler.run_full_pipeline(force=True)

    proc.process_pending_filings.assert_called_once_with(limit=50)
    assert report["discovery"]["checked"] == 1
    assert report["discovery"]["new_filings"] == 1
    assert report["processing"]["processed"] == 1
    assert "timestamp" in report
    datetime.fromisoformat(report["timestamp"])


def test_registry_mode_runs_one_daily_index_discovery_not_per_ticker(tmp_path):
    daily_index = MagicMock()
    daily_index.discover.return_value = {
        "downloaded": 1, "registered": 4, "failed": 0, "errors": [],
    }
    scheduler, _store, proc = _make_scheduler(tmp_path)
    scheduler.daily_index_discovery = daily_index

    result = scheduler.run_discovery()

    assert result["mode"] == "daily_index"
    assert result["new_filings"] == 4
    daily_index.discover.assert_called_once_with()
    proc.discover_new_filings.assert_not_called()


# ── 6. status_report ────────────────────────────────

def test_status_report_never_checked_and_age_hours(tmp_path):
    """status_report reports never_checked and age_hours for fresh entries."""
    scheduler, store, proc = _make_scheduler(tmp_path)
    proc.status_report.return_value = {
        "total_unprocessed": 0,
        "total_parsed": 0,
        "filings_by_ticker": {},
    }

    store.mark_cache_fresh("T1", DISCOVERY_SOURCE, 12)

    with _patch_core_tickers(["T1", "T2"]):
        report = scheduler.status_report()

    by_ticker = {row["ticker"]: row for row in report["discovery"]}
    assert by_ticker["T2"]["status"] == "never_checked"
    assert by_ticker["T2"]["age_hours"] is None

    assert by_ticker["T1"]["status"] == "fresh"
    assert by_ticker["T1"]["age_hours"] is not None
    assert by_ticker["T1"]["age_hours"] >= 0
    assert report["pipeline"] == proc.status_report.return_value


def test_status_report_exposes_section_index_observability(tmp_path):
    scheduler, _store, proc = _make_scheduler(tmp_path)
    proc.status_report.return_value = {
        "total_unprocessed": 0,
        "total_index_pending": 2,
        "total_parsed": 4,
        "indexed_sections": 30,
        "indexed_chunks": 92,
        "filings_by_ticker": {},
    }

    with _patch_core_tickers([]):
        report = scheduler.status_report()

    assert report["pipeline"]["total_index_pending"] == 2
    assert report["pipeline"]["indexed_sections"] == 30
    assert report["pipeline"]["indexed_chunks"] == 92


# ── 7. reset_discovery_cache ────────────────────────

def test_reset_discovery_cache_marks_stale(tmp_path):
    """reset_discovery_cache marks all core tickers stale via upsert_cache_stale."""
    scheduler, store, proc = _make_scheduler(tmp_path)
    store.mark_cache_fresh("T1", DISCOVERY_SOURCE, 12)
    store.mark_cache_fresh("T2", DISCOVERY_SOURCE, 12)

    with _patch_core_tickers(["T1", "T2", "T3"]):
        scheduler.reset_discovery_cache()

    for ticker in ["T1", "T2", "T3"]:
        cache = store.get_cache_status(ticker, DISCOVERY_SOURCE)
        assert cache is not None
        assert cache["status"] == "stale"


# ── 8. LIVE incremental discovery (skip-guarded) ────

@pytest.mark.live
def test_live_scheduler_incremental_discovery(tmp_path):
    """Live: force discovery, second run skips, reset re-checks."""
    if not _endpoint_reachable(f"https://{EDGAR_HOST}"):
        pytest.skip("SEC EDGAR is unreachable")
    if not _model_alive():
        pytest.skip("TraceAlchemy model endpoint (port 8087) is unreachable")

    store = Store(
        db_path=tmp_path / "live.db",
        chroma_path=tmp_path / "live_chroma",
    )
    from src.sec.filing_processor import FilingProcessor

    processor = FilingProcessor(store=store)
    scheduler = FilingScheduler(store=store, processor=processor)

    with _patch_core_tickers(["AAPL"]):
        first = scheduler.run_discovery(force=True)
        assert first["checked"] >= 1, f"expected at least one ticker checked: {first}"

        second = scheduler.run_discovery(force=False)
        assert second["skipped"] >= 1, (
            f"expected cache-fresh skip on second run: {second}"
        )

        scheduler.reset_discovery_cache()

        third = scheduler.run_discovery(force=False)
        assert third["checked"] >= 1, (
            f"expected re-check after cache reset: {third}"
        )

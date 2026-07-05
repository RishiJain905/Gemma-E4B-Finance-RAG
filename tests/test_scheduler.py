"""
tests/test_scheduler.py
Comprehensive pytest suite for the Phase 1.7 unified scheduler.

Test coverage:
  1. UnifiedScheduler — initialization, source ordering, TTL loading
  2. Run modes — daily, hourly, weekly, all_stale
  3. Staggered execution — inter-source delays, partial failures
  4. Status reporting — freshness, errors, last-run timestamps
  5. CLI entry point — argument parsing, mode dispatch

Usage:
    pytest tests/test_scheduler.py -v
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Lightweight chromadb stub so importing Store never needs the real backend.
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


@pytest.fixture
def store(tmp_path):
    with patch("src.storage.store.ChromaStore") as mock_cls:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mock_cls.return_value = inst
        yield Store(db_path=tmp_path / "sched.db", chroma_path=tmp_path / "chroma")


@pytest.fixture
def scheduler(store):
    """Scheduler on an isolated store with no inter-source delay."""
    return UnifiedScheduler(store=store, inter_source_delay=0)


def _stub_run_source(scheduler, side_effect=None, return_value=None):
    """Patch the instance's _run_source so no real ingestion happens."""
    m = MagicMock()
    if side_effect is not None:
        m.side_effect = side_effect
    else:
        m.return_value = return_value if return_value is not None else {"ok": True}
    scheduler._run_source = m
    return m


# ════════════════════════════════════════════════════════
# 1. Initialization
# ════════════════════════════════════════════════════════

class TestUnifiedSchedulerInit:

    def test_import(self):
        from src.scheduler import UnifiedScheduler as US
        assert US is not None

    def test_init(self, scheduler):
        assert set(scheduler.SOURCES) == {
            "yfinance", "sec_filings", "fred", "gdelt", "earnings_transcripts",
            "ir_pages", "estimates",
        }

    def test_source_ordering(self, scheduler):
        ordered = [name for name, _ in scheduler._ordered_sources()]
        assert ordered == [
            "yfinance", "sec_filings", "fred", "gdelt", "earnings_transcripts",
            "ir_pages", "estimates",
        ]

    def test_ttl_loading(self, scheduler):
        # Values come from configs/watchlist.yaml schedule section.
        assert scheduler.ttls["fundamentals"] == 24
        assert scheduler.ttls["gdelt_news"] == 6
        assert scheduler.ttls["transcripts"] == 168

    def test_custom_store(self, store):
        sched = UnifiedScheduler(store=store)
        assert sched.store is store

    def test_custom_delay(self, store):
        sched = UnifiedScheduler(store=store, inter_source_delay=7.5)
        assert sched.inter_source_delay == 7.5


# ════════════════════════════════════════════════════════
# 2. Run modes
# ════════════════════════════════════════════════════════

class TestUnifiedSchedulerRunModes:

    def test_run_all_stale(self, scheduler):
        m = _stub_run_source(scheduler)
        result = scheduler.run_all_stale()
        assert set(result) == set(scheduler.SOURCES)
        assert all(r["status"] == "success" for r in result.values())
        assert m.call_count == len(scheduler.SOURCES)

    def test_run_all_stale_force(self, scheduler):
        # Mark everything fresh; force=True must still run all.
        for name in scheduler.SOURCES:
            scheduler.store.mark_cache_fresh(
                scheduler.SCHEDULER_TICKER, scheduler._scheduler_cache_source(name), 24,
            )
        m = _stub_run_source(scheduler)
        result = scheduler.run_all_stale(force=True)
        assert m.call_count == len(scheduler.SOURCES)
        assert all(r["status"] == "success" for r in result.values())

    def test_run_daily(self, scheduler):
        m = _stub_run_source(scheduler)
        result = scheduler.run_daily()
        assert set(result) == {"yfinance", "fred", "sec_filings", "ir_pages", "estimates"}
        assert "gdelt" not in result
        assert "earnings_transcripts" not in result
        assert m.call_count == 5

    def test_run_hourly(self, scheduler):
        m = _stub_run_source(scheduler)
        result = scheduler.run_hourly()
        assert set(result) == {"gdelt"}
        assert m.call_count == 1

    def test_run_weekly(self, scheduler):
        m = _stub_run_source(scheduler)
        result = scheduler.run_weekly()
        assert set(result) == {"earnings_transcripts", "sec_filings"}
        # SEC runs in deep mode during the weekly batch.
        sec_call = [c for c in m.call_args_list if c.args[0] == "sec_filings"][0]
        assert sec_call.kwargs.get("deep") is True

    def test_run_empty(self, scheduler):
        for name in scheduler.SOURCES:
            scheduler.store.mark_cache_fresh(
                scheduler.SCHEDULER_TICKER, scheduler._scheduler_cache_source(name), 24,
            )
        m = _stub_run_source(scheduler)
        result = scheduler.run_all_stale(force=False)
        assert all(r["status"] == "skipped" for r in result.values())
        m.assert_not_called()


# ════════════════════════════════════════════════════════
# 3. Partial failure / staggering
# ════════════════════════════════════════════════════════

class TestUnifiedSchedulerPartialFailure:

    def test_one_source_fails(self, scheduler):
        def side(name, deep=False, force=False):
            if name == "fred":
                raise RuntimeError("FRED down")
            return {"ok": True}
        _stub_run_source(scheduler, side_effect=side)
        result = scheduler.run_all_stale()
        assert result["fred"]["status"] == "error"
        assert "FRED down" in result["fred"]["error"]
        # Sources after the failing one still ran.
        assert result["gdelt"]["status"] == "success"
        assert result["earnings_transcripts"]["status"] == "success"

    def test_all_sources_fail(self, scheduler):
        _stub_run_source(scheduler, side_effect=RuntimeError("boom"))
        result = scheduler.run_all_stale()
        assert all(r["status"] == "error" for r in result.values())

    def test_source_skipped(self, scheduler):
        scheduler.store.mark_cache_fresh(
            scheduler.SCHEDULER_TICKER,
            scheduler._scheduler_cache_source("yfinance"), 24,
        )
        m = _stub_run_source(scheduler)
        result = scheduler.run_all_stale(force=False)
        assert result["yfinance"]["status"] == "skipped"
        ran = [c.args[0] for c in m.call_args_list]
        assert "yfinance" not in ran

    def test_stagger_delay(self, store):
        sched = UnifiedScheduler(store=store, inter_source_delay=0.01)
        _stub_run_source(sched)
        with patch("src.scheduler.time.sleep") as mock_sleep:
            sched.run_all_stale()
        # N sources => N-1 inter-source delays.
        assert mock_sleep.call_count == len(sched.SOURCES) - 1


# ════════════════════════════════════════════════════════
# 4. Status reporting
# ════════════════════════════════════════════════════════

class TestUnifiedSchedulerStatus:

    def test_status_report(self, scheduler):
        report = scheduler.status_report()
        assert set(report["sources"]) == set(scheduler.SOURCES)
        assert "timestamp" in report

    def test_status_after_run(self, scheduler):
        _stub_run_source(scheduler)
        scheduler.run_all_stale()
        report = scheduler.status_report()
        assert report["sources"]["yfinance"]["status"] == "fresh"
        assert report["sources"]["yfinance"]["age_hours"] is not None

    def test_status_never_run(self, scheduler):
        report = scheduler.status_report()
        assert all(s["status"] == "never_fetched" for s in report["sources"].values())

    def test_status_with_errors(self, scheduler):
        _stub_run_source(scheduler, side_effect=RuntimeError("kaboom"))
        scheduler.run_all_stale()
        report = scheduler.status_report()
        assert report["sources"]["fred"]["status"] == "stale"
        assert report["sources"]["fred"]["error"] is not None


# ════════════════════════════════════════════════════════
# 5. CLI entry point
# ════════════════════════════════════════════════════════

class TestUnifiedSchedulerCLI:

    def _run_cli(self, argv):
        from src import scheduler as sched_mod
        mock_instance = MagicMock()
        mock_instance.run_daily.return_value = {}
        mock_instance.run_hourly.return_value = {}
        mock_instance.run_weekly.return_value = {}
        mock_instance.run_all_stale.return_value = {}
        mock_instance.status_report.return_value = {}
        with patch.object(sched_mod, "UnifiedScheduler", return_value=mock_instance), \
                patch.object(sys, "argv", ["prog"] + argv):
            sched_mod.main()
        return mock_instance

    def test_cli_daily(self):
        inst = self._run_cli(["daily"])
        inst.run_daily.assert_called_once_with(force=False)

    def test_cli_hourly(self):
        inst = self._run_cli(["hourly"])
        inst.run_hourly.assert_called_once_with(force=False)

    def test_cli_weekly(self):
        inst = self._run_cli(["weekly"])
        inst.run_weekly.assert_called_once_with(force=False)

    def test_cli_all(self):
        inst = self._run_cli(["all"])
        inst.run_all_stale.assert_called_once_with(force=False)

    def test_cli_status(self):
        inst = self._run_cli(["status"])
        inst.status_report.assert_called_once()

    def test_cli_force(self):
        inst = self._run_cli(["all", "--force"])
        inst.run_all_stale.assert_called_once_with(force=True)

    def test_cli_invalid_mode(self):
        with pytest.raises(SystemExit):
            self._run_cli(["bogus"])

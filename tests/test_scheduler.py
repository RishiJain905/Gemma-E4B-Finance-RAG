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
from dataclasses import replace
from datetime import datetime, timedelta, timezone
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
from src.scheduler.budget import RunBudget
from src.scheduler.source_registry import SourceRegistry
from src.ingestion.errors import ErrorClass, ProviderError
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
def scheduler(store, monkeypatch):
    """Scheduler on an isolated store with no inter-source delay."""
    for name in (
        "FINNHUB_API_KEY",
        "MASSIVE_API_KEY",
        "BLS_API_KEY",
        "BEA_API_KEY",
        "EIA_API_KEY",
        "OPENFDA_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "FMP_API_KEY",
        "MARKETAUX_API_KEY",
    ):
        monkeypatch.setenv(name, "test-key")
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
        assert {
            "yfinance", "sec_filings", "sec_companyfacts", "fred", "gdelt",
            "earnings_transcripts", "ir_pages", "estimates", "finnhub", "massive",
            "massive_news", "alpha_vantage", "fmp", "marketaux", "openfigi",
            "universe_nasdaq100", "universe_ivv", "universe_sec", "federal_reserve",
            "treasury", "bls", "bea", "eia", "ny_fed", "cftc", "openfda",
            "nhtsa", "usaspending",
        } <= set(scheduler.SOURCES)
        assert scheduler.registry.version == "2.3.4.2-free-adapters"

    def test_source_ordering(self, scheduler):
        ordered = [name for name, _ in scheduler._ordered_sources()]
        assert ordered[:3] == [
            "universe_nasdaq100", "universe_ivv", "universe_sec",
        ]
        assert ordered.index("yfinance") < ordered.index("sec_filings")
        assert ordered.index("gdelt") < ordered.index("earnings_transcripts")

    def test_ttl_loading(self, scheduler):
        # Values come from configs/watchlist.yaml schedule section.
        assert scheduler.ttls["fundamentals"] == 24
        assert scheduler.ttls["gdelt_news"] == 24
        assert scheduler.ttls["transcripts"] == 168
        assert scheduler.ttls["alpha_vantage"] == 24
        assert scheduler.ttls["marketaux_news"] == 24

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
        assert {
            "yfinance", "fred", "sec_filings", "sec_companyfacts", "ir_pages", "estimates",
            "finnhub", "massive", "universe_nasdaq100", "universe_ivv", "universe_sec",
            "federal_reserve", "treasury", "bls", "bea", "eia", "ny_fed",
            "openfda", "nhtsa", "usaspending",
            "alpha_vantage", "fmp", "marketaux", "openfigi", "gdelt",
        } <= set(result)
        assert "earnings_transcripts" not in result
        assert "cftc" not in result
        assert m.call_count == len(result)

    def test_run_hourly(self, scheduler):
        m = _stub_run_source(scheduler)
        result = scheduler.run_hourly()
        assert set(result) == {"massive_news"}
        assert m.call_count == 1
        assert m.call_args.args[0] == "massive_news"

    def test_run_weekly(self, scheduler):
        m = _stub_run_source(scheduler)
        result = scheduler.run_weekly()
        assert set(result) == {"earnings_transcripts", "sec_filings", "cftc"}
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

    def test_massive_news_rate_limit_does_not_stop_other_sources(self, scheduler):
        """A failed Massive news run is stale while later sources still execute."""
        def side(name, deep=False, force=False):
            if name == "massive_news":
                raise RuntimeError("Massive rate limit exhausted")
            return {"ok": True}

        mock_run = _stub_run_source(scheduler, side_effect=side)
        result = scheduler.run_all_stale()

        assert result["massive_news"]["status"] == "error"
        assert "rate limit" in result["massive_news"]["error"]
        assert result["earnings_transcripts"]["status"] == "success"
        assert result["ir_pages"]["status"] == "success"
        assert result["estimates"]["status"] == "success"
        ran_sources = [call.args[0] for call in mock_run.call_args_list]
        assert ran_sources[-3:] == ["earnings_transcripts", "ir_pages", "estimates"]
        news_status = scheduler.status_report()["sources"]["massive_news"]
        assert news_status["status"] == "stale"
        assert "rate limit" in news_status["error"]

    def test_provider_circuit_failure_is_observable_and_next_source_runs(
        self, scheduler
    ):
        reset_at = "2026-07-14T12:05:00Z"

        def side(name, deep=False, force=False):
            if name == "massive_news":
                raise ProviderError(
                    "token=secret provider throttled",
                    error_class=ErrorClass.RATE_LIMITED,
                    retry_after=60,
                    reset_at=reset_at,
                    attempts=3,
                    circuit_open=True,
                )
            return {"ok": True}

        mock_run = _stub_run_source(scheduler, side_effect=side)
        result = scheduler.run_all_stale(force=True)

        massive_news = result["massive_news"]
        assert massive_news["terminal_status"] == "skipped"
        assert massive_news["error_class"] == "rate_limited"
        assert massive_news["attempts"] == 3
        assert massive_news["reset_at"] == reset_at
        assert massive_news["remaining_work_skipped"] is True
        assert "secret" not in massive_news["error"]
        assert result["earnings_transcripts"]["status"] == "success"
        assert "earnings_transcripts" in [
            call.args[0] for call in mock_run.call_args_list
        ]
        provider_status = scheduler.status_report(source="massive_news")["sources"][
            "massive_news"
        ]["provider"]
        assert provider_status["error_class"] == "rate_limited"
        assert provider_status["reset_at"] == reset_at
        assert provider_status["circuit_opened_at"] is not None

    def test_persisted_provider_cooldown_skips_then_resumes(
        self, scheduler
    ):
        now = [datetime(2026, 7, 14, 23, 58, tzinfo=timezone.utc)]
        scheduler._now_fn = lambda: now[0]
        reset_at = now[0] + timedelta(minutes=5)
        scheduler.store.record_source_budget_usage(
            "finnhub",
            day_start=now[0].date().isoformat(),
            minute_start=now[0].strftime("%Y-%m-%dT%H:%MZ"),
            attempted_requests=0,
            successful_requests=0,
            provider_remaining=0,
            provider_reset=reset_at.isoformat().replace("+00:00", "Z"),
        )
        scheduler.store.set_source_cursor(
            "finnhub",
            "__provider__",
            reset_at.isoformat().replace("+00:00", "Z"),
            cursor_type="timestamp",
            status="circuit_open",
            error_class="rate_limited",
            retry_after=300,
        )
        scheduler.coverage.tickers_for = MagicMock(return_value=["AAA"])
        mock_run = _stub_run_source(scheduler, return_value={"status": "ok"})

        during = scheduler.run_daily(force=True, source="finnhub")
        now[0] = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        next_day = scheduler.run_daily(force=True, source="finnhub")
        now[0] = reset_at + timedelta(seconds=1)
        after = scheduler.run_daily(force=True, source="finnhub")

        assert during["finnhub"]["reason"] == "provider_cooldown"
        assert during["finnhub"]["remaining_work_skipped"] is True
        assert next_day["finnhub"]["reason"] == "provider_cooldown"
        assert mock_run.call_count == 1
        assert after["finnhub"]["status"] == "success"

    def test_returned_rate_limit_is_skipped_and_remains_stale(self, scheduler):
        def side(name, deep=False, force=False):
            if name == "massive_news":
                return {
                    "status": "rate_limited",
                    "requests": 1,
                    "retry_after": 60,
                }
            return {"ok": True}

        mock_run = _stub_run_source(scheduler, side_effect=side)
        result = scheduler.run_all_stale(force=True)

        assert result["massive_news"]["status"] == "skipped"
        assert result["massive_news"]["reason"] == "rate_limited"
        assert result["earnings_transcripts"]["status"] == "success"
        assert scheduler.store.get_cache_status(
            "SCHEDULER", "unified:massive_news"
        )["status"] == "stale"
        assert "earnings_transcripts" in [
            call.args[0] for call in mock_run.call_args_list
        ]

    def test_success_does_not_open_circuit_for_stray_error_class(self, scheduler):
        """Healthy run_status must not circuit-break on a leftover detail.error_class."""

        def side(name, deep=False, force=False):
            if name == "fmp":
                return {
                    "status": "ok",
                    "stored": 4,
                    "requests": 2,
                    "error_class": "entitlement",
                }
            return {"ok": True}

        _stub_run_source(scheduler, side_effect=side)
        result = scheduler.run_daily(force=True, source="fmp")

        assert result["fmp"]["status"] == "success"
        assert result["fmp"].get("remaining_work_skipped") is not True
        budget = result["fmp"].get("budget") or {}
        # open_provider_circuit sets provider_remaining to 0
        assert budget.get("provider_remaining") != 0
        assert budget.get("provider_cooldown") is not True
        provider = (
            scheduler.status_report(source="fmp")["sources"]["fmp"].get("provider") or {}
        )
        assert provider.get("status") != "circuit_open"
        cursor = scheduler.store.get_source_cursor_state("fmp", "__provider__")
        if cursor is not None:
            assert cursor.get("status") != "circuit_open"

    def test_nested_provider_failures_are_not_reported_as_partial_success(self):
        status, reason = UnifiedScheduler._classify_detail(
            {
                "market": {"status": "rate_limited"},
                "corporate_actions": {"status": "error"},
            }
        )

        assert (status, reason) == ("error", "provider_error")

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

    def test_missing_key_source_is_visible_and_does_not_block_public_source(
        self, store
    ):
        registry = SourceRegistry.load(environ={})
        sched = UnifiedScheduler(store=store, registry=registry, inter_source_delay=0)
        mock_run = _stub_run_source(sched)

        result = sched.run_daily(force=True)

        assert result["finnhub"]["status"] == "skipped"
        assert result["finnhub"]["reason"] == "disabled_missing_key"
        assert result["finnhub"]["detail"] == "missing FINNHUB_API_KEY"
        assert result["finnhub"]["error_class"] == "authentication"
        assert result["finnhub"]["remaining_work_skipped"] is True
        assert result["fred"]["status"] == "success"
        assert "finnhub" not in [call.args[0] for call in mock_run.call_args_list]

    def test_force_does_not_bypass_source_budget(self, scheduler, monkeypatch):
        monkeypatch.setattr(
            scheduler,
            "_new_budget",
            lambda _spec: RunBudget(
                requests_per_minute=1,
                requests_per_day=1,
                requests_per_run=1,
                max_work_items=1,
                day_requests=1,
            ),
        )
        mock_run = _stub_run_source(scheduler)

        result = scheduler.run_daily(force=True, source="yfinance")

        assert result["yfinance"]["status"] == "skipped"
        assert result["yfinance"]["reason"] == "budget_exhausted"
        mock_run.assert_not_called()

    def test_repeated_forced_runs_share_in_process_safety_usage(self, scheduler):
        scheduler.registry.sources["yfinance"] = replace(
            scheduler.registry.get("yfinance"),
            requests_per_minute=1,
            requests_per_day=1,
            requests_per_run=1,
        )
        scheduler.SOURCES["yfinance"] = scheduler.registry.get("yfinance")
        mock_run = _stub_run_source(scheduler)

        first = scheduler.run_daily(force=True, source="yfinance")
        restarted = UnifiedScheduler(
            store=scheduler.store,
            registry=scheduler.registry,
            coverage_resolver=scheduler.coverage,
            inter_source_delay=0,
        )
        restarted.SOURCES["yfinance"] = scheduler.registry.get("yfinance")
        restarted_run = _stub_run_source(restarted)
        second = restarted.run_daily(force=True, source="yfinance")

        assert first["yfinance"]["status"] == "success"
        assert second["yfinance"]["status"] == "skipped"
        assert second["yfinance"]["reason"] == "budget_exhausted"
        assert mock_run.call_count == 1
        restarted_run.assert_not_called()

    def test_preflight_failure_does_not_stop_later_sources(self, scheduler):
        original_is_stale = scheduler._is_stale

        def is_stale(name):
            if name == "fred":
                raise RuntimeError("freshness database unavailable")
            return original_is_stale(name)

        scheduler._is_stale = is_stale
        mock_run = _stub_run_source(scheduler)

        result = scheduler.run_all_stale()

        assert result["fred"]["status"] == "error"
        assert result["federal_reserve"]["status"] == "success"
        assert "federal_reserve" in [call.args[0] for call in mock_run.call_args_list]

    def test_massive_overlap_is_taken_from_registry(self, scheduler, monkeypatch):
        ingestor_class = MagicMock()
        ingestor_class.return_value.ingest_all.return_value = {"market": {"status": "ok"}}
        budgeted_get = MagicMock(return_value=MagicMock())
        scheduler._budgeted_http_get = budgeted_get
        monkeypatch.setattr(
            "src.ingestion.massive_ingestor.MassiveIngestor", ingestor_class
        )
        scheduler._active_budgets["massive"] = scheduler._new_budget(
            scheduler.registry.get("massive")
        )

        scheduler._run_source("massive")

        assert ingestor_class.call_args.kwargs["overlap_days"] == 2
        budgeted_get.assert_called_once_with("massive", wait_for_minute=True)

    def test_massive_news_shares_daily_massive_provider_usage(self, scheduler):
        now = datetime(2026, 7, 18, 15, 30, tzinfo=timezone.utc)
        scheduler._now_fn = lambda: now
        scheduler.store.record_source_budget_usage(
            "massive",
            day_start=now.date().isoformat(),
            minute_start="2026-07-18T15:29Z",
            attempted_requests=7,
            successful_requests=7,
        )

        budget = scheduler._new_budget(scheduler.registry.get("massive_news"))

        assert budget.day_requests == 7
        assert budget.reserve(requests=1, work_items=1)
        scheduler._remember_budget("massive_news", budget)
        shared = scheduler.store.get_source_budget_usage(
            "massive",
            day_start=now.date().isoformat(),
            minute_start=now.strftime("%Y-%m-%dT%H:%MZ"),
        )
        separate = scheduler.store.get_source_budget_usage(
            "massive_news",
            day_start=now.date().isoformat(),
            minute_start=now.strftime("%Y-%m-%dT%H:%MZ"),
        )
        assert shared["day_requests"] == 8
        assert separate["day_requests"] == 0

    def test_massive_news_dispatch_is_separate_from_daily_massive(
        self, scheduler, monkeypatch
    ):
        ingestor_class = MagicMock()
        ingestor_class.return_value.ingest_news.return_value = {"status": "ok"}
        budgeted_get = MagicMock(return_value=MagicMock())
        scheduler._budgeted_http_get = budgeted_get
        monkeypatch.setattr(
            "src.ingestion.massive_ingestor.MassiveIngestor", ingestor_class
        )

        result = scheduler._run_source("massive_news")

        assert result == {"status": "ok"}
        assert ingestor_class.call_args.kwargs["news_overlap_hours"] == 2
        ingestor_class.return_value.ingest_news.assert_called_once_with()
        ingestor_class.return_value.ingest_all.assert_not_called()
        budgeted_get.assert_called_once_with("massive_news", wait_for_minute=True)

    def test_stagger_delay(self, store):
        registry = SourceRegistry.load(
            environ={
                "FINNHUB_API_KEY": "x", "MASSIVE_API_KEY": "x", "BLS_API_KEY": "x",
                "BEA_API_KEY": "x", "EIA_API_KEY": "x", "OPENFDA_API_KEY": "x",
            }
        )
        sched = UnifiedScheduler(
            store=store, registry=registry, inter_source_delay=0.01
        )
        _stub_run_source(sched)
        with patch("src.scheduler.time.sleep") as mock_sleep:
            sched.run_all_stale()
        available = sum(spec.is_available for spec in sched.SOURCES.values())
        assert mock_sleep.call_count == available - 1

    def test_partial_finnhub_run_resumes_never_fetched_tickers(
        self, scheduler, monkeypatch
    ):
        scheduler.registry.sources["finnhub"] = replace(
            scheduler.registry.get("finnhub"),
            requests_per_minute=2,
            requests_per_day=2,
            requests_per_run=2,
            max_work_items_per_run=2,
        )
        scheduler.SOURCES["finnhub"] = scheduler.registry.get("finnhub")
        scheduler.coverage.tickers_for = MagicMock(
            return_value=["AAA", "BBB", "CCC", "DDD"]
        )
        calls = []

        def ingest_news(*, tickers):
            calls.append(tickers)
            for ticker in tickers:
                scheduler.cursors.advance(
                    "finnhub_news",
                    ticker,
                    f"2026-07-{10 + len(calls):02d}T00:00:00Z",
                    kind="timestamp",
                )
            return {"status": "ok", "requests": len(tickers), "tickers": len(tickers)}

        ingestor = MagicMock()
        ingestor.ingest_news.side_effect = ingest_news
        monkeypatch.setattr(
            "src.ingestion.finnhub_ingestor.FinnhubIngestor",
            MagicMock(return_value=ingestor),
        )

        scheduler.run_daily(force=True, source="finnhub")
        scheduler.run_daily(force=True, source="finnhub")

        assert calls == [["AAA", "BBB"], ["CCC", "DDD"]]
        constructor = __import__(
            "src.ingestion.finnhub_ingestor", fromlist=["FinnhubIngestor"]
        ).FinnhubIngestor
        assert constructor.call_args.kwargs["http_get"] is not None


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
        assert all(
            source["status"] == "never_fetched"
            for source in report["sources"].values()
        )

    def test_status_with_errors(self, scheduler):
        _stub_run_source(scheduler, side_effect=RuntimeError("kaboom"))
        scheduler.run_all_stale()
        report = scheduler.status_report()
        assert report["sources"]["fred"]["status"] == "stale"
        assert report["sources"]["fred"]["error"] is not None

    def test_invalid_registry_entry_is_always_visible_in_status(self, scheduler):
        scheduler.registry.sources["yfinance"] = replace(
            scheduler.registry.get("yfinance"),
            enabled=False,
            run_modes=(),
            status="invalid_configuration",
            disabled_reason="missing fields: cadence",
        )
        scheduler.SOURCES["yfinance"] = scheduler.registry.get("yfinance")

        report = scheduler.status_report(source="yfinance")

        assert report["sources"]["yfinance"]["status"] == "invalid_configuration"
        assert report["sources"]["yfinance"]["error"] == "missing fields: cadence"


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

    def test_cli_source_and_scope_filters(self):
        inst = self._run_cli(["daily", "--source", "fred", "--scope", "global"])
        inst.run_daily.assert_called_once_with(
            force=False, source="fred", scope="global"
        )

    def test_cli_invalid_mode(self):
        with pytest.raises(SystemExit):
            self._run_cli(["bogus"])

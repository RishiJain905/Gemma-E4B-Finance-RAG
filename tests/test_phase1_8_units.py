"""
tests/test_phase1_8_units.py
Phase 1.8.3 — targeted unit tests filling coverage gaps for the intent
parser, store freshness/staleness logic, scheduler run/failure modes, and
the resilience primitives (retry, circuit breaker, dead-letter queue).

These tests are pure-logic: ChromaDB is stubbed so no embedding server is
required.

Usage:
    pytest tests/test_phase1_8_units.py -v
"""

import sys
import time
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

from src.middleware.intent_parser import IntentParser
from src.scheduler import UnifiedScheduler
from src.storage.store import Store
from src.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    DeadLetterQueue,
    retry_with_backoff,
)


@pytest.fixture
def store(tmp_path):
    """Store with real SQLite (tmp) and a mocked ChromaStore."""
    with patch("src.storage.store.ChromaStore") as mock_cls:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mock_cls.return_value = inst
        yield Store(db_path=tmp_path / "units.db", chroma_path=tmp_path / "chroma")


def _backdate(store: Store, ticker: str, cache_source: str, hours_ago: float):
    """Insert a fresh cache row then backdate last_updated to age it."""
    store.mark_source_fresh(ticker, cache_source, 24)
    old = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    with store.sqlite._connect() as conn:
        conn.execute(
            "UPDATE cache_meta SET last_updated=? WHERE ticker=? AND source=?",
            (old, ticker, cache_source),
        )
        conn.commit()


# ════════════════════════════════════════════════════════
# Intent parser
# ════════════════════════════════════════════════════════

class TestIntentParserEdges:

    @pytest.fixture
    def parser(self):
        return IntentParser()

    def test_parse_empty_string(self, parser):
        intent = parser.parse("")
        assert intent["ticker"] is None
        assert intent["metrics"] == []
        assert intent["question_type"] == "general"

    def test_parse_no_ticker(self, parser):
        intent = parser.parse("How is the economy doing overall?")
        assert intent["ticker"] is None

    def test_parse_multiple_tickers(self, parser):
        # Company-name detection returns the first match; ensure both are known.
        intent = parser.parse("Compare NVDA and AMD revenue")
        assert intent["ticker"] in ("NVDA", "AMD")
        assert intent["question_type"] == "comparison"

    def test_parse_company_name_variants(self, parser):
        for q in ["Nvidia revenue", "nVidia revenue", "NVIDIA revenue"]:
            assert parser.parse(q)["ticker"] == "NVDA"

    def test_parse_timeframe_edge_cases(self, parser):
        assert parser.parse("NVDA revenue FY2025")["timeframe"] is not None
        assert parser.parse("NVDA TTM revenue")["timeframe_type"] == "ttm"
        assert parser.parse("NVDA revenue YTD")["timeframe_type"] == "ytd"

    def test_parse_question_type_all(self, parser):
        cases = {
            "What is NVDA revenue?": "fact_lookup",
            "Compare NVDA vs AMD": "comparison",
            "How has NVDA revenue trended over time?": "trend",
            "Why did NVDA stock drop?": "explanation",
            "What is the market sentiment on NVDA?": "sentiment",
            "What is the latest news on NVDA?": "news",
            "What are the risks for NVDA?": "risk",
        }
        for question, expected in cases.items():
            assert parser.parse(question)["question_type"] == expected


# ════════════════════════════════════════════════════════
# Store freshness / staleness
# ════════════════════════════════════════════════════════

class TestStoreFreshness:

    def test_freshness_report_never_fetched(self, store):
        report = store.get_freshness_report("ZZZZ")
        assert report["overall"] == "never_fetched"
        assert all(s["status"] == "never_fetched" for s in report["sources"].values())

    def test_freshness_report_partial(self, store):
        store.mark_source_fresh("NVDA", "yfinance_fundamentals", 24)
        _backdate(store, "NVDA", "yfinance_news", hours_ago=48)
        report = store.get_freshness_report("NVDA")
        assert report["overall"] == "partial"
        assert "yfinance_news" in report["stale_sources"]

    def test_freshness_report_all_fresh(self, store):
        for name, cfg in Store.FRESHNESS_SOURCES.items():
            store.mark_source_fresh("NVDA", cfg["cache_source"], 24)
        report = store.get_freshness_report("NVDA")
        assert report["overall"] == "fresh"
        assert report["stale_sources"] == []

    def test_freshness_report_all_stale(self, store):
        for name, cfg in Store.FRESHNESS_SOURCES.items():
            _backdate(store, "NVDA", cfg["cache_source"], hours_ago=1000)
        report = store.get_freshness_report("NVDA")
        assert report["overall"] == "stale"
        assert len(report["stale_sources"]) == len(Store.FRESHNESS_SOURCES)

    def test_get_stale_tickers_empty(self, store):
        assert store.get_stale_tickers("yfinance_fundamentals") == []

    def test_get_stale_tickers_multiple(self, store):
        _backdate(store, "NVDA", "yfinance_fundamentals", hours_ago=100)
        _backdate(store, "AMD", "yfinance_fundamentals", hours_ago=100)
        store.mark_source_fresh("AAPL", "yfinance_fundamentals", 24)  # fresh
        stale = set(store.get_stale_tickers("yfinance_fundamentals"))
        assert {"NVDA", "AMD"} <= stale
        assert "AAPL" not in stale

    def test_age_hours_none(self):
        assert Store._age_hours(None) is None

    def test_age_hours_fresh(self):
        recent = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        age = Store._age_hours(recent)
        assert age is not None and age < 1

    def test_age_hours_stale(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=100)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        age = Store._age_hours(old)
        assert age is not None and age > 90


# ════════════════════════════════════════════════════════
# Scheduler init / staleness / failure modes
# ════════════════════════════════════════════════════════

class TestSchedulerUnits:

    def test_scheduler_init_custom_store(self, store):
        sched = UnifiedScheduler(store=store)
        assert sched.store is store

    def test_scheduler_init_custom_delay(self, store):
        sched = UnifiedScheduler(store=store, inter_source_delay=9.0)
        assert sched.inter_source_delay == 9.0

    def test_scheduler_ordered_sources(self, store):
        sched = UnifiedScheduler(store=store)
        weights = [cfg["weight"] for _, cfg in sched._ordered_sources()]
        assert weights == sorted(weights)

    def test_scheduler_is_stale_never_run(self, store):
        sched = UnifiedScheduler(store=store)
        assert sched._is_stale("yfinance") is True

    def test_scheduler_is_stale_marked_stale(self, store):
        sched = UnifiedScheduler(store=store)
        store.upsert_cache_stale("SCHEDULER", sched._scheduler_cache_source("fred"))
        assert sched._is_stale("fred") is True

    def test_scheduler_is_stale_fresh(self, store):
        sched = UnifiedScheduler(store=store)
        store.mark_cache_fresh("SCHEDULER", sched._scheduler_cache_source("fred"), 24)
        assert sched._is_stale("fred") is False

    def test_scheduler_is_stale_expired(self, store):
        sched = UnifiedScheduler(store=store)
        _backdate(store, "SCHEDULER", sched._scheduler_cache_source("fred"),
                  hours_ago=1000)
        assert sched._is_stale("fred") is True

    def test_scheduler_run_empty(self, store):
        """All sources fresh => nothing runs without force."""
        sched = UnifiedScheduler(store=store, inter_source_delay=0)
        for name in sched.SOURCES:
            store.mark_cache_fresh("SCHEDULER", sched._scheduler_cache_source(name), 24)
        sched._run_source = MagicMock(return_value={"ok": True})
        result = sched.run_all_stale(force=False)
        assert all(r["status"] == "skipped" for r in result.values())
        sched._run_source.assert_not_called()

    def test_scheduler_reset_schedule(self, store):
        sched = UnifiedScheduler(store=store)
        for name in sched.SOURCES:
            store.mark_cache_fresh("SCHEDULER", sched._scheduler_cache_source(name), 24)
        sched.reset_schedule()
        assert all(sched._is_stale(name) for name in sched.SOURCES)

    def test_scheduler_dlq_unavailable(self, store):
        """Scheduler degrades gracefully when the DLQ cannot initialize."""
        sched = UnifiedScheduler(store=store, inter_source_delay=0)
        with patch("src.utils.resilience.DeadLetterQueue",
                   side_effect=RuntimeError("no dlq")):
            sched._dlq = None  # force re-evaluation
            assert sched.dlq is None

    def test_scheduler_partial_failure_isolated(self, store):
        """One failing source does not stop the others."""
        sched = UnifiedScheduler(store=store, inter_source_delay=0)

        def _side(name, **kw):
            if name == "fred":
                raise RuntimeError("boom")
            return {"ok": True}

        sched._run_source = MagicMock(side_effect=_side)
        result = sched.run_all_stale(force=True)
        assert result["fred"]["status"] == "error"
        assert result["yfinance"]["status"] == "success"


# ════════════════════════════════════════════════════════
# Resilience: retry, circuit breaker, dead-letter queue
# ════════════════════════════════════════════════════════

class TestRetry:

    def test_retry_non_retryable_exception(self):
        calls = {"n": 0}

        @retry_with_backoff(max_attempts=3, base_delay=0.001,
                            retryable_exceptions=(ConnectionError,))
        def boom():
            calls["n"] += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            boom()
        assert calls["n"] == 1  # never retried

    def test_retry_backoff_increases(self):
        delays = []

        @retry_with_backoff(max_attempts=4, base_delay=0.01, jitter=False,
                            retryable_exceptions=(ConnectionError,),
                            on_retry=lambda a, e, d: delays.append(d))
        def always_fail():
            raise ConnectionError("x")

        with pytest.raises(ConnectionError):
            always_fail()
        assert delays == sorted(delays)
        assert delays[-1] > delays[0]

    def test_retry_jitter(self):
        delays = []

        @retry_with_backoff(max_attempts=6, base_delay=1.0, jitter=True,
                            max_delay=1.0,
                            retryable_exceptions=(ConnectionError,),
                            on_retry=lambda a, e, d: delays.append(d))
        def always_fail():
            raise ConnectionError("x")

        with patch("src.utils.resilience.time.sleep"):
            with pytest.raises(ConnectionError):
                always_fail()
        # Jitter capped at max_delay produces varied (not identical) delays.
        assert len(set(round(d, 6) for d in delays)) > 1


class TestCircuitBreaker:

    def test_circuit_breaker_half_open_recovery(self):
        cb = CircuitBreaker("t", failure_threshold=2, recovery_timeout=0.05)
        for _ in range(2):
            with pytest.raises(ConnectionError):
                with cb:
                    raise ConnectionError("fail")
        assert cb.state == "OPEN"
        time.sleep(0.06)
        with cb:  # HALF_OPEN test call succeeds
            pass
        assert cb.state == "CLOSED"

    def test_circuit_breaker_half_open_failure(self):
        cb = CircuitBreaker("t", failure_threshold=2, recovery_timeout=0.05)
        for _ in range(2):
            with pytest.raises(ConnectionError):
                with cb:
                    raise ConnectionError("fail")
        time.sleep(0.06)
        with pytest.raises(ConnectionError):
            with cb:  # HALF_OPEN test call fails -> back to OPEN
                raise ConnectionError("again")
        assert cb.state == "OPEN"

    def test_circuit_breaker_exhausted_half_open(self):
        cb = CircuitBreaker("t", failure_threshold=1, recovery_timeout=0.05,
                            half_open_max_attempts=1)
        with pytest.raises(ConnectionError):
            with cb:
                raise ConnectionError("fail")
        assert cb.state == "OPEN"
        time.sleep(0.06)
        # First HALF_OPEN attempt consumes the single allowed test slot.
        cb.__enter__()
        cb.half_open_attempts = cb.half_open_max_attempts  # simulate exhausted
        with pytest.raises(CircuitBreakerOpenError):
            cb.__enter__()


class TestDeadLetterQueue:

    def test_dead_letter_queue_update_existing(self, store):
        dlq = DeadLetterQueue(store)
        dlq.add("fred", "GDP", "first error")
        dlq.add("fred", "GDP", "second error")
        pending = dlq.get_pending()
        gdp = [p for p in pending if p["item_key"] == "GDP"][0]
        assert gdp["retry_count"] == 1
        assert gdp["last_error"] == "second error"

    def test_dead_letter_queue_retry_nonexistent(self, store):
        dlq = DeadLetterQueue(store)
        assert dlq.retry("fred", "NOPE") is False

    def test_dead_letter_queue_count(self, store):
        dlq = DeadLetterQueue(store)
        assert dlq.count() == 0
        dlq.add("fred", "GDP", "e")
        dlq.add("gdelt", "NVDA", "e")
        assert dlq.count() == 2
        dlq.retry("fred", "GDP")
        assert dlq.count() == 1

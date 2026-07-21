"""
tests/test_refresh_budget.py
Offline tests for the in-query, budgeted, parallel freshness refresh
(src/middleware/app.py). No network: refreshes are faked by monkeypatching
``_refresh_one_source`` so the tests exercise the real parallel/budget/
background-completion/isolation logic through the real ``_refresh_ticker_sources``.
"""

import threading
from concurrent.futures import wait as futures_wait

import pytest

from src.middleware import app as middleware_app


class _FakeStore:
    """Minimal store stub: records ``mark_source_stale`` for isolation checks."""

    def __init__(self, report=None):
        self._report = report
        self.stale_marked: list[tuple] = []
        self._lock = threading.Lock()

    def get_freshness_report(self, ticker):
        return self._report

    def mark_source_stale(self, ticker, source, reason):
        with self._lock:
            self.stale_marked.append((ticker, source, reason))


@pytest.fixture
def fake_store(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(middleware_app, "store", store)
    return store


def test_sources_refresh_in_parallel(monkeypatch, fake_store):
    """All stale sources refresh concurrently, not serially (a)."""
    barrier = threading.Barrier(3)

    def refresher(ticker, logical):
        # If the three tasks ran serially this barrier would never complete
        # and raise BrokenBarrierError; concurrent execution releases it.
        barrier.wait(timeout=3.0)

    monkeypatch.setattr(middleware_app, "_refresh_one_source", refresher)

    refreshed, stale_used, pending = middleware_app._refresh_ticker_sources_budgeted(
        "NVDA", ["a", "b", "c"], budget_s=3.0
    )

    assert refreshed == ["a", "b", "c"]
    assert stale_used == []
    assert pending == []


def test_budget_cutoff_reports_unfinished_as_stale(monkeypatch, fake_store):
    """When the budget expires, unfinished sources are reported stale (b)."""
    release = threading.Event()

    def slow(ticker, logical):
        release.wait(timeout=5.0)

    monkeypatch.setattr(middleware_app, "_refresh_one_source", slow)
    try:
        refreshed, stale_used, pending = (
            middleware_app._refresh_ticker_sources_budgeted(
                "NVDA", ["x", "y"], budget_s=0.3
            )
        )
        assert refreshed == []
        assert stale_used == ["x", "y"]  # original order preserved
        assert len(pending) == 2  # still running, never cancelled
    finally:
        release.set()
        futures_wait(pending, timeout=5.0)


def test_evaluate_and_refresh_budget_warning(monkeypatch, fake_store):
    """_evaluate_and_refresh surfaces unfinished sources + the stale warning."""
    fake_store._report = {
        "overall": "stale",
        "sources": {
            "yfinance_news": {"status": "stale"},
            "finnhub_news": {"status": "fresh"},
        },
    }
    release = threading.Event()

    def slow(ticker, logical):
        release.wait(timeout=5.0)

    monkeypatch.setattr(middleware_app, "_refresh_one_source", slow)
    try:
        meta = middleware_app._evaluate_and_refresh("NVDA", True, budget_s=0.3)
        assert meta["refreshed_during_query"] == []
        assert meta["stale_sources_used"] == ["yfinance_news"]
        assert meta["warning"] and "yfinance_news" in meta["warning"]
        assert meta.get("_write_requested") is True
    finally:
        release.set()


def test_background_completion_still_records(monkeypatch, fake_store):
    """A source unfinished at budget still completes its write later (c)."""
    release = threading.Event()
    completed: list[str] = []
    completed_lock = threading.Lock()

    def slow(ticker, logical):
        release.wait(timeout=5.0)
        with completed_lock:
            completed.append(logical)  # stands in for the ingestor's write

    monkeypatch.setattr(middleware_app, "_refresh_one_source", slow)

    refreshed, stale_used, pending = middleware_app._refresh_ticker_sources_budgeted(
        "NVDA", ["z"], budget_s=0.2
    )
    assert refreshed == []
    assert stale_used == ["z"]
    assert completed == []  # not done yet

    release.set()
    done, not_done = futures_wait(pending, timeout=5.0)
    assert not not_done
    assert completed == ["z"]  # background run landed the data


def test_per_source_failure_is_isolated(monkeypatch, fake_store):
    """One failing source is marked stale; the others still refresh (d)."""

    def maybe_fail(ticker, logical):
        if logical == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr(middleware_app, "_refresh_one_source", maybe_fail)

    refreshed, stale_used, pending = middleware_app._refresh_ticker_sources_budgeted(
        "NVDA", ["good", "bad"], budget_s=3.0
    )

    assert refreshed == ["good"]
    assert stale_used == ["bad"]
    assert pending == []
    # The failing source was marked stale by the isolation path.
    assert [(t, s) for t, s, _ in fake_store.stale_marked] == [("NVDA", "bad")]


def test_empty_sources_is_a_noop(fake_store):
    """No stale sources -> nothing submitted, empty result."""
    assert middleware_app._refresh_ticker_sources_budgeted("NVDA", [], 3.0) == (
        [],
        [],
        [],
    )


def test_refresh_budget_s_clamps(monkeypatch):
    """_refresh_budget_s defends against out-of-range / non-numeric config."""
    from types import SimpleNamespace

    monkeypatch.setattr(middleware_app, "config", SimpleNamespace(refresh_budget_s=999.0))
    assert middleware_app._refresh_budget_s() == 30.0
    monkeypatch.setattr(middleware_app, "config", SimpleNamespace(refresh_budget_s=0.0))
    assert middleware_app._refresh_budget_s() == 0.5
    monkeypatch.setattr(middleware_app, "config", SimpleNamespace(refresh_budget_s="x"))
    assert middleware_app._refresh_budget_s() == 3.0

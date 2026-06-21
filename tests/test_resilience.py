"""
tests/test_resilience.py
Pytest suite for Phase 1.7.3 error handling & resilience layer.

Usage:
    pytest tests/test_resilience.py -v
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

if "chromadb" not in sys.modules:
    _m = type(sys)("chromadb")
    _m.EmbeddingFunction = object
    _m.Documents = list
    _m.Embeddings = list
    _m.PersistentClient = MagicMock
    sys.modules["chromadb"] = _m
    sys.modules["chromadb.api"] = MagicMock()

from src.storage.store import Store
from src.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    DeadLetterQueue,
    retry_with_backoff,
)


@pytest.fixture
def store(tmp_path):
    """Store with real SQLite (tmp_path) and a mocked Chroma backend."""
    with patch("src.storage.store.ChromaStore") as mc:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mc.return_value = inst
        yield Store(db_path=tmp_path / "t.db", chroma_path=tmp_path / "chroma")


# ── Retry with Backoff ─────────────────────────────────────


class TestRetryWithBackoff:

    def test_retry_success(self):
        """Function that succeeds on the first attempt is called once."""
        calls = {"n": 0}

        @retry_with_backoff(max_attempts=3, base_delay=0.0)
        def fn():
            calls["n"] += 1
            return "ok"

        assert fn() == "ok"
        assert calls["n"] == 1

    def test_retry_eventual_success(self):
        """Function fails twice then succeeds on the third attempt."""
        calls = {"n": 0}

        @retry_with_backoff(max_attempts=3, base_delay=0.0)
        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        with patch("src.utils.resilience.time.sleep"):
            assert fn() == "ok"
        assert calls["n"] == 3

    def test_retry_exhausted(self):
        """Function that always fails re-raises after exhausting attempts."""
        calls = {"n": 0}

        @retry_with_backoff(max_attempts=3, base_delay=0.0)
        def fn():
            calls["n"] += 1
            raise TimeoutError("always down")

        with patch("src.utils.resilience.time.sleep"):
            with pytest.raises(TimeoutError):
                fn()
        assert calls["n"] == 3

    def test_retry_backoff_increases(self):
        """Recorded sleep delays are non-decreasing across retries."""

        @retry_with_backoff(
            max_attempts=4, base_delay=1.0, backoff_factor=2.0, jitter=False
        )
        def fn():
            raise ConnectionError("down")

        with patch("src.utils.resilience.time.sleep") as mock_sleep:
            with pytest.raises(ConnectionError):
                fn()

        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert len(delays) == 3
        assert all(b >= a for a, b in zip(delays, delays[1:]))

    def test_retry_non_retryable(self):
        """A non-retryable exception is raised immediately, no retries."""
        calls = {"n": 0}

        @retry_with_backoff(
            max_attempts=3,
            base_delay=0.0,
            retryable_exceptions=(ConnectionError,),
        )
        def fn():
            calls["n"] += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            fn()
        assert calls["n"] == 1


# ── Circuit Breaker ────────────────────────────────────────


class TestCircuitBreaker:

    def test_circuit_breaker_closed(self):
        """Normal operation passes through and resets failure count."""
        breaker = CircuitBreaker(name="t", failure_threshold=3)
        with breaker:
            pass
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    def test_circuit_breaker_opens(self):
        """After threshold failures the breaker opens and rejects calls."""
        breaker = CircuitBreaker(name="t", failure_threshold=3, recovery_timeout=300)

        for _ in range(3):
            with pytest.raises(RuntimeError):
                with breaker:
                    raise RuntimeError("boom")

        assert breaker.state == "OPEN"

        # Subsequent entry is rejected immediately.
        with pytest.raises(CircuitBreakerOpenError):
            with breaker:
                pass

    def test_circuit_breaker_recovers(self):
        """After the recovery timeout: OPEN -> HALF_OPEN -> CLOSED."""
        breaker = CircuitBreaker(
            name="t", failure_threshold=2, recovery_timeout=0.05
        )

        for _ in range(2):
            with pytest.raises(RuntimeError):
                with breaker:
                    raise RuntimeError("boom")
        assert breaker.state == "OPEN"

        # Move the last failure back in time so the timeout has elapsed.
        breaker.last_failure_time -= 1.0

        # A successful test call transitions HALF_OPEN -> CLOSED.
        with breaker:
            assert breaker.state == "HALF_OPEN"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0


# ── Dead-Letter Queue ──────────────────────────────────────


class TestDeadLetterQueue:

    def test_dead_letter_queue_add(self, store):
        """Adding an item stores it; re-adding increments retry_count."""
        dlq = DeadLetterQueue(store)
        dlq.add("fred", "GDP", "API key invalid")

        pending = dlq.get_pending()
        assert len(pending) == 1
        assert pending[0]["source"] == "fred"
        assert pending[0]["item_key"] == "GDP"
        assert pending[0]["retry_count"] == 0

        dlq.add("fred", "GDP", "still failing")
        pending = dlq.get_pending()
        assert len(pending) == 1
        assert pending[0]["retry_count"] == 1
        assert pending[0]["last_error"] == "still failing"

    def test_dead_letter_queue_get_pending(self, store):
        """Multiple pending items are retrievable, honoring the limit."""
        dlq = DeadLetterQueue(store)
        dlq.add("fred", "GDP", "err")
        dlq.add("gdelt", "NVDA", "HTTP 429")
        dlq.add("sec", "AAPL", "403")

        pending = dlq.get_pending()
        assert len(pending) == 3
        keys = {(p["source"], p["item_key"]) for p in pending}
        assert ("fred", "GDP") in keys
        assert ("gdelt", "NVDA") in keys

        limited = dlq.get_pending(limit=2)
        assert len(limited) == 2

    def test_dead_letter_queue_retry(self, store):
        """A successful retry removes the item; returns True only if removed."""
        dlq = DeadLetterQueue(store)
        dlq.add("fred", "GDP", "err")
        assert dlq.count() == 1

        assert dlq.retry("fred", "GDP") is True
        assert dlq.count() == 0

        # Removing a non-existent item returns False.
        assert dlq.retry("fred", "GDP") is False

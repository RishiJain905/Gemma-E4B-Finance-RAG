"""
src/utils/resilience.py
Resilience utilities for data source ingestion — retry with backoff,
circuit breakers, structured error logging, and dead-letter queue.

Usage:
    from src.utils.resilience import retry_with_backoff, CircuitBreaker

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def fetch_fred_data(series_id):
        ...
"""

import logging
import random
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ── Retry with Exponential Backoff ─────────────────────────


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, IOError),
    on_retry: Optional[Callable] = None,
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        backoff_factor: Multiplier applied to delay each retry
        jitter: Add random jitter to avoid thundering herd
        retryable_exceptions: Tuple of exception types that trigger retry
        on_retry: Optional callback: on_retry(attempt, exception, delay)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_attempts, e,
                        )
                        raise
                    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    if on_retry:
                        on_retry(attempt, e, delay)
                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.1fs...",
                        func.__name__, attempt, max_attempts, e, delay,
                    )
                    time.sleep(delay)
            # Should not reach here
            raise last_exception  # type: ignore
        return wrapper
    return decorator


# ── Circuit Breaker ────────────────────────────────────────


class CircuitBreaker:
    """
    Circuit breaker pattern — prevents repeated calls to a failing API.

    States:
      CLOSED  — Normal operation, calls pass through
      OPEN    — Failure threshold exceeded, calls are rejected immediately
      HALF_OPEN — Testing if the API has recovered

    Usage:
        breaker = CircuitBreaker(name="gdelt", failure_threshold=5, recovery_timeout=300)
        with breaker:
            data = fetch_gdelt_data()
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,  # 5 minutes
        half_open_max_attempts: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_attempts = half_open_max_attempts

        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_attempts = 0

    def __enter__(self):
        if self.state == "OPEN":
            if time.monotonic() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.half_open_attempts = 0
                logger.info("Circuit breaker %s: OPEN -> HALF_OPEN", self.name)
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Retry in {self.recovery_timeout - (time.monotonic() - self.last_failure_time):.0f}s."
                )

        if self.state == "HALF_OPEN" and self.half_open_attempts >= self.half_open_max_attempts:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is HALF_OPEN and has exhausted test attempts."
            )

        if self.state == "HALF_OPEN":
            self.half_open_attempts += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

            if self.state == "HALF_OPEN":
                self.state = "OPEN"
                logger.warning(
                    "Circuit breaker %s: HALF_OPEN -> OPEN (test call failed)",
                    self.name,
                )
            elif self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(
                    "Circuit breaker %s: CLOSED -> OPEN (%d consecutive failures)",
                    self.name, self.failure_count,
                )
        else:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker %s: HALF_OPEN -> CLOSED (recovered)", self.name)
            else:
                self.failure_count = 0  # Reset on success in CLOSED state

        return False  # Don't suppress exceptions


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is OPEN and rejecting calls."""
    pass


# ── Dead-Letter Queue ──────────────────────────────────────


class DeadLetterQueue:
    """
    Stores persistently failing items for later review and retry.

    Items are stored in SQLite (dead_letter table) with:
      - source: which data source produced the failure
      - item_key: unique identifier (e.g., ticker + series_id)
      - error: error message
      - failed_at: timestamp
      - retry_count: number of retry attempts
      - last_error: most recent error message

    Usage:
        dlq = DeadLetterQueue(store)
        dlq.add("fred", "GDP", "API key invalid")
        dlq.add("gdelt", "NVDA", "HTTP 429")
        pending = dlq.get_pending()
        dlq.retry("fred", "GDP")  # Remove from DLQ on success
    """

    def __init__(self, store):
        self.store = store
        self._ensure_table()

    def _ensure_table(self):
        """Lazily create the dead_letter table."""
        with self.store.sqlite._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dead_letter (
                    source TEXT NOT NULL,
                    item_key TEXT NOT NULL,
                    error TEXT,
                    failed_at TEXT,
                    retry_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    PRIMARY KEY (source, item_key)
                )
                """
            )
            conn.commit()

    def add(self, source: str, item_key: str, error: str) -> None:
        """Add a failed item, or increment retry_count if it already exists."""
        now = datetime.now(timezone.utc).isoformat()
        with self.store.sqlite._connect() as conn:
            row = conn.execute(
                "SELECT retry_count FROM dead_letter WHERE source = ? AND item_key = ?",
                (source, item_key),
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT INTO dead_letter
                        (source, item_key, error, failed_at, retry_count, last_error)
                    VALUES (?, ?, ?, ?, 0, ?)
                    """,
                    (source, item_key, error, now, error),
                )
            else:
                conn.execute(
                    """
                    UPDATE dead_letter
                       SET retry_count = retry_count + 1,
                           last_error = ?,
                           failed_at = ?
                     WHERE source = ? AND item_key = ?
                    """,
                    (error, now, source, item_key),
                )
            conn.commit()

    def get_pending(self, limit: int = 50) -> list[dict]:
        """Return pending failed items as a list of dicts."""
        with self.store.sqlite._connect() as conn:
            rows = conn.execute(
                """
                SELECT source, item_key, error, failed_at, retry_count, last_error
                  FROM dead_letter
                 ORDER BY failed_at ASC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def retry(self, source: str, item_key: str) -> bool:
        """Remove an item from the DLQ (call on successful retry).

        Returns True if a row was removed, else False.
        """
        with self.store.sqlite._connect() as conn:
            cur = conn.execute(
                "DELETE FROM dead_letter WHERE source = ? AND item_key = ?",
                (source, item_key),
            )
            conn.commit()
            return cur.rowcount > 0

    def count(self) -> int:
        """Return the number of items currently in the DLQ."""
        with self.store.sqlite._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM dead_letter").fetchone()
        return int(row["n"]) if row else 0

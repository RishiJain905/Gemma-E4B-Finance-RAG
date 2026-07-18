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
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Callable, Optional

from src.ingestion.errors import (
    ErrorClass,
    PROVIDER_WIDE_ERROR_CLASSES,
    ProviderError,
    RETRYABLE_ERROR_CLASSES,
    iso_utc,
    safe_message,
    utc_now,
)

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
        now_fn: Callable[[], float] = time.monotonic,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_attempts = half_open_max_attempts
        self._now_fn = now_fn

        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_attempts = 0

    def __enter__(self):
        if self.state == "OPEN":
            if self._now_fn() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.half_open_attempts = 0
                logger.info("Circuit breaker %s: OPEN -> HALF_OPEN", self.name)
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Retry in {self.recovery_timeout - (self._now_fn() - self.last_failure_time):.0f}s."
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
            self.last_failure_time = self._now_fn()

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

    def open(self) -> None:
        """Open the breaker immediately for a known provider-wide failure."""
        self.state = "OPEN"
        self.failure_count = max(self.failure_count, self.failure_threshold)
        self.last_failure_time = self._now_fn()

    def close(self) -> None:
        """Reset the breaker after a confirmed recovery."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_attempts = 0


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is OPEN and rejecting calls."""
    pass


class ProviderRequestPolicy:
    """Execute provider requests with bounded retry and source-circuit semantics."""

    def __init__(
        self,
        *,
        source: str,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_sleep: float = 60.0,
        jitter_ratio: float = 0.2,
        cooldown_seconds: float = 300.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] = utc_now,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self.source = str(source)
        self.max_attempts = max(int(max_attempts), 1)
        self.base_delay = max(float(base_delay), 0.0)
        self.max_sleep = max(float(max_sleep), 0.0)
        self.jitter_ratio = min(max(float(jitter_ratio), 0.0), 1.0)
        self.cooldown_seconds = max(float(cooldown_seconds), 0.0)
        self.sleep_fn = sleep_fn
        self.now_fn = now_fn
        self.random_fn = random_fn
        self.breaker = CircuitBreaker(source, failure_threshold=1)
        self.attempts = 0
        self.retry_timestamps: list[str] = []
        self.reset_at: Optional[str] = None
        self.circuit_opened_at: Optional[str] = None
        self.remaining_work_skipped = False
        self.last_error: Optional[ProviderError] = None

    def request(self, operation: Callable[[], object]) -> object:
        """Run one response-producing operation and reject open-circuit work."""
        if self.breaker.state == "OPEN":
            self.remaining_work_skipped = True
            previous = self.last_error
            raise ProviderError(
                previous.safe_message if previous else f"{self.source} provider circuit is open",
                error_class=(previous.error_class if previous else ErrorClass.QUOTA_EXHAUSTED),
                status_code=previous.status_code if previous else None,
                retry_after=previous.retry_after if previous else None,
                reset_at=self.reset_at,
                attempts=previous.attempts if previous else 1,
                provider_wide=True,
                circuit_open=True,
            )

        for attempt in range(1, self.max_attempts + 1):
            self.attempts += 1
            current_error: Optional[ProviderError] = None
            try:
                response = operation()
                status_code = int(getattr(response, "status_code", 200))
                if status_code >= 400:
                    response_error = ProviderError.from_response(
                        response, now=self.now_fn()
                    )
                    if self.source.startswith("sec") and status_code in {403, 429}:
                        # Two very different SEC 403s share one status code. A
                        # real throttle block serves an HTML block page; a
                        # request for an object EDGAR never published (e.g. a
                        # market-holiday daily index) serves raw S3 XML
                        # AccessDenied. Treating the latter as RATE_LIMITED
                        # retried a file that will never exist, opened the
                        # provider circuit, and fail-fasted the whole run
                        # (live incident 2026-07-17/18).
                        body = str(getattr(response, "text", "") or "")
                        if "<Code>AccessDenied</Code>" in body:
                            response_error = ProviderError(
                                response_error.safe_message,
                                error_class=ErrorClass.PERMANENT,
                                status_code=status_code,
                            )
                        else:
                            response_error = ProviderError(
                                response_error.safe_message,
                                error_class=ErrorClass.RATE_LIMITED,
                                status_code=status_code,
                                retry_after=response_error.retry_after,
                                reset_at=response_error.reset_at,
                            )
                    if response_error.error_class is ErrorClass.PERMANENT:
                        return response
                    raise response_error
                return response
            except ProviderError as exc:
                exc.attempts = attempt
                current_error = exc
                self.last_error = exc
            except (ConnectionError, TimeoutError, OSError) as exc:
                current_error = ProviderError(
                    safe_message(exc),
                    error_class=ErrorClass.TRANSIENT,
                    attempts=attempt,
                    provider_wide=False,
                )
                self.last_error = current_error

            if current_error is None:
                raise RuntimeError("provider request failed without an error")
            if current_error.error_class in RETRYABLE_ERROR_CLASSES and attempt < self.max_attempts:
                delay = self._retry_delay(current_error, attempt)
                self.retry_timestamps.append(iso_utc(self.now_fn()))
                self.sleep_fn(delay)
                continue

            if current_error.error_class in PROVIDER_WIDE_ERROR_CLASSES:
                self._open(current_error)
            current_error.retry_timestamps = list(self.retry_timestamps)
            raise current_error

        raise RuntimeError("provider request loop exhausted unexpectedly")

    def _retry_delay(self, error: ProviderError, attempt: int) -> float:
        delay = (
            float(error.retry_after)
            if error.retry_after is not None
            else self.base_delay * (2 ** (attempt - 1))
        )
        delay = min(max(delay, 0.0), self.max_sleep)
        if self.jitter_ratio and error.retry_after is None:
            spread = delay * self.jitter_ratio
            delay = max(0.0, delay - spread + (2 * spread * self.random_fn()))
        return delay

    def _open(self, error: ProviderError) -> None:
        now = self.now_fn()
        self.breaker.open()
        self.circuit_opened_at = iso_utc(now)
        self.reset_at = error.reset_at or iso_utc(
            now + timedelta(seconds=self.cooldown_seconds)
        )
        error.reset_at = self.reset_at
        error.circuit_open = True


def provider_request_policy(
    source: str,
    policy_name: str,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = utc_now,
) -> ProviderRequestPolicy:
    """Build the small source-policy variants referenced by the registry."""
    policy = str(policy_name or "").lower()
    max_attempts = 3
    if policy in {"vendor_news", "vendor_market", "market_data"}:
        values = {"base_delay": 1.0, "max_sleep": 30.0, "cooldown_seconds": 900.0}
        if source == "massive":
            max_attempts = 1
    elif policy in {"sec", "public_snapshot"}:
        values = {"base_delay": 1.0, "max_sleep": 20.0, "cooldown_seconds": 300.0}
        if policy == "sec":
            max_attempts = 1
    elif policy in {"official_feed", "public_api", "public_news"}:
        values = {"base_delay": 1.0, "max_sleep": 30.0, "cooldown_seconds": 300.0}
    else:
        values = {"base_delay": 1.0, "max_sleep": 30.0, "cooldown_seconds": 300.0}
    return ProviderRequestPolicy(
        source=source,
        max_attempts=max_attempts,
        jitter_ratio=0.2,
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        **values,
    )


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

    def __init__(self, store, *, max_records: int = 1_000):
        self.store = store
        self.max_records = max(int(max_records), 1)
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
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(dead_letter)").fetchall()
            }
            additions = {
                "partition": "TEXT",
                "provider_record_id": "TEXT",
                "cursor_value": "TEXT",
                "error_class": "TEXT",
                "first_seen": "TEXT",
                "last_seen": "TEXT",
                "attempt_count": "INTEGER DEFAULT 0",
            }
            for name, definition in additions.items():
                if name not in columns:
                    conn.execute(
                        f"ALTER TABLE dead_letter ADD COLUMN {name} {definition}"
                    )
            conn.execute(
                "UPDATE dead_letter SET first_seen=COALESCE(first_seen, failed_at), "
                "last_seen=COALESCE(last_seen, failed_at), "
                "attempt_count=CASE WHEN attempt_count IS NULL OR attempt_count=0 "
                "THEN retry_count + 1 ELSE attempt_count END"
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

    def add_failure(
        self,
        *,
        source: str,
        partition: str,
        provider_record_id: Optional[str],
        cursor: Optional[str],
        error_class: str,
        message: str,
        attempts: int = 1,
    ) -> None:
        """Upsert one structured failure identity and cap retained queue rows."""
        partition = str(partition or "__source__")[:200]
        provider_id = str(provider_record_id)[:300] if provider_record_id else None
        cursor_value = str(cursor)[:300] if cursor else None
        normalized_class = str(error_class or ErrorClass.PERMANENT.value)[:50]
        identity_value = provider_id or cursor_value or "__source__"
        item_key = f"{partition}|{identity_value}|{normalized_class}"[:500]
        now = datetime.now(timezone.utc).isoformat()
        safe_error = safe_message(message)
        attempt_count = max(int(attempts), 1)
        with self.store.sqlite._connect() as conn:
            existing = conn.execute(
                "SELECT attempt_count FROM dead_letter WHERE source=? AND item_key=?",
                (str(source), item_key),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """INSERT INTO dead_letter (
                        source, item_key, error, failed_at, retry_count, last_error,
                        partition, provider_record_id, cursor_value, error_class,
                        first_seen, last_seen, attempt_count
                    ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(source), item_key, safe_error, now, safe_error,
                        partition, provider_id, cursor_value, normalized_class,
                        now, now, attempt_count,
                    ),
                )
            else:
                conn.execute(
                    """UPDATE dead_letter SET
                        retry_count=retry_count + 1,
                        last_error=?, failed_at=?, last_seen=?,
                        attempt_count=COALESCE(attempt_count, 0) + ?
                    WHERE source=? AND item_key=?""",
                    (
                        safe_error, now, now, attempt_count, str(source), item_key,
                    ),
                )
            conn.execute(
                """DELETE FROM dead_letter WHERE rowid IN (
                    SELECT rowid FROM dead_letter ORDER BY
                    COALESCE(last_seen, failed_at) DESC LIMIT -1 OFFSET ?
                )""",
                (self.max_records,),
            )
            conn.commit()

    def get_pending(self, limit: int = 50) -> list[dict]:
        """Return pending failed items as a list of dicts."""
        with self.store.sqlite._connect() as conn:
            rows = conn.execute(
                """
                SELECT source, item_key, error, failed_at, retry_count, last_error,
                       partition, provider_record_id, cursor_value, error_class,
                       first_seen, last_seen, attempt_count
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
                "DELETE FROM dead_letter WHERE source = ? "
                "AND (item_key = ? OR partition = ?)",
                (source, item_key, item_key),
            )
            conn.commit()
            return cur.rowcount > 0

    def count(self) -> int:
        """Return the number of items currently in the DLQ."""
        with self.store.sqlite._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM dead_letter").fetchone()
        return int(row["n"]) if row else 0

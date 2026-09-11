"""src/ingestion/vendor_common.py
Shared helpers for free/freemium vendor adapters (Alpha Vantage, FMP, Marketaux).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests

from src.ingestion.errors import ProviderError, error_class_for_http, parse_retry_after
from src.ingestion.normalization import parse_timestamp

logger = logging.getLogger(__name__)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for provenance fields."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def as_iso_timestamp(value: object, field_name: str = "timestamp") -> str:
    """Convert provider seconds, compact stamps, or ISO text to UTC ISO text."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1_000
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace(
                "+00:00", "Z"
            )
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"{field_name} is invalid") from exc
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    # Alpha Vantage NEWS_SENTIMENT uses YYYYMMDDTHHMMSS.
    if re.fullmatch(r"\d{8}T\d{6}", text):
        text = (
            f"{text[0:4]}-{text[4:6]}-{text[6:8]}T"
            f"{text[9:11]}:{text[11:13]}:{text[13:15]}Z"
        )
    if text.isdigit():
        return as_iso_timestamp(float(text), field_name)
    return parse_timestamp(text, field_name).isoformat().replace("+00:00", "Z")


def as_date(value: object, field_name: str = "date") -> str:
    """Return YYYY-MM-DD from a date or timestamp value."""
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    return parse_timestamp(as_iso_timestamp(text, field_name), field_name).date().isoformat()


def safe_identity(value: object, fallback: str) -> str:
    """Create a bounded stable identifier component."""
    text = str(value or "").strip() or fallback
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:160]
    return normalized or hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:32]


def parse_number(value: object) -> Optional[float]:
    """Parse a finite float from provider numeric text."""
    if value is None or value == "" or value == "None":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "N/A", "null", "None"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number or abs(number) == float("inf"):
        return None
    return number


def format_number(value: float) -> str:
    """Render a finite float without scientific notation noise."""
    return f"{value:.12g}"


def error_text(payload: object) -> str:
    """Extract a short provider error message without retaining large payloads."""
    if isinstance(payload, dict):
        for key in ("Error Message", "Note", "Information", "error", "message", "detail"):
            value = payload.get(key)
            if value:
                return str(value)[:500]
    return str(payload)[:500]


class VendorProviderError(ProviderError):
    """Bounded, normalized transport/contract failure for free vendor APIs."""

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
        reset_at: Optional[str] = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(
            message,
            error_class=error_class,
            status_code=status_code,
            retry_after=retry_after,
            reset_at=reset_at,
            attempts=attempts,
        )


def request_json(
    http_get: Callable[..., Any],
    url: str,
    *,
    params: Optional[dict[str, object]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
    max_attempts: int = 3,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], object] = utc_now,
    rate_limit_markers: tuple[str, ...] = (
        "rate limit",
        "thank you for using alpha vantage",
        "api call frequency",
        "limit reached",
    ),
) -> tuple[object, int, list[str]]:
    """GET JSON with bounded retries for transient/429 failures."""
    attempts = 0
    retries: list[str] = []
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            response = http_get(
                url,
                params=params or {},
                headers=headers or {"Accept": "application/json"},
                timeout=timeout,
            )
            status_code = int(getattr(response, "status_code", 200))
            try:
                payload = response.json()
            except Exception as exc:
                raise VendorProviderError(
                    "provider returned malformed JSON",
                    error_class="contract",
                    status_code=status_code,
                    attempts=attempt,
                ) from exc
            message = error_text(payload)
            lowered = message.lower()
            if status_code >= 400 or any(marker in lowered for marker in rate_limit_markers):
                if any(marker in lowered for marker in rate_limit_markers) and status_code < 400:
                    status_code = 429
                error_class = error_class_for_http(status_code).value
                if any(marker in lowered for marker in rate_limit_markers):
                    error_class = "rate_limited"
                retry_after = None
                reset_at = None
                header_value = None
                if hasattr(getattr(response, "headers", {}), "get"):
                    header_value = response.headers.get("Retry-After") or response.headers.get(
                        "retry-after"
                    )
                if header_value is not None:
                    window = parse_retry_after(
                        header_value, now=parse_timestamp(str(now_fn()), "now")
                    )
                    if window is not None:
                        retry_after = window.delay_seconds
                        reset_at = window.reset_at
                error = VendorProviderError(
                    message or f"HTTP {status_code}",
                    error_class=error_class,
                    status_code=status_code,
                    retry_after=retry_after,
                    reset_at=reset_at,
                    attempts=attempt,
                )
                if error_class in {"rate_limited", "transient"} and attempt < max_attempts:
                    retries.append(str(now_fn()))
                    delay = retry_after if retry_after is not None else min(2 ** attempt, 30)
                    sleep_fn(float(delay))
                    continue
                raise error
            if isinstance(payload, dict) and payload.get("Error Message"):
                raise VendorProviderError(
                    str(payload.get("Error Message")),
                    error_class="contract",
                    status_code=status_code,
                    attempts=attempt,
                )
            return payload, attempts, retries
        except VendorProviderError:
            raise
        except (requests.RequestException, ConnectionError, TimeoutError, OSError) as exc:
            error = VendorProviderError(str(exc), error_class="transient", attempts=attempt)
            if attempt < max_attempts:
                retries.append(str(now_fn()))
                sleep_fn(min(2 ** attempt, 30))
                continue
            raise error from exc
    raise VendorProviderError("request exhausted", error_class="transient", attempts=attempts)


def finish_result(result: dict, status: str, **extra: object) -> dict:
    """Attach terminal status fields used by the scheduler."""
    result["status"] = status
    result["terminal_status"] = status
    result.setdefault("remaining_work_skipped", False)
    result.update(extra)
    return result


def empty_result(source: str, status: str = "ok") -> dict:
    """Return a standard aggregate result envelope."""
    return {
        "status": status,
        "source": source,
        "tickers": 0,
        "stored": 0,
        "updated": 0,
        "duplicates": 0,
        "malformed": 0,
        "rejected": 0,
        "requests": 0,
        "attempts": 0,
        "errors": [],
        "facts_stored": 0,
        "observations_stored": 0,
        "narratives_stored": 0,
    }

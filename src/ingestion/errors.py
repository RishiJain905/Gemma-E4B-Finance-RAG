"""src/ingestion/errors.py
Normalized, redacted provider failures and Retry-After parsing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
import json
import re
from typing import Mapping, Optional


class ErrorClass(str, Enum):
    """Stable provider failure classes exposed by ingestion status."""

    AUTHENTICATION = "authentication"
    ENTITLEMENT = "entitlement"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    TRANSIENT = "transient"
    CONTRACT = "contract"
    ITEM = "item"
    PERMANENT = "permanent"


PROVIDER_WIDE_ERROR_CLASSES = frozenset(
    {
        ErrorClass.AUTHENTICATION,
        ErrorClass.ENTITLEMENT,
        ErrorClass.RATE_LIMITED,
        ErrorClass.QUOTA_EXHAUSTED,
    }
)
RETRYABLE_ERROR_CLASSES = frozenset(
    {ErrorClass.RATE_LIMITED, ErrorClass.TRANSIENT}
)

_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|authorization|password|secret)\b"
    r"\s*[:=]\s*([^\s,;&]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def utc_now() -> datetime:
    """Return an aware UTC wall-clock value."""
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    """Serialize one timestamp as a compact UTC ISO-8601 value."""
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RetryWindow:
    """Parsed provider retry delay and its absolute reset timestamp."""

    delay_seconds: float
    reset_at: str


def parse_retry_after(
    value: object,
    *,
    now: Optional[datetime] = None,
) -> Optional[RetryWindow]:
    """Parse numeric seconds or an HTTP-date without imposing a sleep cap."""
    if value is None:
        return None
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    try:
        delay = max(float(str(value).strip()), 0.0)
        reset = current + timedelta(seconds=delay)
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(str(value).strip())
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            reset = parsed.astimezone(timezone.utc)
            delay = max((reset - current.astimezone(timezone.utc)).total_seconds(), 0.0)
        except (TypeError, ValueError, OverflowError):
            return None
    return RetryWindow(delay_seconds=delay, reset_at=iso_utc(reset))


def safe_message(value: object, *, limit: int = 300) -> str:
    """Return a short credential-redacted message suitable for logs/status."""
    if isinstance(value, Mapping):
        selected = next(
            (
                value.get(key)
                for key in ("error", "message", "detail", "status")
                if value.get(key)
            ),
            None,
        )
        text = str(selected) if selected is not None else json.dumps(value, default=str)
    else:
        text = str(value or "provider request failed")
    text = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _BEARER_PATTERN.sub("Bearer [REDACTED]", text)
    text = re.sub(r"([?&](?:api[_-]?key|token|key)=)[^&\s]+", r"\1[REDACTED]", text, flags=re.I)
    return " ".join(text.split())[: max(int(limit), 1)]


def error_class_for_http(status_code: int, message: str = "") -> ErrorClass:
    """Classify one HTTP/provider failure without exposing response bodies."""
    lowered = message.lower()
    if status_code == 401:
        return ErrorClass.AUTHENTICATION
    if status_code == 429:
        return ErrorClass.RATE_LIMITED
    # FMP free-tier symbol blocks commonly return HTTP 402.
    if status_code in {402, 403} or any(
        marker in lowered
        for marker in (
            "entitlement",
            "not entitled",
            "subscription",
            "upgrade your plan",
            "permission denied",
            "forbidden",
        )
    ):
        return ErrorClass.ENTITLEMENT
    if status_code >= 500 or status_code in {408, 425}:
        return ErrorClass.TRANSIENT
    return ErrorClass.PERMANENT


def _response_payload(response: object) -> object:
    reader = getattr(response, "json", None)
    if callable(reader):
        try:
            return reader()
        except Exception:  # noqa: BLE001 - error responses are best-effort only
            pass
    return getattr(response, "text", "")


class ProviderError(RuntimeError):
    """One normalized provider failure with safe operational metadata."""

    def __init__(
        self,
        message: object,
        *,
        error_class: ErrorClass | str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
        reset_at: Optional[str] = None,
        attempts: int = 1,
        provider_wide: Optional[bool] = None,
        circuit_open: bool = False,
        retry_timestamps: Optional[list[str]] = None,
    ) -> None:
        normalized = ErrorClass(error_class)
        self.safe_message = safe_message(message)
        super().__init__(self.safe_message)
        self.error_class = normalized
        self.status_code = status_code
        self.retry_after = retry_after
        self.reset_at = reset_at
        self.attempts = max(int(attempts), 1)
        self.provider_wide = (
            normalized in PROVIDER_WIDE_ERROR_CLASSES
            if provider_wide is None
            else bool(provider_wide)
        )
        self.circuit_open = bool(circuit_open)
        self.retry_timestamps = list(retry_timestamps or [])

    @classmethod
    def from_response(
        cls,
        response: object,
        *,
        now: Optional[datetime] = None,
    ) -> "ProviderError":
        """Build a normalized failure from a response-like object."""
        status_code = int(getattr(response, "status_code", 500))
        message = safe_message(_response_payload(response))
        headers = getattr(response, "headers", {})
        retry_value = None
        if hasattr(headers, "get"):
            retry_value = headers.get("Retry-After") or headers.get("retry-after")
        window = parse_retry_after(retry_value, now=now)
        return cls(
            message or f"provider HTTP {status_code}",
            error_class=error_class_for_http(status_code, message),
            status_code=status_code,
            retry_after=window.delay_seconds if window else None,
            reset_at=window.reset_at if window else None,
        )


def normalize_error_class(value: object) -> Optional[str]:
    """Return one public error class or ``None`` for a healthy result."""
    if value in (None, ""):
        return None
    text = str(value.value if isinstance(value, ErrorClass) else value).lower()
    aliases = {
        "storage": ErrorClass.ITEM.value,
        "cursor": ErrorClass.CONTRACT.value,
        "pagination": ErrorClass.CONTRACT.value,
        "provider_error": ErrorClass.TRANSIENT.value,
    }
    text = aliases.get(text, text)
    try:
        return ErrorClass(text).value
    except ValueError:
        return ErrorClass.PERMANENT.value


__all__ = [
    "ErrorClass",
    "PROVIDER_WIDE_ERROR_CLASSES",
    "ProviderError",
    "RETRYABLE_ERROR_CLASSES",
    "RetryWindow",
    "error_class_for_http",
    "iso_utc",
    "normalize_error_class",
    "parse_retry_after",
    "safe_message",
    "utc_now",
]

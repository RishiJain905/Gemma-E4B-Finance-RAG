"""src/ingestion/finnhub_ingestor.py
Finnhub company-news ingestion with bounded pagination and durable cursors.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import logging
import os
import re
import time
from typing import Any, Callable, Optional

import requests

from src.ingestion.normalization import (
    NORMALIZATION_VERSION,
    content_hash,
    normalize_canonical_url,
    parse_timestamp,
)
from src.ingestion.errors import (
    ErrorClass,
    ProviderError,
    error_class_for_http,
    parse_retry_after,
)
from src.ingestion.records import NarrativeRecord
from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)


class FinnhubProviderError(ProviderError):
    """Bounded, normalized Finnhub transport or contract failure."""

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


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for provenance fields."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_iso_timestamp(value: object, field_name: str = "timestamp") -> str:
    """Convert provider seconds/milliseconds or ISO text to UTC ISO text."""
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
    if isinstance(value, str) and value.strip().isdigit():
        return _as_iso_timestamp(float(value.strip()), field_name)
    parsed = parse_timestamp(str(value), field_name)
    return parsed.isoformat().replace("+00:00", "Z")


def _as_date(value: str) -> str:
    """Return the UTC calendar date represented by an ISO timestamp."""
    return parse_timestamp(value, "timestamp").date().isoformat()


def _retry_after(
    headers: object,
    payload: object,
    *,
    now: datetime,
) -> tuple[Optional[float], Optional[str]]:
    """Read numeric/date Retry-After metadata from headers or JSON."""
    header_value = None
    if hasattr(headers, "get"):
        header_value = headers.get("Retry-After") or headers.get("retry-after")
    if header_value is None and isinstance(payload, dict):
        header_value = payload.get("retry_after") or payload.get("retryAfter")
    if header_value is None:
        return None, None
    window = parse_retry_after(header_value, now=now)
    if window is None:
        return None, None
    return window.delay_seconds, window.reset_at


def _error_text(payload: object) -> str:
    """Extract a safe, short provider error message without retaining payloads."""
    if isinstance(payload, dict):
        for key in ("error", "message", "detail", "status"):
            value = payload.get(key)
            if value:
                return str(value)[:500]
    return str(payload)[:500]


def _is_entitlement_message(message: str) -> bool:
    """Recognize plan/endpoint denials that must not be retried."""
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "entitlement", "plan", "subscription", "upgrade", "permission", "access",
            "not_authorized", "unauthorized", "not authorized", "forbidden", "not entitled",
        )
    )


class FinnhubIngestor:
    """Translate Finnhub news into normalized narratives through ``Store``."""

    BASE_URL = "https://finnhub.io/api/v1"
    SOURCE_NAME = "finnhub"
    CURSOR_SOURCE = "finnhub_news"
    COVERAGE_SOURCE = "finnhub"
    COMPANY_NEWS_PATH = "/company-news"

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        coverage_resolver: Optional[CoverageResolver] = None,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        http_get: Optional[Callable[..., Any]] = None,
        timeout: float = 30.0,
        overlap_hours: int = 6,
        initial_lookback_days: int = 7,
        max_pages: int = 20,
        max_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        now_fn: Callable[[], object] = _utc_now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store or Store()
        self.coverage = coverage_resolver or CoverageResolver(self.store)
        self.api_key = str(
            api_key if api_key is not None else os.environ.get("FINNHUB_API_KEY", "")
        ).strip()
        self.base_url = str(base_url).rstrip("/")
        self.http_get = http_get or requests.get
        self.timeout = max(float(timeout), 0.1)
        self.overlap_hours = max(int(overlap_hours), 0)
        self.initial_lookback_days = max(int(initial_lookback_days), 1)
        self.max_pages = max(int(max_pages), 1)
        self.max_attempts = max(int(max_attempts), 1)
        self.retry_base_delay = max(float(retry_base_delay), 0.0)
        self.retry_max_delay = max(float(retry_max_delay), self.retry_base_delay)
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self._last_request_attempts = 0
        self._last_retry_timestamps: list[str] = []

    def ingest_news(self, tickers: Optional[list[str]] = None) -> dict:
        """Ingest broad-universe news, isolating ticker partitions."""
        selected = list(tickers) if tickers is not None else self.coverage.tickers_for(
            self.COVERAGE_SOURCE
        )
        aggregate = {
            "status": "ok",
            "source": self.SOURCE_NAME,
            "tickers": 0,
            "stored": 0,
            "duplicates": 0,
            "malformed": 0,
            "rejected": 0,
            "requests": 0,
            "attempts": 0,
            "pages": 0,
            "errors": [],
            "cursor_before": {},
            "cursor_after": {},
        }
        terminal_statuses = {
            "disabled_missing_key",
            "disabled_authentication",
            "disabled_entitlement",
            "rate_limited",
        }
        for ticker in selected:
            result = self.ingest_ticker_news(str(ticker).upper())
            aggregate["tickers"] += 1
            for key in ("stored", "duplicates", "malformed", "requests", "pages"):
                aggregate[key] += int(result.get(key, 0) or 0)
            aggregate["attempts"] += int(result.get("attempts", 0) or 0)
            aggregate["rejected"] += int(result.get("rejected_items", 0) or 0)
            ticker_key = str(ticker).upper()
            aggregate["cursor_before"][ticker_key] = result.get("cursor_before")
            aggregate["cursor_after"][ticker_key] = result.get("cursor_after")
            if result.get("error_class"):
                aggregate["error_class"] = result["error_class"]
            if result.get("retry_after") is not None:
                aggregate["retry_after"] = result["retry_after"]
            errors = result.get("errors") or []
            aggregate["errors"].extend(errors)
            if result.get("status") in terminal_statuses:
                aggregate["status"] = result["status"]
                aggregate["remaining_work_skipped"] = True
                break
            if result.get("status") in {"error", "partial"}:
                if aggregate["status"] == "ok":
                    aggregate["status"] = result["status"]
        aggregate["terminal_status"] = aggregate["status"]
        aggregate["accepted_items"] = int(aggregate["stored"]) + int(
            aggregate["duplicates"]
        )
        aggregate["rejected_items"] = int(aggregate["rejected"])
        aggregate.setdefault("remaining_work_skipped", False)
        return aggregate

    def ingest_ticker_news(self, ticker: str) -> dict:
        """Fetch one ticker's overlap window and checkpoint only after storage."""
        ticker = str(ticker or "").strip().upper()
        if not ticker:
            return self._result("error", ticker, errors=["ticker is required"])
        cursor_before = self.store.get_source_cursor(self.CURSOR_SOURCE, ticker)
        result = self._result("ok", ticker, cursor_before=cursor_before)
        if not self.api_key:
            self._set_status(ticker, "disabled_missing_key", "authentication", "FINNHUB_API_KEY is not configured")
            return self._finish(result, "disabled_missing_key", error_class="authentication")

        now = self._now_timestamp()
        from_date, to_date = self._window(cursor_before, now)
        params: dict[str, object] = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": self.api_key,
            "page": 1,
        }
        next_page: object = 1
        newest: Optional[str] = cursor_before
        storage_failed = False
        provider_failure: Optional[FinnhubProviderError] = None
        seen_pages: set[str] = set()
        pagination_incomplete = False

        while next_page is not None and result["pages"] < self.max_pages:
            params["page"] = next_page
            page_key = str(next_page)
            if page_key in seen_pages:
                break
            seen_pages.add(page_key)
            try:
                payload = self._request_json(params)
                result["requests"] += self._last_request_attempts
                result["attempts"] += self._last_request_attempts
                result["retry_timestamps"].extend(self._last_retry_timestamps)
                result["pages"] += 1
            except FinnhubProviderError as exc:
                provider_failure = exc
                result["requests"] += exc.attempts
                result["attempts"] += exc.attempts
                result["retry_timestamps"].extend(self._last_retry_timestamps)
                break

            rows = self._rows_from_payload(payload)
            if rows is None:
                result["malformed"] += 1
                result["errors"].append("Finnhub response did not contain a news list")
                provider_failure = FinnhubProviderError(
                    "Finnhub response did not contain a news list", error_class="contract"
                )
                break

            for raw_row in rows:
                try:
                    record, published_at = self._record_from_row(ticker, raw_row, now)
                except (TypeError, ValueError, KeyError) as exc:
                    result["malformed"] += 1
                    result["errors"].append(f"{ticker}: malformed news row: {exc}")
                    continue
                try:
                    stored = self.store.upsert_narrative(record)
                except Exception as exc:  # noqa: BLE001 - one row cannot abort other sources
                    storage_failed = True
                    result["errors"].append(f"{ticker}: storage failed: {exc}")
                    continue
                if stored.get("indexing_status") == "error":
                    storage_failed = True
                    result["errors"].append(
                        f"{ticker}: narrative indexing failed: {stored.get('index_error', 'unknown error')}"
                    )
                    continue
                if stored.get("created"):
                    result["stored"] += 1
                else:
                    result["duplicates"] += 1
                if newest is None or parse_timestamp(published_at) > parse_timestamp(newest):
                    newest = published_at

            next_page = self._next_page(payload, next_page, len(rows))

        if next_page is not None and provider_failure is None:
            pagination_incomplete = True

        if pagination_incomplete:
            error = FinnhubProviderError(
                "Finnhub pagination limit reached", error_class="pagination"
            )
            self._set_status(ticker, "error", error.error_class, str(error))
            return self._finish(result, "error", error_class=error.error_class)

        if provider_failure is not None:
            status = self._status_for_error(provider_failure)
            self._set_status(
                ticker,
                status,
                provider_failure.error_class,
                str(provider_failure),
                provider_failure.retry_after,
            )
            return self._finish(
                result,
                status,
                error_class=provider_failure.error_class,
                retry_after=provider_failure.retry_after,
                reset_at=provider_failure.reset_at,
            )
        if storage_failed:
            self._set_status(ticker, "error", "storage", "one or more news records failed to store")
            return self._finish(result, "error", error_class="storage")

        if newest is not None and (cursor_before is None or parse_timestamp(newest) >= parse_timestamp(cursor_before)):
            try:
                self.store.set_source_cursor(
                    self.CURSOR_SOURCE,
                    ticker,
                    newest,
                    cursor_type="timestamp",
                    overlap_value=f"{self.overlap_hours}h",
                    last_successful_run_id=hashlib.sha256(
                        f"{ticker}:{now}".encode("utf-8")
                    ).hexdigest()[:32],
                    status="partial" if result["malformed"] else "success",
                )
            except Exception as exc:  # noqa: BLE001 - cursor failure must be visible, not silent
                result["errors"].append(f"{ticker}: cursor commit failed: {exc}")
                return self._finish(result, "error", error_class="cursor")
        else:
            self._set_status(
                ticker,
                "partial" if result["malformed"] else "success",
                "contract" if result["malformed"] else None,
                "malformed rows were skipped" if result["malformed"] else None,
            )
        status = "partial" if result["malformed"] else "ok"
        return self._finish(result, status, cursor_after=newest or cursor_before)

    ingest_ticker = ingest_ticker_news

    def _request_json(self, params: dict[str, object]) -> object:
        """Request one page with bounded transient/429 retries."""
        url = f"{self.base_url}{self.COMPANY_NEWS_PATH}"
        self._last_request_attempts = 0
        self._last_retry_timestamps = []
        for attempt in range(1, self.max_attempts + 1):
            self._last_request_attempts = attempt
            try:
                response = self.http_get(
                    url,
                    params=params,
                    headers={"Accept": "application/json"},
                    timeout=self.timeout,
                )
                status_code = int(getattr(response, "status_code", 200))
                try:
                    payload = response.json()
                except Exception as exc:
                    raise FinnhubProviderError(
                        "Finnhub returned malformed JSON", error_class="contract", status_code=status_code
                    ) from exc
                message = _error_text(payload)
                if status_code >= 400:
                    error_class = self._error_class(status_code, message)
                    retry_after, reset_at = _retry_after(
                        getattr(response, "headers", {}),
                        payload,
                        now=parse_timestamp(self._now_timestamp(), "now"),
                    )
                    error = FinnhubProviderError(
                        message or f"Finnhub HTTP {status_code}",
                        error_class=error_class,
                        status_code=status_code,
                        retry_after=retry_after,
                        reset_at=reset_at,
                        attempts=attempt,
                    )
                    if error_class in {"rate_limited", "transient"} and attempt < self.max_attempts:
                        self._last_retry_timestamps.append(self._now_timestamp())
                        self.sleep_fn(self._retry_delay(error, attempt))
                        continue
                    error.circuit_open = error.provider_wide
                    raise error
                if isinstance(payload, dict) and payload.get("error"):
                    if _is_entitlement_message(message):
                        raise FinnhubProviderError(
                            message,
                            error_class="entitlement",
                            status_code=status_code,
                        )
                    raise FinnhubProviderError(message, error_class="contract", status_code=status_code)
                return payload
            except FinnhubProviderError:
                raise
            except (requests.RequestException, ConnectionError, TimeoutError, OSError) as exc:
                error = FinnhubProviderError(str(exc), error_class="transient")
                if attempt < self.max_attempts:
                    self._last_retry_timestamps.append(self._now_timestamp())
                    self.sleep_fn(self._retry_delay(error, attempt))
                    continue
                error.attempts = attempt
                raise error from exc
        raise FinnhubProviderError("Finnhub request exhausted", error_class="transient")

    def _record_from_row(
        self, ticker: str, row: object, accessed_at: str,
    ) -> tuple[NarrativeRecord, str]:
        """Convert only provider-supplied news fields into a narrative record."""
        if not isinstance(row, dict):
            raise ValueError("row must be an object")
        headline = str(row.get("headline") or row.get("title") or "").strip()
        summary = str(row.get("summary") or row.get("description") or "").strip()
        publisher = str(row.get("source") or row.get("publisher") or "").strip()
        source_url = str(row.get("url") or row.get("link") or "").strip()
        if not headline:
            raise ValueError("headline is required")
        if not publisher:
            raise ValueError("source is required")
        canonical_url = normalize_canonical_url(source_url)
        if not canonical_url:
            raise ValueError("URL is required")
        published_at = _as_iso_timestamp(
            row.get("datetime") or row.get("published_at") or row.get("timestamp"),
            "published_at",
        )
        provider_id = row.get("id") or row.get("news_id") or row.get("uuid") or row.get("article_id")
        provider_id_text = str(provider_id).strip() if provider_id is not None else None
        body = "\n\n".join(value for value in (headline, summary, publisher) if value)
        body_hash = content_hash(body)
        identity = provider_id_text or body_hash
        identity = re.sub(r"[^A-Za-z0-9._-]+", "-", identity).strip("-")[:160] or body_hash
        security = self.store.resolve_security(ticker, provider=self.SOURCE_NAME)
        security_id = security.get("security_id") if isinstance(security, dict) else None
        record = NarrativeRecord(
            corpus_item_id=f"news/{self.SOURCE_NAME}/{ticker}/{identity}",
            source_name=self.SOURCE_NAME,
            source_category="news_vendor",
            provider_record_id=provider_id_text,
            original_publisher=publisher,
            item_type="news",
            title=headline,
            body=body,
            summary=summary or None,
            published_at=published_at,
            observed_at=None,
            accessed_at=accessed_at,
            ingested_at=accessed_at,
            source_url=source_url,
            canonical_url=canonical_url,
            license_label="provider_entitlement",
            normalization_version=NORMALIZATION_VERSION,
            content_hash=body_hash,
            document_family="company_news",
            security_ids=(str(security_id),) if security_id else (),
            tickers=(ticker,),
            evidence_authority="provider",
        )
        return record, published_at

    @staticmethod
    def _rows_from_payload(payload: object) -> Optional[list[object]]:
        """Accept Finnhub list and common paginated envelope shapes."""
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return None
        for key in ("data", "results", "news", "articles"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
        return None

    @staticmethod
    def _next_page(payload: object, current: object, row_count: int) -> Optional[object]:
        """Read a provider continuation token while avoiding implicit infinite loops."""
        if not isinstance(payload, dict):
            return None
        for key in ("next", "next_page", "nextPage", "next_token", "nextToken"):
            value = payload.get(key)
            if isinstance(value, dict):
                value = value.get("page") or value.get("token")
            if value not in (None, "", current):
                return value
        has_more = payload.get("has_more") or payload.get("hasMore")
        if has_more:
            try:
                return int(current) + 1
            except (TypeError, ValueError):
                return None
        return None

    def _window(self, cursor: Optional[str], now: str) -> tuple[str, str]:
        """Build an overlap date window from a publication cursor."""
        now_dt = parse_timestamp(now, "now")
        if cursor:
            start_dt = parse_timestamp(cursor, "cursor") - timedelta(hours=self.overlap_hours)
        else:
            start_dt = now_dt - timedelta(days=self.initial_lookback_days)
        return start_dt.date().isoformat(), now_dt.date().isoformat()

    def _now_timestamp(self) -> str:
        """Normalize the injected clock value."""
        value = self.now_fn()
        if isinstance(value, datetime):
            dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return _as_iso_timestamp(value, "now")

    def _result(self, status: str, ticker: str, **values: object) -> dict:
        return {
            "status": status,
            "source": self.SOURCE_NAME,
            "ticker": ticker,
            "stored": 0,
            "duplicates": 0,
            "malformed": 0,
            "requests": 0,
            "attempts": 0,
            "pages": 0,
            "retry_timestamps": [],
            "errors": [],
            **values,
        }

    def _finish(
        self,
        result: dict,
        status: str,
        *,
        cursor_after: Optional[str] = None,
        error_class: Optional[str] = None,
        retry_after: Optional[float] = None,
        reset_at: Optional[str] = None,
    ) -> dict:
        result["status"] = status
        result["cursor_after"] = cursor_after if cursor_after is not None else result.get("cursor_before")
        if error_class:
            result["error_class"] = error_class
        if retry_after is not None:
            result["retry_after"] = retry_after
        if reset_at is not None:
            result["reset_at"] = reset_at
        result["terminal_status"] = status
        normalized_class = str(
            error_class.value if isinstance(error_class, ErrorClass) else error_class or ""
        )
        result["accepted_items"] = int(result.get("stored", 0)) + int(
            result.get("duplicates", 0)
        )
        result["rejected_items"] = int(result.get("malformed", 0)) + (
            1 if normalized_class == "permanent" else 0
        )
        result["remaining_work_skipped"] = normalized_class in {
            "authentication", "entitlement", "rate_limited", "quota_exhausted"
        }
        return result

    def _set_status(
        self,
        ticker: str,
        status: str,
        error_class: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        try:
            self.store.set_source_status(
                self.CURSOR_SOURCE,
                ticker,
                status,
                error_class=error_class,
                error_message=error_message,
                retry_after=retry_after,
            )
        except Exception:  # noqa: BLE001 - status bookkeeping must not abort another source
            logger.warning("Could not persist Finnhub status for %s", ticker, exc_info=True)

    def _status_for_error(self, error: FinnhubProviderError) -> str:
        return {
            "authentication": "disabled_authentication",
            "entitlement": "disabled_entitlement",
            "rate_limited": "rate_limited",
        }.get(error.error_class, "error")

    @staticmethod
    def _error_class(status_code: int, message: str) -> str:
        return error_class_for_http(status_code, message).value

    def _retry_delay(self, error: FinnhubProviderError, attempt: int) -> float:
        if error.retry_after is not None:
            return error.retry_after
        return min(self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay)


CompanyNewsIngestor = FinnhubIngestor

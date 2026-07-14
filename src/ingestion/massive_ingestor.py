"""src/ingestion/massive_ingestor.py
Massive grouped-market, corporate-action, and optional secondary-news ingestion.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional

import requests

from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash, normalize_canonical_url
from src.ingestion.records import EventRecord, NarrativeRecord, ObservationRecord
from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)


class MassiveProviderError(RuntimeError):
    """Bounded, normalized Massive transport or contract failure."""

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.status_code = status_code
        self.retry_after = retry_after


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for provenance fields."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _error_text(payload: object) -> str:
    """Extract a short provider message without retaining a raw response."""
    if isinstance(payload, dict):
        for key in ("error", "message", "detail", "status"):
            if payload.get(key):
                return str(payload[key])[:500]
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


def _retry_after(headers: object, payload: object) -> Optional[float]:
    """Read and cap a numeric Retry-After value."""
    value = None
    if hasattr(headers, "get"):
        value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None and isinstance(payload, dict):
        value = payload.get("retry_after") or payload.get("retryAfter")
    if value is None:
        return None
    try:
        return max(0.0, min(float(value), 300.0))
    except (TypeError, ValueError):
        return None


def _format_number(value: object) -> str:
    """Format a numeric observation without adding provider semantics."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def _as_date(value: object, field_name: str) -> str:
    """Normalize a provider date while retaining date precision."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    if "T" in text:
        text = text.split("T", 1)[0]
    parsed = date.fromisoformat(text)
    return parsed.isoformat()


def _safe_identity(value: object, fallback: str) -> str:
    """Make a bounded provider identity suitable for a corpus/event id."""
    text = str(value or "").strip()
    if not text:
        text = fallback
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:160]
    return normalized or hashlib.sha256(fallback.encode("utf-8")).hexdigest()


class MassiveIngestor:
    """Translate Massive responses into Store-owned observations, events, and narratives."""

    BASE_URL = "https://api.massive.com"
    SOURCE_NAME = "massive"
    MARKET_CURSOR_SOURCE = "massive_market"
    ACTION_CURSOR_SOURCE = "massive_actions"
    NEWS_STATUS_SOURCE = "massive_news"
    FILINGS_STATUS_SOURCE = "massive_filings"
    COVERAGE_SOURCE = "massive"
    GROUPED_PATH = "/v2/aggs/grouped/locale/us/markets/stocks"
    SPLITS_PATH = "/v3/reference/splits"
    DIVIDENDS_PATH = "/v3/reference/dividends"
    NEWS_PATH = "/v2/reference/news"
    FILINGS_PATH = "/v3/reference/filings"
    BAR_METRICS = (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v"))

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        coverage_resolver: Optional[CoverageResolver] = None,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        http_get: Optional[Callable[..., Any]] = None,
        timeout: float = 30.0,
        overlap_days: int = 3,
        initial_lookback_days: int = 5,
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
            api_key if api_key is not None else os.environ.get("MASSIVE_API_KEY", "")
        ).strip()
        self.base_url = str(base_url).rstrip("/")
        self.http_get = http_get or requests.get
        self.timeout = max(float(timeout), 0.1)
        self.overlap_days = max(int(overlap_days), 0)
        self.initial_lookback_days = max(int(initial_lookback_days), 1)
        self.max_pages = max(int(max_pages), 1)
        self.max_attempts = max(int(max_attempts), 1)
        self.retry_base_delay = max(float(retry_base_delay), 0.0)
        self.retry_max_delay = max(float(retry_max_delay), self.retry_base_delay)
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn

    def ingest_market_data(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[list[str]] = None,
    ) -> dict:
        """Fetch one grouped US summary per market day and filter locally."""
        selected = {str(ticker).upper() for ticker in (tickers or self._coverage_tickers())}
        result = self._result("ok", capability="market_data")
        if not self.api_key:
            self._set_status(self.MARKET_CURSOR_SOURCE, "US", "disabled_missing_key", "authentication", "MASSIVE_API_KEY is not configured")
            return self._finish(result, "disabled_missing_key", error_class="authentication")

        today = self._now_date()
        cursor_before = self.store.get_source_cursor(self.MARKET_CURSOR_SOURCE, "US")
        result["cursor_before"] = cursor_before
        first = date.fromisoformat(start_date) if start_date else (
            date.fromisoformat(cursor_before) - timedelta(days=self.overlap_days)
            if cursor_before else today - timedelta(days=self.initial_lookback_days)
        )
        last = date.fromisoformat(end_date) if end_date else today
        if first > last:
            return self._finish(result, "ok", cursor_after=cursor_before)

        storage_failed = False
        market_dates = self._market_dates(first, last)
        if not market_dates:
            return self._finish(result, "ok", cursor_after=cursor_before)
        for market_date in market_dates:
            url = f"{self.base_url}{self.GROUPED_PATH}/{market_date.isoformat()}"
            try:
                payload = self._request_json(url, {"adjusted": "true", "apiKey": self.api_key})
                result["requests"] += 1
            except MassiveProviderError as exc:
                result["requests"] += 1
                status = self._status_for_error(exc)
                self._set_status(self.MARKET_CURSOR_SOURCE, "US", status, exc.error_class, str(exc), exc.retry_after)
                return self._finish(result, status, error_class=exc.error_class, retry_after=exc.retry_after)

            rows = self._rows_from_payload(payload)
            if rows is None:
                result["malformed"] += 1
                result["errors"].append(f"{market_date}: grouped response has no results list")
                self._set_status(self.MARKET_CURSOR_SOURCE, "US", "error", "contract", "grouped response has no results list")
                return self._finish(result, "error", error_class="contract")
            result["pages"] += 1
            revision = self._provider_revision(payload)
            adjusted = bool(payload.get("adjusted")) if isinstance(payload, dict) else False
            for raw_row in rows:
                try:
                    records = self._observations_from_bar(
                        raw_row, market_date.isoformat(), adjusted, revision, selected, url
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    result["malformed"] += 1
                    result["errors"].append(f"{market_date}: malformed market row: {exc}")
                    continue
                for record in records:
                    try:
                        stored = self.store.upsert_observation(record)
                    except Exception as exc:  # noqa: BLE001 - one bar cannot abort other sources
                        storage_failed = True
                        result["errors"].append(f"{record.tickers[0]}: observation storage failed: {exc}")
                        continue
                    if stored.get("created"):
                        result["stored"] += 1
                    elif stored.get("changed"):
                        result["updated"] += 1
                    else:
                        result["duplicates"] += 1

            if storage_failed:
                self._set_status(self.MARKET_CURSOR_SOURCE, "US", "error", "storage", "one or more bars failed to store")
                return self._finish(result, "error", error_class="storage")
            self.store.set_source_cursor(
                self.MARKET_CURSOR_SOURCE,
                "US",
                market_date.isoformat(),
                cursor_type="date",
                overlap_value=f"{self.overlap_days}d",
                last_successful_run_id=hashlib.sha256(
                    f"massive-market:{market_date}".encode("utf-8")
                ).hexdigest()[:32],
                status="partial" if result["malformed"] else "success",
            )

        status = "partial" if result["malformed"] else "ok"
        return self._finish(result, status, cursor_after=market_dates[-1].isoformat())

    def ingest_corporate_actions(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[list[str]] = None,
    ) -> dict:
        """Fetch entitled splits/dividends and store structured EventRecords."""
        selected = {str(ticker).upper() for ticker in (tickers or self._coverage_tickers())}
        result = self._result("ok", capability="corporate_actions")
        if not self.api_key:
            self._set_status(self.ACTION_CURSOR_SOURCE, "US", "disabled_missing_key", "authentication", "MASSIVE_API_KEY is not configured")
            return self._finish(result, "disabled_missing_key", error_class="authentication")
        today = self._now_date()
        first = start_date or (
            self.store.get_source_cursor(self.ACTION_CURSOR_SOURCE, "US") or
            (today - timedelta(days=self.initial_lookback_days)).isoformat()
        )
        last = end_date or today.isoformat()
        endpoint_specs = (
            ("split", self.SPLITS_PATH, {"execution_date.gte": first, "execution_date.lte": last}),
            ("dividend", self.DIVIDENDS_PATH, {"ex_dividend_date.gte": first, "ex_dividend_date.lte": last}),
        )
        storage_failed = False
        fetch_incomplete = False
        for action_type, path, filters in endpoint_specs:
            url = f"{self.base_url}{path}"
            params = {**filters, "limit": 1_000, "apiKey": self.api_key}
            page_url = url
            page_params = params
            seen_pages: set[str] = set()
            while page_url and len(seen_pages) < self.max_pages:
                if page_url in seen_pages:
                    result["malformed"] += 1
                    result["errors"].append(f"{action_type}: repeated pagination URL")
                    fetch_incomplete = True
                    break
                seen_pages.add(page_url)
                try:
                    payload = self._request_json(page_url, page_params)
                    result["requests"] += 1
                except MassiveProviderError as exc:
                    result["requests"] += 1
                    status = self._status_for_error(exc)
                    self._set_status(self.ACTION_CURSOR_SOURCE, "US", status, exc.error_class, str(exc), exc.retry_after)
                    if result["stored"] or result["duplicates"]:
                        return self._finish(result, "partial", error_class=exc.error_class, retry_after=exc.retry_after)
                    return self._finish(result, status, error_class=exc.error_class, retry_after=exc.retry_after)
                rows = self._rows_from_payload(payload)
                if rows is None:
                    result["malformed"] += 1
                    result["errors"].append(f"{action_type}: response has no results list")
                    fetch_incomplete = True
                    break
                result["pages"] += 1
                for row in rows:
                    try:
                        event = self._event_from_action(
                            action_type, row, selected, page_url
                        )
                    except (TypeError, ValueError, KeyError) as exc:
                        result["malformed"] += 1
                        result["errors"].append(f"{action_type}: malformed action row: {exc}")
                        continue
                    if event is None:
                        continue
                    try:
                        stored = self.store.upsert_event(event)
                    except Exception as exc:  # noqa: BLE001 - action rows are isolated
                        storage_failed = True
                        result["errors"].append(f"{action_type}: event storage failed: {exc}")
                        result["malformed"] += 1
                        continue
                    if stored.get("created"):
                        result["stored"] += 1
                    elif stored.get("changed"):
                        result["updated"] += 1
                    else:
                        result["duplicates"] += 1
                next_url = self._next_page_url(payload)
                if not next_url:
                    break
                page_url = next_url
                page_params = {"apiKey": self.api_key}
            if page_url and len(seen_pages) >= self.max_pages and self._next_page_url(payload):
                result["malformed"] += 1
                result["errors"].append(f"{action_type}: pagination limit reached")
                fetch_incomplete = True
        if storage_failed:
            self._set_status(self.ACTION_CURSOR_SOURCE, "US", "error", "storage", "one or more actions failed to store")
            return self._finish(result, "error", error_class="storage")
        if fetch_incomplete:
            self._set_status(self.ACTION_CURSOR_SOURCE, "US", "partial", "pagination", "one or more action pages were incomplete")
            return self._finish(result, "partial", error_class="pagination")
        if result["malformed"]:
            status = "partial"
        else:
            status = "ok"
        self.store.set_source_cursor(
            self.ACTION_CURSOR_SOURCE,
            "US",
            last,
            cursor_type="date",
            overlap_value=f"{self.overlap_days}d",
            last_successful_run_id=hashlib.sha256(
                f"massive-actions:{last}".encode("utf-8")
            ).hexdigest()[:32],
            status=status,
        )
        return self._finish(result, status, cursor_after=last)

    def ingest_news(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[list[str]] = None,
    ) -> dict:
        """Optionally ingest provider news as a secondary deduplication origin."""
        selected = {str(ticker).upper() for ticker in (tickers or self._coverage_tickers())}
        result = self._result("ok", capability="news")
        if not self.api_key:
            self._set_status(self.NEWS_STATUS_SOURCE, "US", "disabled_missing_key", "authentication", "MASSIVE_API_KEY is not configured")
            return self._finish(result, "disabled_missing_key", error_class="authentication")
        today = self._now_date()
        first = start_date or (today - timedelta(days=self.initial_lookback_days)).isoformat()
        last = end_date or today.isoformat()
        url = f"{self.base_url}{self.NEWS_PATH}"
        try:
            payload = self._request_json(
                url,
                {"published_utc.gte": first, "published_utc.lte": last, "limit": 1_000, "apiKey": self.api_key},
            )
            result["requests"] += 1
        except MassiveProviderError as exc:
            result["requests"] += 1
            status = self._status_for_error(exc)
            self._set_status(self.NEWS_STATUS_SOURCE, "US", status, exc.error_class, str(exc), exc.retry_after)
            return self._finish(result, status, error_class=exc.error_class, retry_after=exc.retry_after)
        rows = self._rows_from_payload(payload)
        if rows is None:
            return self._finish(result, "error", error_class="contract")
        for row in rows:
            try:
                record = self._news_record(row, selected, url)
            except (TypeError, ValueError, KeyError):
                result["malformed"] += 1
                continue
            if record is None:
                continue
            try:
                stored = self.store.upsert_narrative(record)
            except Exception as exc:  # noqa: BLE001 - one secondary item is isolated
                result["malformed"] += 1
                result["errors"].append(f"{record.tickers[0]}: news storage failed: {exc}")
                continue
            if stored.get("indexing_status") == "error":
                result["malformed"] += 1
                result["errors"].append(
                    f"{record.tickers[0]}: narrative indexing failed: {stored.get('index_error', 'unknown error')}"
                )
                continue
            if stored.get("created"):
                result["stored"] += 1
            else:
                result["duplicates"] += 1
        status = "partial" if result["malformed"] else "ok"
        self._set_status(self.NEWS_STATUS_SOURCE, "US", status)
        return self._finish(result, status)

    def ingest_vendor_filings(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[list[str]] = None,
    ) -> dict:
        """Store optional vendor filing metadata without claiming SEC authority."""
        selected = {str(ticker).upper() for ticker in (tickers or self._coverage_tickers())}
        result = self._result("ok", capability="vendor_filings")
        if not self.api_key:
            self._set_status(self.FILINGS_STATUS_SOURCE, "US", "disabled_missing_key", "authentication", "MASSIVE_API_KEY is not configured")
            return self._finish(result, "disabled_missing_key", error_class="authentication")
        today = self._now_date()
        url = f"{self.base_url}{self.FILINGS_PATH}"
        try:
            payload = self._request_json(
                url,
                {
                    "filing_date.gte": start_date or (today - timedelta(days=self.initial_lookback_days)).isoformat(),
                    "filing_date.lte": end_date or today.isoformat(),
                    "limit": 1_000,
                    "apiKey": self.api_key,
                },
            )
            result["requests"] += 1
        except MassiveProviderError as exc:
            result["requests"] += 1
            status = self._status_for_error(exc)
            self._set_status(self.FILINGS_STATUS_SOURCE, "US", status, exc.error_class, str(exc), exc.retry_after)
            return self._finish(result, status, error_class=exc.error_class, retry_after=exc.retry_after)
        rows = self._rows_from_payload(payload)
        if rows is None:
            return self._finish(result, "error", error_class="contract")
        for row in rows:
            try:
                record = self._filing_record(row, selected, url)
            except (TypeError, ValueError, KeyError):
                result["malformed"] += 1
                continue
            if record is None:
                continue
            try:
                stored = self.store.upsert_narrative(record)
            except Exception as exc:  # noqa: BLE001 - one secondary item is isolated
                result["malformed"] += 1
                result["errors"].append(f"{record.tickers[0]}: filing storage failed: {exc}")
                continue
            if stored.get("indexing_status") == "error":
                result["malformed"] += 1
                result["errors"].append(
                    f"{record.tickers[0]}: filing metadata indexing failed: {stored.get('index_error', 'unknown error')}"
                )
                continue
            if stored.get("created"):
                result["stored"] += 1
            else:
                result["duplicates"] += 1
        status = "partial" if result["malformed"] else "ok"
        self._set_status(self.FILINGS_STATUS_SOURCE, "US", status)
        return self._finish(result, status)

    def ingest_all(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_news: bool = False,
        include_filings: bool = False,
    ) -> dict:
        """Run configured Massive capabilities independently."""
        result = {
            "market": self.ingest_market_data(start_date=start_date, end_date=end_date),
            "corporate_actions": self.ingest_corporate_actions(start_date=start_date, end_date=end_date),
        }
        if include_news:
            result["news"] = self.ingest_news(start_date=start_date, end_date=end_date)
        if include_filings:
            result["filings"] = self.ingest_vendor_filings(start_date=start_date, end_date=end_date)
        return result

    def _request_json(self, url: str, params: dict[str, object]) -> object:
        """Request one provider page with bounded retry semantics."""
        for attempt in range(1, self.max_attempts + 1):
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
                    raise MassiveProviderError(
                        "Massive returned malformed JSON", error_class="contract", status_code=status_code
                    ) from exc
                message = _error_text(payload)
                if status_code >= 400:
                    error_class = self._error_class(status_code, message)
                    error = MassiveProviderError(
                        message or f"Massive HTTP {status_code}",
                        error_class=error_class,
                        status_code=status_code,
                        retry_after=_retry_after(getattr(response, "headers", {}), payload),
                    )
                    if error_class in {"rate_limited", "transient"} and attempt < self.max_attempts:
                        self.sleep_fn(self._retry_delay(error, attempt))
                        continue
                    raise error
                if isinstance(payload, dict) and payload.get("error"):
                    if _is_entitlement_message(message):
                        raise MassiveProviderError(message, error_class="entitlement", status_code=status_code)
                    raise MassiveProviderError(message, error_class="contract", status_code=status_code)
                return payload
            except MassiveProviderError:
                raise
            except (requests.RequestException, ConnectionError, TimeoutError, OSError) as exc:
                error = MassiveProviderError(str(exc), error_class="transient")
                if attempt < self.max_attempts:
                    self.sleep_fn(self._retry_delay(error, attempt))
                    continue
                raise error from exc
        raise MassiveProviderError("Massive request exhausted", error_class="transient")

    def _observations_from_bar(
        self,
        row: object,
        market_date: str,
        adjusted: bool,
        revision: Optional[str],
        selected: set[str],
        source_url: str,
    ) -> list[ObservationRecord]:
        """Convert one grouped bar into five structured observations."""
        if not isinstance(row, dict):
            raise ValueError("row must be an object")
        ticker = str(row.get("T") or row.get("ticker") or "").strip().upper()
        if not ticker or ticker not in selected:
            return []
        if not isinstance(row.get("o"), (int, float)) or not isinstance(row.get("h"), (int, float)):
            raise ValueError("open/high must be numeric")
        if not isinstance(row.get("l"), (int, float)) or not isinstance(row.get("c"), (int, float)):
            raise ValueError("low/close must be numeric")
        if not isinstance(row.get("v"), (int, float)):
            raise ValueError("volume must be numeric")
        security = self.store.resolve_security(ticker, provider=self.SOURCE_NAME)
        security_id = security.get("security_id") if isinstance(security, dict) else None
        if not security_id:
            raise ValueError(f"no canonical security for {ticker}")
        row_revision = revision
        if row_revision is None:
            for key in ("provider_revision", "revision", "r", "updated_at"):
                if row.get(key) is not None:
                    row_revision = str(row[key])
                    break
        row_adjusted = row.get("adjusted")
        adjusted_value = bool(row_adjusted) if isinstance(row_adjusted, bool) else adjusted
        metadata = {"adjusted": adjusted_value, "market_date": market_date}
        if row_revision is not None:
            metadata["provider_revision"] = row_revision
        records = []
        for metric_id, field_name in self.BAR_METRICS:
            value = float(row[field_name])
            records.append(ObservationRecord(
                observation_id=f"observation/{self.SOURCE_NAME}/{ticker}/{market_date}/{metric_id}",
                metric_id=metric_id,
                series_id=f"{ticker}:{metric_id}",
                value_text=_format_number(value),
                value_numeric=value,
                unit="shares" if metric_id == "volume" else "usd",
                frequency="daily",
                period_start=market_date,
                period_end=market_date,
                vintage_at=self._now_timestamp(),
                as_of_at=market_date,
                scope="security",
                security_ids=(str(security_id),),
                tickers=(ticker,),
                sector=None,
                source_name=self.SOURCE_NAME,
                source_category="market_data",
                provider_record_id=f"{ticker}:{market_date}:{metric_id}",
                original_publisher=None,
                source_url=source_url,
                canonical_url=None,
                published_at=None,
                observed_at=self._now_timestamp(),
                accessed_at=self._now_timestamp(),
                ingested_at=self._now_timestamp(),
                license_label="provider_entitlement",
                normalization_version=NORMALIZATION_VERSION,
                metadata=metadata,
                evidence_authority="provider",
            ))
        return records

    def _event_from_action(
        self,
        action_type: str,
        row: object,
        selected: set[str],
        source_url: str,
    ) -> Optional[EventRecord]:
        """Convert one split/dividend row into a structured event."""
        if not isinstance(row, dict):
            raise ValueError("row must be an object")
        ticker = str(row.get("ticker") or row.get("T") or "").strip().upper()
        if ticker not in selected:
            return None
        security = self.store.resolve_security(ticker, provider=self.SOURCE_NAME)
        security_id = security.get("security_id") if isinstance(security, dict) else None
        if not security_id:
            raise ValueError(f"no canonical security for {ticker}")
        if action_type == "split":
            effective = _as_date(row.get("execution_date") or row.get("effective_date"), "execution_date")
            ex_date = _optional_date(row, "ex_date") or effective
            declared = _optional_date(row, "declared_date") or _optional_date(row, "declaration_date")
            record_date = _optional_date(row, "record_date")
            payable = _optional_date(row, "payable_date") or _optional_date(row, "pay_date")
            split_from = _number_or_none(row.get("split_from"))
            split_to = _number_or_none(row.get("split_to"))
            ratio = split_to / split_from if split_from not in (None, 0) and split_to is not None else None
            metadata = _action_metadata(
                declaration_date=declared,
                ex_date=ex_date,
                payable_date=payable,
                record_date=record_date,
                split_from=split_from,
                split_to=split_to,
            )
            amount = None
            currency = None
        else:
            effective = _as_date(
                row.get("ex_dividend_date") or row.get("ex_date") or row.get("effective_date"),
                "ex_dividend_date",
            )
            ex_date = effective
            declared = _optional_date(row, "declaration_date") or _optional_date(row, "declared_date")
            record_date = _optional_date(row, "record_date")
            payable = _optional_date(row, "pay_date") or _optional_date(row, "payable_date")
            metadata = _action_metadata(
                declaration_date=declared,
                ex_date=ex_date,
                payable_date=payable,
                record_date=record_date,
                frequency=str(row.get("frequency")) if row.get("frequency") is not None else None,
            )
            amount = _number_or_none(row.get("cash_amount") or row.get("amount"))
            currency = str(row.get("currency") or "USD").strip() or None
            ratio = None
        raw_identity = row.get("id") or row.get("ticker") or json.dumps(row, sort_keys=True, default=str)
        provider_id = str(raw_identity).strip()
        event_id = f"event/{self.SOURCE_NAME}/{_safe_identity(provider_id, json.dumps(row, sort_keys=True, default=str))}"
        return EventRecord(
            event_id=event_id,
            event_type=action_type,
            effective_at=effective,
            announced_at=declared,
            status="announced",
            security_ids=(str(security_id),),
            source_corpus_item_ids=(),
            source_name=self.SOURCE_NAME,
            source_category="corporate_action",
            provider_record_id=provider_id,
            original_publisher=None,
            source_url=source_url,
            canonical_url=None,
            published_at=declared,
            observed_at=self._now_timestamp(),
            accessed_at=self._now_timestamp(),
            ingested_at=self._now_timestamp(),
            license_label="provider_entitlement",
            normalization_version=NORMALIZATION_VERSION,
            amount=amount,
            currency=currency,
            ratio=ratio,
            action_date=effective,
            metadata=metadata,
            evidence_authority="provider",
        )

    def _news_record(
        self, row: object, selected: set[str], source_url: str,
    ) -> Optional[NarrativeRecord]:
        """Convert optional Massive news into the shared narrative contract."""
        if not isinstance(row, dict):
            raise ValueError("row must be an object")
        ticker_values = row.get("tickers") or row.get("ticker") or []
        if isinstance(ticker_values, str):
            ticker_values = [ticker_values]
        ticker = next((str(value).upper() for value in ticker_values if str(value).upper() in selected), None)
        if not ticker:
            return None
        title = str(row.get("title") or "").strip()
        summary = str(row.get("description") or row.get("summary") or "").strip()
        url = normalize_canonical_url(str(row.get("article_url") or row.get("url") or "").strip())
        published = str(row.get("published_utc") or row.get("published_at") or "").strip()
        if not title or not url or not published:
            raise ValueError("title, URL, and publication timestamp are required")
        provider_id = str(row.get("id") or row.get("article_id") or "").strip() or None
        body = "\n\n".join(value for value in (title, summary) if value)
        security = self.store.resolve_security(ticker, provider=self.SOURCE_NAME)
        security_id = security.get("security_id") if isinstance(security, dict) else None
        identity = _safe_identity(provider_id, url)
        return NarrativeRecord(
            corpus_item_id=f"news/{self.SOURCE_NAME}/{ticker}/{identity}",
            source_name=self.SOURCE_NAME,
            source_category="news_vendor",
            provider_record_id=provider_id,
            original_publisher=str(row.get("publisher") or "").strip() or None,
            item_type="news",
            title=title,
            body=body,
            summary=summary or None,
            published_at=published,
            observed_at=None,
            accessed_at=self._now_timestamp(),
            ingested_at=self._now_timestamp(),
            source_url=url,
            canonical_url=url,
            license_label="provider_entitlement",
            normalization_version=NORMALIZATION_VERSION,
            content_hash=content_hash(body),
            document_family="company_news",
            security_ids=(str(security_id),) if security_id else (),
            tickers=(ticker,),
            evidence_authority="provider",
        )

    def _filing_record(
        self, row: object, selected: set[str], source_url: str,
    ) -> Optional[NarrativeRecord]:
        """Convert optional vendor filing metadata without direct SEC authority."""
        if not isinstance(row, dict):
            raise ValueError("row must be an object")
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker not in selected:
            return None
        title = str(row.get("title") or row.get("filing_type") or "").strip()
        summary = str(row.get("summary") or row.get("description") or "").strip()
        filing_date = _as_date(row.get("filing_date") or row.get("filed_at"), "filing_date")
        provider_id = str(row.get("id") or row.get("accession_number") or "").strip() or None
        body = "\n\n".join(value for value in (title, summary) if value)
        if not body:
            raise ValueError("filing title or summary is required")
        url = normalize_canonical_url(str(row.get("url") or row.get("source_url") or source_url))
        security = self.store.resolve_security(ticker, provider=self.SOURCE_NAME)
        security_id = security.get("security_id") if isinstance(security, dict) else None
        identity = _safe_identity(provider_id, f"{ticker}:{filing_date}:{body}")
        metadata = {"form": str(row.get("filing_type") or row.get("form") or "") or None,
                    "accession": str(row.get("accession_number") or row.get("accession") or "") or None}
        metadata = {key: value for key, value in metadata.items() if value is not None}
        return NarrativeRecord(
            corpus_item_id=f"filing/{self.SOURCE_NAME}/{ticker}/{identity}",
            source_name=self.SOURCE_NAME,
            source_category="vendor_filing_metadata",
            provider_record_id=provider_id,
            original_publisher=None,
            item_type="filing",
            title=title,
            body=body,
            summary=summary or None,
            published_at=filing_date,
            observed_at=None,
            accessed_at=self._now_timestamp(),
            ingested_at=self._now_timestamp(),
            source_url=url,
            canonical_url=url,
            license_label="provider_entitlement",
            normalization_version=NORMALIZATION_VERSION,
            content_hash=content_hash(body),
            document_family="vendor_filing_metadata",
            security_ids=(str(security_id),) if security_id else (),
            tickers=(ticker,),
            metadata=metadata,
            evidence_authority="provider",
        )

    def _coverage_tickers(self) -> list[str]:
        """Resolve the configured broad universe at the ingestion boundary."""
        return [str(value).upper() for value in self.coverage.tickers_for(self.COVERAGE_SOURCE)]

    def _now_timestamp(self) -> str:
        """Normalize the injected clock value."""
        value = self.now_fn()
        if isinstance(value, datetime):
            dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        text = str(value)
        if text.isdigit():
            return datetime.fromtimestamp(float(text), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def _now_date(self) -> date:
        """Return the injected UTC date."""
        return date.fromisoformat(self._now_timestamp()[:10])

    @staticmethod
    def _market_dates(first: date, last: date) -> list[date]:
        """Return weekdays only; grouped US equity summaries do not exist on weekends."""
        return [
            first + timedelta(days=offset)
            for offset in range((last - first).days + 1)
            if (first + timedelta(days=offset)).weekday() < 5
        ]

    @staticmethod
    def _rows_from_payload(payload: object) -> Optional[list[object]]:
        """Accept Massive result envelopes and direct lists."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("results", "data", "items"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        return None

    def _next_page_url(self, payload: object) -> Optional[str]:
        """Read a bounded provider continuation URL/token."""
        if not isinstance(payload, dict):
            return None
        value = payload.get("next_url") or payload.get("next") or payload.get("next_page")
        if isinstance(value, dict):
            value = value.get("url") or value.get("href")
        if not value:
            return None
        text = str(value).strip()
        return text if text.startswith("http") else f"{self.base_url}{text if text.startswith('/') else '/' + text}"

    @staticmethod
    def _provider_revision(payload: object) -> Optional[str]:
        """Extract only an explicit provider revision marker."""
        if isinstance(payload, dict):
            for key in ("provider_revision", "revision", "updated_at"):
                if payload.get(key) is not None:
                    return str(payload[key])
        return None

    @staticmethod
    def _result(status: str, *, capability: str) -> dict:
        return {
            "status": status,
            "source": MassiveIngestor.SOURCE_NAME,
            "capability": capability,
            "stored": 0,
            "updated": 0,
            "duplicates": 0,
            "malformed": 0,
            "requests": 0,
            "pages": 0,
            "errors": [],
        }

    @staticmethod
    def _finish(result: dict, status: str, **values: object) -> dict:
        result["status"] = status
        result.update(values)
        return result

    def _set_status(
        self,
        source: str,
        partition_key: str,
        status: str,
        error_class: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        try:
            self.store.set_source_status(
                source,
                partition_key,
                status,
                error_class=error_class,
                error_message=error_message,
                retry_after=retry_after,
            )
        except Exception:  # noqa: BLE001 - status is observability, never source coupling
            logger.warning("Could not persist Massive status for %s/%s", source, partition_key, exc_info=True)

    @staticmethod
    def _status_for_error(error: MassiveProviderError) -> str:
        return {
            "authentication": "disabled_authentication",
            "entitlement": "disabled_entitlement",
            "rate_limited": "rate_limited",
        }.get(error.error_class, "error")

    @staticmethod
    def _error_class(status_code: int, message: str) -> str:
        if status_code == 401:
            return "authentication"
        if status_code == 403 or _is_entitlement_message(message):
            return "entitlement"
        if status_code == 429:
            return "rate_limited"
        if status_code >= 500:
            return "transient"
        return "permanent"

    def _retry_delay(self, error: MassiveProviderError, attempt: int) -> float:
        if error.retry_after is not None:
            return error.retry_after
        return min(self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay)


def _optional_date(row: dict, key: str) -> Optional[str]:
    """Read one optional provider action date."""
    value = row.get(key)
    return _as_date(value, key) if value else None


def _number_or_none(value: object) -> Optional[float]:
    """Convert one optional provider number without guessing missing values."""
    if value is None or value == "":
        return None
    number = float(value)
    return number


def _action_metadata(**values: object) -> dict[str, object]:
    """Keep corporate-action dates and ratios in the bounded metadata contract."""
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != ""
    }


CorporateActionsIngestor = MassiveIngestor

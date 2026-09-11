"""src/ingestion/marketaux_ingestor.py
Marketaux ticker-linked finance news ingestion (headline/snippet/URL only).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import timedelta
from typing import Any, Callable, Optional

import requests

from src.ingestion.normalization import (
    NORMALIZATION_VERSION,
    content_hash,
    normalize_canonical_url,
    parse_timestamp,
)
from src.ingestion.records import NarrativeRecord
from src.ingestion.vendor_common import (
    VendorProviderError,
    as_iso_timestamp,
    empty_result,
    finish_result,
    parse_number,
    request_json,
    safe_identity,
    utc_now,
)
from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)


class MarketauxIngestor:
    """Ingest Marketaux news into the corpus without redistributing full article bodies."""

    BASE_URL = "https://api.marketaux.com/v1/news/all"
    SOURCE_NAME = "marketaux"
    CURSOR_SOURCE = "marketaux_news"
    COVERAGE_SOURCE = "marketaux"
    API_KEY_ENV = "MARKETAUX_API_KEY"
    # Marketaux docs also accept API_TOKEN; we honor both env names.
    API_KEY_ENV_ALIASES = ("MARKETAUX_API_KEY", "MARKETAUX_API_TOKEN", "API_TOKEN")

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        coverage_resolver: Optional[CoverageResolver] = None,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        http_get: Optional[Callable[..., Any]] = None,
        timeout: float = 30.0,
        max_attempts: int = 3,
        overlap_hours: int = 24,
        initial_lookback_days: int = 3,
        articles_per_request: int = 3,
        now_fn: Callable[[], object] = utc_now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store or Store()
        self.coverage = coverage_resolver or CoverageResolver(self.store)
        self.api_key = str(api_key).strip() if api_key is not None else self._resolve_api_key()
        self.base_url = str(base_url).rstrip("/")
        self.http_get = http_get or requests.get
        self.timeout = max(float(timeout), 0.1)
        self.max_attempts = max(int(max_attempts), 1)
        self.overlap_hours = max(int(overlap_hours), 0)
        self.initial_lookback_days = max(int(initial_lookback_days), 1)
        self.articles_per_request = max(min(int(articles_per_request), 3), 1)
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn

    @classmethod
    def _resolve_api_key(cls) -> str:
        for name in cls.API_KEY_ENV_ALIASES:
            value = str(os.environ.get(name, "")).strip()
            if value:
                return value
        return ""

    def ingest_news(self, tickers: Optional[list[str]] = None) -> dict:
        """Ingest ticker-linked news for the coverage universe."""
        selected = list(tickers) if tickers is not None else self.coverage.tickers_for(
            self.COVERAGE_SOURCE
        )
        aggregate = empty_result(self.SOURCE_NAME)
        terminal = {
            "disabled_missing_key",
            "disabled_authentication",
            "disabled_entitlement",
            "rate_limited",
        }
        for ticker in selected:
            result = self.ingest_ticker_news(str(ticker).upper())
            aggregate["tickers"] += 1
            for key in (
                "stored",
                "duplicates",
                "malformed",
                "requests",
                "attempts",
                "narratives_stored",
            ):
                aggregate[key] += int(result.get(key, 0) or 0)
            aggregate["errors"].extend(result.get("errors") or [])
            if result.get("status") in terminal:
                aggregate["status"] = result["status"]
                aggregate["remaining_work_skipped"] = True
                if result.get("error_class"):
                    aggregate["error_class"] = result["error_class"]
                break
            if result.get("status") in {"error", "partial"} and aggregate["status"] == "ok":
                aggregate["status"] = result["status"]
        return finish_result(aggregate, aggregate["status"])

    def ingest_ticker_news(self, ticker: str) -> dict:
        """Fetch one ticker's recent Marketaux articles (free tier: ~3/request)."""
        ticker = str(ticker or "").strip().upper()
        result = empty_result(self.SOURCE_NAME)
        result["ticker"] = ticker
        if not ticker:
            return finish_result(result, "error", errors=["ticker is required"])
        if not self.api_key:
            self._set_status(
                ticker,
                "disabled_missing_key",
                "authentication",
                f"{self.API_KEY_ENV} is not configured",
            )
            return finish_result(result, "disabled_missing_key", error_class="authentication")

        cursor_before = self.store.get_source_cursor(self.CURSOR_SOURCE, ticker)
        now = self._now()
        published_after = self._published_after(cursor_before, now)
        accessed = now
        try:
            payload = self._get(
                {
                    "symbols": ticker,
                    "filter_entities": "true",
                    "language": "en",
                    "limit": self.articles_per_request,
                    "published_after": published_after,
                }
            )
            result["requests"] += 1
            result["attempts"] += 1
        except VendorProviderError as exc:
            status = self._status_for_error(exc)
            self._set_status(ticker, status, exc.error_class, str(exc), exc.retry_after)
            return finish_result(
                result,
                status,
                error_class=exc.error_class,
                retry_after=exc.retry_after,
                reset_at=exc.reset_at,
            )

        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            result["malformed"] += 1
            result["errors"].append("Marketaux response missing data list")
            self._set_status(ticker, "error", "contract", "Marketaux response missing data list")
            return finish_result(result, "error", error_class="contract")

        newest = cursor_before
        for row in rows:
            try:
                record, published_at = self._record_from_row(ticker, row, accessed)
            except (TypeError, ValueError, KeyError) as exc:
                result["malformed"] += 1
                result["errors"].append(f"{ticker}: malformed news row: {exc}")
                continue
            try:
                stored = self.store.upsert_narrative(record)
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"{ticker}: storage failed: {exc}")
                continue
            if stored.get("indexing_status") == "error":
                result["errors"].append(
                    f"{ticker}: narrative indexing failed: {stored.get('index_error', 'unknown')}"
                )
                continue
            if stored.get("created"):
                result["stored"] += 1
                result["narratives_stored"] += 1
            else:
                result["duplicates"] += 1
            if newest is None or parse_timestamp(published_at) > parse_timestamp(newest):
                newest = published_at

        if newest is not None and (
            cursor_before is None or parse_timestamp(newest) >= parse_timestamp(cursor_before)
        ):
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
                    status="partial" if result["malformed"] or result["errors"] else "success",
                )
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"{ticker}: cursor commit failed: {exc}")
                return finish_result(result, "error", error_class="cursor")
        else:
            self._set_status(
                ticker,
                "partial" if result["malformed"] or result["errors"] else "success",
                "contract" if result["malformed"] else None,
                "malformed rows were skipped" if result["malformed"] else None,
            )
        status = "partial" if result["malformed"] or result["errors"] else "ok"
        return finish_result(result, status)

    def _record_from_row(
        self, ticker: str, row: object, accessed_at: str,
    ) -> tuple[NarrativeRecord, str]:
        if not isinstance(row, dict):
            raise ValueError("row must be an object")
        title = str(row.get("title") or "").strip()
        snippet = str(
            row.get("snippet") or row.get("description") or row.get("summary") or ""
        ).strip()
        url = normalize_canonical_url(str(row.get("url") or "").strip())
        publisher_obj = row.get("source")
        if isinstance(publisher_obj, dict):
            publisher = str(publisher_obj.get("name") or publisher_obj.get("domain") or "").strip()
        else:
            publisher = str(publisher_obj or "").strip()
        publisher = publisher or "Marketaux"
        if not title or not url:
            raise ValueError("title and url are required")
        published_at = as_iso_timestamp(
            row.get("published_at") or row.get("published"),
            "published_at",
        )
        sentiment = None
        entities = row.get("entities") or []
        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                if str(entity.get("symbol") or "").strip().upper() == ticker:
                    sentiment = parse_number(entity.get("sentiment_score"))
                    break
        sentiment_text = f"Sentiment {sentiment:.4g}." if sentiment is not None else ""
        summary = " ".join(part for part in (sentiment_text, snippet) if part).strip()
        body = "\n\n".join(value for value in (title, summary[:500], publisher, url) if value)
        provider_id = str(row.get("uuid") or row.get("id") or url).strip()
        identity = safe_identity(provider_id, content_hash(body))
        security = self.store.resolve_security(ticker, provider=self.SOURCE_NAME)
        security_id = security.get("security_id") if isinstance(security, dict) else None
        record = NarrativeRecord(
            corpus_item_id=f"news/{self.SOURCE_NAME}/{ticker}/{identity}",
            source_name=self.SOURCE_NAME,
            source_category="news_vendor",
            provider_record_id=identity,
            original_publisher=publisher,
            item_type="news",
            title=title[:1000],
            body=body,
            summary=(summary[:4000] or None),
            published_at=published_at,
            observed_at=None,
            accessed_at=accessed_at,
            ingested_at=accessed_at,
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
        return record, published_at

    def _published_after(self, cursor: Optional[str], now: str) -> str:
        now_dt = parse_timestamp(now, "now")
        if cursor:
            start = parse_timestamp(cursor, "cursor") - timedelta(hours=self.overlap_hours)
        else:
            start = now_dt - timedelta(days=self.initial_lookback_days)
        return start.strftime("%Y-%m-%dT%H:%M")

    def _get(self, params: dict[str, object]) -> object:
        payload, _attempts, _retries = request_json(
            self.http_get,
            self.base_url,
            params={**params, "api_token": self.api_key},
            timeout=self.timeout,
            max_attempts=self.max_attempts,
            sleep_fn=self.sleep_fn,
            now_fn=self.now_fn,
        )
        return payload

    def _set_status(
        self,
        partition: str,
        status: str,
        error_class: Optional[str],
        message: Optional[str],
        retry_after: Optional[float] = None,
    ) -> None:
        try:
            self.store.set_source_status(
                self.CURSOR_SOURCE,
                partition,
                status,
                error_class=error_class,
                error_message=message,
                retry_after=retry_after,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Could not persist Marketaux status", exc_info=True)

    @staticmethod
    def _status_for_error(error: VendorProviderError) -> str:
        return {
            "authentication": "disabled_authentication",
            "entitlement": "disabled_entitlement",
            "rate_limited": "rate_limited",
        }.get(error.error_class, "error")

    def _now(self) -> str:
        value = self.now_fn()
        if isinstance(value, str):
            return value
        return utc_now()

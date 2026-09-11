"""src/ingestion/alpha_vantage_ingestor.py
Alpha Vantage OVERVIEW, INCOME_STATEMENT, and NEWS_SENTIMENT ingestion.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

import requests

from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash, normalize_canonical_url
from src.ingestion.records import NarrativeRecord, ObservationRecord
from src.ingestion.vendor_common import (
    VendorProviderError,
    as_date,
    as_iso_timestamp,
    empty_result,
    finish_result,
    format_number,
    parse_number,
    request_json,
    safe_identity,
    utc_now,
)
from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)

OVERVIEW_METRICS = (
    ("MarketCapitalization", "market_cap", "usd"),
    ("EBITDA", "ebitda", "usd"),
    ("PERatio", "pe_ratio", "ratio"),
    ("PEGRatio", "peg_ratio", "ratio"),
    ("BookValue", "book_value", "usd"),
    ("DividendPerShare", "dividend_per_share", "usd"),
    ("DividendYield", "dividend_yield", "ratio"),
    ("EPS", "eps", "usd"),
    ("RevenueTTM", "revenue_ttm", "usd"),
    ("GrossProfitTTM", "gross_profit_ttm", "usd"),
    ("ProfitMargin", "profit_margin", "ratio"),
    ("OperatingMarginTTM", "operating_margin_ttm", "ratio"),
    ("ReturnOnAssetsTTM", "return_on_assets_ttm", "ratio"),
    ("ReturnOnEquityTTM", "return_on_equity_ttm", "ratio"),
    ("TrailingPE", "trailing_pe", "ratio"),
    ("ForwardPE", "forward_pe", "ratio"),
    ("PriceToSalesRatioTTM", "price_to_sales_ttm", "ratio"),
    ("PriceToBookRatio", "price_to_book", "ratio"),
    ("Beta", "beta", "ratio"),
    ("52WeekHigh", "week_52_high", "usd"),
    ("52WeekLow", "week_52_low", "usd"),
)

INCOME_METRICS = (
    ("totalRevenue", "total_revenue", "usd"),
    ("grossProfit", "gross_profit", "usd"),
    ("operatingIncome", "operating_income", "usd"),
    ("netIncome", "net_income", "usd"),
    ("ebitda", "ebitda", "usd"),
    ("ebit", "ebit", "usd"),
)


class AlphaVantageIngestor:
    """Ingest Alpha Vantage fundamentals and bounded news sentiment into Store."""

    BASE_URL = "https://www.alphavantage.co/query"
    SOURCE_NAME = "alpha_vantage"
    COVERAGE_SOURCE = "alpha_vantage"
    API_KEY_ENV = "ALPHA_VANTAGE_API_KEY"

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
        include_news: bool = True,
        now_fn: Callable[[], object] = utc_now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store or Store()
        self.coverage = coverage_resolver or CoverageResolver(self.store)
        self.api_key = str(
            api_key if api_key is not None else os.environ.get(self.API_KEY_ENV, "")
        ).strip()
        self.base_url = str(base_url).rstrip("/")
        self.http_get = http_get or requests.get
        self.timeout = max(float(timeout), 0.1)
        self.max_attempts = max(int(max_attempts), 1)
        self.include_news = bool(include_news)
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn

    def ingest(self, tickers: Optional[list[str]] = None) -> dict:
        """Ingest OVERVIEW + annual income + optional NEWS_SENTIMENT for coverage tickers."""
        selected = list(tickers) if tickers is not None else self.coverage.tickers_for(
            self.COVERAGE_SOURCE
        )
        result = empty_result(self.SOURCE_NAME)
        if not self.api_key:
            self._set_status(
                "GLOBAL",
                "disabled_missing_key",
                "authentication",
                f"{self.API_KEY_ENV} is not configured",
            )
            return finish_result(result, "disabled_missing_key", error_class="authentication")

        terminal = {
            "disabled_missing_key",
            "disabled_authentication",
            "disabled_entitlement",
            "rate_limited",
        }
        for ticker in selected:
            ticker_result = self.ingest_ticker(str(ticker).upper())
            result["tickers"] += 1
            for key in (
                "stored",
                "updated",
                "duplicates",
                "malformed",
                "requests",
                "attempts",
                "facts_stored",
                "observations_stored",
                "narratives_stored",
            ):
                result[key] += int(ticker_result.get(key, 0) or 0)
            result["errors"].extend(ticker_result.get("errors") or [])
            if ticker_result.get("status") in terminal:
                result["status"] = ticker_result["status"]
                result["remaining_work_skipped"] = True
                if ticker_result.get("error_class"):
                    result["error_class"] = ticker_result["error_class"]
                break
            if ticker_result.get("status") in {"error", "partial"} and result["status"] == "ok":
                result["status"] = ticker_result["status"]

        if self.include_news and result.get("status") not in terminal and selected:
            news_result = self.ingest_news([str(t).upper() for t in selected])
            for key in (
                "stored",
                "duplicates",
                "malformed",
                "requests",
                "attempts",
                "narratives_stored",
            ):
                result[key] += int(news_result.get(key, 0) or 0)
            result["errors"].extend(news_result.get("errors") or [])
            if news_result.get("status") in terminal:
                result["status"] = news_result["status"]
                result["remaining_work_skipped"] = True
            elif news_result.get("status") in {"error", "partial"} and result["status"] == "ok":
                result["status"] = news_result["status"]

        return finish_result(result, result["status"])

    def ingest_ticker(self, ticker: str) -> dict:
        """Fetch OVERVIEW and annual INCOME_STATEMENT for one ticker."""
        ticker = str(ticker or "").strip().upper()
        result = empty_result(self.SOURCE_NAME)
        result["ticker"] = ticker
        if not ticker:
            return finish_result(result, "error", errors=["ticker is required"])
        if not self.api_key:
            return finish_result(result, "disabled_missing_key", error_class="authentication")

        accessed = self._now()
        try:
            overview = self._get({"function": "OVERVIEW", "symbol": ticker})
            result["requests"] += 1
            result["attempts"] += 1
            self._store_overview(ticker, overview, accessed, result)
            income = self._get({"function": "INCOME_STATEMENT", "symbol": ticker})
            result["requests"] += 1
            result["attempts"] += 1
            self._store_income(ticker, income, accessed, result)
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

        status = "partial" if result["malformed"] or result["errors"] else "ok"
        if result["facts_stored"] or result["observations_stored"]:
            try:
                self.store.mark_cache_fresh(ticker, "alpha_vantage_fundamentals", ttl_hours=24)
            except Exception as exc:  # noqa: BLE001
                logger.debug("mark_cache_fresh failed for %s: %s", ticker, exc)
        self._set_status(ticker, "success" if status == "ok" else "partial", None, None)
        return finish_result(result, status)

    def ingest_news(self, tickers: list[str]) -> dict:
        """Fetch NEWS_SENTIMENT once for a ticker batch (free-tier friendly)."""
        result = empty_result(self.SOURCE_NAME)
        symbols = [str(t).upper() for t in tickers if str(t).strip()]
        if not symbols:
            return finish_result(result, "ok")
        if not self.api_key:
            return finish_result(result, "disabled_missing_key", error_class="authentication")
        accessed = self._now()
        try:
            payload = self._get(
                {
                    "function": "NEWS_SENTIMENT",
                    "tickers": ",".join(symbols[:12]),
                    "limit": 50,
                    "sort": "LATEST",
                }
            )
            result["requests"] += 1
            result["attempts"] += 1
        except VendorProviderError as exc:
            status = self._status_for_error(exc)
            return finish_result(result, status, error_class=exc.error_class)

        rows = payload.get("feed") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            result["malformed"] += 1
            result["errors"].append("NEWS_SENTIMENT response missing feed list")
            return finish_result(result, "error", error_class="contract")

        selected = set(symbols)
        for row in rows:
            try:
                records = self._news_records(row, selected, accessed)
            except (TypeError, ValueError, KeyError) as exc:
                result["malformed"] += 1
                result["errors"].append(f"malformed news row: {exc}")
                continue
            for record in records:
                try:
                    stored = self.store.upsert_narrative(record)
                except Exception as exc:  # noqa: BLE001
                    result["errors"].append(f"news storage failed: {exc}")
                    continue
                if stored.get("created"):
                    result["stored"] += 1
                    result["narratives_stored"] += 1
                else:
                    result["duplicates"] += 1
        status = "partial" if result["malformed"] or result["errors"] else "ok"
        return finish_result(result, status)

    def _store_overview(self, ticker: str, payload: object, accessed: str, result: dict) -> None:
        if not isinstance(payload, dict) or not payload.get("Symbol"):
            result["malformed"] += 1
            result["errors"].append(f"{ticker}: empty OVERVIEW payload")
            return
        period = as_date(payload.get("LatestQuarter") or accessed[:10], "LatestQuarter")
        for field, metric_id, unit in OVERVIEW_METRICS:
            value = parse_number(payload.get(field))
            if value is None:
                continue
            self._persist_metric(
                ticker=ticker,
                metric_id=metric_id,
                value=value,
                unit=unit,
                period=period,
                frequency="ttm",
                period_type="ttm",
                provider_record_id=f"overview:{ticker}:{metric_id}:{period}",
                source_url=f"{self.base_url}?function=OVERVIEW&symbol={ticker}",
                accessed=accessed,
                result=result,
                metadata={"report_period": period, "fiscal_period": "TTM"},
            )

    def _store_income(self, ticker: str, payload: object, accessed: str, result: dict) -> None:
        if not isinstance(payload, dict):
            result["malformed"] += 1
            result["errors"].append(f"{ticker}: INCOME_STATEMENT was not an object")
            return
        reports = payload.get("annualReports") or []
        if not isinstance(reports, list) or not reports:
            result["malformed"] += 1
            result["errors"].append(f"{ticker}: INCOME_STATEMENT missing annualReports")
            return
        report = reports[0]
        if not isinstance(report, dict):
            result["malformed"] += 1
            return
        period = as_date(report.get("fiscalDateEnding"), "fiscalDateEnding")
        for field, metric_id, unit in INCOME_METRICS:
            value = parse_number(report.get(field))
            if value is None:
                continue
            self._persist_metric(
                ticker=ticker,
                metric_id=metric_id,
                value=value,
                unit=unit,
                period=period,
                frequency="annual",
                period_type="annual",
                provider_record_id=f"income:{ticker}:{metric_id}:{period}",
                source_url=f"{self.base_url}?function=INCOME_STATEMENT&symbol={ticker}",
                accessed=accessed,
                result=result,
                metadata={"fiscal_period": "FY", "report_period": period},
            )

    def _persist_metric(
        self,
        *,
        ticker: str,
        metric_id: str,
        value: float,
        unit: str,
        period: str,
        frequency: str,
        period_type: str,
        provider_record_id: str,
        source_url: str,
        accessed: str,
        result: dict,
        metadata: dict[str, object],
    ) -> None:
        try:
            if self.store.save_fundamental(
                ticker=ticker,
                metric=metric_id,
                value=value,
                unit=unit,
                period=period,
                period_type=period_type,
                source_type=self.SOURCE_NAME,
                source_url=source_url,
            ):
                result["facts_stored"] += 1
                result["stored"] += 1
            else:
                result["duplicates"] += 1
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"{ticker}/{metric_id} fundamental store failed: {exc}")

        security = self.store.resolve_security(ticker, provider=self.SOURCE_NAME)
        security_id = security.get("security_id") if isinstance(security, dict) else None
        if not security_id:
            return
        record = ObservationRecord(
            observation_id=f"observation/{self.SOURCE_NAME}/{ticker}/{metric_id}/{period}",
            metric_id=metric_id,
            series_id=f"{ticker}:{metric_id}",
            value_text=format_number(value),
            value_numeric=value,
            unit=unit,
            frequency=frequency,
            period_start=None,
            period_end=period,
            vintage_at=accessed,
            as_of_at=period,
            scope="security",
            security_ids=(str(security_id),),
            tickers=(ticker,),
            sector=None,
            source_name=self.SOURCE_NAME,
            source_category="fundamentals",
            provider_record_id=provider_record_id,
            original_publisher="Alpha Vantage",
            source_url=source_url,
            canonical_url=None,
            published_at=None,
            observed_at=accessed,
            accessed_at=accessed,
            ingested_at=accessed,
            license_label="provider_entitlement",
            normalization_version=NORMALIZATION_VERSION,
            metadata=metadata,
            evidence_authority="provider",
        )
        try:
            stored = self.store.upsert_observation(record)
            if stored.get("created") or stored.get("changed"):
                result["observations_stored"] += 1
                if stored.get("created"):
                    result["stored"] += 1
                else:
                    result["updated"] += 1
            else:
                result["duplicates"] += 1
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"{ticker}/{metric_id} observation store failed: {exc}")

    def _news_records(
        self, row: object, selected: set[str], accessed: str,
    ) -> list[NarrativeRecord]:
        if not isinstance(row, dict):
            raise ValueError("row must be an object")
        title = str(row.get("title") or "").strip()
        summary = str(row.get("summary") or "").strip()
        url = normalize_canonical_url(str(row.get("url") or "").strip())
        publisher = str(row.get("source") or "").strip() or "Alpha Vantage"
        if not title or not url:
            raise ValueError("title and url are required")
        published = as_iso_timestamp(row.get("time_published"), "time_published")
        overall = parse_number(row.get("overall_sentiment_score"))
        ticker_rows = row.get("ticker_sentiment") or []
        matched: list[tuple[str, Optional[float]]] = []
        if isinstance(ticker_rows, list):
            for item in ticker_rows:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("ticker") or "").strip().upper()
                if symbol in selected:
                    matched.append((symbol, parse_number(item.get("ticker_sentiment_score"))))
        if not matched:
            return []
        provider_id = str(row.get("url") or title).strip()
        records: list[NarrativeRecord] = []
        for ticker, sentiment in matched:
            sentiment_text = (
                f"Sentiment {sentiment:.4g}."
                if sentiment is not None
                else (f"Overall sentiment {overall:.4g}." if overall is not None else "")
            )
            snippet = " ".join(part for part in (sentiment_text, summary) if part).strip()
            # Prefer headline + snippet + URL provenance; never redistribute full bodies.
            body = "\n\n".join(value for value in (title, snippet[:500], publisher, url) if value)
            security = self.store.resolve_security(ticker, provider=self.SOURCE_NAME)
            security_id = security.get("security_id") if isinstance(security, dict) else None
            identity = safe_identity(provider_id, content_hash(body))
            records.append(
                NarrativeRecord(
                    corpus_item_id=f"news/{self.SOURCE_NAME}/{ticker}/{identity}",
                    source_name=self.SOURCE_NAME,
                    source_category="news_vendor",
                    provider_record_id=identity,
                    original_publisher=publisher,
                    item_type="news",
                    title=title[:1000],
                    body=body,
                    summary=(snippet[:4000] or None),
                    published_at=published,
                    observed_at=None,
                    accessed_at=accessed,
                    ingested_at=accessed,
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
            )
        return records

    def _get(self, params: dict[str, object]) -> object:
        payload, _attempts, _retries = request_json(
            self.http_get,
            self.base_url,
            params={**params, "apikey": self.api_key},
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
                self.SOURCE_NAME,
                partition,
                status,
                error_class=error_class,
                error_message=message,
                retry_after=retry_after,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Could not persist Alpha Vantage status", exc_info=True)

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

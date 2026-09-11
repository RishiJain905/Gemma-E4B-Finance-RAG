"""src/ingestion/fmp_ingestor.py
Financial Modeling Prep annual income statement and TTM key ratios ingestion.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

import requests

from src.ingestion.errors import ErrorClass, ProviderError
from src.ingestion.normalization import NORMALIZATION_VERSION
from src.ingestion.records import ObservationRecord
from src.ingestion.vendor_common import (
    VendorProviderError,
    as_date,
    empty_result,
    finish_result,
    format_number,
    parse_number,
    request_json,
    utc_now,
)
from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)

INCOME_METRICS = (
    ("revenue", "total_revenue", "usd"),
    ("grossProfit", "gross_profit", "usd"),
    ("operatingIncome", "operating_income", "usd"),
    ("netIncome", "net_income", "usd"),
    ("ebitda", "ebitda", "usd"),
    ("eps", "eps", "usd"),
    ("epsdiluted", "eps_diluted", "usd"),
)

RATIO_METRICS = (
    ("peRatioTTM", "pe_ratio_ttm", "ratio"),
    ("priceToSalesRatioTTM", "price_to_sales_ttm", "ratio"),
    ("priceToBookRatioTTM", "price_to_book_ttm", "ratio"),
    ("returnOnEquityTTM", "return_on_equity_ttm", "ratio"),
    ("returnOnAssetsTTM", "return_on_assets_ttm", "ratio"),
    ("grossProfitMarginTTM", "gross_margin_ttm", "ratio"),
    ("operatingProfitMarginTTM", "operating_margin_ttm", "ratio"),
    ("netProfitMarginTTM", "net_margin_ttm", "ratio"),
    ("currentRatioTTM", "current_ratio_ttm", "ratio"),
    ("debtEquityRatioTTM", "debt_equity_ttm", "ratio"),
    ("dividendYielTTM", "dividend_yield_ttm", "ratio"),
    ("dividendYieldTTM", "dividend_yield_ttm", "ratio"),
)


class FMPIngestor:
    """Ingest FMP income statements and TTM ratios into fundamentals/observations."""

    # Stable API (legacy /api/v3 paths are entitlement-blocked for new free keys).
    BASE_URL = "https://financialmodelingprep.com/stable"
    SOURCE_NAME = "fmp"
    COVERAGE_SOURCE = "fmp"
    API_KEY_ENV = "FMP_API_KEY"

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
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn

    def ingest(self, tickers: Optional[list[str]] = None) -> dict:
        """Ingest annual income + TTM ratios for coverage tickers."""
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
        return finish_result(result, result["status"])

    def ingest_ticker(self, ticker: str) -> dict:
        """Fetch annual income statement and TTM ratios for one ticker."""
        ticker = str(ticker or "").strip().upper()
        result = empty_result(self.SOURCE_NAME)
        result["ticker"] = ticker
        if not ticker:
            return finish_result(result, "error", errors=["ticker is required"])
        if not self.api_key:
            return finish_result(result, "disabled_missing_key", error_class="authentication")

        accessed = self._now()
        try:
            income = self._get(
                "/income-statement",
                {"symbol": ticker, "period": "annual", "limit": 1},
            )
            result["requests"] += 1
            result["attempts"] += 1
            self._store_income(ticker, income, accessed, result)
            ratios = self._get("/ratios-ttm", {"symbol": ticker})
            result["requests"] += 1
            result["attempts"] += 1
            self._store_ratios(ticker, ratios, accessed, result)
        except (VendorProviderError, ProviderError) as exc:
            # Budgeted scheduler HTTP raises base ProviderError (ENTITLEMENT)
            # on free-tier 402 before request_json can wrap VendorProviderError.
            # Treat symbol-scoped entitlement/contract blocks as per-ticker skips
            # so working tickers are not aborted by a provider-wide cooldown.
            if self._is_per_ticker_entitlement(exc):
                skip_msg = (
                    f"{ticker}: free-tier entitlement skip "
                    f"(symbol unavailable under current plan)"
                )
                result["errors"].append(skip_msg)
                self._set_status(ticker, "skipped_entitlement", "item", str(exc))
                finished = finish_result(result, "ok")
                finished.pop("error_class", None)
                finished.pop("retry_after", None)
                finished.pop("reset_at", None)
                finished["remaining_work_skipped"] = False
                return finished

            status = self._status_for_error(exc)
            error_class = self._error_class_value(exc)
            self._set_status(
                ticker,
                status,
                error_class,
                str(exc),
                getattr(exc, "retry_after", None),
            )
            return finish_result(
                result,
                status,
                error_class=error_class,
                retry_after=getattr(exc, "retry_after", None),
                reset_at=getattr(exc, "reset_at", None),
            )

        status = "partial" if result["malformed"] or result["errors"] else "ok"
        if result["facts_stored"] or result["observations_stored"]:
            try:
                self.store.mark_cache_fresh(ticker, "fmp_fundamentals", ttl_hours=24)
            except Exception as exc:  # noqa: BLE001
                logger.debug("mark_cache_fresh failed for %s: %s", ticker, exc)
        self._set_status(ticker, "success" if status == "ok" else "partial", None, None)
        return finish_result(result, status)

    def _store_income(self, ticker: str, payload: object, accessed: str, result: dict) -> None:
        rows = payload if isinstance(payload, list) else []
        if not rows or not isinstance(rows[0], dict):
            result["malformed"] += 1
            result["errors"].append(f"{ticker}: income-statement response empty")
            return
        report = rows[0]
        period = as_date(report.get("date") or report.get("fillingDate") or accessed[:10], "date")
        source_url = f"{self.base_url}/income-statement?symbol={ticker}"
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
                source_url=source_url,
                accessed=accessed,
                result=result,
                metadata={"fiscal_period": "FY", "report_period": period},
            )

    def _store_ratios(self, ticker: str, payload: object, accessed: str, result: dict) -> None:
        rows = payload if isinstance(payload, list) else []
        if not rows or not isinstance(rows[0], dict):
            result["malformed"] += 1
            result["errors"].append(f"{ticker}: ratios-ttm response empty")
            return
        report = rows[0]
        period = as_date(accessed[:10], "as_of")
        source_url = f"{self.base_url}/ratios-ttm?symbol={ticker}"
        seen: set[str] = set()
        for field, metric_id, unit in RATIO_METRICS:
            if metric_id in seen:
                continue
            value = parse_number(report.get(field))
            if value is None:
                continue
            seen.add(metric_id)
            self._persist_metric(
                ticker=ticker,
                metric_id=metric_id,
                value=value,
                unit=unit,
                period=period,
                frequency="ttm",
                period_type="ttm",
                provider_record_id=f"ratios:{ticker}:{metric_id}:{period}",
                source_url=source_url,
                accessed=accessed,
                result=result,
                metadata={"fiscal_period": "TTM", "report_period": period},
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
            original_publisher="Financial Modeling Prep",
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

    def _get(self, path: str, params: dict[str, object]) -> object:
        payload, _attempts, _retries = request_json(
            self.http_get,
            f"{self.base_url}{path}",
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
            logger.debug("Could not persist FMP status", exc_info=True)

    @staticmethod
    def _error_class_value(error: ProviderError) -> str:
        error_class = getattr(error, "error_class", "")
        if isinstance(error_class, ErrorClass):
            return error_class.value
        if hasattr(error_class, "value"):
            return str(error_class.value)
        return str(error_class or "")

    @classmethod
    def _status_for_error(cls, error: ProviderError) -> str:
        return {
            "authentication": "disabled_authentication",
            "entitlement": "disabled_entitlement",
            "rate_limited": "rate_limited",
        }.get(cls._error_class_value(error), "error")

    @classmethod
    def _is_per_ticker_entitlement(cls, error: ProviderError) -> bool:
        """Return True when a failure is a single-symbol free-tier block.

        FMP free keys return HTTP 402 (and entitlement-class payloads) for some
        symbols while others remain available. Budgeted scheduler HTTP raises
        base ProviderError before VendorProviderError wrapping — both must skip.
        """
        error_class = cls._error_class_value(error)
        message = str(error).lower()
        if int(getattr(error, "status_code", 0) or 0) == 402:
            return True
        if any(marker in message for marker in ("subscription", "premium")):
            return True
        if error_class in {"entitlement", "contract"}:
            return True
        return False

    def _now(self) -> str:
        value = self.now_fn()
        if isinstance(value, str):
            return value
        return utc_now()

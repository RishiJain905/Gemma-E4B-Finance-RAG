"""
src/macros/estimates_ingestor.py
Analyst estimates ingestion - fetches forward-looking consensus estimates and
price targets, then stores them as estimate-period fundamentals.
"""

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf
import yaml

from src.storage.store import Store
from src.utils.env import load_env

logger = logging.getLogger(__name__)

ESTIMATE_METRICS = (
    "estimate_revenue_current_q",
    "estimate_revenue_next_q",
    "estimate_revenue_current_y",
    "estimate_revenue_next_y",
    "estimate_eps_current_q",
    "estimate_eps_next_q",
    "estimate_eps_current_y",
    "estimate_eps_next_y",
)

PRICE_TARGET_METRICS = (
    "price_target_mean",
    "price_target_high",
    "price_target_low",
    "num_analysts",
    "recommendation_mean",
)


class EstimatesIngestor:
    """Fetch analyst estimates and store them as forward-dated facts."""

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs/estimates.yaml"

    def __init__(
        self,
        store: Optional[Store] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        load_env()
        self.store = store or Store()
        self.config = self._load_config(config_path)
        self.provider = str(self.config.get("provider", "yfinance")).lower()
        self.request_delay = float(self.config.get("request_delay", 3.0))
        self.timeout = int(self.config.get("timeout", 30))
        self.max_retries = int(self.config.get("max_retries", 3))
        self.session = requests.Session()

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load estimates configuration from YAML, with inline defaults."""
        path = config_path or self.DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {
            "provider": "yfinance",
            "request_delay": 3.0,
            "max_retries": 3,
            "timeout": 30,
            "fmp": {
                "api_key": "${FMP_API_KEY}",
                "base_url": "https://financialmodelingprep.com/api/v3",
            },
            "finnhub": {
                "api_key": "${FINNHUB_API_KEY}",
                "base_url": "https://finnhub.io/api/v1",
            },
        }

    def fetch_for_ticker(self, ticker: str) -> dict:
        """Fetch estimates for one ticker and return a status dict; never raises."""
        ticker = ticker.upper()
        errors: list[str] = []

        try:
            if self.provider in {"fmp", "finnhub"} and not self._api_key(self.provider):
                logger.warning("Skipping %s estimates for %s: missing API key", self.provider, ticker)
                return {
                    "ticker": ticker,
                    "status": "skipped_no_key",
                    "facts_stored": 0,
                    "errors": [],
                }

            dispatch = {
                "yfinance": self._fetch_yfinance,
                "fmp": self._fetch_fmp,
                "finnhub": self._fetch_finnhub,
            }
            fetcher = dispatch.get(self.provider)
            if fetcher is None:
                return {
                    "ticker": ticker,
                    "status": "error",
                    "facts_stored": 0,
                    "errors": [f"unknown provider: {self.provider}"],
                }

            facts = fetcher(ticker, errors)
            stored, store_errors = self._store_facts(ticker, facts)
            errors.extend(store_errors)

            if stored:
                try:
                    ttl = int(self.config.get("ttl_hours", 24))
                    self.store.mark_cache_fresh(ticker, "estimates", ttl_hours=ttl)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("mark_cache_fresh failed for %s estimates: %s", ticker, exc)

            status = "success" if stored else "no_data"
            if errors and not stored:
                status = "error"
            return {
                "ticker": ticker,
                "status": status,
                "facts_stored": stored,
                "errors": errors,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Estimates ingestion failed for %s: %s", ticker, exc)
            return {
                "ticker": ticker,
                "status": "error",
                "facts_stored": 0,
                "errors": [str(exc)],
            }

    def fetch_all_core(self) -> dict[str, dict]:
        """Fetch estimates for all core tickers with the configured delay."""
        from src.ingestion.yfinance_ingestor import YFinanceIngestor

        ingestor = YFinanceIngestor(store=self.store)
        results: dict[str, dict] = {}
        for ticker in ingestor.core_tickers:
            results[ticker] = self.fetch_for_ticker(ticker)
            if self.request_delay > 0:
                time.sleep(self.request_delay)
        return results

    def _fetch_yfinance(self, ticker: str, errors: list[str]) -> list[dict]:
        facts: list[dict] = []
        yf_ticker = yf.Ticker(ticker)

        earnings = self._safe_yf_call(ticker, "get_earnings_estimate", yf_ticker.get_earnings_estimate, errors)
        revenue = self._safe_yf_call(ticker, "get_revenue_estimate", yf_ticker.get_revenue_estimate, errors)
        price_targets = self._safe_yf_attr(ticker, "analyst_price_targets", yf_ticker, errors)
        recommendations = self._safe_yf_attr(ticker, "recommendations", yf_ticker, errors)

        facts.extend(self._estimate_rows_from_frame(revenue, "estimate_revenue", "usd"))
        facts.extend(self._estimate_rows_from_frame(earnings, "estimate_eps", "usd"))

        analyst_count = self._row_value(self._frame_row(earnings, "0q"), ("numberOfAnalysts", "numberofanalysts"))
        if analyst_count is not None:
            facts.append(self._fact("num_analysts", analyst_count, "count", self._target_period()))

        facts.extend(self._price_target_rows(price_targets))

        recommendation_mean = self._recommendation_mean(recommendations)
        if recommendation_mean is not None:
            facts.append(
                self._fact(
                    "recommendation_mean",
                    recommendation_mean,
                    "score",
                    self._target_period(),
                )
            )

        return facts

    def _fetch_fmp(self, ticker: str, errors: list[str]) -> list[dict]:
        facts: list[dict] = []
        key = self._api_key("fmp")
        cfg = self.config.get("fmp", {}) or {}
        base_url = str(cfg.get("base_url", "https://financialmodelingprep.com/api/v3")).rstrip("/")

        try:
            resp = self.session.get(
                f"{base_url}/analyst-estimates/{ticker}",
                params={"apikey": key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            records = self._as_records(resp.json())
            facts.extend(self._provider_estimate_rows(records))
        except Exception as exc:  # noqa: BLE001
            logger.debug("FMP analyst estimates failed for %s: %s", ticker, exc)
            errors.append(f"fmp_estimates: {exc}")

        price_base = base_url.replace("/api/v3", "/api/v4")
        try:
            resp = self.session.get(
                f"{price_base}/price-target-consensus",
                params={"symbol": ticker, "apikey": key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = self._first_record(resp.json())
            facts.extend(
                self._price_target_rows(
                    {
                        "mean": self._value_from(data, ("targetConsensus", "targetMean", "priceTargetMean")),
                        "high": self._value_from(data, ("targetHigh", "priceTargetHigh")),
                        "low": self._value_from(data, ("targetLow", "priceTargetLow")),
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("FMP price targets failed for %s: %s", ticker, exc)
            errors.append(f"fmp_price_targets: {exc}")

        return facts

    def _fetch_finnhub(self, ticker: str, errors: list[str]) -> list[dict]:
        facts: list[dict] = []
        key = self._api_key("finnhub")
        cfg = self.config.get("finnhub", {}) or {}
        base_url = str(cfg.get("base_url", "https://finnhub.io/api/v1")).rstrip("/")

        try:
            data = self._get_json(f"{base_url}/stock/eps-estimate", {"symbol": ticker, "token": key})
            facts.extend(self._provider_estimate_rows(self._as_records(data), eps_only=True))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Finnhub EPS estimates failed for %s: %s", ticker, exc)
            errors.append(f"finnhub_eps: {exc}")

        try:
            data = self._get_json(f"{base_url}/stock/revenue-estimate", {"symbol": ticker, "token": key})
            facts.extend(self._provider_estimate_rows(self._as_records(data), revenue_only=True))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Finnhub revenue estimates failed for %s: %s", ticker, exc)
            errors.append(f"finnhub_revenue: {exc}")

        try:
            data = self._get_json(f"{base_url}/stock/recommendation", {"symbol": ticker, "token": key})
            mean = self._recommendation_mean(pd.DataFrame(self._as_records(data)))
            if mean is not None:
                facts.append(self._fact("recommendation_mean", mean, "score", self._target_period()))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Finnhub recommendations failed for %s: %s", ticker, exc)
            errors.append(f"finnhub_recommendation: {exc}")

        try:
            data = self._get_json(f"{base_url}/stock/price-target", {"symbol": ticker, "token": key})
            facts.extend(
                self._price_target_rows(
                    {
                        "mean": self._value_from(data, ("targetMean", "mean")),
                        "high": self._value_from(data, ("targetHigh", "high")),
                        "low": self._value_from(data, ("targetLow", "low")),
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Finnhub price targets failed for %s: %s", ticker, exc)
            errors.append(f"finnhub_price_target: {exc}")

        return facts

    def _get_json(self, url: str, params: dict) -> Any:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _store_facts(self, ticker: str, facts: list[dict]) -> tuple[int, list[str]]:
        stored = 0
        errors: list[str] = []
        for fact in facts:
            value = self._to_float(fact.get("value"))
            if value is None:
                continue
            try:
                self.store.save_fundamental(
                    ticker=ticker,
                    metric=fact["metric"],
                    value=value,
                    unit=fact.get("unit", "usd"),
                    period=fact["period"],
                    period_type=fact.get("period_type", "estimate"),
                    source_type="estimates",
                    source_url=fact.get("source_url"),
                )
                stored += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to store estimate for %s/%s: %s", ticker, fact.get("metric"), exc)
                errors.append(f"store {fact.get('metric')}: {exc}")
        return stored, errors

    def _api_key(self, provider: str) -> str:
        cfg = self.config.get(provider, {}) or {}
        raw = str(cfg.get("api_key", "") or "")
        if raw.startswith("${") and raw.endswith("}"):
            return os.environ.get(raw[2:-1], "")
        return raw

    @staticmethod
    def _safe_yf_call(ticker: str, name: str, accessor: Any, errors: list[str]) -> Any:
        try:
            return accessor()
        except Exception as exc:  # noqa: BLE001
            logger.debug("yfinance %s failed for %s: %s", name, ticker, exc)
            errors.append(f"{name}: {exc}")
            return None

    @staticmethod
    def _safe_yf_attr(ticker: str, name: str, obj: Any, errors: list[str]) -> Any:
        try:
            return getattr(obj, name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("yfinance %s failed for %s: %s", name, ticker, exc)
            errors.append(f"{name}: {exc}")
            return None

    def _estimate_rows_from_frame(self, frame: Any, prefix: str, unit: str) -> list[dict]:
        rows: list[dict] = []
        mapping = (
            ("0q", f"{prefix}_current_q", self._quarter_period(0)),
            ("+1q", f"{prefix}_next_q", self._quarter_period(1)),
            ("0y", f"{prefix}_current_y", self._year_period(0)),
            ("+1y", f"{prefix}_next_y", self._year_period(1)),
        )
        for row_key, metric, period in mapping:
            value = self._row_value(self._frame_row(frame, row_key), ("avg", "average"))
            if value is not None:
                rows.append(self._fact(metric, value, unit, period))
        return rows

    def _provider_estimate_rows(
        self,
        records: list[dict],
        eps_only: bool = False,
        revenue_only: bool = False,
    ) -> list[dict]:
        rows: list[dict] = []
        for index, record in enumerate(records[:2]):
            suffix = "current_y" if index == 0 else "next_y"
            period = self._provider_year_period(record, index)
            if not eps_only:
                revenue = self._value_from(
                    record,
                    ("estimatedRevenueAvg", "revenueAvg", "revenue", "revenueEstimate"),
                )
                if revenue is not None:
                    rows.append(self._fact(f"estimate_revenue_{suffix}", revenue, "usd", period))
            if not revenue_only:
                eps = self._value_from(record, ("estimatedEpsAvg", "epsAvg", "eps", "epsEstimate"))
                if eps is not None:
                    rows.append(self._fact(f"estimate_eps_{suffix}", eps, "usd", period))
        return rows

    def _price_target_rows(self, data: Any) -> list[dict]:
        if not data:
            return []
        return [
            self._fact(metric, value, "usd", self._target_period())
            for metric, value in (
                ("price_target_mean", self._value_from(data, ("mean", "targetMean", "targetConsensus"))),
                ("price_target_high", self._value_from(data, ("high", "targetHigh"))),
                ("price_target_low", self._value_from(data, ("low", "targetLow"))),
            )
            if value is not None
        ]

    def _recommendation_mean(self, recommendations: Any) -> Optional[float]:
        if recommendations is None:
            return None
        if isinstance(recommendations, pd.DataFrame):
            if recommendations.empty:
                return None
            row = recommendations.iloc[-1]
        elif isinstance(recommendations, list):
            if not recommendations:
                return None
            row = recommendations[-1]
        elif isinstance(recommendations, dict):
            row = recommendations
        else:
            return None

        weighted = (
            (self._value_from(row, ("strongBuy", "strong_buy")) or 0) * 1
            + (self._value_from(row, ("buy",)) or 0) * 2
            + (self._value_from(row, ("hold",)) or 0) * 3
            + (self._value_from(row, ("sell",)) or 0) * 4
            + (self._value_from(row, ("strongSell", "strong_sell")) or 0) * 5
        )
        total = sum(
            self._value_from(row, keys) or 0
            for keys in (
                ("strongBuy", "strong_buy"),
                ("buy",),
                ("hold",),
                ("sell",),
                ("strongSell", "strong_sell"),
            )
        )
        if not total:
            return None
        return float(weighted) / float(total)

    @staticmethod
    def _frame_row(frame: Any, row_key: str) -> Any:
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        try:
            if row_key in frame.index:
                row = frame.loc[row_key]
                return row.iloc[0] if isinstance(row, pd.DataFrame) else row
            if "period" in frame.columns:
                matches = frame[frame["period"].astype(str).str.lower() == row_key.lower()]
                if not matches.empty:
                    return matches.iloc[0]
            position = {"0q": 0, "+1q": 1, "0y": 2, "+1y": 3}.get(row_key)
            if position is not None and len(frame) > position:
                return frame.iloc[position]
        except Exception:  # noqa: BLE001
            return None
        return None

    def _row_value(self, row: Any, keys: tuple[str, ...]) -> Optional[float]:
        if row is None:
            return None
        return self._value_from(row, keys)

    def _value_from(self, data: Any, keys: tuple[str, ...]) -> Optional[float]:
        if data is None:
            return None
        getter = data.get if hasattr(data, "get") else None
        normalized = {}
        if isinstance(data, dict):
            normalized = {self._normalize_key(k): v for k, v in data.items()}
        elif hasattr(data, "index"):
            normalized = {self._normalize_key(k): data.get(k) for k in data.index}
        for key in keys:
            value = getter(key) if getter else None
            if value is None:
                value = normalized.get(self._normalize_key(key))
            parsed = self._to_float(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:  # noqa: BLE001
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_key(key: Any) -> str:
        return str(key).replace("_", "").replace(" ", "").lower()

    @staticmethod
    def _as_records(data: Any) -> list[dict]:
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [row for row in data["data"] if isinstance(row, dict)]
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
        return []

    @staticmethod
    def _first_record(data: Any) -> dict:
        records = EstimatesIngestor._as_records(data)
        return records[0] if records else {}

    def _provider_year_period(self, record: dict, offset: int) -> str:
        raw = str(record.get("period") or record.get("date") or "")
        if len(raw) >= 4 and raw[:4].isdigit():
            return f"FY{raw[:4]}E"
        return self._year_period(offset)

    @staticmethod
    def _fact(metric: str, value: Any, unit: str, period: str) -> dict:
        return {
            "metric": metric,
            "value": value,
            "unit": unit,
            "period": period,
            "period_type": "estimate",
        }

    @staticmethod
    def _quarter_period(offset: int) -> str:
        today = date.today()
        quarter = ((today.month - 1) // 3) + 1
        zero_based = quarter - 1 + offset
        year = today.year + zero_based // 4
        quarter = (zero_based % 4) + 1
        return f"{year}-Q{quarter}E"

    @staticmethod
    def _year_period(offset: int) -> str:
        return f"FY{date.today().year + offset}E"

    @staticmethod
    def _target_period() -> str:
        today = date.today()
        zero_based = today.month - 1 + 12
        year = today.year + zero_based // 12
        month = (zero_based % 12) + 1
        return f"{year}-{month:02d}E"

    def close(self) -> None:
        """Close HTTP resources."""
        self.session.close()

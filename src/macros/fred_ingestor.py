"""
src/macros/fred_ingestor.py
FRED economic data ingestion — fetches macro indicators and
stores them in the hybrid RAG store as structured facts.

Usage:
    ingestor = FREDIngestor()
    ingestor.fetch_all_indicators()
    ingestor.fetch_indicator("GDP")
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import yaml

from src.storage.store import Store
from src.utils.resilience import retry_with_backoff

logger = logging.getLogger(__name__)


class FREDIngestor:
    """
    Fetches macro-economic indicators from FRED (Federal Reserve
    Economic Data) and stores them as structured facts.

    Each indicator is stored with:
      ticker = "MACRO" (special ticker for macro data)
      metric = indicator_id (e.g., "GDP", "FEDFUNDS")
      value = latest observation value
      period = observation date
      source_type = "fred"
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs/fred.yaml"

    def __init__(
        self,
        store: Optional[Store] = None,
        config_path: Optional[Path] = None,
        api_key: Optional[str] = None,
    ):
        from src.utils.env import load_env
        load_env()  # ensure .env credentials are available in os.environ

        self.store = store or Store()
        self.config = self._load_config(config_path)
        self._client = None  # Lazy init

        cfg_key = self.config.get("api_key", "")
        if isinstance(cfg_key, str) and cfg_key.startswith("${") and cfg_key.endswith("}"):
            cfg_key = os.environ.get(cfg_key[2:-1], "")
        self.api_key = api_key or cfg_key or os.environ.get("FRED_API_KEY", "")

    # ── Lazy Client ────────────────────────────────────

    @property
    def client(self):
        """Lazy-init FRED API client."""
        if self._client is None:
            from fredapi import Fred
            self._client = Fred(api_key=self.api_key)
        return self._client

    # ── Config Loading ─────────────────────────────────

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load FRED configuration from YAML."""
        path = config_path or self.DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {
            "request_delay": 0.25,
            "max_retries": 3,
            "indicators": {
                "GDP": "Gross Domestic Product",
                "FEDFUNDS": "Federal Funds Rate",
                "CPIAUCSL": "CPI (All Urban Consumers)",
                "UNRATE": "Unemployment Rate",
                "DGS10": "10-Year Treasury Rate",
                "T10Y2Y": "10Y-2Y Spread",
            },
        }

    # ── Public API ─────────────────────────────────────

    def fetch_indicator(
        self,
        series_id: str,
        limit: int = 5,
        *,
        store_history: bool = False,
    ) -> Optional[float]:
        """
        Fetch the latest observation for a single FRED indicator.

        Args:
            series_id: FRED series ID (e.g., "GDP", "FEDFUNDS")
            limit: Max observations to retrieve

        Returns:
            The latest value, or None on failure
        """
        label = self.config.get("indicators", {}).get(series_id, series_id)

        @retry_with_backoff(
            max_attempts=self.config.get("max_retries", 3),
            base_delay=self.config.get("request_delay", 0.25),
            retryable_exceptions=(ConnectionError, TimeoutError, IOError),
        )
        def _get_series():
            # sort_order="desc" makes FRED return the *most recent* observations
            # (the API's default `limit` returns the oldest observations).
            return self.client.get_series(
                series_id, limit=limit, sort_order="desc",
            )

        try:
            series = _get_series()

            if series is None or series.empty:
                logger.warning("No data returned for FRED series %s", series_id)
                return None

            # fredapi returns the Series indexed (and sorted) by date ascending,
            # so the last row is always the most recent observation regardless
            # of the API sort order.
            series = series.sort_index()
            latest = series.iloc[-1]
            latest_date = series.index[-1]

            # Store the current observation for the normal path. Bootstrap can
            # explicitly retain the bounded history returned by the same call.
            rows = series if store_history else [(latest_date, latest)]
            for observed_date, observed_value in rows:
                period = (
                    observed_date.date().isoformat()
                    if hasattr(observed_date, "date")
                    else str(observed_date)
                )
                self.store.save_fundamental(
                    ticker="MACRO",
                    metric=series_id,
                    value=float(observed_value),
                    unit=self._guess_unit(series_id),
                    period=period,
                    period_type="daily",
                    source_type="fred",
                    source_url=f"https://fred.stlouisfed.org/series/{series_id}",
                )

            logger.info(
                "FRED %s (%s): %.4f on %s",
                series_id, label, float(latest), latest_date.date(),
            )

            self._respect_rate_limit()
            return float(latest)

        except Exception as e:
            logger.error("Failed to fetch FRED series %s: %s", series_id, e)
            return None

    def fetch_all_indicators(
        self,
        limit: int = 5,
        *,
        store_history: bool = False,
    ) -> dict[str, Optional[float]]:
        """
        Fetch all configured indicators.

        Returns:
            {series_id: latest_value_or_None}
        """
        results = {}
        indicators = self.config.get("indicators", {})

        for series_id in indicators:
            value = self.fetch_indicator(
                series_id,
                limit=max(int(limit), 1),
                store_history=store_history,
            )
            results[series_id] = value

        return results

    def fetch_by_category(self, category: str) -> dict[str, Optional[float]]:
        """
        Fetch indicators in a specific category.

        Categories: gdp, inflation, interest_rates, employment,
                    housing, consumer, manufacturing
        """
        category_map = {
            "gdp": ["GDP", "GDPC1"],
            "inflation": ["CPIAUCSL", "PCEPILFE", "T10YIE"],
            "interest_rates": ["FEDFUNDS", "DFF", "DGS1", "DGS10", "DGS2", "T10Y2Y"],
            "employment": ["UNRATE", "PAYEMS", "IC4WSA"],
            "housing": ["HOUST", "PERMIT"],
            "consumer": ["UMCSENT", "DSPIC96"],
            "manufacturing": ["INDPRO", "TCU"],
        }

        series_ids = category_map.get(category.lower(), [])
        if not series_ids:
            logger.warning("Unknown category: %s", category)
            return {}

        results = {}
        for series_id in series_ids:
            if series_id in self.config.get("indicators", {}):
                value = self.fetch_indicator(series_id)
                results[series_id] = value

        return results

    # ── Quick Macros Summary ────────────────────────────

    def get_macro_snapshot(self) -> dict:
        """
        Get a quick snapshot of key macro indicators.

        Returns structured dict for prompt injection:
            {
                "gdp": 29.5,
                "inflation_cpi": 3.2,
                "fed_rate": 4.5,
                "unemployment": 3.9,
                "ten_year_treasury": 4.2,
            }
        """
        indicators = self.fetch_all_indicators()
        return {
            "gdp": indicators.get("GDP"),
            "inflation_cpi": indicators.get("CPIAUCSL"),
            "fed_rate": indicators.get("FEDFUNDS"),
            "unemployment": indicators.get("UNRATE"),
            "ten_year_treasury": indicators.get("DGS10"),
            "ten_two_spread": indicators.get("T10Y2Y"),
        }

    # ── Helpers ────────────────────────────────────────

    def _guess_unit(self, series_id: str) -> str:
        """Guess the unit for a FRED series ID."""
        rate_series = {"FEDFUNDS", "DFF", "DGS1", "DGS10", "DGS2",
                       "T10Y2Y", "UNRATE", "T10YIE"}
        if series_id in rate_series:
            return "percent"
        if series_id in {"PAYEMS", "HOUST", "PERMIT", "IC4WSA"}:
            return "thousands"
        if series_id in {"CPIAUCSL", "GDPC1", "DSPIC96", "INDPRO", "TCU"}:
            return "index"
        if series_id in {"UMCSENT"}:
            return "index_points"
        return "units"

    def _respect_rate_limit(self):
        delay = self.config.get("request_delay", 0.25)
        if delay > 0:
            time.sleep(delay)

    def health_check(self) -> bool:
        """Quick check that FRED API is reachable."""
        try:
            self.client.get_series("GDP", limit=1)
            return True
        except Exception:
            return False

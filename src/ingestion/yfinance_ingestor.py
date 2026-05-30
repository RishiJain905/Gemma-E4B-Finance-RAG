"""
src/ingestion/yfinance_ingestor.py
Yahoo Finance ingestion module.

Fetches fundamentals, news, and summaries for tracked tickers
and stores them via the unified Store abstraction.

Usage:
    ingestor = YFinanceIngestor()
    ingestor.ingest_all()           # Ingest everything
    ingestor.ingest_ticker("NVDA")  # Ingest a single ticker
    ingestor.ingest_fundamentals()  # Fundamentals only
"""

import logging
from pathlib import Path
from typing import Optional, Any
import yfinance as yf
import yaml

from src.storage.store import Store

logger = logging.getLogger(__name__)


class YFinanceIngestor:
    """Ingests Yahoo Finance data into the unified Store."""

    # ── Config Paths ──────────────────────────────────
    DEFAULT_WATCHLIST_PATH = Path(__file__).parent.parent.parent / "configs/watchlist.yaml"

    def __init__(
        self,
        store: Optional[Store] = None,
        watchlist_path: Optional[Path] = None,
    ):
        self.store = store or Store()
        self.watchlist_path = watchlist_path or self.DEFAULT_WATCHLIST_PATH
        self.watchlist = self._load_watchlist()

    # ── Config Loading ────────────────────────────────

    def _load_watchlist(self) -> dict:
        """Load ticker watchlist from YAML config."""
        if not self.watchlist_path.exists():
            logger.warning("Watchlist not found at %s, using defaults", self.watchlist_path)
            return {
                "core": ["NVDA", "AMD", "AAPL", "MSFT", "META", "CRWD"],
                "extended": [],
                "macro_tickers": [],
                "schedule": {"fundamentals": 24, "news": 6, "macro": 24},
            }
        with open(self.watchlist_path) as f:
            return yaml.safe_load(f)

    @property
    def all_tickers(self) -> list[str]:
        """All tickers (core + extended + macro) as a flat list."""
        tickers = []
        tickers.extend(self.watchlist.get("core", []))
        tickers.extend(self.watchlist.get("extended", []))
        tickers.extend(self.watchlist.get("macro_tickers", []))
        return tickers

    @property
    def core_tickers(self) -> list[str]:
        """Core tickers only (get full ingestion)."""
        return self.watchlist.get("core", [])

    # ── Ticker Helpers ────────────────────────────────

    def _fetch_ticker(self, ticker: str) -> Optional[Any]:
        """Fetch a yfinance Ticker object with error handling."""
        try:
            t = yf.Ticker(ticker)
            # Quick validation — fetch info to confirm ticker exists
            info = t.info
            if not info or info.get("regularMarketPrice") is None:
                logger.warning("Ticker %s returned no price data, skipping", ticker)
                return None
            return t
        except Exception as e:
            logger.error("Failed to fetch ticker %s: %s", ticker, e)
            return None

    # ── Fundamentals Ingestion ────────────────────────

    FUNDAMENTAL_METRICS = [
        # (metric_name, info_key, unit, period_type)
        ("market_cap",        "marketCap",         "usd",      "point_in_time"),
        ("pe_ratio_ttm",      "trailingPE",        "ratio",    "ttm"),
        ("forward_pe",        "forwardPE",         "ratio",    "forward"),
        ("eps_ttm",           "trailingEps",       "usd",      "ttm"),
        ("dividend_yield",    "dividendYield",     "percent",  "ttm"),
        ("price_to_book",     "priceToBook",       "ratio",    "ttm"),
        ("debt_to_equity",    "debtToEquity",      "ratio",    "ttm"),
        ("revenue_ttm",       "totalRevenue",      "usd",      "ttm"),
        ("gross_margin_ttm",  "grossMargins",      "percent",  "ttm"),
        ("operating_margin",  "operatingMargins",  "percent",  "ttm"),
        ("profit_margin",     "profitMargins",     "percent",  "ttm"),
        ("revenue_growth",    "revenueGrowth",     "percent",  "yoy"),
        ("earnings_growth",   "earningsGrowth",    "percent",  "yoy"),
        ("return_on_equity",  "returnOnEquity",    "percent",  "ttm"),
        ("free_cash_flow",    "freeCashflow",      "usd",      "ttm"),
        ("operating_cf",      "operatingCashflow",  "usd",     "ttm"),
        ("current_ratio",     "currentRatio",      "ratio",    "ttm"),
        ("quick_ratio",       "quickRatio",        "ratio",    "ttm"),
    ]

    # ── Normalization Helpers ───────────────────────────

    def _normalize_value(self, raw_value):
        """Normalize a raw Yahoo Finance value for storage."""
        if raw_value is None:
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None

    def _current_period_label(self) -> str:
        """Generate a period label like '2026-Q1' for the current quarter."""
        from datetime import datetime
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        return f"{now.year}-Q{quarter}"

    # ── Ingestion Pipeline Methods (stubs) ────────────

    def ingest_all(self):
        """Run all ingestion steps for all tickers."""
        logger.info("Starting full Yahoo Finance ingestion...")
        self.ingest_fundamentals()
        self.ingest_news()
        self.ingest_macro()
        logger.info("Full ingestion complete.")

    def ingest_fundamentals(self):
        """Ingest fundamentals for all core tickers."""
        logger.info("Ingesting fundamentals for %d core tickers...", len(self.core_tickers))
        success_count = 0
        skip_count = 0

        for ticker in self.core_tickers:
            # Cache check — skip if fundamentals are still fresh
            if self._fundamentals_fresh(ticker):
                logger.debug("Ticker %s fundamentals are fresh, skipping", ticker)
                skip_count += 1
                continue

            t = self._fetch_ticker(ticker)
            if t is None:
                continue

            self._ingest_ticker_fundamentals(ticker, t)
            success_count += 1

        logger.info(
            "Fundamentals ingestion complete: %d ingested, %d skipped (fresh)",
            success_count, skip_count
        )

    def _fundamentals_fresh(self, ticker: str) -> bool:
        """Check if fundamentals cache is still fresh for this ticker."""
        ttl_hours = self.watchlist.get("schedule", {}).get("fundamentals", 24)
        status = self.store.get_cache_status(ticker, "yfinance_fundamentals")
        if status and status.get("status") == "fresh":
            # Verify it isn't expired by checking the stored TTL
            from datetime import datetime, timezone
            last_updated = status.get("last_updated")
            if last_updated:
                try:
                    if isinstance(last_updated, str):
                        # Parse SQLite datetime string
                        try:
                            from dateutil import parser
                            updated_dt = parser.parse(last_updated)
                        except Exception:
                            updated_dt = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
                        if updated_dt.tzinfo is None:
                            updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                    else:
                        updated_dt = last_updated
                    age_hours = (datetime.now(timezone.utc) - updated_dt).total_seconds() / 3600
                    return age_hours < ttl_hours
                except Exception:
                    return False
        return False

    def ingest_news(self):
        """Ingest recent news and summaries for core tickers."""
        # Will be implemented in 1.3.3
        raise NotImplementedError("Implement in 1.3.3")

    def ingest_macro(self):
        """Ingest macro indicators."""
        # Will be implemented in 1.3.4
        raise NotImplementedError("Implement in 1.3.4")

    def ingest_ticker(self, ticker: str):
        """Ingest everything for a single ticker."""
        t = self._fetch_ticker(ticker)
        if t is None:
            return
        logger.info("Ingesting single ticker: %s", ticker)
        self._ingest_ticker_fundamentals(ticker, t)
        self._ingest_ticker_news(ticker, t)

    def _ingest_ticker_fundamentals(self, ticker: str, t):
        """Ingest fundamentals for a single ticker."""
        info = t.info
        period_label = self._current_period_label()
        saved = 0

        for metric_name, info_key, unit, period_type in self.FUNDAMENTAL_METRICS:
            raw_value = info.get(info_key)
            if raw_value is None:
                logger.debug("Ticker %s has no %s (%s), skipping", ticker, metric_name, info_key)
                continue

            # Normalize the value
            value = self._normalize_value(raw_value)
            if value is None:
                continue

            self.store.save_fundamental(
                ticker=ticker,
                metric=metric_name,
                value=value,
                unit=unit,
                period=period_label,
                period_type=period_type,
                source_type="yfinance",
            )
            saved += 1

        # Mark cache as fresh
        ttl_hours = self.watchlist.get("schedule", {}).get("fundamentals", 24)
        self.store.mark_cache_fresh(ticker, "yfinance_fundamentals", ttl_hours)
        logger.info("Ticker %s: saved %d fundamentals, cache marked fresh (%dh)", ticker, saved, ttl_hours)

    def _ingest_ticker_news(self, ticker: str, t: Any):
        """Ingest news for one ticker (stub)."""
        raise NotImplementedError("Implement in 1.3.3")

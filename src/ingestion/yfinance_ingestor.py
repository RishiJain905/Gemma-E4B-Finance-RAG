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
        # Will be implemented in 1.3.2
        raise NotImplementedError("Implement in 1.3.2")

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

    def _ingest_ticker_fundamentals(self, ticker: str, t: Any):
        """Ingest fundamentals for one ticker (stub)."""
        raise NotImplementedError("Implement in 1.3.2")

    def _ingest_ticker_news(self, ticker: str, t: Any):
        """Ingest news for one ticker (stub)."""
        raise NotImplementedError("Implement in 1.3.3")

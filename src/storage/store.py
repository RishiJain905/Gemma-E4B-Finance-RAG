"""
src/storage/store.py
Minimal working Store wrapper over SQLiteStore.

This will be replaced by the full unified Store implementation from Phase 1.2.4.
Until then, it provides the public API surface that ingestion modules expect.
"""

from pathlib import Path
from typing import Optional

from .sqlite_store import SQLiteStore


class Store:
    """Unified storage abstraction (minimal viable implementation)."""

    def __init__(self, db_path: Optional[Path] = None):
        self.sqlite = SQLiteStore(db_path=db_path)

    # ── Structured Facts (SQLite) ──────────────────────

    def save_fundamental(self, ticker: str, metric: str, value: float,
                         unit: str = "usd", period: str = None,
                         period_type: str = "quarterly",
                         source_type: str = "yfinance",
                         source_url: str = None) -> int:
        """Save or update a single financial metric."""
        return self.sqlite.upsert_fundamental(
            ticker, metric, value, unit, period,
            period_type, source_type, source_url
        )

    def get_fundamental(self, ticker: str, metric: str,
                        period: str = None) -> Optional[dict]:
        """Get the latest value for a metric."""
        return self.sqlite.get_fundamental(ticker, metric, period)

    def get_fundamentals_batch(self, ticker: str,
                               metrics: list[str] = None) -> dict:
        """Get multiple metrics for a ticker at once."""
        return self.sqlite.get_fundamentals_batch(ticker, metrics)

    # ── Cache Management ──────────────────────────────

    def get_cache_status(self, ticker: str, source: str) -> Optional[dict]:
        return self.sqlite.get_cache_status(ticker, source)

    def mark_cache_fresh(self, ticker: str, source: str, ttl_hours: int = 24):
        self.sqlite.mark_cache_fresh(ticker, source, ttl_hours)

    def mark_cache_stale(self, ticker: str, source: str, error: str = None):
        self.sqlite.mark_cache_stale(ticker, source, error)

    def get_stale_entries(self, limit: int = 20) -> list[dict]:
        return self.sqlite.get_stale_cache_entries(limit)

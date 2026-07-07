"""
src/middleware/on_demand.py
Fetch cheap ticker data on demand when query freshness has no cache.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TRACKED_DYNAMIC_PATH = Path(__file__).resolve().parents[2] / "data" / "tracked_dynamic.json"


def read_dynamic_tracked() -> list[str]:
    """Read dynamically discovered tickers persisted by fetch-on-miss."""
    try:
        data = json.loads(TRACKED_DYNAMIC_PATH.read_text(encoding="utf-8"))
        tickers = data.get("tickers", [])
        if not isinstance(tickers, list):
            return []
        return [str(t).upper() for t in tickers if str(t).strip()]
    except Exception as exc:  # noqa: BLE001 - best-effort helper
        logger.debug("Failed to read dynamic tracked tickers: %s", exc)
        return []


def _persist_dynamic_tracked(ticker: str) -> None:
    """Append a ticker to data/tracked_dynamic.json without duplicating it."""
    try:
        normalized = ticker.upper().strip()
        tickers = read_dynamic_tracked()
        if normalized not in tickers:
            tickers.append(normalized)
        TRACKED_DYNAMIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRACKED_DYNAMIC_PATH.write_text(
            json.dumps({"tickers": tickers}, indent=2) + "\n",
            encoding="utf-8",
        )
        # Scheduler pickup is deliberately deferred to future work.
    except Exception as exc:  # noqa: BLE001 - persistence must not break queries
        logger.warning("Failed to persist dynamic tracked ticker %s: %s", ticker, exc)


def fetch_ticker_on_miss(store, ticker: str) -> dict:
    """Fetch yfinance fundamentals/news for a valid untracked ticker."""
    normalized = ticker.upper().strip()
    try:
        from src.ingestion.yfinance_ingestor import YFinanceIngestor

        ing = YFinanceIngestor(store=store)
        t = ing._fetch_ticker(normalized)
        if t is None:
            return {
                "fetched": False,
                "ticker": normalized,
                "sources": [],
                "error": "invalid_ticker",
            }

        sources: list[str] = []
        ing._ingest_ticker_fundamentals(normalized, t)
        sources.append("yfinance_fundamentals")

        news_error = None
        try:
            ing._ingest_ticker_news(normalized, t)
            sources.append("yfinance_news")
        except Exception as exc:  # noqa: BLE001 - partial data beats none
            news_error = str(exc)
            logger.warning("Fetch-on-miss news ingestion failed for %s: %s", normalized, exc)

        _persist_dynamic_tracked(normalized)
        return {
            "fetched": True,
            "ticker": normalized,
            "sources": sources,
            "error": news_error,
        }
    except Exception as exc:  # noqa: BLE001 - never raise into query handling
        logger.warning("Fetch-on-miss failed for %s: %s", normalized, exc)
        return {
            "fetched": False,
            "ticker": normalized,
            "sources": [],
            "error": str(exc),
        }

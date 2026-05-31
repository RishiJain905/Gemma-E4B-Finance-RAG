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
        """Ingest recent news for all core tickers."""
        logger.info("Ingesting news for %d core tickers...", len(self.core_tickers))
        success_count = 0
        skip_count = 0

        for ticker in self.core_tickers:
            if self._news_fresh(ticker):
                logger.debug("Ticker %s news is fresh, skipping", ticker)
                skip_count += 1
                continue

            status = self.store.get_cache_status(ticker, "yfinance_fundamentals")
            if not status or status.get("status") != "fresh":
                logger.info(
                    "Ticker %s fundamentals not yet ingested, doing fundamentals first",
                    ticker,
                )
                t = self._fetch_ticker(ticker)
                if t:
                    self._ingest_ticker_fundamentals(ticker, t)

            t = self._fetch_ticker(ticker)
            if t is None:
                continue

            self._ingest_ticker_news(ticker, t)
            success_count += 1

        logger.info(
            "News ingestion complete: %d ingested, %d skipped (fresh)",
            success_count,
            skip_count,
        )

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

    # ── News Ingestion ────────────────────────────────

    def _news_fresh(self, ticker: str) -> bool:
        """Check if news cache is still fresh for this ticker."""
        ttl_hours = self.watchlist.get("schedule", {}).get("news", 6)
        status = self.store.get_cache_status(ticker, "yfinance_news")
        if status and status.get("status") == "fresh":
            from datetime import datetime, timezone
            last_updated = status.get("last_updated")
            if last_updated:
                try:
                    if isinstance(last_updated, str):
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

    def _extract_news_fields(self, article: dict) -> dict | None:
        """Normalize nested (yfinance>=0.2.50) or flat news article schemas."""
        if not isinstance(article, dict):
            return None

        content = article.get("content")
        if isinstance(content, dict):
            return {
                "id": content.get("id") or article.get("id"),
                "title": content.get("title", ""),
                "summary": content.get("summary") or content.get("description") or "",
                "publisher": (content.get("provider") or {}).get("displayName", ""),
                "link": (
                    (content.get("canonicalUrl") or {}).get("url")
                    or (content.get("clickThroughUrl") or {}).get("url")
                    or ""
                ),
                "pub_raw": content.get("pubDate") or content.get("displayTime"),
                "type": content.get("contentType", "STORY"),
            }

        return {
            "id": article.get("uuid") or article.get("id"),
            "title": article.get("title", ""),
            "summary": article.get("summary", ""),
            "publisher": article.get("publisher", ""),
            "link": article.get("link", ""),
            "pub_raw": article.get("providerPublishTime"),
            "type": article.get("type", "news"),
        }

    def _make_news_doc_id(self, ticker: str, article: dict) -> str:
        """Create a unique document ID for a news article."""
        fields = self._extract_news_fields(article)
        article_id = fields.get("id") if fields else None
        if article_id:
            return f"news/{ticker}/{article_id}"

        link = fields.get("link", "") if fields else article.get("link", "")
        if link:
            url_id = link.rstrip("/").rsplit("/", 1)[-1]
        else:
            title = fields.get("title", "") if fields else article.get("title", "")
            url_id = str(hash(title))
        return f"news/{ticker}/{url_id}"

    def _format_news_article(self, article: dict) -> str | None:
        """Format a news article into a searchable document text."""
        fields = self._extract_news_fields(article)
        if not fields:
            return None

        title = fields.get("title", "")
        summary = fields.get("summary", "")
        publisher = fields.get("publisher", "")

        if not title and not summary:
            return None

        parts = []
        if title:
            parts.append(f"# {title}")
        if summary:
            parts.append(summary)
        if publisher:
            parts.append(f"—— Source: {publisher}")

        return "\n\n".join(parts)

    def _format_news_date(self, pub_raw) -> str:
        """Convert pubDate (ISO8601) or Unix timestamp to YYYY-MM-DD."""
        if not pub_raw:
            return ""

        from datetime import datetime, timezone

        if isinstance(pub_raw, (int, float)):
            return datetime.fromtimestamp(pub_raw, tz=timezone.utc).strftime("%Y-%m-%d")

        if isinstance(pub_raw, str):
            try:
                from dateutil import parser
                dt = parser.parse(pub_raw)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        return ""

    def _ingest_ticker_news(self, ticker: str, t: Any):
        """Ingest recent news for a single ticker and store as ChromaDB docs."""
        try:
            news_raw = t.news
        except Exception as e:
            logger.warning("Ticker %s news fetch failed: %s", ticker, e)
            return

        if not news_raw:
            logger.info("Ticker %s has no news articles to ingest", ticker)
            return

        saved = 0
        skipped = 0
        for article in news_raw:
            doc_id = self._make_news_doc_id(ticker, article)
            text = self._format_news_article(article)
            if text is None:
                skipped += 1
                continue

            existing = self.store.chroma.get_document(doc_id)
            if existing:
                skipped += 1
                continue

            fields = self._extract_news_fields(article)
            date_str = self._format_news_date(fields.get("pub_raw") if fields else None)

            self.store.save_document(
                document_id=doc_id,
                text=text,
                ticker=ticker,
                source="yfinance_news",
                date=date_str,
                metadata={
                    "title": fields.get("title", "") if fields else "",
                    "publisher": fields.get("publisher", "") if fields else "",
                    "link": fields.get("link", "") if fields else "",
                    "type": fields.get("type", "news") if fields else "news",
                },
            )
            saved += 1

        ttl_hours = self.watchlist.get("schedule", {}).get("news", 6)
        self.store.mark_cache_fresh(ticker, "yfinance_news", ttl_hours)
        logger.info(
            "Ticker %s: %d news articles saved, %d skipped, cache marked fresh (%dh)",
            ticker,
            saved,
            skipped,
            ttl_hours,
        )

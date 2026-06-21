"""
src/macros/ir_ingestor.py
Company IR pages ingestion — fetches press releases, investor presentations,
and earnings materials from company investor relations pages.

Sources:
  - Company IR RSS feeds (preferred — structured, reliable)
  - Company IR page scraping (fallback — HTML parsing)

Usage:
    ingestor = IRIngestor()
    ingestor.fetch_for_ticker("NVDA")
    ingestor.fetch_all_core()
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup

from src.storage.store import Store

logger = logging.getLogger(__name__)


class IRIngestor:
    """
    Fetches forward-looking company communications from investor relations
    pages — press releases, investor presentations, earnings materials, and
    corporate events — and stores them as ChromaDB documents.

    Each item is stored as a ChromaDB document with metadata:
      - ticker: the core ticker
      - source: "ir"
      - doc_type: press_release | presentation | earnings_material | event | other
      - date: publication date
      - url: original item URL

    RSS feeds are preferred (structured, reliable); HTML scraping of the IR
    page is used as a best-effort fallback when no usable feed is available.
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs/ir.yaml"

    # ── Ticker-to-IR Mapping ───────────────────────────

    TICKER_IR_MAP = {
        "NVDA": {
            "name": "NVIDIA",
            "ir_url": "https://investor.nvidia.com/",
            "rss_url": "https://investor.nvidia.com/rss/",
            "press_releases_url": "https://investor.nvidia.com/news-and-events/press-releases/",
        },
        "AMD": {
            "name": "Advanced Micro Devices",
            "ir_url": "https://ir.amd.com/",
            "rss_url": "https://ir.amd.com/rss/",
        },
        "AAPL": {
            "name": "Apple",
            "ir_url": "https://investor.apple.com/",
            "rss_url": "https://investor.apple.com/rss/",
        },
        "MSFT": {
            "name": "Microsoft",
            "ir_url": "https://www.microsoft.com/en-us/Investor/",
            "rss_url": "https://www.microsoft.com/en-us/Investor/RSS/",
        },
        "META": {
            "name": "Meta Platforms",
            "ir_url": "https://investor.fb.com/",
            "rss_url": "https://investor.fb.com/rss/",
        },
        "CRWD": {
            "name": "CrowdStrike",
            "ir_url": "https://ir.crowdstrike.com/",
            "rss_url": "https://ir.crowdstrike.com/rss/",
        },
    }

    # ── Classification Keywords ────────────────────────

    _PRESS_RELEASE_KEYWORDS = (
        "press release", "launch", "launches", "partnership", "partner",
        "acquisition", "acquires", "announces", "announcement", "introduces",
    )
    _PRESENTATION_KEYWORDS = (
        "presentation", "deck", "slides", "slide deck", "fireside",
    )
    _EARNINGS_KEYWORDS = (
        "earnings", "supplemental", "non-gaap", "results", "quarterly",
        "financial results", "q1", "q2", "q3", "q4",
    )
    _EVENT_KEYWORDS = (
        "analyst day", "conference", "shareholder meeting", "annual meeting",
        "investor day", "webcast", "event",
    )

    def __init__(
        self,
        store: Optional[Store] = None,
        config_path: Optional[Path] = None,
    ):
        self.store = store or Store()
        self.config = self._load_config(config_path)
        self.request_delay = float(self.config.get("request_delay", 3.0))
        self.timeout = int(self.config.get("timeout", 30))
        self.max_items = int(self.config.get("max_items_per_ticker", 20))

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.get(
                "user_agent",
                "TraceAlchemy Research (contact@tracealchemy.example.com)",
            ),
        })

    # ── Config Loading ─────────────────────────────────

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load IR configuration from YAML, with sensible defaults."""
        path = config_path or self.DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {
            "request_delay": 3.0,
            "max_retries": 3,
            "timeout": 30,
            "user_agent": "TraceAlchemy Research (contact@tracealchemy.example.com)",
            "doc_types": [
                "press_release",
                "presentation",
                "earnings_material",
                "event",
            ],
            "max_items_per_ticker": 20,
        }

    # ── Public API ─────────────────────────────────────

    def fetch_for_ticker(self, ticker: str) -> dict:
        """
        Fetch IR items for a single ticker and store them.

        Tries the RSS feed first, then falls back to scraping the IR page.
        Network and parsing errors are caught and reported in the result
        (never raised).

        Returns:
            {
                "ticker": "NVDA",
                "status": "success" | "unknown_ticker" | "no_items" | "error",
                "items_found": int,
                "items_stored": int,
                "errors": list[str],
            }
        """
        ticker = ticker.upper()
        info = self.TICKER_IR_MAP.get(ticker)
        if info is None:
            logger.warning("Unknown ticker for IR ingestion: %s", ticker)
            return {"ticker": ticker, "status": "unknown_ticker"}

        errors: list[str] = []
        items: list[dict] = []

        # Step 1: try the RSS feed (structured, reliable)
        rss_url = info.get("rss_url")
        if rss_url:
            try:
                items = self._parse_rss(rss_url)
            except Exception as e:
                logger.debug("RSS parse failed for %s (%s): %s", ticker, rss_url, e)
                errors.append(f"rss: {e}")

        # Step 2: fall back to scraping the IR page
        if not items:
            ir_url = info.get("ir_url")
            if ir_url:
                try:
                    items = self._scrape_ir_page(ir_url)
                except Exception as e:
                    logger.debug("IR scrape failed for %s (%s): %s", ticker, ir_url, e)
                    errors.append(f"scrape: {e}")

        # If both paths failed with errors and produced nothing, report error.
        if not items and errors:
            return {
                "ticker": ticker,
                "status": "error",
                "items_found": 0,
                "items_stored": 0,
                "errors": errors,
            }

        items = items[: self.max_items]

        # Step 3: store each parsed item
        stored = 0
        for item in items:
            try:
                self._store_item(ticker, item)
                stored += 1
            except Exception as e:
                logger.warning("Failed to store IR item for %s: %s", ticker, e)
                errors.append(f"store: {e}")

        if stored:
            try:
                self.store.mark_cache_fresh(ticker, "ir_pages", ttl_hours=24)
            except Exception as e:
                logger.debug("mark_cache_fresh failed for %s: %s", ticker, e)

        status = "success" if items else "no_items"
        return {
            "ticker": ticker,
            "status": status,
            "items_found": len(items),
            "items_stored": stored,
            "errors": errors,
        }

    def fetch_all_core(self) -> dict[str, dict]:
        """
        Fetch IR items for all core tickers that have an IR mapping.

        Respects the configured ``request_delay`` between tickers.

        Returns:
            {ticker: result_dict}
        """
        from src.ingestion.yfinance_ingestor import YFinanceIngestor

        ingestor = YFinanceIngestor(store=self.store)
        results: dict[str, dict] = {}

        for ticker in ingestor.core_tickers:
            if ticker in self.TICKER_IR_MAP:
                results[ticker] = self.fetch_for_ticker(ticker)
                if self.request_delay > 0:
                    time.sleep(self.request_delay)

        return results

    # ── RSS Parsing ────────────────────────────────────

    def _parse_rss(self, feed_url: str) -> list[dict]:
        """
        Fetch and parse an RSS/Atom feed via feedparser.

        Returns a list of dicts with keys:
            title, url, published_date, content, doc_type
        """
        parsed = feedparser.parse(feed_url)
        items: list[dict] = []

        for entry in parsed.get("entries", []):
            title = (entry.get("title") or "").strip()
            url = entry.get("link") or entry.get("id") or ""

            published_date = (
                entry.get("published")
                or entry.get("updated")
                or entry.get("pubDate")
                or ""
            )

            content = (
                entry.get("summary")
                or entry.get("description")
                or ""
            )
            if not content:
                content_list = entry.get("content") or []
                if content_list and isinstance(content_list, list):
                    content = content_list[0].get("value", "")

            if not title and not content:
                continue

            items.append({
                "title": title,
                "url": url,
                "published_date": published_date,
                "content": content,
                "doc_type": self._classify_document(title, content),
            })

        return items

    # ── HTML Scraping Fallback ─────────────────────────

    def _scrape_ir_page(self, ir_url: str) -> list[dict]:
        """
        Best-effort HTML fallback for IR pages without a usable RSS feed.

        Looks for press-release / news / event anchors and returns whatever
        is available. Returns [] when nothing relevant is found.
        """
        items: list[dict] = []

        resp = self.session.get(ir_url, timeout=self.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        seen_urls: set[str] = set()
        keyword_hints = (
            "press-release", "press_release", "news", "presentation",
            "earnings", "investor", "event",
        )

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            text = anchor.get_text(strip=True)
            if not href or not text:
                continue

            href_lower = href.lower()
            if not any(hint in href_lower for hint in keyword_hints):
                continue

            full_url = urljoin(ir_url, href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            items.append({
                "title": text,
                "url": full_url,
                "published_date": "",
                "content": text,
                "doc_type": self._classify_document(text, ""),
            })

            if len(items) >= self.max_items:
                break

        return items

    # ── Classification ─────────────────────────────────

    def _classify_document(self, title: str, content: str) -> str:
        """
        Classify a document type from title/content keywords.

        Returns one of: press_release, presentation, earnings_material,
        event, other.
        """
        text = f"{title} {content}".lower()

        if any(kw in text for kw in self._PRESENTATION_KEYWORDS):
            return "presentation"
        if any(kw in text for kw in self._EARNINGS_KEYWORDS):
            return "earnings_material"
        if any(kw in text for kw in self._EVENT_KEYWORDS):
            return "event"
        if any(kw in text for kw in self._PRESS_RELEASE_KEYWORDS):
            return "press_release"
        return "other"

    # ── Storage ────────────────────────────────────────

    def _store_item(self, ticker: str, item: dict) -> str:
        """Store a single parsed IR item as a ChromaDB document."""
        url = item.get("url", "")
        date = self._normalize_date(item.get("published_date", ""))
        digest = hashlib.sha1(
            (url or item.get("title", "")).encode("utf-8")
        ).hexdigest()[:12]
        doc_id = f"ir/{ticker}/{digest}"

        title = item.get("title", "")
        content = item.get("content", "")
        text = f"{title}\n\n{content}".strip() if content else title

        self.store.save_document(
            document_id=doc_id,
            text=text,
            ticker=ticker,
            source="ir",
            date=date,
            metadata={
                "ticker": ticker,
                "source": "ir",
                "doc_type": item.get("doc_type", "other"),
                "date": date,
                "url": url,
            },
        )
        return doc_id

    @staticmethod
    def _normalize_date(raw: str) -> str:
        """Normalize a feed date string to ISO (YYYY-MM-DD) when possible."""
        if not raw:
            return ""
        try:
            from dateutil import parser as date_parser

            return date_parser.parse(raw).date().isoformat()
        except Exception:
            return raw

    # ── Cleanup ────────────────────────────────────────

    def close(self):
        self.session.close()

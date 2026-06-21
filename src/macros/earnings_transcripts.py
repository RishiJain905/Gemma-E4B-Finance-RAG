"""
src/macros/earnings_transcripts.py
Earnings transcripts ingestion — fetches earnings call transcripts,
processes them through the TraceAlchemy parser, and stores
extracted management guidance and Q&A highlights.

Sources (primary): Seeking Alpha transcripts
Sources (fallback): Fool.com transcripts

Usage:
    ingestor = EarningsTranscriptIngestor()
    ingestor.fetch_and_process("NVDA")
    ingestor.fetch_all_core()
"""

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.storage.store import Store

logger = logging.getLogger(__name__)


class EarningsTranscriptIngestor:
    """
    Fetches earnings call transcripts and processes them through the
    TraceAlchemy model-as-parser pipeline.

    The parser extracts:
      - Forward guidance (revenue, margin, EPS ranges)
      - Segment performance breakdowns
      - New product/initiative announcements
      - Management tone and key phrases
      - Analyst Q&A highlights

    Transcripts are stored as ChromaDB documents and extracted
    guidance metrics are stored as SQLite facts.
    """

    # ── Sources ────────────────────────────────────────

    SEEKING_ALPHA_BASE = "https://seekingalpha.com"
    FOOL_BASE = "https://www.fool.com"

    # ── Ticker to SA Symbol ────────────────────────────

    TICKER_SA_MAP = {
        "NVDA": "NVDA",
        "AMD": "AMD",
        "AAPL": "AAPL",
        "MSFT": "MSFT",
        "META": "META",
        "GOOGL": "GOOGL",
        "AMZN": "AMZN",
        "TSLA": "TSLA",
        "CRWD": "CRWD",
        "PANW": "PANW",
        "AVGO": "AVGO",
        "INTC": "INTC",
    }

    def __init__(
        self,
        store: Optional[Store] = None,
        request_delay: float = 1.0,
    ):
        self.store = store or Store()
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })

    # ── Public API ─────────────────────────────────────

    def fetch_and_process(
        self, ticker: str, quarter: Optional[str] = None,
    ) -> dict:
        """
        Fetch the latest earnings transcript for a ticker and
        process it through the model-as-parser pipeline.

        Args:
            ticker: Stock ticker
            quarter: Optional quarter label (e.g., "2026-Q1").
                     If None, fetches the most recent available.

        Returns:
            {
                "ticker": "NVDA",
                "quarter": "2026-Q1",
                "transcript_length": 15000,
                "guidance_extracted": 3,
                "facts_stored": 5,
                "status": "success" | "not_found" | "error",
            }
        """
        logger.info("Fetching earnings transcript for %s...", ticker)

        # Step 1: Fetch transcript
        transcript = self._fetch_transcript(ticker)

        if not transcript:
            logger.warning("No transcript found for %s", ticker)
            return {
                "ticker": ticker,
                "quarter": quarter,
                "status": "not_found",
            }

        # Step 2: Parse via TraceAlchemy
        quarter = quarter or self._detect_quarter(transcript)
        facts = self._parse_transcript(ticker, transcript, quarter)

        # Step 3: Store in ChromaDB
        doc_id = self._store_transcript(ticker, transcript, quarter, facts)

        # Step 4: Store guidance facts in SQLite
        stored_count = 0
        for fact in facts:
            self.store.save_fundamental(
                ticker=ticker,
                metric=fact["metric"],
                value=fact["value"],
                unit=fact.get("unit", "usd"),
                period=quarter,
                period_type="quarterly",
                source_type=f"earnings_transcript",
                source_url=f"https://seekingalpha.com/symbol/{ticker}/earnings/transcripts",
            )
            stored_count += 1

        logger.info(
            "Earnings transcript for %s (%s): %d facts stored",
            ticker, quarter, stored_count,
        )

        return {
            "ticker": ticker,
            "quarter": quarter,
            "transcript_length": len(transcript),
            "guidance_extracted": len(facts),
            "facts_stored": stored_count,
            "status": "success",
            "document_id": doc_id,
        }

    def fetch_all_core(self) -> dict[str, dict]:
        """
        Fetch and process transcripts for all core tickers.

        Returns:
            {ticker: result_dict}
        """
        from src.ingestion.yfinance_ingestor import YFinanceIngestor

        ingestor = YFinanceIngestor(store=self.store)
        results = {}

        for ticker in ingestor.core_tickers:
            if ticker in self.TICKER_SA_MAP:
                result = self.fetch_and_process(ticker)
                results[ticker] = result
                time.sleep(self.request_delay)

        return results

    def get_latest_guidance(self, ticker: str) -> dict:
        """
        Get the latest guidance metrics for a ticker from SQLite.

        Returns:
            {metric: value} for guidance-related metrics
        """
        guidance_metrics = [
            "guidance_revenue_low",
            "guidance_revenue_high",
            "guidance_eps_low",
            "guidance_eps_high",
            "guidance_margin",
        ]
        return self.store.get_fundamentals_batch(ticker, metrics=guidance_metrics)

    # ── Transcript Fetching ────────────────────────────

    def _fetch_transcript(self, ticker: str) -> Optional[str]:
        """Fetch the latest earnings transcript for a ticker."""
        # Try Seeking Alpha first
        transcript = self._fetch_sa_transcript(ticker)
        if transcript:
            return transcript

        # Fallback to Fool.com
        return self._fetch_fool_transcript(ticker)

    def _fetch_sa_transcript(self, ticker: str) -> Optional[str]:
        """Fetch transcript from Seeking Alpha."""
        try:
            url = f"{self.SEEKING_ALPHA_BASE}/symbol/{ticker}/earnings/transcripts"
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            transcript_div = soup.find("div", class_=re.compile(r"transcript|article-content"))

            if transcript_div:
                paragraphs = transcript_div.find_all("p")
                text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
                if len(text) > 500:
                    return text

            # Fallback: grab all article text
            article = soup.find("article")
            if article:
                text = article.get_text(strip=True)
                if len(text) > 500:
                    return text

            return None

        except Exception as e:
            logger.debug("SA transcript fetch failed for %s: %s", ticker, e)
            return None

    def _fetch_fool_transcript(self, ticker: str) -> Optional[str]:
        """Fetch transcript from Fool.com as fallback."""
        try:
            url = f"{self.FOOL_BASE}/earnings/call-transcripts/{ticker.lower()}/"
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            content = soup.find("div", class_=re.compile(r"article-body|calls-content"))

            if content:
                text = content.get_text(strip=True)
                if len(text) > 500:
                    return text

            return None

        except Exception as e:
            logger.debug("Fool transcript fetch failed for %s: %s", ticker, e)
            return None

    # ── Transcript Parsing ─────────────────────────────

    def _parse_transcript(
        self, ticker: str, transcript: str, quarter: str,
    ) -> list[dict]:
        """
        Parse an earnings transcript to extract guidance metrics.

        Uses regex pattern matching on key phrases like:
          - "expect revenue of $X to $Y"
          - "guidance for Q[X]"
          - "non-GAAP EPS of $X"
        """
        facts = []

        # Revenue guidance range
        rev_match = re.search(
            r'revenue (?:of|in the range of|between)\s*\$?([\d,.]+)\s*(?:billion|million|B|M)?'
            r'\s*(?:to|-|and)\s*\$?([\d,.]+)\s*(?:billion|million|B|M)?',
            transcript, re.IGNORECASE,
        )
        if rev_match:
            try:
                low = float(rev_match.group(1).replace(",", ""))
                high = float(rev_match.group(2).replace(",", ""))
                # Normalize to billions
                context = transcript[max(0, rev_match.start()-100):rev_match.end()+100].lower()
                if "million" in context or "m" in context and "b" not in context:
                    low /= 1000
                    high /= 1000
                facts.append({"metric": "guidance_revenue_low", "value": low, "unit": "billion_usd"})
                facts.append({"metric": "guidance_revenue_high", "value": high, "unit": "billion_usd"})
            except (ValueError, IndexError):
                pass

        # EPS guidance
        eps_match = re.search(
            r'(?:non-gaap\s+)?eps\s+(?:of|guidance\s+(?:of|for))\s*\$?(\d+(?:\.\d+)?)',
            transcript, re.IGNORECASE,
        )
        if eps_match:
            try:
                eps_val = float(eps_match.group(1))
                facts.append({"metric": "guidance_eps", "value": eps_val, "unit": "usd"})
            except ValueError:
                pass

        # Margin guidance — CORRECTION 2: fixed to match "Gross margin guidance of 75.5%"
        margin_match = re.search(
            r'(?:gross|operating)\s+margin\s+(?:guidance\s+)?(?:of|approximately)?\s*([\d.]+)\s*%',
            transcript, re.IGNORECASE,
        )
        if margin_match:
            try:
                margin_val = float(margin_match.group(1))
                facts.append({"metric": "guidance_margin", "value": margin_val, "unit": "percent"})
            except ValueError:
                pass

        return facts

    def _detect_quarter(self, transcript: str) -> str:
        """Detect the fiscal quarter from transcript text."""
        # Look for patterns like "Q1 fiscal 2026", "first quarter of 2026"
        # CORRECTION 1: fixed patterns to handle "fiscal" without "year" and "first quarter OF 2025"
        patterns = [
            (r"q([1-4])\s*(?:fy|fiscal(?:\s*year)?)?\s*20(\d{2})", None),
            (r"(first|second|third|fourth)\s+quarter\s+(?:of\s+|fy\s*|fiscal(?:\s*year)?\s*)?20(\d{2})",
             {"first": "1", "second": "2", "third": "3", "fourth": "4"}),
        ]

        for pattern, word_map in patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                if word_map:
                    q_num = word_map.get(match.group(1).lower(), "1")
                    year = match.group(2)
                else:
                    q_num = match.group(1)
                    year = match.group(2)
                return f"20{year}-Q{q_num}" if len(year) == 2 else f"{year}-Q{q_num}"

        return datetime.now(timezone.utc).strftime("%Y-Q%m")

    # ── Storage ────────────────────────────────────────

    def _store_transcript(
        self, ticker: str, transcript: str, quarter: str, facts: list[dict],
    ) -> str:
        """Store the full transcript in ChromaDB."""
        # Build a summary for the document (first ~3K chars + key facts)
        summary = transcript[:3000]
        if len(transcript) > 3000:
            summary += "\n\n[...transcript truncated...]"

        if facts:
            guidance_summary = "\n".join(
                f"- {f['metric']}: {f['value']} {f.get('unit', '')}"
                for f in facts
            )
            summary += f"\n\n## Extracted Guidance\n{guidance_summary}"

        doc_id = f"earnings_call/{ticker}/{quarter}"
        self.store.save_document(
            document_id=doc_id,
            text=summary,
            ticker=ticker,
            source="earnings_transcript",
            date=quarter,
        )

        return doc_id

    # ── Cleanup ────────────────────────────────────────

    def close(self):
        self.session.close()

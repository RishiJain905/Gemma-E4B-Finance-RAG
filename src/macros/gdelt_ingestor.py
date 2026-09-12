"""
src/macros/gdelt_ingestor.py
GDELT news ingestion — fetches global financial news with geo-tags,
entity analysis, and tone scores, and stores them as ChromaDB documents
with sentiment metadata.

GDELT DOC API allows ~1 request per 5 seconds; bursts return HTTP 429.
All DOC API and api.gdeltproject.org traffic shares a module-level throttle.

Live verification: use mocks in CI. Manual checks = ONE fetch, then wait 5s+
before any further GDELT HTTP call.

Usage:
    ingestor = GDELTIngestor()
    ingestor.fetch_news_for_ticker("NVDA")
    ingestor.fetch_financial_news()
"""

import csv
import io
import json
import logging
import re
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import yaml

from src.ingestion.errors import ErrorClass, ProviderError, parse_retry_after, safe_message
from src.storage.store import Store
from src.universe.coverage import CoverageResolver

logger = logging.getLogger(__name__)


class GDELTRateLimitError(ProviderError):
    """Raised when GDELT keeps returning HTTP 429 after all retries."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: Optional[float] = None,
        reset_at: Optional[str] = None,
        attempts: int = 1,
        retry_timestamps: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            message,
            error_class=ErrorClass.RATE_LIMITED,
            status_code=429,
            retry_after=retry_after,
            reset_at=reset_at,
            attempts=attempts,
            provider_wide=True,
            circuit_open=True,
            retry_timestamps=retry_timestamps,
        )


# Shared across all GDELTIngestor instances — serializes DOC API + GKG downloads.
_last_gdelt_request_at: float = 0.0
_gdelt_request_lock = threading.Lock()


class GDELTIngestor:
    """
    Fetches global financial news from GDELT 2.0 and stores articles
    in ChromaDB with sentiment analysis (tone score).

    Each article is stored as a ChromaDB document with metadata:
      - ticker: detected ticker(s)
      - source: "gdelt"
      - date: publication date
      - tone: GDELT tone score (-100 to +100)
      - url: original article URL
      - entities: extracted entity names
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs/gdelt.yaml"
    DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    GKG_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
    GKG_DOC_COL = 4
    GKG_V2TONE_COL = 15

    POSITIVE_TITLE_WORDS = frozenset({
        "surge", "soar", "soars", "beat", "beats", "growth", "gain", "gains",
        "rise", "rises", "rising", "bull", "bullish", "upgrade", "record",
        "strong", "outperform", "rally", "rallies", "jump", "jumps", "boost",
        "profit", "profits", "win", "wins", "positive", "optimistic", "buy",
        "buying", "opportunity", "breakthrough", "milestone",
    })
    NEGATIVE_TITLE_WORDS = frozenset({
        "fall", "falls", "drop", "drops", "plunge", "plunges", "miss", "misses",
        "loss", "losses", "decline", "declines", "bear", "bearish", "downgrade",
        "weak", "weakness", "cut", "cuts", "crash", "crashes", "slump", "slumps",
        "sell-off", "selloff", "fear", "warning", "lawsuit", "probe", "investigation",
        "layoff", "layoffs", "recall", "bankrupt", "default",
    })

    def __init__(
        self,
        store: Optional[Store] = None,
        config_path: Optional[Path] = None,
        coverage_resolver: Optional[CoverageResolver] = None,
    ):
        self.store = store or Store()
        self._coverage_injected = coverage_resolver is not None
        self.coverage = coverage_resolver or CoverageResolver(self.store)
        self.config = self._load_config(config_path)
        self._gkg_file_cache: dict[str, dict[str, float]] = {}

    # ── Config Loading ─────────────────────────────────

    def _load_config(self, config_path: Optional[Path]) -> dict:
        path = config_path or self.DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {
            "max_records": 250,
            "lookback_days": 7,
            "request_delay": 7.0,
            "max_retries_on_429": 4,
            "max_query_terms_per_ticker": 1,
            "financial_topics": ["FINANCE", "MARKETS", "CORPORATE"],
        }

    # ── Ticker-to-Query Mapping ────────────────────────

    TICKER_TO_QUERY = {
        "NVDA": ["NVIDIA", "NVDA"],
        "AMD": ["AMD", "Advanced Micro Devices"],
        "AAPL": ["Apple", "AAPL"],
        "MSFT": ["Microsoft", "MSFT"],
        "META": ["Meta", "Facebook", "META"],
        "GOOGL": ["Google", "Alphabet", "GOOGL"],
        "AMZN": ["Amazon", "AMZN"],
        "TSLA": ["Tesla", "TSLA"],
        "CRWD": ["CrowdStrike", "CRWD"],
        "PANW": ["Palo Alto Networks", "PANW"],
        "PLTR": ["Palantir", "PLTR"],
        "SNOW": ["Snowflake", "SNOW"],
        "AVGO": ["Broadcom", "AVGO"],
        "ORCL": ["Oracle", "ORCL"],
        "INTC": ["Intel", "INTC"],
        "QCOM": ["Qualcomm", "QCOM"],
        "MU": ["Micron", "MU"],
        "MRVL": ["Marvell", "MRVL"],
        "NFLX": ["Netflix", "NFLX"],
        "SPOT": ["Spotify", "SPOT"],
        "UBER": ["Uber", "UBER"],
        "SHOP": ["Shopify", "SHOP"],
        "SQ": ["Block", "Square", "SQ"],
        "MSTR": ["MicroStrategy", "Strategy", "MSTR"],
    }

    def _query_terms_for_ticker(self, ticker: str) -> list[str]:
        """Return DOC query terms for a ticker, capped to limit API fan-out."""
        terms = list(self.TICKER_TO_QUERY.get(ticker.upper(), [ticker]))
        try:
            limit = int(self.config.get("max_query_terms_per_ticker", 1))
        except (TypeError, ValueError):
            limit = 1
        if limit < 1:
            limit = 1
        return terms[:limit] or [ticker]

    # ── Public API ─────────────────────────────────────

    def fetch_news_for_ticker(
        self,
        ticker: str,
        max_records: Optional[int] = None,
        lookback_days: Optional[int] = None,
    ) -> list[dict]:
        """
        Fetch recent GDELT news articles for a specific ticker.

        Args:
            ticker: Stock ticker (e.g., "NVDA")
            max_records: Max articles to return
            lookback_days: How far back to search

        Returns:
            List of processed article dicts
        """
        query_terms = self._query_terms_for_ticker(ticker)
        max_records = max_records or self.config.get("max_records", 250)
        lookback_days = lookback_days or self.config.get("lookback_days", 7)

        all_articles = []

        for term in query_terms:
            articles = self._search_gdelt(
                term=term,
                max_records=max_records // len(query_terms),
                lookback_days=lookback_days,
            )
            all_articles.extend(articles)

        self._enrich_articles_with_tone(all_articles, query_terms, lookback_days)
        processed = self._process_articles(all_articles, ticker)

        logger.info(
            "GDELT: %d articles fetched for %s (%d after dedup/processing)",
            len(all_articles), ticker, len(processed),
        )

        return processed

    def fetch_financial_news(self, max_records: int = 500) -> list[dict]:
        """
        Fetch broad financial news across all topics.

        Returns:
            Combined list of processed articles
        """
        topics = self.config.get("financial_topics", ["FINANCE", "MARKETS"])
        all_articles = []

        for topic in topics:
            articles = self._search_gdelt(
                term=topic,
                max_records=max_records // len(topics),
                lookback_days=1,
            )
            all_articles.extend(articles)

        self._enrich_articles_with_tone(all_articles, topics, lookback_days=1)
        processed = self._process_articles(all_articles, ticker="FINANCE")

        logger.info(
            "GDELT financial news: %d articles fetched (%d processed)",
            len(all_articles), len(processed),
        )

        return processed

    def fetch_and_store_for_ticker(
        self,
        ticker: str,
        max_records: Optional[int] = None,
    ) -> int:
        """
        Fetch news for a ticker AND store in ChromaDB.

        Args:
            ticker: Stock ticker
            max_records: Max articles

        Returns:
            Number of articles stored
        """
        articles = self.fetch_news_for_ticker(ticker, max_records)
        stored = self._store_articles(articles)
        logger.info("GDELT: Stored %d/%d articles for %s", stored, len(articles), ticker)
        return stored

    def fetch_and_store_all(self) -> dict[str, int]:
        """
        Fetch news for ALL core tickers and store in ChromaDB.

        Returns:
            {ticker: articles_stored}
        """
        results = {}

        for ticker in self._batch_tickers("gdelt"):
            count = self.fetch_and_store_for_ticker(
                ticker, max_records=100,
            )
            results[ticker] = count

        return results

    def _batch_tickers(self, source_name: str) -> list[str]:
        """Use the legacy core list only while a fresh registry is empty."""
        rows = self.store.list_securities(active=None, limit=1, offset=0)
        if not self._coverage_injected and (not isinstance(rows, list) or not rows):
            from src.ingestion.yfinance_ingestor import YFinanceIngestor

            return list(YFinanceIngestor(store=self.store).core_tickers)
        return self.coverage.tickers_for(source_name)

    # ── Tone Analysis ─────────────────────────────────

    def get_sentiment_summary(self, ticker: str, days: int = 7) -> dict:
        """
        Get a sentiment summary for a ticker based on GDELT tone scores.

        Args:
            ticker: Stock ticker
            days: Lookback period

        Returns:
            {
                "ticker": "NVDA",
                "average_tone": 1.5,
                "article_count": 42,
                "positive_ratio": 0.55,
                "negative_ratio": 0.20,
                "most_positive": "Article title...",
                "most_negative": "Article title...",
            }
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        articles = self._query_stored_articles(ticker, since=cutoff)

        if not articles:
            return {
                "ticker": ticker,
                "average_tone": None,
                "article_count": 0,
                "positive_ratio": 0,
                "negative_ratio": 0,
            }

        tones = []
        for art in articles:
            tone = art.get("metadata", {}).get("tone")
            if tone is not None:
                try:
                    tones.append(float(tone))
                except (ValueError, TypeError):
                    pass

        if not tones:
            return {"ticker": ticker, "article_count": len(articles),
                    "average_tone": None}

        avg_tone = sum(tones) / len(tones)
        positive = sum(1 for t in tones if t > 1)
        negative = sum(1 for t in tones if t < -1)

        return {
            "ticker": ticker,
            "average_tone": round(avg_tone, 2),
            "article_count": len(articles),
            "positive_ratio": round(positive / len(tones), 2),
            "negative_ratio": round(negative / len(tones), 2),
        }

    # ── Internal ──────────────────────────────────────

    def _search_gdelt(
        self, term: str, max_records: int, lookback_days: int,
    ) -> list[dict]:
        """Query GDELT DOC 2.0 API for articles matching a term."""
        query = self._build_doc_query(term)
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": max(1, min(max_records, 250)),
            "format": "json",
            "timespan": f"{max(1, lookback_days)}d",
        }

        try:
            response = self._doc_api_get(params)
            articles = (
                self._parse_doc_api_articles(response, term) if response is not None else []
            )
            if articles:
                return articles

            # Domain-scoped queries may hit length limits (HTTP 200, plain-text
            # error); retry a bare keyword once via the shared throttle.
            if query != term:
                logger.info(
                    "GDELT domain query returned no articles for '%s'; retrying without domain filter",
                    term,
                )
                fallback = {**params, "query": term}
                response = self._doc_api_get(fallback)
                if response is not None:
                    return self._parse_doc_api_articles(response, term)

            return []

        except GDELTRateLimitError:
            raise
        except Exception as e:
            logger.error("GDELT search failed for '%s': %s", term, safe_message(e))
            return []

    def _doc_api_get(self, params: dict):
        """GET the GDELT DOC API with shared throttling and 429 backoff."""
        return self._gdelt_http_get(
            self.DOC_API_URL,
            params=params,
            timeout=30.0,
            throttle=True,
            log_context=params.get("query", "?"),
        )

    def _gdelt_http_get(
        self,
        url: str,
        *,
        params: Optional[dict] = None,
        throttle: bool = True,
        log_context: str = "",
        **kwargs,
    ):
        """
        Serialized GET for GDELT endpoints.

        Enforces minimum gap between requests and retries 429 with exponential
        backoff (request_delay * 2**attempt), capped by max_retries_on_429.
        """
        delay = float(self.config.get("request_delay", 7.0))
        max_retries = int(self.config.get("max_retries_on_429", 4))
        label = log_context or url
        retry_timestamps: list[str] = []

        with _gdelt_request_lock:
            if throttle and delay > 0:
                self._wait_for_rate_limit_unlocked(delay)

            for attempt in range(max_retries + 1):
                response = httpx.get(url, params=params, **kwargs)
                self._mark_gdelt_request()

                if response.status_code != 429:
                    response.raise_for_status()
                    return response

                if attempt >= max_retries:
                    retry_after = response.headers.get("retry-after")
                    retry_window = parse_retry_after(
                        retry_after,
                        now=datetime.now(timezone.utc),
                    )
                    retry_hint = (
                        f"; retry after {retry_after}s" if retry_after else ""
                    )
                    logger.error(
                        "GDELT rate limit (429) for '%s' after %d attempts; giving up%s",
                        label,
                        attempt + 1,
                        retry_hint,
                    )
                    raise GDELTRateLimitError(
                        f"GDELT rate limit exhausted for '{label}' after "
                        f"{attempt + 1} attempts{retry_hint}",
                        retry_after=(
                            retry_window.delay_seconds if retry_window else None
                        ),
                        reset_at=retry_window.reset_at if retry_window else None,
                        attempts=attempt + 1,
                        retry_timestamps=retry_timestamps,
                    )

                backoff = delay * (2 ** attempt)
                retry_after = response.headers.get("retry-after")
                retry_window = parse_retry_after(
                    retry_after,
                    now=datetime.now(timezone.utc),
                )
                if retry_window is not None:
                    backoff = retry_window.delay_seconds
                backoff = min(
                    max(backoff, 0.0),
                    float(self.config.get("max_retry_sleep", 300.0)),
                )
                logger.warning(
                    "GDELT rate limit (429) for '%s' (attempt %d/%d); "
                    "waiting %.1fs before retry",
                    label,
                    attempt + 1,
                    max_retries + 1,
                    backoff,
                )
                retry_timestamps.append(
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                )
                time.sleep(backoff)

        return None

    def _enrich_articles_with_tone(
        self,
        articles: list[dict],
        search_terms: list[str],
        lookback_days: int,
    ) -> None:
        """Attach GDELT V2Tone from GKG files; title lexicon as fallback."""
        if not articles:
            return

        tone_map = self._fetch_gkg_tone_map(articles, search_terms, lookback_days)
        use_fallback = self.config.get("use_title_tone_fallback", True)

        for art in articles:
            if art.get("tone") not in (None, "", 0, 0.0):
                continue

            url = art.get("url", "") or art.get("sourceurl", "")
            if url and url in tone_map:
                art["tone"] = tone_map[url]
                continue

            if use_fallback:
                title = art.get("title", "") or art.get("name", "")
                art["tone"] = self._estimate_title_tone(title)

    def _fetch_gkg_tone_map(
        self,
        articles: list[dict],
        search_terms: list[str],
        lookback_days: int,
    ) -> dict[str, float]:
        """Build URL -> tone map from GDELT GKG 2.0 CSV files."""
        target_urls = {
            art.get("url", "") or art.get("sourceurl", "")
            for art in articles
        }
        target_urls.discard("")

        file_stamps = self._gkg_file_stamps_for_articles(articles, lookback_days)
        max_files = self.config.get("gkg_max_files_per_fetch", 24)
        file_stamps = file_stamps[:max_files]

        tone_map: dict[str, float] = {}
        term_patterns = [
            re.compile(re.escape(term), re.IGNORECASE)
            for term in search_terms
            if term
        ]

        for stamp in file_stamps:
            rows = self._load_gkg_file(stamp)
            for url, tone in rows.items():
                if url in target_urls:
                    tone_map[url] = tone
                    continue
                if term_patterns and any(
                    p.search(url) for p in term_patterns
                ):
                    tone_map[url] = tone

        return tone_map

    def _gkg_file_stamps_for_articles(
        self, articles: list[dict], lookback_days: int,
    ) -> list[str]:
        """Derive GKG file timestamps from article seendates (+ recent fallback)."""
        stamps: list[str] = []
        seen: set[str] = set()

        for art in articles:
            seendate = art.get("seendate", "") or art.get("date", "")
            dt = self._parse_seendate(seendate)
            if dt is None:
                continue
            for candidate in self._gkg_stamp_candidates(dt):
                if candidate not in seen:
                    seen.add(candidate)
                    stamps.append(candidate)

        if not stamps:
            now = datetime.now(timezone.utc)
            for offset in range(0, min(lookback_days, 2) * 96):
                dt = now - timedelta(minutes=15 * offset)
                stamp = dt.strftime("%Y%m%d%H%M%S")
                if stamp not in seen:
                    seen.add(stamp)
                    stamps.append(stamp)

        return stamps

    def _gkg_stamp_candidates(self, dt: datetime) -> list[str]:
        """Return likely GKG file stamps for an article timestamp."""
        minute_bucket = (dt.minute // 15) * 15
        base = dt.replace(minute=minute_bucket, second=0, microsecond=0)
        candidates = [base, base + timedelta(minutes=15), base - timedelta(minutes=15)]
        return [c.strftime("%Y%m%d%H%M%S") for c in candidates]

    def _load_gkg_file(self, stamp: str) -> dict[str, float]:
        """Download and parse one GKG CSV zip; cached per stamp."""
        if stamp in self._gkg_file_cache:
            return self._gkg_file_cache[stamp]

        url = f"{self.GKG_BASE_URL}/{stamp}.gkg.csv.zip"
        rows: dict[str, float] = {}

        try:
            response = self._gdelt_http_get(
                url,
                timeout=60.0,
                follow_redirects=True,
                throttle=True,
                log_context=f"GKG {stamp}",
            )
            if response is None or response.status_code != 200:
                self._gkg_file_cache[stamp] = rows
                return rows

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as raw:
                    reader = csv.reader(
                        io.TextIOWrapper(raw, encoding="utf-8", errors="replace"),
                        delimiter="\t",
                    )
                    for row in reader:
                        if len(row) <= self.GKG_V2TONE_COL:
                            continue
                        doc_url = row[self.GKG_DOC_COL].strip()
                        tone = self._parse_v2tone(row[self.GKG_V2TONE_COL])
                        if doc_url and tone is not None:
                            rows[doc_url] = tone

        except GDELTRateLimitError:
            raise
        except Exception as e:
            logger.debug("GKG file %s unavailable: %s", stamp, e)

        self._gkg_file_cache[stamp] = rows
        return rows

    def _parse_v2tone(self, v2tone: str) -> Optional[float]:
        """Parse V2Tone CSV field; first value is average document tone."""
        if not v2tone:
            return None
        parts = v2tone.split(",")
        if not parts:
            return None
        try:
            return float(parts[0])
        except (ValueError, TypeError):
            return None

    def _parse_seendate(self, seendate: str) -> Optional[datetime]:
        """Parse GDELT seendate like 20260608T034632Z."""
        if not seendate:
            return None
        cleaned = seendate.strip()
        if len(cleaned) >= 15 and "T" in cleaned:
            try:
                return datetime.strptime(cleaned[:15], "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc,
                )
            except ValueError:
                pass
        if len(cleaned) >= 10 and cleaned[:10].count("-") == 2:
            try:
                return datetime.strptime(cleaned[:10], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc,
                )
            except ValueError:
                pass
        return None

    def _estimate_title_tone(self, title: str) -> float:
        """Lightweight title sentiment when GKG tone is unavailable."""
        if not title:
            return 0.0

        words = re.findall(r"[a-zA-Z']+", title.lower())
        if not words:
            return 0.0

        pos = sum(1 for w in words if w in self.POSITIVE_TITLE_WORDS)
        neg = sum(1 for w in words if w in self.NEGATIVE_TITLE_WORDS)
        if pos == neg == 0:
            return 0.0

        score = (pos - neg) / max(len(words), 1) * 10.0
        return round(max(-10.0, min(10.0, score)), 2)

    def _build_doc_query(self, term: str) -> str:
        """
        Build a GDELT DOC query, optionally scoped to finance domains.

        GDELT expects ``KEYWORD (domain:a.com OR domain:b.com)``. Long domain
        OR lists exceed API query-length limits and return HTTP 200 with a
        plain-text error body instead of JSON.
        """
        keyword = f'"{term}"' if " " in term.strip() else term.strip()
        domains = self.config.get("finance_domains") or []
        if not domains:
            return keyword

        max_domains = int(self.config.get("max_finance_domains", 5))
        selected = domains[:max_domains]
        domain_clause = " OR ".join(f"domain:{d}" for d in selected)
        return f"{keyword} ({domain_clause})"

    def _parse_doc_api_articles(self, response, term: str) -> list[dict]:
        """Parse ArtList JSON; log and return [] on empty or non-JSON bodies."""
        body = (response.text or "").strip()
        if not body:
            logger.warning("GDELT returned empty response for '%s'", term)
            return []

        if body.startswith("Please limit requests"):
            logger.warning(
                "GDELT rate limit message for '%s': %s",
                term,
                safe_message(body),
            )
            return []

        content_type = (response.headers.get("content-type") or "").lower()
        if "text/html" in content_type or body.lstrip().startswith("<"):
            logger.warning(
                "GDELT non-JSON HTML response for '%s' (content-type=%s): %s",
                term,
                content_type or "unknown",
                safe_message(body),
            )
            return []

        if not body.startswith("{") and not body.startswith("["):
            logger.warning(
                "GDELT non-JSON response for '%s' (content-type=%s): %s",
                term,
                content_type or "unknown",
                safe_message(body),
            )
            return []

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            logger.warning(
                "GDELT JSON parse failed for '%s': %s; body[:200]=%s",
                term,
                e,
                safe_message(body),
            )
            return []

        articles = payload.get("articles", []) if isinstance(payload, dict) else []
        return articles if isinstance(articles, list) else []

    def _process_articles(
        self, articles: list[dict], ticker: str,
    ) -> list[dict]:
        """Normalize GDELT articles into our standard format."""
        processed = []
        seen_urls = set()

        for art in articles:
            url = art.get("url", "") or art.get("sourceurl", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = art.get("title", "") or art.get("name", "")
            content = (
                art.get("content", "")
                or art.get("text", "")
                or art.get("snippet", "")
            )

            if not title and not content:
                continue
            if not content:
                content = title

            tone_raw = art.get("tone", art.get("avgtone"))
            tone = None
            if tone_raw is not None and tone_raw != "":
                try:
                    tone = float(tone_raw)
                except (ValueError, TypeError):
                    tone = None
            if tone is None:
                title = art.get("title", "") or art.get("name", "")
                tone = self._estimate_title_tone(title)

            date = art.get("date", art.get("seendate", ""))
            if date and len(date) > 10:
                date = date[:10]

            processed.append({
                "title": title,
                "content": content[:3000],
                "url": url,
                "tone": tone,
                "date": date,
                "ticker": ticker,
                "source": "gdelt",
                "entities": art.get("persons", art.get("organizations", "")),
            })

        return processed

    def _store_articles(self, articles: list[dict]) -> int:
        """Store processed articles in ChromaDB."""
        stored = 0
        for art in articles:
            try:
                doc_id = f"gdelt/{art['ticker']}/{art['date']}/{hash(art['url']) % 10000:04d}"
                self.store.save_document(
                    document_id=doc_id,
                    text=f"{art['title']}\n\n{art['content']}",
                    ticker=art["ticker"],
                    source="gdelt",
                    date=art["date"],
                    metadata={
                        "tone": art["tone"],
                        "url": art["url"],
                        "entities": art["entities"],
                    },
                )
                stored += 1
            except Exception as e:
                logger.warning("Failed to store GDELT article: %s", safe_message(e))
        return stored

    def _query_stored_articles(
        self, ticker: str, since: datetime,
    ) -> list[dict]:
        """Query ChromaDB for stored articles matching a ticker."""
        try:
            results = self.store.chroma.search(
                query=f"{ticker} financial news",
                n_results=100,
                filter_dict={
                    "$and": [
                        {"ticker": ticker.upper()},
                        {"source": "gdelt"},
                    ],
                },
            )
            return results
        except Exception:
            return []

    def _respect_rate_limit(self) -> None:
        """Enforce minimum gap since the last GDELT HTTP call."""
        delay = float(self.config.get("request_delay", 7.0))
        if delay <= 0:
            return
        with _gdelt_request_lock:
            self._wait_for_rate_limit_unlocked(delay)
            self._mark_gdelt_request()

    def _wait_for_rate_limit_unlocked(self, delay: float) -> None:
        """Caller must hold _gdelt_request_lock."""
        global _last_gdelt_request_at
        now = time.monotonic()
        if _last_gdelt_request_at > 0:
            elapsed = now - _last_gdelt_request_at
            if elapsed < delay:
                time.sleep(delay - elapsed)

    @staticmethod
    def _mark_gdelt_request() -> None:
        """Record that a GDELT HTTP request was just sent."""
        global _last_gdelt_request_at
        _last_gdelt_request_at = time.monotonic()

"""
src/storage/store.py
Unified storage facade over SQLiteStore + ChromaStore.

Usage:
    store = Store()

    # Save structured facts
    store.save_fundamental("NVDA", "revenue_q1", 26.0, "usd", "2026-Q1")

    # Save a document with embedding
    doc_id = store.save_document("NVDA 10-Q", "NVIDIA reported...",
                                   ticker="NVDA", source="sec")

    # Search everything
    results = store.search("What is NVDA's revenue?")
    # Returns: {"facts": [...], "documents": [...]}
"""

import logging
from pathlib import Path
from typing import Optional

from .chroma_store import ChromaStore
from .sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class Store:
    """
    Unified storage layer combining structured (SQLite) and
    semantic (ChromaDB) storage.
    """

    def __init__(self,
                 db_path: Optional[Path] = None,
                 chroma_path: Optional[Path] = None,
                 collection_name: str = "tracealchemy_docs",
                 embedding_endpoint: str = "http://127.0.0.1:8087/v1/embeddings"):

        self.sqlite = SQLiteStore(db_path=db_path)
        self.chroma = ChromaStore(
            persist_directory=chroma_path,
            collection_name=collection_name,
            embedding_endpoint=embedding_endpoint,
        )

    # ── Health ─────────────────────────────────────────

    def heartbeat(self) -> dict:
        """Check both storage backends are responsive."""
        return {
            "sqlite": self._check_sqlite(),
            "chroma": self.chroma.heartbeat(),
            "chroma_doc_count": self.chroma.count(),
        }

    def _check_sqlite(self) -> bool:
        try:
            with self.sqlite._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("SQLite heartbeat failed: %s", e)
            return False

    # ─── Structured Facts (SQLite) ────────────────────

    def save_fundamental(self, ticker: str, metric: str, value: float,
                         unit: str = "usd", period: str = None,
                         period_type: str = "quarterly",
                         source_type: str = "yfinance",
                         source_url: str = None) -> bool:
        """Save or update a single financial metric."""
        return bool(self.sqlite.upsert_fundamental(
            ticker, metric, value, unit, period,
            period_type, source_type, source_url
        ))

    def get_fundamental(self, ticker: str, metric: str,
                        period: str = None) -> Optional[dict]:
        """Get the latest value for a metric."""
        return self.sqlite.get_fundamental(ticker, metric, period)

    def get_fundamentals_batch(self, ticker: str,
                               metrics: list[str] = None) -> dict:
        """Get multiple metrics for a ticker at once."""
        return self.sqlite.get_fundamentals_batch(ticker, metrics)

    # ── Document Storage (ChromaDB) ──────────────────

    def save_document(self,
                      document_id: str,
                      text: str,
                      ticker: str = None,
                      source: str = None,
                      date: str = None,
                      metadata: dict = None) -> str:
        """
        Store a document with its embedding in ChromaDB.

        Args:
            document_id: Unique ID (e.g., 'sec/NVDA/10-K-2025')
            text: Document content
            ticker: Associated stock ticker
            source: Source type
            date: Document date (ISO format)
            metadata: Additional metadata

        Returns:
            The document ID (confirmation of storage)
        """
        self.chroma.add_document(
            document_id=document_id,
            text=text,
            ticker=ticker,
            source=source,
            date=date,
            metadata=metadata,
        )
        return document_id

    def save_documents_batch(self,
                             ids: list[str],
                             texts: list[str],
                             metadatas: list[dict] = None):
        """Store multiple documents at once."""
        self.chroma.add_documents_batch(ids, texts, metadatas)

    # ── Hybrid Search ─────────────────────────────────

    def search(self, query: str, n_results: int = 5,
               ticker: str = None) -> dict:
        """
        Search BOTH stores and return combined results.

        Args:
            query: Natural language query
            n_results: Max semantic results
            ticker: Optional ticker filter

        Returns:
            {
                "documents": [...],   # ChromaDB semantic matches
                "facts": [...],       # SQLite structured matches
                "ticker": "NVDA",     # Detected or provided ticker
            }
        """
        # Detect ticker from query if not provided
        detected_ticker = ticker or self._detect_ticker(query)

        # Parallel search
        documents = self.chroma.search(
            query=query,
            n_results=n_results,
            filter_dict={"ticker": detected_ticker} if detected_ticker else None
        )

        # Also search without ticker filter for broader context
        if detected_ticker:
            broad_results = self.chroma.search(
                query=query,
                n_results=n_results // 2,
            )
            # Merge: ticker-filtered first, then broad results (deduped)
            seen_ids = {d["id"] for d in documents}
            for doc in broad_results:
                if doc["id"] not in seen_ids:
                    documents.append(doc)
                    seen_ids.add(doc["id"])

        # Get structured facts if ticker detected
        facts = []
        if detected_ticker:
            facts = self.sqlite.search_facts(
                ticker=detected_ticker,
                limit=n_results
            )

        return {
            "documents": documents,
            "facts": facts,
            "ticker": detected_ticker,
        }

    def search_by_ticker(self, query: str, ticker: str,
                         n_results: int = 5) -> dict:
        """Search specifically within one ticker's data."""
        return self.search(query=query, n_results=n_results, ticker=ticker)

    # ── Filing Pipeline Support ───────────────────────

    def register_filing(self, ticker: str, filing_type: str,
                        filing_date: str, period: str,
                        accession: str, source_url: str) -> bool:
        """Register a filing as received (before parsing)."""
        return self.sqlite.register_filing(
            ticker, filing_type, filing_date, period,
            accession, source_url
        )

    def process_filing(self, filing_record: dict,
                       extracted_text: str,
                       extracted_facts: list[dict]):
        """
        Process a full filing through the model-as-parser pipeline.

        1. Saves the complete text as a ChromaDB document
        2. Saves each extracted fact to SQLite
        3. Marks the filing as parsed

        Args:
            filing_record: The dict from filings table (must include source_type)
            extracted_text: Full document text (or summary)
            extracted_facts: List of {metric, value, unit, period} dicts
        """
        # 1. Store the document embedding
        doc_id = (
            f"{filing_record['source_type']}/{filing_record['ticker']}/"
            f"{filing_record['filing_type']}-{filing_record['period']}"
        )
        self.save_document(
            document_id=doc_id,
            text=extracted_text,
            ticker=filing_record["ticker"],
            source=filing_record["filing_type"],
            date=filing_record["filing_date"],
        )

        # 2. Save each extracted fact
        for fact in extracted_facts:
            self.save_fundamental(
                ticker=filing_record["ticker"],
                metric=fact.get("metric"),
                value=fact.get("value"),
                unit=fact.get("unit", "usd"),
                period=fact.get("period", filing_record["period"]),
                source_type=filing_record["source_type"],
                source_url=filing_record["source_url"],
            )

        # 3. Mark as parsed
        self.sqlite.mark_filing_parsed(
            filing_record["accession"],
            embedding_id=doc_id
        )

    # ── Cache Management ──────────────────────────────

    def get_cache_status(self, ticker: str, source: str) -> Optional[dict]:
        """Get cache freshness metadata for a ticker + source."""
        return self.sqlite.get_cache_status(ticker, source)

    def mark_cache_fresh(self, ticker: str, source: str, ttl_hours: int = 24):
        self.sqlite.mark_cache_fresh(ticker, source, ttl_hours)

    def mark_cache_stale(self, ticker: str, source: str, error: str = None):
        self.sqlite.mark_cache_stale(ticker, source, error)

    def upsert_cache_stale(self, ticker: str, source: str, error: str = None):
        self.sqlite.upsert_cache_stale(ticker, source, error)

    def get_stale_entries(self, limit: int = 20) -> list[dict]:
        return self.sqlite.get_stale_cache_entries(limit)

    # ── Freshness / Staleness-Aware Querying (Phase 1.7.4) ─

    # Maps a logical data source to its cache_meta source identifier and the
    # watchlist.yaml schedule key that defines its TTL (in hours).
    FRESHNESS_SOURCES = {
        "yfinance_fundamentals": {"cache_source": "yfinance_fundamentals", "ttl_key": "fundamentals"},
        "yfinance_news":         {"cache_source": "yfinance_news",         "ttl_key": "news"},
        "sec_filings":           {"cache_source": "sec_filings_discovery", "ttl_key": "sec_filings"},
        "gdelt_news":            {"cache_source": "gdelt_news",            "ttl_key": "gdelt_news"},
        "earnings_transcripts":  {"cache_source": "earnings_transcripts",  "ttl_key": "transcripts"},
        "ir_pages":              {"cache_source": "ir_pages",              "ttl_key": "ir_pages"},
        "estimates":             {"cache_source": "estimates",             "ttl_key": "estimates"},
    }

    _DEFAULT_TTLS = {
        "fundamentals": 24, "news": 6, "macro": 24, "sec_filings": 12,
        "gdelt_news": 6, "transcripts": 168, "ir_pages": 24, "estimates": 24,
    }

    def _schedule_ttls(self) -> dict:
        """Lazily load the schedule TTL map from configs/watchlist.yaml."""
        cached = getattr(self, "_ttl_cache", None)
        if cached is not None:
            return cached
        ttls = dict(self._DEFAULT_TTLS)
        try:
            import yaml
            wl_path = Path(__file__).parent.parent.parent / "configs/watchlist.yaml"
            if wl_path.exists():
                with open(wl_path) as f:
                    config = yaml.safe_load(f) or {}
                ttls.update(config.get("schedule", {}) or {})
        except Exception:
            pass
        self._ttl_cache = ttls
        return ttls

    @staticmethod
    def _age_hours(last_updated) -> Optional[float]:
        """Compute age in hours from a cache_meta last_updated value."""
        if not last_updated:
            return None
        from datetime import datetime, timezone
        try:
            if isinstance(last_updated, str):
                try:
                    from dateutil import parser
                    dt = parser.parse(last_updated)
                except Exception:
                    dt = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = last_updated
            return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception:
            return None

    def is_ticker_fresh(self, ticker: str, source: str, ttl_hours: int) -> bool:
        """Check if a specific source for a ticker exists and is within TTL.

        Args:
            ticker: Ticker symbol
            source: cache_meta source identifier (e.g., "yfinance_fundamentals")
            ttl_hours: TTL in hours

        Returns:
            True if data exists and is within TTL, False otherwise.
        """
        status = self.sqlite.get_cache_status(ticker, source)
        if not status:
            return False
        if status.get("status") == "stale":
            return False
        age = self._age_hours(status.get("last_updated"))
        if age is None:
            return False
        return age < ttl_hours

    def mark_source_fresh(self, ticker: str, source: str, ttl_hours: int):
        """Mark a source as freshly updated (now + ttl_hours)."""
        self.sqlite.mark_cache_fresh(ticker, source, ttl_hours)

    def mark_source_stale(self, ticker: str, source: str, error: str = ""):
        """Mark a source as stale (e.g., after a failed fetch). Upserts."""
        self.sqlite.upsert_cache_stale(ticker, source, error or None)

    def get_freshness_report(self, ticker: str) -> dict:
        """Get freshness status for all data sources for a given ticker.

        Returns a dict with per-source status ("fresh" | "stale" |
        "never_fetched"), age_hours, ttl_hours, last_updated, plus an
        overall rollup and the list of stale source names.
        """
        ttls = self._schedule_ttls()
        sources: dict[str, dict] = {}
        stale_sources: list[str] = []
        fresh_count = 0
        present_count = 0

        for name, cfg in self.FRESHNESS_SOURCES.items():
            ttl_hours = ttls.get(cfg["ttl_key"], self._DEFAULT_TTLS.get(cfg["ttl_key"], 24))
            cache = self.sqlite.get_cache_status(ticker, cfg["cache_source"])

            if not cache:
                sources[name] = {
                    "status": "never_fetched",
                    "last_updated": None,
                    "age_hours": None,
                    "ttl_hours": ttl_hours,
                }
                continue

            present_count += 1
            age = self._age_hours(cache.get("last_updated"))
            if cache.get("status") == "stale":
                status = "stale"
            elif age is not None and age < ttl_hours:
                status = "fresh"
            else:
                status = "stale"

            if status == "fresh":
                fresh_count += 1
            else:
                stale_sources.append(name)

            sources[name] = {
                "status": status,
                "last_updated": str(cache.get("last_updated")) if cache.get("last_updated") else None,
                "age_hours": round(age, 2) if age is not None else None,
                "ttl_hours": ttl_hours,
            }

        if present_count == 0:
            overall = "never_fetched"
        elif fresh_count == present_count:
            overall = "fresh"
        elif fresh_count == 0:
            overall = "stale"
        else:
            overall = "partial"

        return {
            "ticker": ticker,
            "sources": sources,
            "overall": overall,
            "stale_sources": stale_sources,
        }

    def get_stale_tickers(self, source: str) -> list[str]:
        """Get all tickers whose data for a given logical source is stale.

        ``source`` may be a logical name from FRESHNESS_SOURCES (e.g.
        "yfinance_fundamentals") or a raw cache_meta source identifier.
        Tickers with no cache entry are not included (use the watchlist for
        never-fetched tickers).
        """
        cfg = self.FRESHNESS_SOURCES.get(source)
        cache_source = cfg["cache_source"] if cfg else source
        ttl_key = cfg["ttl_key"] if cfg else None
        ttl_hours = self._schedule_ttls().get(ttl_key, 24) if ttl_key else 24

        sql = "SELECT ticker, last_updated, status FROM cache_meta WHERE source=?"
        with self.sqlite._connect() as conn:
            rows = conn.execute(sql, (cache_source,)).fetchall()

        stale = []
        for row in rows:
            r = dict(row)
            if r.get("status") == "stale":
                stale.append(r["ticker"])
                continue
            age = self._age_hours(r.get("last_updated"))
            if age is None or age >= ttl_hours:
                stale.append(r["ticker"])
        return stale

    # ── Utilities ─────────────────────────────────────

    @staticmethod
    def _detect_ticker(text: str) -> Optional[str]:
        """
        Simple ticker detection from query text.
        Looks for known company names or uppercase 1-4 letter stock symbols.
        """
        import re
        # Known major tickers to prioritize
        known_tickers = {
            "NVDA", "AMD", "AAPL", "MSFT", "GOOGL", "GOOG", "META",
            "AMZN", "TSLA", "INTC", "CRM", "AVGO", "ORCL", "CSCO",
            "IBM", "QCOM", "TXN", "MU", "MRVL", "PLTR", "SNOW",
            "CRWD", "PANW", "UBER", "SQ", "HOOD", "COIN", "MSTR",
        }

        # Resolve common company names to ticker symbols first.
        # This keeps hybrid search useful for natural-language queries.
        company_name_map = {
            "nvidia": "NVDA",
            "advanced micro devices": "AMD",
            "amd": "AMD",
            "apple": "AAPL",
            "microsoft": "MSFT",
            "alphabet": "GOOGL",
            "google": "GOOGL",
            "meta": "META",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "intel": "INTC",
            "salesforce": "CRM",
            "broadcom": "AVGO",
            "oracle": "ORCL",
            "cisco": "CSCO",
            "ibm": "IBM",
            "qualcomm": "QCOM",
            "texas instruments": "TXN",
            "micron": "MU",
            "marvell": "MRVL",
            "palantir": "PLTR",
            "snowflake": "SNOW",
            "crowdstrike": "CRWD",
            "palo alto networks": "PANW",
            "uber": "UBER",
            "block": "SQ",
            "robinhood": "HOOD",
            "coinbase": "COIN",
            "microstrategy": "MSTR",
            "strategy": "MSTR",
        }
        normalized_text = text.lower()
        for company_name, ticker in company_name_map.items():
            if company_name in normalized_text:
                return ticker

        # Find all uppercase words that look like tickers
        candidates = set(re.findall(r'\b[A-Z]{1,4}\b', text))

        # Match against known tickers first
        for c in candidates:
            if c in known_tickers:
                return c

        # Fallback: return the first match that isn't a common word
        common_words = {"I", "A", "AN", "THE", "IT", "IS", "BE", "TO",
                        "OF", "IN", "ON", "AT", "BY", "AS", "OR", "IF",
                        "NO", "GO", "DO", "WE", "HE", "SHE", "ALL"}

        for c in candidates:
            if c not in common_words and len(c) >= 1:
                return c

        return None

    # ── Cleanup / Reset ───────────────────────────────

    def reset(self):
        """Clear all data (for testing)."""
        self.chroma.reset_collection()
        # For SQLite, just drop and recreate tables
        with self.sqlite._connect() as conn:
            conn.executescript("""
                DROP TABLE IF EXISTS fundamentals;
                DROP TABLE IF EXISTS filings;
                DROP TABLE IF EXISTS cache_meta;
                DROP TABLE IF EXISTS ingestion_log;
            """)
            conn.commit()
        self.sqlite._init_schema()

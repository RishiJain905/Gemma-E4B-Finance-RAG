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

from pathlib import Path
from typing import Optional

from .chroma_store import ChromaStore
from .sqlite_store import SQLiteStore


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
        except Exception:
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

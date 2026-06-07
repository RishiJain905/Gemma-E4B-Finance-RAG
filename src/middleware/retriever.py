"""
src/middleware/retriever.py
Hybrid retriever — queries both SQLite (structured facts) and
ChromaDB (semantic documents) based on parsed intent.

Usage:
    retriever = Retriever(store=store, config=config)
    results = retriever.retrieve(
        query="What is NVDA's revenue?",
        intent={"ticker": "NVDA", "metrics": ["total_revenue"], ...},
    )
    # Returns:
    # {
    #     "facts": [{"metric": "total_revenue", "value": 26.0, ...}],
    #     "documents": [{"id": "...", "text": "...", "metadata": {...}}, ...],
    #     "ticker": "NVDA",
    # }
"""

import logging
from typing import Optional

from src.storage.store import Store
from .config import MiddlewareConfig

logger = logging.getLogger(__name__)


class Retriever:
    """
    Hybrid retriever — queries SQLite + ChromaDB based on intent.

    Retrieval strategy by question type:
      - fact_lookup: SQLite first (exact metrics), ChromaDB for context
      - comparison: Both stores, multi-ticker
      - trend: ChromaDB (semantic), SQLite for historical data points
      - sentiment/news: ChromaDB (semantic documents)
      - explanation: Both stores
      - risk: ChromaDB (SEC filings, risk factors)
      - general: Both stores, broad search
    """

    def __init__(self, store: Store, config: Optional[MiddlewareConfig] = None):
        self.store = store
        self.config = config or MiddlewareConfig()

    # ── Public API ─────────────────────────────────────

    def retrieve(self, query: str, intent: dict,
                 top_k_documents: int = 5,
                 top_k_facts: int = 10) -> dict:
        """
        Perform hybrid retrieval based on parsed intent.

        Args:
            query: Original user question
            intent: Parsed intent dict from IntentParser
            top_k_documents: Max ChromaDB results
            top_k_facts: Max SQLite fact results

        Returns:
            {
                "facts": list[dict],
                "documents": list[dict],
                "ticker": str or None,
                "strategy": str,  # Which retrieval strategy was used
            }
        """
        ticker = intent.get("ticker")
        metrics = intent.get("metrics", [])
        question_type = intent.get("question_type", "general")
        timeframe = intent.get("timeframe")

        strategy = self._select_strategy(question_type, ticker, metrics)

        facts = []
        documents = []

        if strategy == "facts_only":
            facts = self._retrieve_facts(ticker, metrics, timeframe, top_k_facts)

        elif strategy == "documents_only":
            documents = self._retrieve_documents(query, ticker, top_k_documents)

        elif strategy == "hybrid":
            facts = self._retrieve_facts(ticker, metrics, timeframe, top_k_facts)
            documents = self._retrieve_documents(query, ticker, top_k_documents)

        elif strategy == "comparison":
            # Multi-ticker: extract all tickers from query
            tickers = self._extract_all_tickers(query)
            for t in tickers:
                t_facts = self._retrieve_facts(t, metrics, timeframe, top_k_facts // len(tickers))
                facts.extend(t_facts)
                t_docs = self._retrieve_documents(query, t, top_k_documents // len(tickers))
                documents.extend(t_docs)

        elif strategy == "broad":
            # No ticker detected — search everything
            documents = self._retrieve_documents(query, ticker=None, n_results=top_k_documents)
            facts = self._retrieve_all_facts(top_k_facts)

        logger.info(
            "Retrieval strategy=%s ticker=%s: %d facts, %d documents",
            strategy, ticker, len(facts), len(documents),
        )

        return {
            "facts": facts,
            "documents": documents,
            "ticker": ticker,
            "strategy": strategy,
        }

    # ── Strategy Selection ────────────────────────────

    def _select_strategy(self, question_type: str, ticker: Optional[str],
                         metrics: list[str]) -> str:
        """Select the retrieval strategy based on intent."""
        if question_type == "comparison":
            return "comparison"
        if question_type in ("sentiment", "news"):
            return "documents_only"
        if question_type == "fact_lookup" and metrics and ticker:
            return "facts_only"
        if question_type == "fact_lookup" and ticker:
            return "hybrid"
        if question_type == "trend" and ticker:
            return "hybrid"
        if question_type == "risk":
            return "documents_only"
        if ticker:
            return "hybrid"
        return "broad"

    # ── Fact Retrieval (SQLite) ────────────────────────

    def _retrieve_facts(self, ticker: Optional[str], metrics: list[str],
                        timeframe: Optional[str], limit: int) -> list[dict]:
        """Retrieve structured facts from SQLite."""
        if not ticker:
            return []

        facts = []

        # If specific metrics were detected, fetch those
        if metrics:
            batch = self.store.get_fundamentals_batch(ticker, metrics=metrics)
            for metric, value in batch.items():
                if value is not None:
                    facts.append({
                        "metric": metric,
                        "value": value,
                        "ticker": ticker,
                        "source_type": "sqlite",
                    })

        # Also get the latest N facts for context
        if len(facts) < limit:
            recent = self.store.sqlite.search_facts(ticker=ticker, limit=limit)
            for r in recent:
                # Avoid duplicates
                if not any(f.get("metric") == r.get("metric") for f in facts):
                    facts.append(dict(r))

        return facts[:limit]

    def _retrieve_all_facts(self, limit: int) -> list[dict]:
        """Retrieve facts across all tickers (broad search fallback)."""
        # Get the most recent facts from any ticker
        facts = []
        for ticker in self._get_all_tracked_tickers():
            batch = self.store.get_fundamentals_batch(ticker)
            for metric, value in batch.items():
                if value is not None:
                    facts.append({
                        "metric": metric,
                        "value": value,
                        "ticker": ticker,
                        "source_type": "sqlite",
                    })
            if len(facts) >= limit:
                break
        return facts[:limit]

    # ── Document Retrieval (ChromaDB) ──────────────────

    def _retrieve_documents(self, query: str, ticker: Optional[str],
                            n_results: int) -> list[dict]:
        """Retrieve semantically relevant documents from ChromaDB."""
        results = self.store.search(
            query=query,
            n_results=n_results,
            ticker=ticker,
        )
        return results.get("documents", [])

    # ── Multi-Ticker Extraction ────────────────────────

    def _extract_all_tickers(self, text: str) -> list[str]:
        """Extract ALL ticker mentions from a comparison query."""
        import re
        from .intent_parser import IntentParser

        parser = IntentParser()
        normalized = text.lower()
        tickers = set()

        # Check company names
        for company_name, ticker in parser.COMPANY_TO_TICKER.items():
            if company_name in normalized:
                tickers.add(ticker)

        # Check uppercase symbols
        candidates = set(re.findall(r'\b[A-Z]{1,5}\b', text))
        for c in candidates:
            if c in parser.KNOWN_TICKERS:
                tickers.add(c)

        return list(tickers) if tickers else ["NVDA", "AMD"]  # Fallback

    # ── Helpers ────────────────────────────────────────

    def _get_all_tracked_tickers(self) -> list[str]:
        """Get the list of all tracked tickers from the watchlist."""
        try:
            from src.ingestion.yfinance_ingestor import YFinanceIngestor
            ingestor = YFinanceIngestor(store=self.store)
            return ingestor.core_tickers
        except Exception:
            return ["NVDA", "AMD", "AAPL", "MSFT", "META", "CRWD"]

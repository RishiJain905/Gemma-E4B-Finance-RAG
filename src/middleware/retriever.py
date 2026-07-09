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
import time
from typing import Optional

from src.storage.store import Store
from .config import MiddlewareConfig
from .lexical_index import LexicalIndex

logger = logging.getLogger(__name__)

ESTIMATE_METRICS = [
    "estimate_revenue_current_q",
    "estimate_revenue_next_q",
    "estimate_revenue_current_y",
    "estimate_revenue_next_y",
    "estimate_eps_current_q",
    "estimate_eps_next_q",
    "estimate_eps_current_y",
    "estimate_eps_next_y",
]

PRICE_TARGET_METRICS = [
    "price_target_mean",
    "price_target_high",
    "price_target_low",
    "num_analysts",
    "recommendation_mean",
]

PROJECTION_METRICS = ESTIMATE_METRICS + PRICE_TARGET_METRICS


def rrf_fuse(vector_hits: list[dict], lexical_hits: list[dict],
             k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked lists (Phase 2.1.2.1).

    Each hit needs an ``id``. Returns ``(id, rrf_score)`` sorted high → low.
    An id ranked highly by *both* channels beats one ranked highly by only one.
    """
    scores: dict[str, float] = {}
    for rank, h in enumerate(vector_hits):
        hid = h.get("id")
        if hid is None:
            continue
        scores[hid] = scores.get(hid, 0.0) + 1.0 / (k + rank + 1)
    for rank, h in enumerate(lexical_hits):
        hid = h.get("id")
        if hid is None:
            continue
        scores[hid] = scores.get(hid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


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

    MACRO_KEYWORDS = [
        "interest rate", "fed rate", "federal reserve", "inflation",
        "cpi", "gdp", "economy", "economic", "recession", "unemployment",
        "treasury yield", "bond yield", "yield curve", "macro",
        "market outlook", "economic indicator", "gross domestic product",
        "jobs report", "nonfarm payroll", "consumer sentiment",
        "housing market", "industrial production",
        "fed funds", "interest rates", "rate hike", "rate cut",
        "fed funds rate",
    ]

    def __init__(self, store: Store, config: Optional[MiddlewareConfig] = None):
        self.store = store
        self.config = config or MiddlewareConfig()
        # Phase 2.1.2 — lazy singletons for the lexical channel + re-ranker.
        self._lexical: Optional[LexicalIndex] = None
        self._reranker = None
        # Tracks the document-retrieval path actually used for a retrieve() call
        # ("vector" | "hybrid" | "hybrid+rerank") — surfaced in the /query response.
        self._doc_retrieval_strategy = "vector"
        self._timings = {"embedding": 0.0, "chroma": 0.0, "sqlite": 0.0}

    @property
    def lexical(self) -> LexicalIndex:
        """Lazily-built, cached BM25 index over the chroma corpus."""
        if self._lexical is None:
            self._lexical = LexicalIndex(self.store)
        return self._lexical

    @property
    def reranker(self):
        """Lazily-built, cached cross-encoder re-ranker (Phase 2.1.2.2)."""
        if self._reranker is None:
            from .reranker import Reranker
            self._reranker = Reranker(self.config)
        return self._reranker

    def refresh_lexical_index(self) -> None:
        """Rebuild the BM25 index (call after ingestion / scheduler runs).

        Always invokes ``LexicalIndex.refresh()`` so callers/tests can assert a
        rebuild happened. No-op when the lexical channel is disabled.
        """
        if not self.config.enable_lexical:
            return
        _ = self.lexical  # ensure the index exists
        self._lexical.refresh()  # noqa: SLF001

    def warm_lexical_index(self) -> None:
        """Eagerly build the BM25 index (best-effort, never raises).

        Called on middleware startup so the first query doesn't pay the build
        cost and the index is ready for fusion.
        """
        try:
            self.lexical._ensure()  # noqa: SLF001
        except Exception as e:  # noqa: BLE001 - never fail startup
            logger.warning("Lexical index warm-up failed: %s", e)

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
                "strategy": str,  # Logical retrieval strategy used
                "retrieval_strategy": str,  # doc path: vector|hybrid|hybrid+rerank
            }
        """
        self._doc_retrieval_strategy = "vector"  # reset; upgraded by hybrid path
        self._timings = {"embedding": 0.0, "chroma": 0.0, "sqlite": 0.0}
        ticker = intent.get("ticker")
        metrics = intent.get("metrics", [])
        question_type = intent.get("question_type", "general")
        timeframe = intent.get("timeframe")

        strategy = self._select_strategy(question_type, ticker, metrics)

        # Text-based macro detection: _select_strategy only sees metrics, not the
        # raw question, so upgrade the strategy when the query mentions macro topics.
        if question_type not in ("sentiment", "news") and self._query_mentions_macro(query):
            if ticker is None:
                strategy = "macro"
            elif strategy in ("hybrid", "facts_only", "broad"):
                strategy = "macro_hybrid"

        facts = []
        documents = []

        if strategy == "facts_only":
            facts = self._time_sqlite(
                self._retrieve_facts, ticker, metrics, timeframe, top_k_facts,
            )

        elif strategy == "documents_only":
            documents = self._retrieve_documents(query, ticker, top_k_documents)

        elif strategy == "macro":
            facts = self._time_sqlite(self._retrieve_macro_facts, top_k_facts)
            documents = self._retrieve_documents(query, ticker=None, n_results=top_k_documents)

        elif strategy == "macro_hybrid":
            facts = self._time_sqlite(
                self._retrieve_facts, ticker, metrics, timeframe, top_k_facts,
            )
            facts.extend(self._time_sqlite(self._retrieve_macro_facts, top_k_facts))
            documents = self._retrieve_documents(query, ticker, top_k_documents)

        elif strategy == "hybrid":
            facts = self._time_sqlite(
                self._retrieve_facts, ticker, metrics, timeframe, top_k_facts,
            )
            documents = self._retrieve_documents(query, ticker, top_k_documents)

        elif strategy == "comparison":
            # Multi-ticker: extract all tickers from query
            tickers = self._extract_all_tickers(query)
            for t in tickers:
                t_facts = self._time_sqlite(
                    self._retrieve_facts, t, metrics, timeframe, top_k_facts // len(tickers),
                )
                facts.extend(t_facts)
                t_docs = self._retrieve_documents(query, t, top_k_documents // len(tickers))
                documents.extend(t_docs)

        elif strategy == "broad":
            # No ticker detected — search everything
            documents = self._retrieve_documents(query, ticker=None, n_results=top_k_documents)
            facts = self._time_sqlite(self._retrieve_all_facts, top_k_facts)

        if question_type == "projection" and ticker:
            facts = self._time_sqlite(self._merge_projection_facts, ticker, facts)

        logger.info(
            "Retrieval strategy=%s ticker=%s: %d facts, %d documents",
            strategy, ticker, len(facts), len(documents),
        )

        return {
            "facts": facts,
            "documents": documents,
            "ticker": ticker,
            "strategy": strategy,
            "retrieval_strategy": self._doc_retrieval_strategy,
            "timings": {k: round(v, 1) for k, v in self._timings.items()},
        }

    def _time_sqlite(self, func, *args, **kwargs):
        """Run a SQLite-backed retrieval function and accumulate elapsed time."""
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self._timings["sqlite"] += (time.perf_counter() - start) * 1000

    def _search_chroma(self, *args, **kwargs) -> list[dict]:
        """Run Chroma search and accumulate embedding/vector-store timings."""
        start = time.perf_counter()
        results = self.store.chroma.search(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        search_timings = getattr(self.store.chroma, "last_search_timings", {}) or {}
        embedding_ms = float(search_timings.get("embedding", 0.0) or 0.0)
        chroma_ms = float(search_timings.get("chroma", 0.0) or 0.0)
        if embedding_ms == 0.0 and chroma_ms == 0.0:
            chroma_ms = elapsed_ms
        self._timings["embedding"] += embedding_ms
        self._timings["chroma"] += chroma_ms
        return results

    # ── Strategy Selection ────────────────────────────

    def _select_strategy(self, question_type: str, ticker: Optional[str],
                         metrics: list[str]) -> str:
        """Select the retrieval strategy based on intent."""
        if question_type == "comparison":
            return "comparison"
        if question_type == "projection":
            return "hybrid" if ticker else "broad"
        if question_type in ("sentiment", "news"):
            return "documents_only"
        if question_type == "risk":
            return "documents_only"
        if question_type == "fact_lookup" and metrics and ticker:
            return "facts_only"
        if question_type == "fact_lookup" and ticker:
            return "hybrid"
        if ticker is None and self._is_macro_question(metrics, question_type):
            return "macro"  # NEW: macro-only strategy
        if ticker and self._is_macro_question(metrics, question_type):
            return "macro_hybrid"  # NEW: company + macro
        if question_type == "trend" and ticker:
            return "hybrid"
        if ticker:
            return "hybrid"
        return "broad"

    def _is_macro_question(self, metrics: list[str], question_type: str) -> bool:
        """Detect if a question is about macro-economic topics."""
        if question_type in ("sentiment", "news"):
            return False
        # Check if any extracted metric maps to FRED indicators
        macro_metrics = {"gdp", "inflation", "cpi", "interest_rates",
                         "unemployment", "yield"}
        if metrics and any(m in macro_metrics for m in metrics):
            return True
        # The question text has macro keywords — this info comes
        # from the intent parser which we don't have here directly,
        # but the higher-level query method can pass it
        return False

    def _query_mentions_macro(self, query: str) -> bool:
        """Return True if the query text mentions any macro-economic keyword."""
        lowered = query.lower()
        return any(keyword in lowered for keyword in self.MACRO_KEYWORDS)

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

    def _merge_projection_facts(self, ticker: str, facts: list[dict]) -> list[dict]:
        """Add full estimate rows for projection queries, failing per metric."""
        by_metric = {
            fact.get("metric"): dict(fact)
            for fact in facts
            if fact.get("metric")
        }
        for metric in PROJECTION_METRICS:
            try:
                row = self.store.get_fundamental(ticker, metric)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Projection fact lookup failed for %s %s: %s",
                    ticker,
                    metric,
                    e,
                )
                continue
            if row and row.get("source_type") == "estimates":
                by_metric[metric] = dict(row)
        return list(by_metric.values())

    def _retrieve_macro_facts(self, limit: int = 15) -> list[dict]:
        """Retrieve macro-economic facts from FRED data in SQLite."""
        macro_metrics = [
            "GDP", "FEDFUNDS", "CPIAUCSL", "UNRATE",
            "DGS10", "T10Y2Y", "UMCSENT",
        ]
        batch = self.store.get_fundamentals_batch("MACRO", metrics=macro_metrics)
        facts = []
        for metric, value in batch.items():
            if value is not None:
                facts.append({
                    "metric": metric,
                    "value": value,
                    "ticker": "MACRO",
                    "source_type": "fred",
                })
        # Fallback: query recent facts from MACRO
        if len(facts) < limit:
            recent = self.store.sqlite.search_facts(ticker="MACRO", limit=limit)
            for r in recent:
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

    def retrieve_documents(self, query: str, ticker: Optional[str],
                           n_results: int) -> list[dict]:
        """Public document retrieval (hybrid + re-rank per config).

        Used by ``/search`` so it exposes ``fusion_score`` / ``rerank_score``.
        """
        return self._retrieve_documents(query, ticker, n_results)

    def _retrieve_documents(self, query: str, ticker: Optional[str],
                            n_results: int) -> list[dict]:
        """Retrieve relevant documents — hybrid (vector+BM25+RRF, optional
        re-rank) when enabled, else the original vector-only path.

        ``n_results`` is the final count returned. The hybrid path retrieves
        ``rerank_candidates`` broadly, fuses, then re-ranks / truncates to
        ``n_results``.
        """
        if not self.config.enable_lexical:
            start = time.perf_counter()
            results = self.store.search(query=query, n_results=n_results,
                                        ticker=ticker)
            elapsed_ms = (time.perf_counter() - start) * 1000
            chroma = getattr(self.store, "chroma", None)
            search_timings = getattr(chroma, "last_search_timings", {}) or {}
            self._timings["embedding"] += float(search_timings.get("embedding", 0.0) or 0.0)
            self._timings["chroma"] += float(search_timings.get("chroma", elapsed_ms) or 0.0)
            return results.get("documents", [])
        return self._retrieve_documents_hybrid(query, ticker, n_results)

    def _retrieve_documents_hybrid(self, query: str, ticker: Optional[str],
                                   n_results: int) -> list[dict]:
        """Vector + BM25 → RRF → (optional) cross-encoder re-rank → top n."""
        candidates = max(self.config.rerank_candidates, n_results)
        vfilter = {"ticker": ticker} if ticker else None
        broad_n = max(candidates // 2, n_results)

        # 1. Vector channel (ranked by cosine). Mirror store.search: ticker-
        #    filtered first, then a broad (unfiltered) pass appended (deduped)
        #    so false-positive tickers (e.g. "P", "GDP") still surface docs.
        vector_hits = self._search_chroma(
            query=query, n_results=candidates, filter_dict=vfilter,
        )
        if ticker:
            seen = {h["id"] for h in vector_hits}
            for h in self._search_chroma(query=query, n_results=broad_n,
                                         filter_dict=None):
                if h["id"] not in seen:
                    vector_hits.append(h)
                    seen.add(h["id"])

        # 2. Lexical (BM25) channel — same ticker + broad shape; best-effort.
        lexical_hits: list[dict] = []
        try:
            lexical_hits = self.lexical.search(query=query, k=candidates,
                                                where=vfilter)
            if ticker:
                seen = {h["id"] for h in lexical_hits}
                for h in self.lexical.search(query=query, k=broad_n, where=None):
                    if h["id"] not in seen:
                        lexical_hits.append(h)
                        seen.add(h["id"])
        except Exception as e:  # noqa: BLE001 - never let lexical break a query
            logger.warning("Lexical search failed, vector-only fallback: %s", e)

        # 3. RRF fuse.
        fused = rrf_fuse(vector_hits, lexical_hits, k=self.config.rrf_k)

        # 4. Hydrate fused ids to full doc dicts (text + metadata).
        by_id: dict[str, dict] = {}
        for h in vector_hits:
            by_id.setdefault(h["id"], h)
        for h in lexical_hits:
            by_id.setdefault(h["id"], h)
        fused_docs: list[dict] = []
        for doc_id, rrf_score in fused:
            d = dict(by_id.get(doc_id, {"id": doc_id, "document": "",
                                        "metadata": {}}))
            d["fusion_score"] = float(rrf_score)
            fused_docs.append(d)

        # Record the strategy actually used (vector if lexical produced nothing).
        if lexical_hits:
            self._doc_retrieval_strategy = "hybrid"
        else:
            self._doc_retrieval_strategy = "vector"

        # 5. Re-rank or truncate to the final count.
        if self.config.enable_reranker and fused_docs:
            try:
                fused_docs = self.reranker.rerank(
                    query, fused_docs, top_n=n_results,
                )
                self._doc_retrieval_strategy = "hybrid+rerank"
            except Exception as e:  # noqa: BLE001 - fallback to fused order
                logger.warning("Reranker failed, using fused order: %s", e)
                fused_docs = fused_docs[:n_results]
        else:
            fused_docs = fused_docs[:n_results]

        return fused_docs

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

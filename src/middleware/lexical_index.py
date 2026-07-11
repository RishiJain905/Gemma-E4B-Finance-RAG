"""
src/middleware/lexical_index.py
BM25 lexical (keyword) retrieval channel — Phase 2.1.2.1.

Indexes the ChromaDB document corpus by id + text with BM25Okapi so exact
tokens (tickers, metric names, product names like "MI300X", "10-Q", "CRWD")
that vector search misses still surface. Fused with vector hits via RRF in
the retriever.

Pure-Python (``rank_bm25``) — no service dependency. Built lazily from
``Store.chroma.iter_documents()`` and cached; rebuilt when the corpus count
changes (so it stays fresh after ingestion) or on explicit ``refresh()``.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lowercase + keep alphanumeric runs. "MI300X" -> "mi300x" (one token),
# "10-Q" -> "10","q", "CRWD" -> "crwd". Tickers and product codes survive.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumerics, keeping tickers intact."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class LexicalIndex:
    """BM25 index over the ChromaDB document corpus."""

    def __init__(self, store):
        self.store = store
        self._bm25 = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._token_sets: list[set[str]] = []
        self._count_at_build: Optional[int] = None

    # ── build / freshness ──────────────────────────────

    def _build(self) -> None:
        from rank_bm25 import BM25Okapi

        ids, texts, metas = self.store.chroma.iter_documents()
        self._ids, self._texts, self._metas = ids, texts, metas
        tokenized = [tokenize(t) for t in texts]
        self._token_sets = [set(toks) for toks in tokenized]
        if not tokenized or not any(tokenized):
            self._bm25 = None
            self._count_at_build = 0
            logger.warning("LexicalIndex: empty/blank corpus, BM25 disabled")
            return
        self._bm25 = BM25Okapi(tokenized)
        try:
            self._count_at_build = self.store.chroma.count()
        except Exception:  # noqa: BLE001
            self._count_at_build = len(ids)
        logger.info("LexicalIndex built over %d documents", len(ids))

    def _ensure(self) -> None:
        """Lazy build + freshness check (rebuild if the corpus count changed)."""
        if self._bm25 is None:
            self._build()
            return
        try:
            cur = self.store.chroma.count()
        except Exception:  # noqa: BLE001 - stay safe; never block a query
            return
        if self._count_at_build is None or cur != self._count_at_build:
            logger.info("LexicalIndex stale (count %s -> %s), rebuilding",
                        self._count_at_build, cur)
            self._build()

    def refresh(self) -> None:
        """Force a rebuild (call after ingestion / scheduler runs)."""
        self._bm25 = None
        self._build()

    @property
    def ready(self) -> bool:
        return self._bm25 is not None

    # ── search ─────────────────────────────────────────

    def search(self, query: str, k: int = 10,
               where: Optional[dict] = None) -> list[dict]:
        """Return up to ``k`` docs ranked by BM25 score (highest first).

        Args:
            query: the natural-language query.
            k: max results.
            where: optional metadata filter, e.g. {"ticker": "NVDA"}.

        Returns dicts shaped like ChromaStore.search: {id, document, metadata,
        score}, only docs that share at least one query token (ranked by BM25
        score). Empty list if the index is empty / failed to build.
        """
        self._ensure()
        if self._bm25 is None or not self._ids:
            return []
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(list(q_tokens))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        out: list[dict] = []
        for idx, score in ranked:
            # Keep docs that actually share a token with the query. BM25Okapi
            # can yield negative scores (negative IDF for terms in most docs),
            # so filtering on score sign would drop real matches; use overlap.
            doc_tokens = self._token_sets[idx] if idx < len(self._token_sets) else set()
            if not (doc_tokens & q_tokens):
                continue
            meta = self._metas[idx] if idx < len(self._metas) else {}
            if where and not all(meta.get(mk) == mv for mk, mv in where.items()):
                continue
            out.append({
                "id": self._ids[idx],
                "document": self._texts[idx] if idx < len(self._texts) else "",
                "metadata": meta,
                "score": float(score),
            })
            if len(out) >= k:
                break
        return out
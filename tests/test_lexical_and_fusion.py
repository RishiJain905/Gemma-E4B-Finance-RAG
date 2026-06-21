"""
tests/test_lexical_and_fusion.py
Phase 2.1.2.1 — BM25 lexical index + RRF fusion.

Unit tests stub the chroma corpus (no live embedding endpoint needed). A
corpus-building test against real chroma is marked ``integration``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.middleware.lexical_index import LexicalIndex, tokenize
from src.middleware.retriever import Retriever, rrf_fuse


# ── Fakes ──────────────────────────────────────────────────────────────

class FakeChroma:
    """Minimal chroma stand-in: iter_documents + search + count."""

    def __init__(self, docs: list[tuple[str, str, dict]]):
        # docs: (id, text, metadata)
        self._docs = docs

    def iter_documents(self, where=None, limit=None):
        ids, texts, metas = [], [], []
        for did, text, meta in self._docs:
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            ids.append(did); texts.append(text); metas.append(meta)
        if limit is not None:
            ids, texts, metas = ids[:limit], texts[:limit], metas[:limit]
        return ids, texts, metas

    def count(self):
        return len(self._docs)

    def search(self, query, n_results=5, filter_dict=None):
        # Predictable vector stand-in: return docs in corpus order that match
        # the ticker filter, each with a fake distance.
        out = []
        for did, text, meta in self._docs:
            if filter_dict and not all(meta.get(k) == v for k, v in filter_dict.items()):
                continue
            out.append({"id": did, "document": text, "metadata": meta,
                        "distance": float(len(out))})
            if len(out) >= n_results:
                break
        return out


class FakeStore:
    def __init__(self, docs):
        self.chroma = FakeChroma(docs)


# ── Tokenizer ──────────────────────────────────────────────────────────

def test_tokenize_keeps_tickers_and_splits_punctuation():
    assert tokenize("MI300X 10-Q CRWD revenue") == ["mi300x", "10", "q", "crwd", "revenue"]
    assert tokenize("") == []
    assert tokenize("P/E ratio!") == ["p", "e", "ratio"]


# ── BM25 ───────────────────────────────────────────────────────────────

def test_bm25_exact_token_ranks_top():
    docs = [
        ("d1", "NVIDIA reported record datacenter growth and GPU shipments.", {"ticker": "NVDA"}),
        ("d2", "AMD launched the new MI300X accelerator for AI workloads.", {"ticker": "AMD"}),
        ("d3", "Apple announced a buyback program.", {"ticker": "AAPL"}),
    ]
    lx = LexicalIndex(FakeStore(docs))
    hits = lx.search("MI300X", k=3)
    assert hits and hits[0]["id"] == "d2"   # exact token "mi300x" only in d2
    assert all(h["score"] > 0 for h in hits)


def test_bm25_filters_by_metadata_where():
    docs = [
        ("a", "risk factors competition", {"ticker": "NVDA"}),
        ("b", "risk factors competition", {"ticker": "AMD"}),
    ]
    lx = LexicalIndex(FakeStore(docs))
    hits = lx.search("risk factors", k=5, where={"ticker": "AMD"})
    assert [h["id"] for h in hits] == ["b"]


def test_bm25_empty_corpus_returns_empty():
    lx = LexicalIndex(FakeStore([]))
    assert lx.search("anything", k=5) == []
    assert lx.ready is False


def test_lexical_search_result_shape():
    docs = [("d1", "NVIDIA revenue growth", {"ticker": "NVDA"})]
    lx = LexicalIndex(FakeStore(docs))
    hits = lx.search("revenue", k=5)
    assert hits
    h = hits[0]
    assert set(h.keys()) == {"id", "document", "metadata", "score"}
    assert h["document"] == "NVIDIA revenue growth"


# ── RRF fusion ─────────────────────────────────────────────────────────

def test_rrf_orders_by_combined_rank():
    # 'b' and 'a' appear in both channels → beat 'c'/'d' which appear once.
    v = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    l = [{"id": "b"}, {"id": "d"}, {"id": "a"}]
    fused = rrf_fuse(v, l, k=60)
    ids = [i for i, _ in fused]
    assert ids[:2] == ["b", "a"]   # both-channels rank above single-channel
    # b is rank-1 in lexical and rank-2 in vector; a is rank-1 in vector and
    # rank-3 in lexical. b's combined reciprocal rank is higher.
    assert ids[0] == "b"


def test_rrf_handles_disjoint_lists():
    v = [{"id": "x"}, {"id": "y"}]
    l = [{"id": "z"}, {"id": "w"}]
    fused = dict(rrf_fuse(v, l, k=60))
    assert {"x", "y", "z", "w"} <= set(fused)   # all unique ids appear


def test_rrf_empty_inputs():
    assert rrf_fuse([], [], k=60) == []


# ── Retriever fallback ─────────────────────────────────────────────────

def test_retriever_vector_only_fallback_on_empty_lexical(monkeypatch):
    """Empty lexical corpus → documents come from the vector channel only,
    in vector order (no exception, no reranker dependency)."""
    docs = [("v1", "NVDA risk factors", {"ticker": "NVDA"}),
            ("v2", "NVDA competition", {"ticker": "NVDA"})]
    store = FakeStore(docs)
    cfg = MagicMock()
    cfg.enable_lexical = True
    cfg.enable_reranker = False
    cfg.rrf_k = 60
    cfg.rerank_candidates = 10
    ret = Retriever(store=store, config=cfg)   # type: ignore[arg-type]
    # LexicalIndex over an empty corpus would return []; here the corpus is
    # non-empty so force lexical to return [] to simulate a build failure.
    monkeypatch.setattr(ret.lexical, "search", lambda **kw: [])
    intent = {"ticker": "NVDA", "metrics": [], "question_type": "risk",
              "timeframe": None, "original_question": "risks?"}
    out = ret.retrieve(query="risks?", intent=intent, top_k_documents=2, top_k_facts=10)
    assert out["retrieval_strategy"] == "vector"   # lexical produced nothing
    ids = [d["id"] for d in out["documents"]]
    assert ids == ["v1", "v2"]   # vector order preserved


def test_retriever_hybrid_uses_both_channels(monkeypatch):
    """With lexical hits present, retrieval_strategy is 'hybrid' and fused
    docs carry a fusion_score."""
    docs = [("v1", "NVIDIA datacenter growth", {"ticker": "NVDA"}),
            ("v2", "NVIDIA GPU shipments", {"ticker": "NVDA"})]
    store = FakeStore(docs)
    cfg = MagicMock()
    cfg.enable_lexical = True
    cfg.enable_reranker = False
    cfg.rrf_k = 60
    cfg.rerank_candidates = 10
    ret = Retriever(store=store, config=cfg)   # type: ignore[arg-type]
    intent = {"ticker": "NVDA", "metrics": [], "question_type": "risk",
              "timeframe": None, "original_question": "growth?"}
    out = ret.retrieve(query="datacenter growth", intent=intent,
                       top_k_documents=2, top_k_facts=10)
    assert out["retrieval_strategy"] == "hybrid"
    assert out["documents"]
    assert all("fusion_score" in d for d in out["documents"])


@pytest.mark.integration
@pytest.mark.network
def test_lexical_index_real_corpus():
    """Build the BM25 index over the real chroma corpus (needs :8087 up for
    the Store init). Skips if the embedding endpoint is unreachable."""
    import httpx
    try:
        r = httpx.get("http://127.0.0.1:8087/health", timeout=3)
        if r.status_code != 200:
            pytest.skip("model/embedding server not on :8087")
    except Exception:
        pytest.skip("model/embedding server not on :8087")
    from src.storage.store import Store
    from src.middleware.config import MiddlewareConfig
    cfg = MiddlewareConfig()
    store = Store(embedding_endpoint=cfg.embedding_endpoint)
    lx = LexicalIndex(store)
    hits = lx.search("risk factors", k=5, where={"ticker": "NVDA"})
    assert lx.ready
    # The corpus is non-empty; a common-term query should find something.
    assert isinstance(hits, list)
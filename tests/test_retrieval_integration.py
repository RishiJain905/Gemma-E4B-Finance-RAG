"""
tests/test_retrieval_integration.py
Phase 2.1.2.3 — Integration of the lexical channel + re-ranker into the live
``/query`` and ``/search`` paths.

Marked ``integration``. The vector/lexical/reranker/model stages are patched
so the tests exercise the wiring without needing the embedding/model server.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.middleware.app import app


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def hybrid_app(monkeypatch):
    """TestClient with the full hybrid+rerank path stubbed at every stage."""
    from src.middleware import app as m
    with TestClient(app) as client:
        ret = m.retriever
        assert ret is not None, "retriever not built on startup"
        monkeypatch.setattr(ret.config, "enable_lexical", True)
        monkeypatch.setattr(ret.config, "enable_reranker", True)

        vhits = [
            {"id": "v1", "document": "NVIDIA datacenter growth surged.",
             "metadata": {"ticker": "NVDA"}, "distance": 0.10},
            {"id": "v2", "document": "NVIDIA GPU shipments rose.",
             "metadata": {"ticker": "NVDA"}, "distance": 0.20},
        ]
        monkeypatch.setattr(ret.store.chroma, "search", MagicMock(return_value=vhits))

        lhits = [
            {"id": "v1", "document": "NVIDIA datacenter growth surged.",
             "metadata": {"ticker": "NVDA"}, "score": 3.0},
            {"id": "l1", "document": "NVIDIA 10-Q risk factors.",
             "metadata": {"ticker": "NVDA"}, "score": 2.0},
        ]
        lex_mock = MagicMock(return_value=lhits)
        monkeypatch.setattr(ret.lexical, "search", lex_mock)

        rr = MagicMock()
        def _rerank(query, docs, top_n):
            for i, d in enumerate(docs):
                d["rerank_score"] = round(1.0 - i * 0.1, 3)
            return docs[:top_n]
        rr.rerank.side_effect = _rerank
        ret._reranker = rr  # type: ignore[attr-defined]

        monkeypatch.setattr(m, "_check_model_health", AsyncMock(return_value=True))
        monkeypatch.setattr(m, "_call_model",
                            AsyncMock(return_value=("answer [Source: yfinance/NVDA]", [])))

        yield client, ret, lex_mock, rr


# ── /query ─────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_query_hybrid_path(hybrid_app):
    """/query runs vector → lexical → RRF → rerank and reports hybrid+rerank."""
    client, ret, lex_mock, rr = hybrid_app
    resp = client.post("/query", json={"question": "NVIDIA risks?", "refresh": False,
                                       "ticker": "NVDA"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieval_strategy"] == "hybrid+rerank"
    assert data["documents_used"] >= 1
    assert data["model_available"] is True
    assert "answer" in data and data["answer"]
    # Both channels and the reranker were actually called.
    assert ret.store.chroma.search.called
    assert lex_mock.called
    assert rr.rerank.called


@pytest.mark.integration
def test_strategy_flag_reports_vector_when_lexical_empty(monkeypatch):
    """When the lexical channel returns nothing, the strategy flag is 'vector'."""
    from src.middleware import app as m
    with TestClient(app) as client:
        ret = m.retriever
        monkeypatch.setattr(ret.config, "enable_lexical", True)
        monkeypatch.setattr(ret.config, "enable_reranker", False)
        monkeypatch.setattr(
            ret.store.chroma, "search",
            MagicMock(return_value=[{"id": "v1", "document": "doc",
                                     "metadata": {"ticker": "NVDA"}, "distance": 0.1}]))
        monkeypatch.setattr(ret.lexical, "search", MagicMock(return_value=[]))  # empty
        monkeypatch.setattr(m, "_check_model_health", AsyncMock(return_value=True))
        monkeypatch.setattr(m, "_call_model",
                            AsyncMock(return_value=("ans [Source: yfinance/NVDA]", [])))
        resp = client.post("/query", json={"question": "x?", "refresh": False,
                                           "ticker": "NVDA"})
        assert resp.status_code == 200
        assert resp.json()["retrieval_strategy"] == "vector"


# ── /search ────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_search_includes_rerank_score(hybrid_app):
    """/search exposes rerank_score on documents."""
    client, _ret, _lex, _rr = hybrid_app
    resp = client.post("/search", json={"query": "NVIDIA growth", "ticker": "NVDA",
                                        "n_results": 5})
    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert docs, "expected documents from /search"
    assert any("rerank_score" in d for d in docs)


# ── Lexical index freshness ────────────────────────────────────────────

@pytest.mark.integration
def test_index_refresh_after_refresh_endpoint(monkeypatch):
    """POST /refresh rebuilds the lexical index (LexicalIndex.refresh called)."""
    from src.middleware import app as m
    with TestClient(app) as client:
        ret = m.retriever
        # Avoid real ingestion; just exercise the rebuild hook.
        monkeypatch.setattr(m, "_refresh_ticker_sources",
                            MagicMock(return_value=(["yfinance_fundamentals"], [])))
        refresh_mock = MagicMock()
        monkeypatch.setattr(ret._lexical, "refresh", refresh_mock)  # type: ignore[attr-defined]
        # /refresh with no sources would refresh all stale; the patched
        # _refresh_ticker_sources makes that a no-op.
        resp = client.post("/refresh/NVDA", json={})
        assert resp.status_code == 200
        assert refresh_mock.called, "LexicalIndex.refresh was not called after /refresh"


# ── Regression: features disabled preserves vector-only behavior ──────

@pytest.mark.integration
def test_features_disabled_preserves_vector_only(monkeypatch):
    """With lexical+reranker disabled, retrieval_strategy is 'vector' and the
    original store.search path is used (no lexical/reranker calls)."""
    from src.middleware import app as m
    with TestClient(app) as client:
        ret = m.retriever
        monkeypatch.setattr(ret.config, "enable_lexical", False)
        monkeypatch.setattr(ret.config, "enable_reranker", False)
        # Spy on store.search (the vector-only path) instead of real chroma.
        store_search_mock = MagicMock(return_value={
            "documents": [{"id": "v1", "document": "doc", "metadata": {"ticker": "NVDA"}}],
            "facts": [], "ticker": "NVDA"})
        monkeypatch.setattr(ret.store, "search", store_search_mock)
        lex_search_mock = MagicMock(return_value=[])
        monkeypatch.setattr(ret.lexical, "search", lex_search_mock)
        monkeypatch.setattr(m, "_check_model_health", AsyncMock(return_value=True))
        monkeypatch.setattr(m, "_call_model",
                            AsyncMock(return_value=("ans [Source: yfinance/NVDA]", [])))
        resp = client.post("/query", json={"question": "x?", "refresh": False,
                                           "ticker": "NVDA"})
        assert resp.status_code == 200
        assert resp.json()["retrieval_strategy"] == "vector"
        assert store_search_mock.called
        assert not lex_search_mock.called  # lexical channel skipped when disabled
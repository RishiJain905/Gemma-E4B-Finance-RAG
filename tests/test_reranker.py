# tests/test_reranker.py
# Unit tests for the cross-encoder / LLM re-ranker — Phase 2.1.2.2
#
# The re-ranker model is STUBBED in every unit test: ``_score`` (and where
# needed ``_load_model``) is monkeypatched so no model downloads and no
# network calls happen. The one ``@pytest.mark.live @pytest.mark.slow`` test
# loads the real cross-encoder and is skipped by default (needs ``--live``).

import pytest

from src.middleware.reranker import Reranker, _parse_score


class TestReranker:
    """Tests for the Reranker module."""

    def test_import(self):
        """Reranker imports successfully."""
        assert Reranker is not None

    # ── Constructor / defaults ─────────────────────────

    def test_defaults_without_config(self):
        """Default construction uses cross-encoder defaults and lazy model."""
        r = Reranker()
        assert r.backend == "cross-encoder"
        assert r.model_id == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert r.llm_endpoint == "http://127.0.0.1:8087/v1/chat/completions"
        assert r.llm_model == "tracealchemy"
        assert r._model is None  # lazy

    def test_backend_normalized_lowercase(self):
        """Backend is normalized to lowercase."""
        assert Reranker(backend="LLM").backend == "llm"
        assert Reranker(backend="Cross-Encoder").backend == "cross-encoder"

    def test_explicit_args_override_config(self):
        """Explicit non-None args override config values."""
        from src.middleware.config import MiddlewareConfig
        cfg = MiddlewareConfig()
        r = Reranker(config=cfg, backend="llm")
        assert r.backend == "llm"
        assert r.model_id == cfg.reranker_model  # not overridden
        assert r.llm_model == cfg.model_name      # read from config

    def test_reads_from_config(self):
        """With config and no explicit args, values come from the config."""
        from src.middleware.config import MiddlewareConfig
        cfg = MiddlewareConfig()
        r = Reranker(config=cfg)
        assert r.backend == cfg.reranker_backend
        assert r.model_id == cfg.reranker_model
        assert r.llm_endpoint == cfg.llama_endpoint
        assert r.llm_model == cfg.model_name

    # ── rerank() ordering ──────────────────────────────

    def test_rerank_orders_by_score(self):
        """rerank returns docs sorted by score desc, truncated to top_n,
        each carrying a float rerank_score."""
        r = Reranker()
        # Stub the scoring seam — no model load, no network.
        r._score = lambda query, docs: [0.1, 0.9, 0.5]  # noqa: E731
        docs = [
            {"document": "doc one"},
            {"document": "doc two"},
            {"document": "doc three"},
        ]
        out = r.rerank("query", docs, top_n=2)
        assert len(out) == 2
        # 0.9 > 0.5 > 0.1 → "doc two" then "doc three"
        assert out[0]["document"] == "doc two"
        assert out[0]["rerank_score"] == 0.9
        assert out[1]["document"] == "doc three"
        assert out[1]["rerank_score"] == 0.5
        # scores are floats
        assert isinstance(out[0]["rerank_score"], float)

    def test_rerank_empty_docs(self):
        """rerank on an empty doc list returns [] without calling the model."""
        r = Reranker()
        called = {"flag": False}

        def _fail(query, docs):
            called["flag"] = True
            return [0.5]

        r._score = _fail
        assert r.rerank("query", [], top_n=5) == []
        assert called["flag"] is False  # short-circuit before scoring

    def test_reranker_load_failure_falls_back(self):
        """If _score raises, rerank falls back to fused order with
        rerank_score=None and never raises."""
        r = Reranker()

        def _boom(query, docs):
            raise RuntimeError("model down")

        r._score = _boom
        docs = [{"document": "a"}, {"document": "b"}, {"document": "c"}]
        out = r.rerank("query", docs, top_n=2)
        assert len(out) == 2
        # Original order preserved (truncated to top_n).
        assert out[0]["document"] == "a"
        assert out[1]["document"] == "b"
        assert out[0]["rerank_score"] is None
        assert out[1]["rerank_score"] is None

    def test_reranker_load_failure_preserves_full_order_when_top_n_large(self):
        """Fallback returns up to top_n docs in original order."""
        r = Reranker()

        def _raise(query, docs):
            raise RuntimeError("model down")

        r._score = _raise
        docs = [{"document": str(i)} for i in range(4)]
        out = r.rerank("query", docs, top_n=10)
        assert [d["document"] for d in out] == ["0", "1", "2", "3"]
        assert all(d["rerank_score"] is None for d in out)

    # ── LLM backend (mocked) ───────────────────────────

    def test_rerank_llm_backend_mocked(self):
        """The llm backend path is wired: with _score stubbed it orders docs
        correctly without touching the network."""
        r = Reranker(backend="llm")
        assert r.backend == "llm"
        r._score = lambda query, docs: [0.3, 0.8, 0.1]  # noqa: E731
        docs = [{"text": "x"}, {"text": "y"}, {"text": "z"}]
        out = r.rerank("query", docs, top_n=3)
        assert [d["text"] for d in out] == ["y", "x", "z"]
        assert out[0]["rerank_score"] == 0.8
        assert out[1]["rerank_score"] == 0.3
        assert out[2]["rerank_score"] == 0.1

    # ── _doc_text ──────────────────────────────────────

    def test_doc_text_extracts_fields(self):
        """_doc_text pulls document, else text, else content, else empty."""
        assert Reranker._doc_text({"document": "doc"}) == "doc"
        assert Reranker._doc_text({"text": "txt"}) == "txt"
        assert Reranker._doc_text({"content": "cnt"}) == "cnt"
        # Precedence: document > text > content.
        assert Reranker._doc_text({"document": "d", "text": "t"}) == "d"
        assert Reranker._doc_text({"text": "t", "content": "c"}) == "t"
        assert Reranker._doc_text({}) == ""
        # Falsy values fall through to the next key.
        assert Reranker._doc_text({"document": "", "text": "t"}) == "t"

    # ── refresh() ──────────────────────────────────────

    def test_refresh_resets_model(self):
        """refresh() clears the cached model so the next use reloads."""
        r = Reranker()
        r._model = "stub"  # pretend a model is loaded
        r.refresh()
        assert r._model is None

    # ── _parse_score ───────────────────────────────────

    @pytest.mark.parametrize("text,expected", [
        ("blah\nScore: 0.83", 0.83),
        ("Score: 0.95 extra", 0.95),
        ("score = 0.5", 0.5),
        ("Score: 8", 0.8),           # 0–10 scale normalized
        ("Score: 10", 1.0),          # 0–10 scale top
        ("Score: 2", 0.2),           # 0–10 scale bottom
        ("Score: 0", 0.0),
        ("Score: 1.5", None),        # (1, 2) ambiguous → None
        ("Score: 12", None),         # > 10 → None
        ("no number here", None),
        ("", None),
        (None, None),
        # Fallback to last number when no Score: tag is present.
        ("the answer is 0.7 overall", 0.7),
        ("I rate this 7", 0.7),        # last number 7 → 0–10 scale → 0.7
        ("got a 9", 0.9),              # last number 9 → 0–10 scale → 0.9
        ("rating 8 out of 10", 1.0),   # last number is 10 → 1.0
    ])
    def test_parse_score_formats(self, text, expected):
        assert _parse_score(text) == expected

    def test_parse_score_clamps_to_unit_range(self):
        """Result is clamped to [0, 1]."""
        assert _parse_score("Score: 0.0") == 0.0
        assert _parse_score("Score: 1.0") == 1.0

    # ── Live (skipped by default) ──────────────────────

    @pytest.mark.live
    @pytest.mark.slow
    def test_rerank_live_cross_encoder(self):
        """Load the real cross-encoder and rerank 3 tiny docs.

        Skipped unless ``--live`` is passed. A model-download failure skips
        cleanly rather than erroring.
        """
        reranker = Reranker()
        try:
            _ = reranker.model  # triggers _load_model (may download)
        except Exception as e:
            pytest.skip(f"cross-encoder model unavailable: {e}")

        docs = [
            {"document": "NVIDIA reported record datacenter revenue growth "
                         "driven by AI accelerator demand."},
            {"document": "The weekend weather forecast predicts rain "
                         "across the coastal region."},
            {"document": "Apple announced a new iPhone model at its "
                         "annual fall event."},
        ]
        out = reranker.rerank("What is NVIDIA datacenter revenue?", docs, top_n=3)
        assert len(out) == 3
        assert all(d["rerank_score"] is not None for d in out)
        # The NVIDIA doc should rank first for this query.
        assert "NVIDIA" in out[0]["document"]
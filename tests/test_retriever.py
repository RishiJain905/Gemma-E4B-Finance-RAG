# tests/test_retriever.py
# Unit tests for hybrid Retriever — Phase 1.5.3

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.storage.store import Store


@pytest.fixture
def mock_chroma():
    """Patch ChromaStore construction; yield the mock instance."""
    with patch("src.storage.store.ChromaStore") as mock_cls:
        instance = MagicMock()
        instance.heartbeat.return_value = True
        instance.count.return_value = 0
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def store(mock_chroma, tmp_path: Path):
    """Store with real SQLite (tmp_path) and mocked Chroma."""
    return Store(db_path=tmp_path / "test.db", chroma_path=tmp_path / "chroma")


class TestRetriever:
    """Tests for the Retriever module."""

    def test_import(self):
        """Retriever imports successfully."""
        from src.middleware.retriever import Retriever
        assert Retriever is not None

    def test_init(self, store):
        """Default init creates retriever with store and config."""
        from src.middleware.retriever import Retriever
        retriever = Retriever(store=store)
        assert retriever.store is store
        assert retriever.config is not None

    # ── Strategy Selection ─────────────────────────────

    @pytest.mark.parametrize("qtype,ticker,metrics,expected_strategy", [
        ("fact_lookup", "NVDA", ["total_revenue"], "facts_only"),
        ("fact_lookup", "NVDA", [], "hybrid"),
        ("sentiment", "AMD", [], "documents_only"),
        ("news", "META", [], "documents_only"),
        ("risk", "CRWD", [], "documents_only"),
        ("comparison", "NVDA", [], "comparison"),
        ("trend", "NVDA", ["revenue"], "hybrid"),
        ("explanation", "NVDA", [], "hybrid"),
        ("general", "NVDA", [], "hybrid"),
        ("general", None, [], "broad"),
    ])
    def test_strategy_selection(self, qtype, ticker, metrics, expected_strategy, store):
        """Retrieval strategy is correctly selected based on intent."""
        from src.middleware.retriever import Retriever
        retriever = Retriever(store=store)
        strategy = retriever._select_strategy(qtype, ticker, metrics)
        assert strategy == expected_strategy, (
            f"Expected '{expected_strategy}' for ({qtype}, {ticker}, {metrics}), "
            f"got '{strategy}'"
        )

    def test_retrieve_empty_db(self, store):
        """Retrieve returns empty results when DB is empty."""
        from src.middleware.retriever import Retriever
        retriever = Retriever(store=store)
        result = retriever.retrieve(
            query="What is NVDA revenue?",
            intent={"ticker": "NVDA", "metrics": ["total_revenue"],
                     "question_type": "fact_lookup"},
        )
        assert result["facts"] == []
        assert result["documents"] == []
        assert result["ticker"] == "NVDA"
        assert result["strategy"] == "facts_only"

    def test_retrieve_with_seeded_data(self, store):
        """Retrieve returns facts when data exists in SQLite."""
        from src.middleware.retriever import Retriever

        # Seed some data
        store.save_fundamental("NVDA", "total_revenue", 26.0, "billion_usd",
                               "2026-Q1", "quarterly", "sec_10q")

        retriever = Retriever(store=store)
        result = retriever.retrieve(
            query="What is NVDA revenue?",
            intent={"ticker": "NVDA", "metrics": ["total_revenue"],
                     "question_type": "fact_lookup"},
        )
        assert len(result["facts"]) >= 1
        revenue_facts = [f for f in result["facts"] if f.get("metric") == "total_revenue"]
        assert len(revenue_facts) >= 1
        assert revenue_facts[0]["value"] == 26.0

    def test_multi_ticker_extraction(self, store):
        """Comparison queries extract multiple tickers."""
        from src.middleware.retriever import Retriever
        retriever = Retriever(store=store)
        tickers = retriever._extract_all_tickers("Compare NVDA and AMD")
        assert "NVDA" in tickers
        assert "AMD" in tickers

    def test_carried_intent_drives_retrieval_for_pronoun_followup(self, store):
        """A pronoun follow-up carries the entity so retrieval still finds it (2.2.2.2).

        The raw turn ("Why did it grow?") names no ticker, so the parsed intent
        alone retrieves nothing. compile_question carries NVDA from the grounded
        prior answer; retrieving with that effective intent returns the fact.
        """
        from src.middleware.conversation import compile_question
        from src.middleware.models import ChatTurn
        from src.middleware.retriever import Retriever

        store.save_fundamental("NVDA", "total_revenue", 26.0, "billion_usd",
                               "2026-Q1", "quarterly", "sec_10q")
        history = [
            ChatTurn(role="user", content="Show NVDA revenue for FY2025"),
            ChatTurn(role="assistant", content="an answer",
                     context={"grounding": "grounded", "ticker": "NVDA"}),
        ]
        compiled = compile_question("Why did it grow?", history)
        assert compiled.entity == "NVDA"

        retriever = Retriever(store=store)
        # Without the carried entity, retrieving for an unrelated ticker finds
        # nothing...
        raw = retriever.retrieve(
            query="Why did it grow?",
            intent={"ticker": "ZZZZ", "metrics": ["total_revenue"],
                    "question_type": "fact_lookup"},
        )
        assert raw["facts"] == []
        # ...but the effective (carried) intent recovers the fact.
        effective = retriever.retrieve(
            query=compiled.retrieval_query,
            intent={"ticker": compiled.entity, "metrics": compiled.metrics,
                    "question_type": "explanation"},
        )
        assert any(f.get("metric") == "total_revenue" for f in effective["facts"])

    def test_post_relevance_policy_prefers_sec_and_packs_duplicate_coverage(self, store):
        from src.middleware.retriever import Retriever

        retriever = Retriever(store=store)
        docs = [
            {
                "id": "vendor", "document": "Investor reaction to the notes financing.",
                "fusion_score": 0.032,
                "metadata": {
                    "ticker": "ORCL", "source": "finnhub",
                    "source_category": "news_vendor", "item_type": "news",
                    "event_type": "debt_raise", "event_id": "notes-1",
                    "published_at": "2026-07-10T14:00:00Z", "authority_tier": "provider",
                },
            },
            {
                "id": "sec", "document": "The filed prospectus for the notes financing.",
                "fusion_score": 0.032,
                "metadata": {
                    "ticker": "ORCL", "source": "sec",
                    "source_category": "regulatory_filing", "item_type": "sec_filing",
                    "event_type": "debt_raise", "event_id": "notes-1",
                    "published_at": "2026-07-10T13:00:00Z", "authority_tier": "direct_sec",
                },
            },
            {
                "id": "duplicate-a", "document": "Oracle launches notes.",
                "fusion_score": 0.031,
                "metadata": {
                    "ticker": "ORCL", "source": "massive",
                    "source_category": "news_vendor", "item_type": "news",
                    "event_type": "debt_raise", "event_id": "notes-1",
                    "syndicated_key": "same-story", "authority_tier": "provider",
                },
            },
            {
                "id": "duplicate-b", "document": "Oracle launches notes.",
                "fusion_score": 0.030,
                "metadata": {
                    "ticker": "ORCL", "source": "gdelt",
                    "source_category": "global_news", "item_type": "news",
                    "event_type": "debt_raise", "event_id": "notes-1",
                    "syndicated_key": "same-story", "authority_tier": "discovery",
                },
            },
        ]
        ranked = retriever._postprocess_documents(
            "Oracle debt financing", docs,
            {"ticker": "ORCL", "question_type": "news", "evidence_filters": {
                "event_type": "debt_raise"
            }},
            limit=4,
        )
        ids = [row["id"] for row in ranked]
        assert ids[:2] == ["sec", "vendor"]
        assert not ({"duplicate-a", "duplicate-b"} <= set(ids))

    def test_taxonomy_policy_flags_off_preserve_document_order_and_shape(self, store):
        from src.middleware.retriever import Retriever

        retriever = Retriever(store=store)
        retriever.config.enable_evidence_taxonomy = False
        retriever.config.enable_authority_ranking = False
        retriever.config.enable_duplicate_coverage_packing = False
        documents = [
            {"id": "vendor", "document": "secondary"},
            {"id": "sec", "document": "primary"},
        ]
        assert retriever._postprocess_documents(
            "query", documents, {}, limit=2,
        ) == documents


# ── 2.2.3.4 review D: retrieve_candidates channel ids are request-local ──

from types import SimpleNamespace as _NS  # noqa: E402


class _FakeSearchStore:
    """Minimal store whose vector search returns a fixed doc set."""

    def __init__(self, docs):
        self._docs = docs
        self.chroma = _NS(last_search_timings={})

    def search(self, query, n_results, ticker=None):
        return {"documents": [dict(d) for d in self._docs], "facts": []}


def _candidates_config():
    from src.middleware.config import MiddlewareConfig
    cfg = MiddlewareConfig()
    cfg.enable_lexical = False       # vector-only path keeps the test hermetic
    cfg.enable_reranker = False
    return cfg


def test_retrieve_candidates_channel_ids_are_request_local():
    from src.middleware.retriever import Retriever

    cfg = _candidates_config()
    intent = {"ticker": "NVDA", "question_type": "news", "metrics": []}

    r = Retriever(store=_FakeSearchStore(
        [{"id": "a1", "document": "x"}, {"id": "a2", "document": "y"}]), config=cfg)
    res_a = r.retrieve_candidates("q", intent, top_k_documents=5, top_k_facts=10)

    # A DIFFERENT store/result on the SAME retriever instance must not bleed the
    # previous call's channel ids (the bug was shared _last_* instance fields).
    r.store = _FakeSearchStore([{"id": "b1", "document": "z"}])
    res_b = r.retrieve_candidates("q", intent, top_k_documents=5, top_k_facts=10)

    assert res_a["vector_ids"] == ["a1", "a2"]
    assert res_a["lexical_ids"] == []
    assert res_b["vector_ids"] == ["b1"]
    # The ids came from the returned dict, not from instance state.
    assert not hasattr(r, "_last_vector_ids")
    assert not hasattr(r, "_last_lexical_ids")


def test_retrieve_omits_channel_ids_from_legacy_dict():
    from src.middleware.retriever import Retriever

    cfg = _candidates_config()
    intent = {"ticker": "NVDA", "question_type": "news", "metrics": []}
    r = Retriever(store=_FakeSearchStore([{"id": "a1", "document": "x"}]), config=cfg)

    legacy = r.retrieve("q", intent, top_k_documents=5, top_k_facts=10)
    # Legacy retrieve() shape is unchanged — no channel-id keys leak in.
    assert "vector_ids" not in legacy
    assert "lexical_ids" not in legacy


# ── CompanyFacts integration (Phase 2.2.5.3) ───────────────

class TestCompanyFactsMerge:
    """Authoritative SEC CompanyFacts prefer/merge into structured retrieval."""

    def _authoritative(self, value=26000000000.0, unit="USD", period="2025-12-31",
                       conflict=False, alternatives=None):
        return [{
            "metric": "total_revenue", "value": value, "value_text": str(value),
            "ticker": "NVDA", "period": period, "period_type": "annual",
            "unit": unit, "source_type": "sec_companyfacts",
            "source_url": "https://sec.gov/x", "as_of": "2026-07-01",
            "taxonomy": "us-gaap", "concept": "Revenues",
            "accession": "0001-25", "form": "10-K", "filed_at": "2026-02-01",
            "conflict": conflict,
        }]

    def test_disabled_source_is_noop(self, store):
        from src.middleware.retriever import Retriever
        store.save_fundamental("NVDA", "total_revenue", 25.0, "usd", "FY2025",
                               source_type="yfinance")
        store.companyfacts_evidence = MagicMock(return_value=[])
        r = Retriever(store=store)
        facts = r._retrieve_facts("NVDA", ["total_revenue"], None, 10)
        # Legacy fact preserved; no CompanyFacts rows added.
        assert any(f.get("source_type") in ("yfinance", "sqlite") for f in facts)
        assert not any(f.get("source_type") == "sec_companyfacts" for f in facts)

    def test_authoritative_fact_included(self, store):
        from src.middleware.retriever import Retriever
        store.companyfacts_evidence = MagicMock(return_value=self._authoritative())
        r = Retriever(store=store)
        facts = r._retrieve_facts("NVDA", ["total_revenue"], None, 10)
        auth = [f for f in facts if f.get("source_type") == "sec_companyfacts"]
        assert len(auth) == 1
        assert auth[0]["value"] == 26000000000.0
        assert auth[0]["concept"] == "Revenues"

    def test_conflicting_legacy_value_kept_separate(self, store):
        from src.middleware.retriever import Retriever
        # Legacy fundamental for the exact same metric+period, different value.
        store.save_fundamental("NVDA", "total_revenue", 30000000000.0, "usd",
                               "2025-12-31", source_type="yfinance")
        store.companyfacts_evidence = MagicMock(return_value=self._authoritative())
        r = Retriever(store=store)
        facts = r._retrieve_facts("NVDA", ["total_revenue"], None, 10)
        auth = [f for f in facts if f.get("source_type") == "sec_companyfacts"]
        disputed = [f for f in facts if f.get("conflict")]
        assert auth and disputed
        # Never averaged: both distinct values survive as separate items.
        values = {f["value"] for f in facts if f.get("metric") == "total_revenue"}
        assert 26000000000.0 in values and 30000000000.0 in values

    def test_agreeing_legacy_value_deduped(self, store):
        from src.middleware.retriever import Retriever
        store.save_fundamental("NVDA", "total_revenue", 26000000000.0, "usd",
                               "2025-12-31", source_type="yfinance")
        store.companyfacts_evidence = MagicMock(return_value=self._authoritative())
        r = Retriever(store=store)
        facts = r._retrieve_facts("NVDA", ["total_revenue"], None, 10)
        rev = [f for f in facts if f.get("metric") == "total_revenue"]
        # CompanyFacts wins; the duplicate legacy value is dropped.
        assert len(rev) == 1
        assert rev[0]["source_type"] == "sec_companyfacts"

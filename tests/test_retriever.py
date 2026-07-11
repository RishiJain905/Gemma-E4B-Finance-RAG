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

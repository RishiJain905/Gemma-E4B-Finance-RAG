"""
tests/test_middleware.py
Comprehensive pytest suite for Phase 1.5 FastAPI middleware.

Test coverage:
  1. IntentParser — ticker detection, metric extraction, question classification
  2. Retriever — strategy selection, fact/document retrieval
  3. PromptAugmenter — prompt assembly, formatting, truncation
  4. API Endpoints — /health, /query, /search
  5. Integration — end-to-end query flow with mocked model

Usage:
    pytest tests/test_middleware.py -v              # All tests (live skipped if :8087 down)
    pytest tests/test_middleware.py -v -m "not live"  # Offline only
    pytest tests/test_middleware.py -v -m live        # Live model on :8087

NOTE: Live model tests (marked @pytest.mark.live) require llama-server on :8087.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.storage.store import Store

MODEL_BASE = "http://127.0.0.1:8087"
HEALTH_URL = f"{MODEL_BASE}/health"


# ── Shared fixtures ───────────────────────────────────


@pytest.fixture
def mock_chroma():
    """Patch ChromaStore construction; yield the mock instance."""
    with patch("src.storage.store.ChromaStore") as mock_cls:
        instance = MagicMock()
        instance.heartbeat.return_value = True
        instance.count.return_value = 0
        instance.search.return_value = []
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def store(mock_chroma, tmp_path: Path):
    """Store with real SQLite (tmp_path) and mocked Chroma."""
    return Store(db_path=tmp_path / "test.db", chroma_path=tmp_path / "chroma")


def _model_alive(timeout: float = 5.0) -> bool:
    """Confirm llama-server responds on :8087/health."""
    try:
        resp = httpx.get(HEALTH_URL, timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


# ============================================================
# 1. IntentParser Tests
# ============================================================


class TestIntentParser:
    """Tests for the IntentParser module."""

    def test_import(self):
        """IntentParser imports successfully."""
        from src.middleware.intent_parser import IntentParser
        assert IntentParser is not None

    def test_init(self):
        """Default init creates parser with compiled regexes."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        assert parser._metric_regexes is not None
        assert len(parser._metric_regexes) > 10
        assert parser._type_regexes is not None
        assert len(parser._type_regexes) >= 7

    # ── Ticker Detection ───────────────────────────────

    @pytest.mark.parametrize("question,expected_ticker", [
        ("What is NVDA revenue?", "NVDA"),
        ("How is Apple doing?", "AAPL"),
        ("Compare AMD and NVDA", "AMD"),  # First match
        ("Tell me about Microsoft", "MSFT"),
        ("What is Meta's PE ratio?", "META"),
        ("CrowdStrike earnings report", "CRWD"),
        ("Palantir outlook", "PLTR"),
        ("Broadcom dividend yield", "AVGO"),
        ("What is the market cap of Tesla?", "TSLA"),
        ("How is Amazon's cloud business?", "AMZN"),
    ])
    def test_ticker_detection_symbols_and_names(self, question, expected_ticker):
        """Ticker is detected from both symbols and company names."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        assert result["ticker"] == expected_ticker, (
            f"Expected {expected_ticker} for '{question}', got {result['ticker']}"
        )

    @pytest.mark.parametrize("question", [
        "What is the weather today?",
        "How do I cook pasta?",
        "Tell me about the economy",
        "What time is it?",
    ])
    def test_no_ticker_detected(self, question):
        """Questions without ticker references return None."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        assert result["ticker"] is None

    def test_override_ticker(self):
        """Override ticker takes precedence over detected ticker."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("What is revenue?", override_ticker="CRWD")
        assert result["ticker"] == "CRWD"

    # ── Metric Extraction ──────────────────────────────

    @pytest.mark.parametrize("question,expected_metrics", [
        ("What is NVDA revenue?", ["total_revenue"]),
        ("What is the PE ratio and EPS for MSFT?", ["pe_ratio", "eps_diluted"]),
        ("Show me gross margin and operating margin", ["gross_margin_pct", "operating_margin_pct"]),
        ("What is free cash flow?", ["free_cash_flow"]),
        ("ROE and ROA for AMD", ["roe", "roa"]),
        ("Market cap and enterprise value", ["market_cap", "enterprise_value"]),
        ("What is the dividend yield?", ["dividend_yield"]),
    ])
    def test_metric_extraction(self, question, expected_metrics):
        """Financial metrics are extracted from the question."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        for metric in expected_metrics:
            assert metric in result["metrics"], (
                f"Expected metric '{metric}' in {result['metrics']} for '{question}'"
            )

    def test_no_metrics_detected(self):
        """Questions without financial metrics return empty list."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("What is the outlook for NVDA?")
        assert result["metrics"] == []

    # ── Question Type Classification ───────────────────

    @pytest.mark.parametrize("question,expected_type", [
        ("What is NVDA revenue?", "fact_lookup"),
        ("How much did AMD earn?", "fact_lookup"),
        ("Compare NVDA and AMD", "comparison"),
        ("NVDA vs AMD which is better", "comparison"),
        ("What is the trend for NVDA revenue?", "trend"),
        ("How has NVDA performed over time?", "trend"),
        ("Why did NVDA stock drop?", "explanation"),
        ("Explain NVDA's competitive advantage", "explanation"),
        ("What is the market sentiment on AMD?", "sentiment"),
        ("Analyst outlook for META", "sentiment"),
        ("Any news on CRWD?", "news"),
        ("What happened with Palantir?", "news"),
        ("What are the risks for NVDA?", "risk"),
        ("NVDA risk factors", "risk"),
        ("Tell me about AI chips", "general"),
    ])
    def test_question_type_classification(self, question, expected_type):
        """Question type is correctly classified."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        assert result["question_type"] == expected_type, (
            f"Expected '{expected_type}' for '{question}', got '{result['question_type']}'"
        )

    # ── Timeframe Extraction ──────────────────────────

    @pytest.mark.parametrize("question,expected_tf", [
        ("What was revenue in Q1 2026?", "q1 2026"),
        ("Revenue for FY 2025", "fy 2025"),
        ("Latest quarter results", "latest quarter"),
        ("TTM revenue", "ttm"),
        ("YTD performance", "ytd"),
    ])
    def test_timeframe_extraction(self, question, expected_tf):
        """Timeframe is extracted from the question."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse(question)
        assert result["timeframe"] is not None
        assert expected_tf in result["timeframe"].lower()

    def test_no_timeframe(self):
        """Questions without timeframe return None."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("What is NVDA's competitive advantage?")
        assert result["timeframe"] is None


# ============================================================
# 2. Retriever Tests
# ============================================================


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


# ============================================================
# 3. PromptAugmenter Tests
# ============================================================


class TestPromptAugmenter:
    """Tests for the PromptAugmenter module."""

    def test_import(self):
        """PromptAugmenter imports successfully."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        assert PromptAugmenter is not None

    def test_build_prompt_with_facts(self):
        """Prompt includes structured facts with citations."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="What is NVDA revenue?",
            intent={"ticker": "NVDA", "metrics": ["total_revenue"],
                    "question_type": "fact_lookup"},
            retrieval={
                "facts": [
                    {"metric": "total_revenue", "value": 26.0, "unit": "billion_usd",
                     "period": "2026-Q1", "source_type": "sec_10q"},
                ],
                "documents": [],
                "ticker": "NVDA",
            },
        )

        assert "Retrieved Financial Facts" in prompt
        assert "total_revenue" in prompt
        assert "26.0" in prompt
        assert "billion_usd" in prompt
        assert "Source: sec_10q/NVDA" in prompt
        assert "User Question" in prompt
        assert "What is NVDA revenue?" in prompt
        assert "Output Format" in prompt

    def test_build_prompt_with_documents(self):
        """Prompt includes document context with metadata."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="What is the outlook for NVDA?",
            intent={"ticker": "NVDA", "question_type": "sentiment"},
            retrieval={
                "facts": [],
                "documents": [
                    {
                        "id": "sec_10k/NVDA/10-K-2025",
                        "text": "NVIDIA reported strong growth in datacenter revenue.",
                        "ticker": "NVDA",
                        "source": "10-K",
                        "date": "2025-03-15",
                        "metadata": {"ticker": "NVDA", "source": "10-K"},
                    },
                ],
                "ticker": "NVDA",
            },
        )

        assert "Retrieved Documents" in prompt
        assert "NVIDIA reported" in prompt
        assert "Document 1" in prompt
        assert "User Question" in prompt

    def test_build_prompt_empty_retrieval(self):
        """Empty retrieval generates a 'no data found' note."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="What about AAPL?",
            intent={"ticker": "AAPL", "question_type": "general"},
            retrieval={"facts": [], "documents": [], "ticker": "AAPL"},
        )

        assert "No data was found" in prompt
        assert "AAPL" in prompt

    def test_prompt_excludes_duplicated_policy_rules(self):
        """The augmented user prompt no longer duplicates the authoritative
        answer-policy rules — those live solely in the system message built
        by prompt_policy.py (2.2.1.1 step 4)."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="Test?",
            intent={"ticker": None, "question_type": "general"},
            retrieval={"facts": [], "documents": [], "ticker": None},
        )

        assert "## Instructions" in prompt
        assert "financial research assistant" not in prompt
        assert "Answer using ONLY the provided context" not in prompt

    def test_question_type_instructions(self):
        """Question-type-specific instructions are included."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        for qtype in ("fact_lookup", "comparison", "trend", "explanation",
                      "sentiment", "news", "risk"):
            prompt = augmenter.build_prompt(
                question="Test?",
                intent={"ticker": "NVDA", "question_type": qtype},
                retrieval={"facts": [], "documents": [], "ticker": "NVDA"},
            )
            assert "## Instructions" in prompt

    def test_truncation(self):
        """Very long prompts are truncated gracefully."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        long_docs = [
            {
                "id": f"doc_{i}",
                "text": "X" * 5000,
                "ticker": "NVDA",
                "source": "10-K",
                "metadata": {"ticker": "NVDA", "source": "10-K"},
            }
            for i in range(10)
        ]

        prompt = augmenter.build_prompt(
            question="Test?",
            intent={"ticker": "NVDA", "question_type": "general"},
            retrieval={"facts": [], "documents": long_docs, "ticker": "NVDA"},
        )

        truncated = augmenter.truncate_if_needed(prompt, max_tokens=2000)
        assert len(truncated) < len(prompt)
        assert "truncated for length" in truncated or len(truncated) < len(prompt)

    def test_estimate_tokens(self):
        """Token estimation is reasonable."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()
        text = "Hello world, this is a test " * 100
        estimate = augmenter.estimate_tokens(text)
        assert estimate > 0
        assert estimate < len(text)  # Should be less than char count


# ============================================================
# 4. API Endpoint Tests
# ============================================================


class TestAPIEndpoints:
    """Tests for the FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked store and model."""
        from src.middleware.app import app
        with TestClient(app) as c:
            yield c

    def test_health_endpoint(self, client):
        """GET /health returns 200 with storage status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "storage" in data
        assert "model_available" in data

    def test_root_endpoint(self, client):
        """GET / returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "Gemma-E4B-Finance-RAG" in data["service"]

    def test_search_endpoint(self, client):
        """POST /search returns search results."""
        response = client.post("/search", json={
            "query": "NVDA revenue",
            "n_results": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "facts" in data
        assert "ticker" in data

    def test_search_with_ticker(self, client):
        """POST /search with ticker filter works."""
        response = client.post("/search", json={
            "query": "revenue",
            "ticker": "NVDA",
            "n_results": 3,
        })
        assert response.status_code == 200

    def test_query_endpoint(self, client):
        """POST /query returns a response with mocked model."""
        from src.middleware.models import SourceCitation

        mock_citations = [SourceCitation(source_type="sec_10q", ticker="NVDA")]
        with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call, \
                patch("src.middleware.app._check_model_health", new_callable=AsyncMock, return_value=True), \
                patch("src.middleware.app._refresh_ticker_sources", return_value=([], [])):
            mock_call.return_value = ("NVDA revenue was $26B in Q1 2026.", mock_citations)
            response = client.post("/query", json={
                "question": "What is NVDA revenue?",
            })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"]
        assert "citations" in data
        assert "latency_ms" in data
        assert "timestamp" in data

    def test_query_with_ticker_override(self, client):
        """POST /query with ticker override works."""
        with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call, \
                patch("src.middleware.app._check_model_health", new_callable=AsyncMock, return_value=True), \
                patch("src.middleware.app._refresh_ticker_sources", return_value=([], [])):
            mock_call.return_value = ("AMD revenue data.", [])
            response = client.post("/query", json={
                "question": "What is revenue?",
                "ticker": "AMD",
            })
        assert response.status_code == 200

    def test_query_empty_question_rejected(self, client):
        """Empty question returns 422 validation error."""
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_missing_question_rejected(self, client):
        """Missing question field returns 422."""
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_search_empty_query_rejected(self, client):
        """Empty search query returns 422."""
        response = client.post("/search", json={"query": ""})
        assert response.status_code == 422

    def test_openapi_docs_available(self, client):
        """OpenAPI docs endpoint is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


# ============================================================
# 5. Integration Tests
# ============================================================


class TestMiddlewareIntegration:
    """End-to-end integration tests with seeded data."""

    @pytest.fixture
    def seeded_store(self, mock_chroma, tmp_path):
        """Create a store with seeded data for integration tests."""
        store = Store(db_path=tmp_path / "test.db", chroma_path=tmp_path / "chroma")

        store.save_fundamental("NVDA", "total_revenue", 26.0, "billion_usd",
                               "2026-Q1", "quarterly", "sec_10q")
        store.save_fundamental("NVDA", "net_income", 7.8, "billion_usd",
                               "2026-Q1", "quarterly", "sec_10q")
        store.save_fundamental("NVDA", "eps_diluted", 3.12, "usd",
                               "2026-Q1", "quarterly", "sec_10q")
        store.save_fundamental("AMD", "total_revenue", 5.8, "billion_usd",
                               "2026-Q1", "quarterly", "sec_10q")
        store.save_fundamental("AMD", "net_income", 1.2, "billion_usd",
                               "2026-Q1", "quarterly", "sec_10q")

        store.save_document(
            document_id="test/NVDA/10-Q-2026-Q1",
            text="NVIDIA Corporation reported strong revenue growth driven by "
                 "datacenter and AI computing demand. Revenue reached $26.0B "
                 "for the first quarter of fiscal 2026.",
            ticker="NVDA",
            source="10-Q",
            date="2026-05-15",
        )

        return store

    def test_intent_parser_integration(self, seeded_store):
        """Intent parser works with the full middleware stack."""
        from src.middleware.intent_parser import IntentParser
        parser = IntentParser()

        result = parser.parse("What is NVDA's revenue and EPS for Q1 2026?")
        assert result["ticker"] == "NVDA"
        assert "total_revenue" in result["metrics"]
        assert "eps_diluted" in result["metrics"]
        assert result["question_type"] == "fact_lookup"
        assert result["timeframe"] is not None

    def test_retriever_integration(self, seeded_store):
        """Retriever returns seeded data from both stores."""
        from src.middleware.retriever import Retriever
        retriever = Retriever(store=seeded_store)

        result = retriever.retrieve(
            query="What is NVDA revenue?",
            intent={"ticker": "NVDA", "metrics": ["total_revenue"],
                     "question_type": "fact_lookup"},
        )

        assert len(result["facts"]) >= 1
        assert result["ticker"] == "NVDA"

        revenue_facts = [f for f in result["facts"] if f.get("metric") == "total_revenue"]
        assert len(revenue_facts) >= 1
        assert revenue_facts[0]["value"] == 26.0

    def test_prompt_augmenter_integration(self, seeded_store):
        """Full pipeline: parse → retrieve → augment produces a valid prompt."""
        from src.middleware.intent_parser import IntentParser
        from src.middleware.retriever import Retriever
        from src.middleware.prompt_augmenter import PromptAugmenter

        parser = IntentParser()
        retriever = Retriever(store=seeded_store)
        augmenter = PromptAugmenter()

        question = "What is NVDA's revenue and net income?"
        intent = parser.parse(question)
        retrieval = retriever.retrieve(query=question, intent=intent)
        prompt = augmenter.build_prompt(question, intent, retrieval)

        assert "NVDA" in prompt
        assert "total_revenue" in prompt or "revenue" in prompt
        assert "26.0" in prompt
        assert "7.8" in prompt or "net_income" in prompt
        assert "Source:" in prompt

    def test_api_integration(self, seeded_store, mock_chroma, monkeypatch):
        """API endpoint returns correct response with seeded data."""
        from src.middleware import app as middleware_app

        mock_chroma.search.return_value = [{
            "id": "test/NVDA/10-Q-2026-Q1",
            "text": "NVIDIA Corporation reported strong revenue growth.",
            "ticker": "NVDA",
            "source": "10-Q",
            "metadata": {"ticker": "NVDA", "source": "10-Q"},
        }]

        monkeypatch.setattr(middleware_app, "store", seeded_store)

        client = TestClient(middleware_app.app)

        response = client.post("/search", json={
            "query": "NVDA revenue",
            "n_results": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "NVDA"

    def test_comparison_flow(self, seeded_store):
        """Comparison intent triggers multi-ticker retrieval."""
        from src.middleware.intent_parser import IntentParser
        from src.middleware.retriever import Retriever

        parser = IntentParser()
        retriever = Retriever(store=seeded_store)

        question = "Compare NVDA and AMD revenue"
        intent = parser.parse(question)
        assert intent["question_type"] == "comparison"

        retrieval = retriever.retrieve(query=question, intent=intent)
        assert retrieval["strategy"] == "comparison"

    def test_unknown_ticker_graceful(self, seeded_store):
        """Querying an unknown ticker returns gracefully."""
        from src.middleware.intent_parser import IntentParser
        from src.middleware.retriever import Retriever
        from src.middleware.prompt_augmenter import PromptAugmenter

        parser = IntentParser()
        retriever = Retriever(store=seeded_store)
        augmenter = PromptAugmenter()

        question = "What is ZZZ Corp revenue?"
        intent = parser.parse(question)
        retrieval = retriever.retrieve(query=question, intent=intent)
        prompt = augmenter.build_prompt(question, intent, retrieval)

        assert len(prompt) > 0

    def test_middleware_coexists_with_phase13_data(self, store, mock_chroma):
        """Phase 1.3 yfinance + Phase 1.4 SEC facts coexist in middleware pipeline."""
        store.save_fundamental(
            ticker="NVDA",
            metric="revenue_q1",
            value=26.0,
            unit="billion_usd",
            period="2026-Q1",
            period_type="quarterly",
            source_type="yfinance",
        )
        store.save_fundamental(
            ticker="NVDA",
            metric="total_revenue",
            value=26.5,
            unit="billion_usd",
            period="2026-Q1",
            period_type="quarterly",
            source_type="sec_10q",
        )

        mock_chroma.search.return_value = [{
            "id": "sec_10q/NVDA/10-Q-2026",
            "text": "NVIDIA reported revenue growth in datacenter and AI.",
            "ticker": "NVDA",
            "source": "10-Q",
            "metadata": {"ticker": "NVDA", "source": "10-Q"},
        }]

        from src.middleware.intent_parser import IntentParser
        from src.middleware.retriever import Retriever
        from src.middleware.prompt_augmenter import PromptAugmenter

        parser = IntentParser()
        retriever = Retriever(store=store)
        augmenter = PromptAugmenter()

        question = "What is NVDA revenue?"
        intent = parser.parse(question)
        retrieval = retriever.retrieve(query=question, intent=intent)
        prompt = augmenter.build_prompt(question, intent, retrieval)

        assert "revenue_q1" in prompt or "26.0" in prompt
        assert "total_revenue" in prompt or "26.5" in prompt

        search_result = store.search("NVDA revenue")
        assert search_result["facts"] or search_result["documents"]


# ============================================================
# 5b. Conversation history contract (2.2.2.1)
# ============================================================


class TestConversationContract:
    """Additive /query history contract — single-turn stays byte-for-byte."""

    def _mock_plain_query_app(self, monkeypatch, answer="NVDA revenue is 26B"):
        """Wire a minimal mocked /query pipeline (no tools/stream/model server)."""
        from types import SimpleNamespace

        import src.middleware.app as middleware_app

        config = SimpleNamespace(
            model_name="tracealchemy",
            llama_endpoint="http://test/v1/chat/completions",
            default_temperature=0.3, max_tokens=256,
            top_k_documents=5, top_k_facts=10,
            enable_tools=False, enable_streaming=True, enable_fetch_on_miss=False,
            answer_policy="graded", allow_general_fallback=True, return_timings=True,
            conversation_max_turns=8, conversation_max_history_chars=8000,
        )
        parser = MagicMock()
        parser.parse.return_value = {
            "ticker": "NVDA", "ticker_confidence": 1.0,
            "question_type": "fact_lookup", "metrics": ["total_revenue"],
        }
        monkeypatch.setattr("src.middleware.intent_parser.IntentParser",
                            MagicMock(return_value=parser))
        monkeypatch.setattr(middleware_app, "config", config)
        monkeypatch.setattr(middleware_app, "store", object())
        monkeypatch.setattr(middleware_app, "retriever", SimpleNamespace(
            retrieve=lambda **_k: {
                "facts": [{"metric": "total_revenue", "value": 26.0}],
                "documents": [], "retrieval_strategy": "vector", "timings": {}}))
        monkeypatch.setattr(middleware_app, "_evaluate_and_refresh",
                            MagicMock(return_value={"overall": "fresh", "fetched_on_miss": []}))
        monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
        monkeypatch.setattr(middleware_app, "_task_params", lambda _t: {})

        completion = MagicMock()
        completion.raise_for_status.return_value = None
        completion.json.return_value = {"choices": [{"message": {"content": answer}}]}
        monkeypatch.setattr(middleware_app, "model_client",
                            SimpleNamespace(post=AsyncMock(return_value=completion)))
        return middleware_app

    def test_query_request_without_history_is_backward_compatible(self, monkeypatch):
        from src.middleware.models import QueryRequest

        # Model defaults: additive fields are empty/None.
        req = QueryRequest(question="What is NVDA revenue?")
        assert req.history == []
        assert req.session_id is None

        app = self._mock_plain_query_app(monkeypatch)
        client = TestClient(app.app)

        # No history -> conversation metadata omitted (single-turn behavior).
        r0 = client.post("/query", json={"question": "What is NVDA revenue?"})
        assert r0.status_code == 200
        assert r0.json()["conversation"] is None

        # With history -> metadata reports what was received/used.
        r1 = client.post("/query", json={
            "question": "And AMD?",
            "session_id": "sess-1",
            "history": [
                {"role": "user", "content": "What is NVDA revenue?"},
                {"role": "assistant", "content": "NVDA revenue is 26B"},
            ],
        })
        assert r1.status_code == 200
        convo = r1.json()["conversation"]
        assert convo == {
            "history_turns_received": 2,
            "history_turns_used": 2,
            "history_truncated": False,
            "topic_reset": False,
        }

    def test_current_question_is_never_silently_truncated(self):
        from src.middleware.models import MAX_QUESTION_CHARS, QueryRequest

        # At the limit: accepted verbatim, no truncation.
        at_limit = "a" * MAX_QUESTION_CHARS
        req = QueryRequest(question=at_limit)
        assert req.question == at_limit
        assert len(req.question) == MAX_QUESTION_CHARS

        # Over the limit: explicit validation error, not a shortened question.
        with pytest.raises(ValidationError):
            QueryRequest(question="a" * (MAX_QUESTION_CHARS + 1))


# ============================================================
# 6. Live Model Tests
# ============================================================


@pytest.mark.live
class TestLiveMiddleware:
    """Live tests requiring llama-server on :8087."""

    pytestmark = pytest.mark.skipif(
        not _model_alive(),
        reason="Requires llama-server :8087",
    )

    @pytest.fixture
    def client(self):
        from src.middleware.app import app
        with TestClient(app) as c:
            yield c

    def test_live_query_pipeline(self, client):
        """POST /query with real model on :8087."""
        response = client.post("/query", json={
            "question": "What is NVDA revenue?",
        })
        assert response.status_code == 200
        data = response.json()
        assert data.get("answer")
        assert data.get("detected_ticker") == "NVDA"


# ══════════════════════════════════════════════════════════════════════
# Phase 2.2.3.4 — adaptive orchestration integration (offline, deterministic)
#
# These exercise the ONE request-path switch in _build_query_context and the
# response/trace/health wiring. `orchestrate` is patched with a fake that
# returns a crafted OrchestrationResult, so the app-level wiring is tested in
# isolation from the orchestrator (covered by test_adaptive_orchestrator.py).
# No model, network, or ChromaDB process is started.
# ══════════════════════════════════════════════════════════════════════

import pytest as _pytest  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402
from unittest.mock import AsyncMock as _AsyncMock  # noqa: E402

from src.middleware import app as _app  # noqa: E402
from src.middleware import adaptive_orchestrator as _ao  # noqa: E402
from src.middleware.adaptive_orchestrator import (  # noqa: E402
    ContextSelection as _ContextSelection,
    Lane as _Lane,
    OrchestrationResult as _OrchResult,
)
from src.middleware.config import MiddlewareConfig as _MW  # noqa: E402
from src.middleware.deterministic_router import (  # noqa: E402
    ExecutedInvocation as _ExecInv,
    ExecutionResult as _ExecResult,
)
from src.middleware.models import QueryRequest as _QReq  # noqa: E402


def _adaptive_config(**overrides) -> _MW:
    cfg = _MW()
    cfg.enable_adaptive_rag = True
    cfg.enable_fetch_on_miss = False
    cfg.enable_conversation_rewrite = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class _FakeRetriever:
    """Records retrieve() calls; used to prove the legacy fallback ran once."""

    def __init__(self, *, facts=None, documents=None, strategy="vector"):
        self.retrieve_calls = 0
        self._facts = facts or []
        self._documents = documents or []
        self._strategy = strategy

    def retrieve(self, query, intent, top_k_documents=5, top_k_facts=10):
        self.retrieve_calls += 1
        return {
            "facts": [dict(f) for f in self._facts],
            "documents": [dict(d) for d in self._documents],
            "ticker": intent.get("ticker"),
            "strategy": "hybrid",
            "retrieval_strategy": self._strategy,
            "timings": {"embedding": 0.0, "chroma": 0.0, "sqlite": 0.0},
        }


def _fake_result(plan, *, lane=_Lane.STANDARD, facts=None, documents=None,
                 tool_execution=None, fallback_reason=None,
                 subqueries=("sq0",), rounds=1, rerank=False, planning=False,
                 strategy="hybrid", dropped_facts=0, dropped_documents=0):
    facts = list(facts or [])
    documents = list(documents or [])
    ctx = _ContextSelection(
        facts=facts, documents=documents,
        context_chars=100 + len(facts) * 10, estimated_tokens=25,
        dropped_facts=dropped_facts, dropped_documents=dropped_documents,
    )
    return _OrchResult(
        lane=lane, plan=plan, reason_codes=[f"lane_{lane.value}"],
        merged_facts=facts, merged_documents=documents,
        tool_execution=tool_execution,
        subqueries_executed=list(subqueries),
        retrieval_rounds_used=rounds, planning_ran=planning, rerank_ran=rerank,
        retrieval_strategy=strategy, context=ctx,
        context_size=ctx.context_chars, estimated_tokens=ctx.estimated_tokens,
        fallback_reason=fallback_reason,
    )


def _wire_adaptive(monkeypatch, config, *, retriever=None, fake_orchestrate=None):
    """Patch middleware globals + freshness/model so the adaptive path can run
    fully in-process. Leaves the real IntentParser + _build_*_context in place."""
    store = _NS(sqlite=_NS(list_metrics=lambda: ["total_revenue", "gross_margin"]))
    monkeypatch.setattr(_app, "store", store)
    monkeypatch.setattr(_app, "config", config)
    monkeypatch.setattr(_app, "retriever", retriever or _FakeRetriever())
    monkeypatch.setattr(
        _app, "_evaluate_and_refresh",
        MagicMock(return_value={"overall": "fresh", "fetched_on_miss": []}))
    monkeypatch.setattr(_app, "_task_params", lambda _t: {})
    monkeypatch.setattr(_app, "_check_model_health", _AsyncMock(return_value=True))
    monkeypatch.setattr(
        _app, "_call_model",
        _AsyncMock(return_value=("adaptive answer", [])))
    if fake_orchestrate is not None:
        monkeypatch.setattr(_ao, "orchestrate", fake_orchestrate)


@_pytest.mark.asyncio
async def test_adaptive_feature_disabled_matches_legacy_route(monkeypatch):
    """enable_adaptive_rag=false -> pure legacy path, orchestration omitted."""
    cfg = _adaptive_config(enable_adaptive_rag=False)
    retriever = _FakeRetriever(
        facts=[{"metric": "total_revenue", "value": 26.0, "ticker": "NVDA"}],
        documents=[{"id": "d1", "document": "NVDA revenue context",
                    "metadata": {"ticker": "NVDA"}}])
    _wire_adaptive(monkeypatch, cfg, retriever=retriever)

    resp = await _app.query(_QReq(question="What is NVDA revenue?"))

    assert resp.orchestration is None
    assert retriever.retrieve_calls == 1
    assert resp.facts_used == 1


@_pytest.mark.asyncio
async def test_adaptive_path_populates_actual_counts(monkeypatch):
    """Metadata reports ACTUAL executed counters, not configured maxima."""
    cfg = _adaptive_config()

    def fake_orch(plan, store, config, **kwargs):
        return _fake_result(
            plan, lane=_Lane.COMPLEX,
            facts=[{"metric": "total_revenue", "value": 26.0, "ticker": "NVDA"}],
            documents=[{"id": "d1", "document": "body",
                        "metadata": {"ticker": "NVDA"}}],
            subqueries=("sq0",), rounds=1, rerank=False, planning=False,
            dropped_documents=7)

    _wire_adaptive(monkeypatch, cfg, fake_orchestrate=fake_orch)

    resp = await _app.query(_QReq(question="compare NVDA AMD revenue and risks"))

    orch = resp.orchestration
    assert orch is not None
    assert orch["lane"] == "complex"
    assert orch["subqueries_executed"] == 1       # actual, cap is 3
    assert orch["retrieval_rounds"] == 1          # actual, cap is 2
    assert orch["planning_calls"] == 0
    assert orch["reranker_calls"] == 0
    assert orch["evidence_dropped"] == 7
    assert orch["fallback_reason"] is None


@_pytest.mark.asyncio
async def test_adaptive_deterministic_tools_in_evidence_and_trace(monkeypatch):
    """Deterministic tool results reach the final evidence AND the trace."""
    cfg = _adaptive_config()
    tool_facts = [{"metric": "total_revenue", "value": 26.0, "ticker": "NVDA",
                   "source_type": "tool"}]
    execution = _ExecResult(
        invocations=[_ExecInv(
            "get_fundamentals", {"ticker": "NVDA", "metrics": ["total_revenue"]},
            "sq0", "route_get_fundamentals",
            result={"ticker": "NVDA", "fundamentals": {"total_revenue": 26.0}})],
        calculations=[], error=False)

    def fake_orch(plan, store, config, **kwargs):
        return _fake_result(plan, lane=_Lane.FAST, facts=tool_facts,
                            tool_execution=execution, rounds=0)

    _wire_adaptive(monkeypatch, cfg, fake_orchestrate=fake_orch)

    # Real _call_model records the trace prompt on the successful path; mirror
    # that here so finalize() produces a trace (the mock otherwise records none).
    async def _model(prompt, temperature, max_tokens, intent, grounding_level):
        _app._record_trace_prompt("SYS", prompt)
        return "adaptive answer", []

    monkeypatch.setattr(_app, "_call_model", _model)

    resp = await _app.query(_QReq(question="What is NVDA revenue?",
                                  include_evidence_trace=True))

    # Final evidence carries the tool fact.
    assert resp.facts_used == 1
    assert resp.orchestration["deterministic_tools"] == ["get_fundamentals"]
    # Trace records the executed tool on the successful path.
    trace = resp.evidence_trace
    assert trace is not None
    orch_trace = trace["orchestration"]
    names = [t["name"] for t in orch_trace["deterministic_tool_results"]]
    assert names == ["get_fundamentals"]
    assert orch_trace["query_plan"]["retrieval_query"]
    assert any(f.get("metric") == "total_revenue" for f in trace["facts"])


@_pytest.mark.asyncio
async def test_adaptive_plan_failure_falls_back_to_legacy_once(monkeypatch):
    """An invalid plan -> legacy retrieval runs once, raw question preserved."""
    cfg = _adaptive_config()
    retriever = _FakeRetriever(
        documents=[{"id": "legacy-doc", "document": "legacy body",
                    "metadata": {"ticker": "NVDA"}}])
    _wire_adaptive(monkeypatch, cfg, retriever=retriever)

    from src.middleware.query_plan import QueryPlanError

    def boom(self, question, retrieval_query=None, override_ticker=None):
        raise QueryPlanError(["blank_retrieval_query"])

    monkeypatch.setattr(
        "src.middleware.intent_parser.IntentParser.parse_plan", boom)

    resp = await _app.query(_QReq(question="What is NVDA revenue?"))

    assert retriever.retrieve_calls == 1
    assert resp.orchestration == {"lane": None, "fallback_reason": "adaptive_fallback"}
    assert resp.answer == "adaptive answer"


@_pytest.mark.asyncio
async def test_adaptive_normal_and_stream_share_context(monkeypatch):
    """Both answer functions consume the same compiled adaptive context."""
    cfg = _adaptive_config()

    def fake_orch(plan, store, config, **kwargs):
        return _fake_result(
            plan, lane=_Lane.STANDARD,
            documents=[{"id": "d1", "document": "shared body",
                        "metadata": {"ticker": "NVDA"}}])

    _wire_adaptive(monkeypatch, cfg, fake_orchestrate=fake_orch)

    request = _QReq(question="why did NVDA drop")
    context = await _app._build_query_context(request)

    # Normal path response.
    normal = await _app._answer_query_context(request, context)
    # Streaming terminal metadata is built from the SAME context via the shared
    # _build_query_response, so its orchestration/grounding must match.
    streamed = _app._build_query_response(
        context=context, answer_text="x", citations=[], model_available=True)

    assert context["orchestration"]["lane"] == "standard"
    assert normal.orchestration == streamed.orchestration
    assert normal.grounding == streamed.grounding
    assert normal.documents_used == streamed.documents_used == 1


def test_health_reports_adaptive_capabilities(monkeypatch):
    """/health advertises effective adaptive_rag + deterministic routing flags."""
    cfg = _adaptive_config(enable_deterministic_tool_routing=True)
    monkeypatch.setattr(_app, "config", cfg)
    monkeypatch.setattr(
        _app, "store", _NS(heartbeat=lambda: {"sqlite": True, "chroma": True}))
    monkeypatch.setattr(_app, "_check_model_health", _AsyncMock(return_value=True))
    monkeypatch.setattr(
        _app, "_cached_health_summary", lambda: {"scheduler": None, "freshness": {}})

    caps = TestClient(_app.app).get("/health").json()["capabilities"]

    assert caps["adaptive_rag"] is True
    assert caps["deterministic_tool_routing"] is True


def test_health_can_skip_expensive_scheduler_details(monkeypatch):
    cfg = _adaptive_config(enable_deterministic_tool_routing=True)
    monkeypatch.setattr(_app, "config", cfg)
    monkeypatch.setattr(
        _app, "store", _NS(heartbeat=lambda: {"sqlite": True, "chroma": True}))
    monkeypatch.setattr(_app, "_check_model_health", _AsyncMock(return_value=True))
    monkeypatch.setattr(
        _app,
        "_cached_health_summary",
        lambda: (_ for _ in ()).throw(AssertionError("scheduler details were loaded")),
    )

    data = TestClient(_app.app).get("/health?details=false").json()

    assert data["status"] == "ok"
    assert data["scheduler"] is None
    assert data["freshness"] == {}


def test_old_client_tolerates_omitted_orchestration_metadata():
    """A QueryResponse without orchestration validates and dumps to null; the
    chat renderer tolerates a response dict missing the field entirely."""
    from src.middleware.models import QueryResponse
    from scripts import chat

    resp = QueryResponse(answer="hi")
    assert resp.orchestration is None
    assert resp.model_dump()["orchestration"] is None

    # No orchestration key at all -> renderer must not raise.
    chat._render_metadata({"answer": "hi", "grounding": "grounded"}, verbose=True)

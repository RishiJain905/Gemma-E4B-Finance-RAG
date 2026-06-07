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

    def test_prompt_includes_system_instruction(self):
        """Prompt always includes system instruction with rules."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="Test?",
            intent={"ticker": None, "question_type": "general"},
            retrieval={"facts": [], "documents": [], "ticker": None},
        )

        assert "financial research assistant" in prompt
        assert "Answer using ONLY the provided context" in prompt
        assert "Cite sources inline" in prompt

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
        with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call:
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
        with patch("src.middleware.app._call_model", new_callable=AsyncMock) as mock_call:
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

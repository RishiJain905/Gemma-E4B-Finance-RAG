"""
tests/test_financebot_adapter.py
Offline tests for the FinanceBot RAG adapter: hit/miss contract, tool
preservation (including classify_trade_bias), and HTTP surface.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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

from src.middleware import app as middleware_app
from src.middleware.app import app
from src.middleware.financebot import (
    REQUIRED_RAG_TOOLS,
    classify_rag_hit,
    invoke_financebot_tool,
    list_financebot_tools,
)
from src.middleware.tools import REGISTRY, openai_schema
from src.storage.store import Store


@pytest.fixture
def mock_chroma():
    with patch("src.storage.store.ChromaStore") as mock_cls:
        instance = MagicMock()
        instance.heartbeat.return_value = True
        instance.count.return_value = 0
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def store(mock_chroma, tmp_path: Path):
    return Store(db_path=tmp_path / "financebot.db", chroma_path=tmp_path / "chroma")


def test_required_rag_tools_remain_registered():
    names = set(REGISTRY)
    missing = REQUIRED_RAG_TOOLS - names
    assert not missing, f"RAG tools missing: {sorted(missing)}"
    assert "classify_trade_bias" in names
    listed = {row["name"] for row in list_financebot_tools()}
    assert REQUIRED_RAG_TOOLS <= listed
    assert listed == names
    schema_names = {entry["function"]["name"] for entry in openai_schema()}
    assert REQUIRED_RAG_TOOLS <= schema_names
    assert schema_names == names


def test_classify_rag_hit_with_facts():
    status, docs, score = classify_rag_hit(
        [{"metric": "price", "value": 100}],
        [],
    )
    assert status == "hit"
    assert docs == []
    assert score is None


def test_classify_rag_miss_when_empty():
    status, docs, score = classify_rag_hit([], [])
    assert status == "miss"
    assert docs == []
    assert score is None


def test_classify_rag_hit_from_scored_document():
    status, docs, score = classify_rag_hit(
        [],
        [{"id": "d1", "fusion_score": 0.4, "text": "NVIDIA revenue grew"}],
        min_document_score=0.1,
    )
    assert status == "hit"
    assert len(docs) == 1
    assert score == 0.4


def test_classify_rag_miss_below_score_threshold():
    status, docs, score = classify_rag_hit(
        [],
        [{"id": "d1", "fusion_score": 0.05}],
        min_document_score=0.2,
    )
    assert status == "miss"
    assert docs == []
    assert score is None


def test_financebot_rag_endpoint_hit_vs_miss(monkeypatch):
    fake_store = MagicMock()
    fake_store.search.return_value = {
        "facts": [{"metric": "price", "value": 120.0, "ticker": "NVDA"}],
        "ticker": "NVDA",
    }
    fake_retriever = MagicMock()
    fake_retriever.retrieve_documents.return_value = []
    fake_retriever._doc_retrieval_strategy = "hybrid"

    monkeypatch.setattr(middleware_app, "config", MagicMock(
        financebot_min_facts=1,
        financebot_min_documents=1,
        financebot_min_document_score=0.0,
    ))

    with TestClient(app) as client:
        monkeypatch.setattr(middleware_app, "store", fake_store)
        monkeypatch.setattr(middleware_app, "retriever", fake_retriever)
        hit = client.post("/financebot/rag", json={"query": "NVDA price", "ticker": "NVDA"})
        assert hit.status_code == 200
        body = hit.json()
        assert body["status"] == "hit"
        assert body["web_search_allowed"] is False
        assert body["source_of_truth"] == "rag"
        assert body["model_generation"] is False
        assert body["fact_count"] == 1

        fake_store.search.return_value = {"facts": [], "ticker": "NVDA"}
        miss = client.post(
            "/financebot/rag",
            json={"query": "obscure private company with no filings"},
        )
        assert miss.status_code == 200
        miss_body = miss.json()
        assert miss_body["status"] == "miss"
        assert miss_body["web_search_allowed"] is True
        assert miss_body["source_of_truth"] is None
        assert miss_body.get("trade_bias") is None


def test_financebot_rag_long_or_short_attaches_classify_trade_bias(store, monkeypatch):
    store.sqlite.upsert_fundamental(
        ticker="NVDA",
        metric="recommendation_mean",
        value=1.5,
        unit="score",
        period="2027-07E",
        period_type="estimate",
        source_type="estimates",
    )
    from src.middleware.tools import data_tools

    monkeypatch.setattr(
        data_tools,
        "get_sentiment_handler",
        lambda store, ticker, days=7: {"average_tone": None, "article_count": 0},
    )
    fake_retriever = MagicMock()
    fake_retriever.retrieve_documents.return_value = []
    fake_retriever._doc_retrieval_strategy = "hybrid"

    with TestClient(app) as client:
        monkeypatch.setattr(middleware_app, "store", store)
        monkeypatch.setattr(middleware_app, "retriever", fake_retriever)
        monkeypatch.setattr(middleware_app, "config", MagicMock(
            financebot_min_facts=1,
            financebot_min_documents=1,
            financebot_min_document_score=0.0,
        ))
        response = client.post(
            "/financebot/rag",
            json={"query": "is NVDA a long or short trade", "ticker": "NVDA"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["trade_bias"]["bias"] == "long"
    assert body["trade_bias"]["must_answer"] is True
    assert body["status"] == "hit"
    assert body["web_search_allowed"] is False
    assert "MUST answer long or short" in body["message"]


def test_financebot_tools_dispatch_classify_trade_bias(store, monkeypatch):
    store.sqlite.upsert_fundamental(
        ticker="NVDA",
        metric="recommendation_mean",
        value=1.5,
        unit="score",
        period="2027-07E",
        period_type="estimate",
        source_type="estimates",
    )
    from src.middleware.tools import data_tools

    monkeypatch.setattr(
        data_tools,
        "get_sentiment_handler",
        lambda store, ticker, days=7: {"average_tone": None, "article_count": 0},
    )

    payload = invoke_financebot_tool(
        store=store,
        name="classify_trade_bias",
        arguments={"ticker": "NVDA"},
        allow_write=False,
    )
    assert payload["tool"] == "classify_trade_bias"
    assert payload["result"]["bias"] == "long"
    assert payload["result"]["must_answer"] is True


def test_financebot_tools_http_lists_and_invokes(monkeypatch, store):
    from src.middleware.tools import data_tools

    monkeypatch.setattr(
        data_tools,
        "get_sentiment_handler",
        lambda store, ticker, days=7: {"average_tone": None, "article_count": 0},
    )
    with TestClient(app) as client:
        monkeypatch.setattr(middleware_app, "store", store)
        monkeypatch.setattr(
            middleware_app,
            "config",
            MagicMock(enable_tools=True, allow_write_tools=False, max_refreshes_per_query=2),
        )
        listed = client.get("/financebot/tools")
        assert listed.status_code == 200
        names = {row["name"] for row in listed.json()["tools"]}
        assert REQUIRED_RAG_TOOLS <= names
        tools_endpoint = client.get("/tools")
        assert tools_endpoint.status_code == 200
        assert {row["name"] for row in tools_endpoint.json()["tools"]} == names

        invoked = client.post(
            "/financebot/tools",
            json={"name": "classify_trade_bias", "arguments": {"ticker": "MSFT"}},
        )
        assert invoked.status_code == 200
        body = invoked.json()
        assert body["tool"] == "classify_trade_bias"
        assert body["result"]["evidence_status"] == "miss"
        assert body["result"]["web_search_allowed"] is True

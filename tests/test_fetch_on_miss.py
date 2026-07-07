import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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

from src.middleware import app as middleware_app  # noqa: E402
from src.middleware.app import app  # noqa: E402
from src.storage.store import Store  # noqa: E402


@pytest.fixture
def tmp_store(tmp_path: Path):
    with patch("src.storage.store.ChromaStore") as mock_cls:
        instance = MagicMock()
        instance.heartbeat.return_value = True
        instance.count.return_value = 0
        instance.get_document.return_value = None
        mock_cls.return_value = instance
        yield Store(db_path=tmp_path / "fetch.db", chroma_path=tmp_path / "chroma")


def _config(**overrides):
    values = {
        "model_name": "tracealchemy",
        "llama_endpoint": "http://test/v1/chat/completions",
        "default_temperature": 0.3,
        "max_tokens": 2048,
        "top_k_documents": 5,
        "top_k_facts": 10,
        "enable_tools": False,
        "enable_fetch_on_miss": True,
        "fetch_on_miss_timeout_s": 10.0,
        "fetch_on_miss_per_query": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ticker(info=None, news=None):
    t = MagicMock()
    t.info = info or {
        "regularMarketPrice": 3.25,
        "forwardPE": 7.5,
        "marketCap": 5000000000,
    }
    t.news = news or [
        {
            "uuid": "bb-news-1",
            "title": "BlackBerry reports progress",
            "summary": "BlackBerry updated investors on operations.",
            "publisher": "TestWire",
            "link": "https://example.test/bb-news-1",
            "providerPublishTime": 1717000000,
            "type": "news",
        }
    ]
    return t


def _query_with_intent(tmp_store, monkeypatch, intent, freshness, cfg=None):
    parser = MagicMock()
    parser.parse.return_value = intent
    parser_cls = MagicMock(return_value=parser)

    monkeypatch.setattr("src.middleware.intent_parser.IntentParser", parser_cls)

    with TestClient(app) as client:
        monkeypatch.setattr(middleware_app, "store", tmp_store)
        monkeypatch.setattr(middleware_app, "config", cfg or _config())
        monkeypatch.setattr(middleware_app, "model_client", SimpleNamespace(aclose=AsyncMock()))
        monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=False))
        monkeypatch.setattr(middleware_app, "_evaluate_and_refresh", MagicMock(return_value=freshness))
        monkeypatch.setattr(
            middleware_app,
            "retriever",
            SimpleNamespace(
                retrieve=lambda **_kwargs: {
                    "facts": [],
                    "documents": [],
                    "retrieval_strategy": "test",
                }
            ),
        )
        return client.post("/query", json={"question": "What is BB's forward P/E?"})


def test_untracked_valid_ticker_triggers_fetch(tmp_store, tmp_path, monkeypatch):
    from src.middleware.on_demand import fetch_ticker_on_miss, read_dynamic_tracked

    monkeypatch.setattr(
        "src.middleware.on_demand.TRACKED_DYNAMIC_PATH",
        tmp_path / "tracked_dynamic.json",
    )

    with patch("src.ingestion.yfinance_ingestor.yf.Ticker", return_value=_ticker()):
        result = fetch_ticker_on_miss(tmp_store, "BB")

    assert result == {
        "fetched": True,
        "ticker": "BB",
        "sources": ["yfinance_fundamentals", "yfinance_news"],
        "error": None,
    }
    facts = tmp_store.get_fundamentals_batch("BB")
    assert facts["forward_pe"] == 7.5
    assert tmp_store.get_cache_status("BB", "yfinance_fundamentals")["status"] == "fresh"
    assert tmp_store.get_cache_status("BB", "yfinance_news")["status"] == "fresh"
    assert read_dynamic_tracked() == ["BB"]


def test_tracked_ticker_no_fetch(tmp_store, monkeypatch):
    tmp_store.mark_cache_fresh("BB", "yfinance_fundamentals", 24)
    fetch = MagicMock()
    monkeypatch.setattr(middleware_app, "_maybe_fetch_on_miss", fetch)

    response = _query_with_intent(
        tmp_store,
        monkeypatch,
        {
            "ticker": "BB",
            "ticker_confidence": 1.0,
            "question_type": "fact_lookup",
            "metrics": ["forward_pe"],
        },
        {
            "overall": "fresh",
            "refreshed_during_query": [],
            "stale_sources_used": [],
            "fetched_on_miss": [],
            "warning": None,
        },
    )

    assert response.status_code == 200
    fetch.assert_not_called()


def test_invalid_ticker_no_fetch(tmp_store, tmp_path, monkeypatch):
    from src.middleware.on_demand import fetch_ticker_on_miss

    monkeypatch.setattr(
        "src.middleware.on_demand.TRACKED_DYNAMIC_PATH",
        tmp_path / "tracked_dynamic.json",
    )

    with patch(
        "src.ingestion.yfinance_ingestor.yf.Ticker",
        return_value=_ticker(info={"regularMarketPrice": None}),
    ):
        result = fetch_ticker_on_miss(tmp_store, "NOPE")

    assert result["fetched"] is False
    assert result["error"] == "invalid_ticker"
    assert tmp_store.get_fundamentals_batch("NOPE") == {}
    assert tmp_store.get_cache_status("NOPE", "yfinance_fundamentals") is None
    assert tmp_store.get_cache_status("NOPE", "yfinance_news") is None


def test_fetch_failure_graceful(tmp_store):
    from src.middleware.on_demand import fetch_ticker_on_miss

    with patch("src.ingestion.yfinance_ingestor.yf.Ticker", side_effect=RuntimeError("boom")):
        result = fetch_ticker_on_miss(tmp_store, "BB")

    assert result["fetched"] is False
    assert result["ticker"] == "BB"
    assert result["sources"] == []
    assert tmp_store.get_fundamentals_batch("BB") == {}


def test_disabled_flag(tmp_store, monkeypatch):
    fetch = MagicMock()
    monkeypatch.setattr(middleware_app, "_maybe_fetch_on_miss", fetch)

    response = _query_with_intent(
        tmp_store,
        monkeypatch,
        {
            "ticker": "BB",
            "ticker_confidence": 1.0,
            "question_type": "fact_lookup",
            "metrics": ["forward_pe"],
        },
        {
            "overall": "never_fetched",
            "refreshed_during_query": [],
            "stale_sources_used": [],
            "fetched_on_miss": [],
            "warning": None,
        },
        cfg=_config(enable_fetch_on_miss=False),
    )

    assert response.status_code == 200
    fetch.assert_not_called()


def test_low_confidence_no_fetch(tmp_store, monkeypatch):
    fetch = MagicMock()
    monkeypatch.setattr(middleware_app, "_maybe_fetch_on_miss", fetch)

    response = _query_with_intent(
        tmp_store,
        monkeypatch,
        {
            "ticker": "BB",
            "ticker_confidence": 0.3,
            "question_type": "fact_lookup",
            "metrics": ["forward_pe"],
        },
        {
            "overall": "never_fetched",
            "refreshed_during_query": [],
            "stale_sources_used": [],
            "fetched_on_miss": [],
            "warning": None,
        },
    )

    assert response.status_code == 200
    fetch.assert_not_called()


def test_only_cheap_sources(tmp_store, tmp_path, monkeypatch):
    from src.middleware.on_demand import fetch_ticker_on_miss

    monkeypatch.setattr(
        "src.middleware.on_demand.TRACKED_DYNAMIC_PATH",
        tmp_path / "tracked_dynamic.json",
    )

    with (
        patch("src.ingestion.yfinance_ingestor.yf.Ticker", return_value=_ticker()),
        patch("src.sec.SECEdgarFilingFetcher") as sec,
        patch("src.macros.gdelt_ingestor.GDELTIngestor") as gdelt,
        patch("src.macros.earnings_transcripts.EarningsTranscriptIngestor") as transcripts,
        patch("src.macros.ir_ingestor.IRIngestor") as ir,
        patch("src.macros.estimates_ingestor.EstimatesIngestor") as estimates,
    ):
        result = fetch_ticker_on_miss(tmp_store, "BB")

    assert result["fetched"] is True
    sec.assert_not_called()
    gdelt.assert_not_called()
    transcripts.assert_not_called()
    ir.assert_not_called()
    estimates.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_degrades(tmp_store, monkeypatch):
    from src.middleware.app import _maybe_fetch_on_miss

    def slow_fetch(_store, ticker):
        time.sleep(0.05)
        return {"fetched": True, "ticker": ticker, "sources": [], "error": None}

    monkeypatch.setattr(middleware_app, "store", tmp_store)
    monkeypatch.setattr(middleware_app, "config", _config(fetch_on_miss_timeout_s=0.001))
    monkeypatch.setattr("src.middleware.on_demand.fetch_ticker_on_miss", slow_fetch)

    result = await _maybe_fetch_on_miss("BB")

    assert result == {
        "fetched": False,
        "ticker": "BB",
        "sources": [],
        "error": "timeout",
    }


def test_golden_untracked_case_present():
    golden = Path("eval/golden/finance_qa.jsonl")
    cases = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines()]
    case = next((case for case in cases if case.get("id") == "bb-fwd-pe"), None)

    assert case is not None
    assert case["expected_ticker"] == "BB"
    assert "fetch_on_miss" in case.get("tags", [])

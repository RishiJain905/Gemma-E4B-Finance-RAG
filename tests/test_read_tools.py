import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

import pytest  # noqa: E402

from src.middleware import app as middleware_app  # noqa: E402
from src.middleware.app import app  # noqa: E402
from src.middleware.tools import REGISTRY, openai_schema  # noqa: E402
from src.middleware.tools import data_tools  # noqa: E402
from src.storage.store import Store  # noqa: E402


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
    return Store(db_path=tmp_path / "read_tools.db", chroma_path=tmp_path / "chroma")


def _seed_metric(
    store,
    ticker,
    metric,
    value,
    period="2026-Q1",
    unit="ratio",
    period_type="quarterly",
    source_type="test",
):
    store.sqlite.upsert_fundamental(
        ticker=ticker,
        metric=metric,
        value=value,
        unit=unit,
        period=period,
        period_type=period_type,
        source_type=source_type,
    )


def test_get_fundamentals(store):
    _seed_metric(store, "NVDA", "forward_pe", 16.5)
    _seed_metric(store, "NVDA", "revenue_growth", 0.22)

    result = data_tools.get_fundamentals_handler(store, ticker="nvda", metrics=["forward_pe"])

    assert result == {"ticker": "NVDA", "fundamentals": {"forward_pe": 16.5}}


def test_search_documents_truncates(store):
    text = "x" * 1600
    store.search = MagicMock(
        return_value={
            "documents": [
                {"id": "doc-1", "document": text, "metadata": {"ticker": "NVDA"}},
            ],
        }
    )

    result = data_tools.search_documents_handler(
        store, query="risk factors", ticker="NVDA", n_results=99
    )

    store.search.assert_called_once_with(query="risk factors", n_results=10, ticker="NVDA")
    assert result == {
        "documents": [
            {"id": "doc-1", "text": text[:1500], "metadata": {"ticker": "NVDA"}},
        ],
    }
    assert len(result["documents"][0]["text"]) == 1500


def test_get_macro_snapshot(store, monkeypatch):
    for metric, value in {
        "GDP": 29000.0,
        "CPIAUCSL": 314.2,
        "FEDFUNDS": 5.25,
        "UNRATE": 4.0,
        "DGS10": 4.35,
        "T10Y2Y": -0.3,
    }.items():
        _seed_metric(store, "MACRO", metric, value, unit="index")
    monkeypatch.setattr(middleware_app, "store", store)

    result = data_tools.get_macro_snapshot_handler(store)

    assert result == {
        "macro": {
            "GDP": 29000.0,
            "CPIAUCSL": 314.2,
            "FEDFUNDS": 5.25,
            "UNRATE": 4.0,
            "DGS10": 4.35,
            "T10Y2Y": -0.3,
        },
    }


def test_get_sentiment_empty(store, monkeypatch):
    summary = {
        "ticker": "NVDA",
        "average_tone": None,
        "article_count": 0,
        "positive_ratio": 0,
        "negative_ratio": 0,
    }
    monkeypatch.setattr(middleware_app, "store", store)
    with patch("src.macros.gdelt_ingestor.GDELTIngestor") as mock_ingestor:
        mock_ingestor.return_value.get_sentiment_summary.return_value = summary

        result = data_tools.get_sentiment_handler(store, ticker="nvda", days=0)

    mock_ingestor.return_value.get_sentiment_summary.assert_called_once_with("NVDA", days=1)
    assert result == summary


def test_get_guidance_not_found(store, monkeypatch):
    monkeypatch.setattr(middleware_app, "store", store)
    with patch("src.macros.earnings_transcripts.EarningsTranscriptIngestor") as mock_ingestor:
        mock_ingestor.return_value.get_latest_guidance.return_value = {}

        result = data_tools.get_guidance_handler(store, ticker="nvda")

    mock_ingestor.return_value.get_latest_guidance.assert_called_once_with("NVDA")
    assert result == {"ticker": "NVDA", "guidance": {}, "status": "not_found"}


def test_get_estimates_returns_estimate_facts_and_growth(store):
    _seed_metric(
        store,
        "NVDA",
        "estimate_revenue_next_y",
        120.0,
        period="FY2027E",
        unit="usd",
        period_type="estimate",
        source_type="estimates",
    )
    _seed_metric(
        store,
        "NVDA",
        "estimate_eps_next_y",
        6.0,
        period="FY2027E",
        unit="usd",
        period_type="estimate",
        source_type="estimates",
    )
    _seed_metric(
        store,
        "NVDA",
        "estimate_revenue_current_q",
        25.0,
        period="2026-Q3E",
        unit="usd",
        period_type="estimate",
        source_type="other",
    )
    _seed_metric(store, "NVDA", "total_revenue", 100.0, period="FY2026", unit="usd")
    _seed_metric(store, "NVDA", "eps_diluted", 5.0, period="FY2026", unit="usd")

    result = data_tools.get_estimates_handler(store, ticker="nvda", horizon="year")

    assert result["ticker"] == "NVDA"
    assert result["horizon"] == "year"
    assert result["estimates"] == {
        "estimate_revenue_next_y": {"value": 120.0, "period": "FY2027E"},
        "estimate_eps_next_y": {"value": 6.0, "period": "FY2027E"},
    }
    assert result["growth_vs_realized"] == {
        "estimate_revenue_next_y": 0.2,
        "estimate_eps_next_y": 0.2,
    }


def test_get_price_targets_returns_estimate_facts_with_periods(store):
    for metric, value in {
        "price_target_mean": 185.0,
        "price_target_high": 220.0,
        "price_target_low": 150.0,
        "num_analysts": 42.0,
        "recommendation_mean": 1.8,
    }.items():
        _seed_metric(
            store,
            "NVDA",
            metric,
            value,
            period="2027-07E",
            unit="usd",
            period_type="estimate",
            source_type="estimates",
        )

    result = data_tools.get_price_targets_handler(store, ticker="nvda")

    assert result["ticker"] == "NVDA"
    assert result["price_targets"] == {
        "price_target_mean": {"value": 185.0, "period": "2027-07E"},
        "price_target_high": {"value": 220.0, "period": "2027-07E"},
        "price_target_low": {"value": 150.0, "period": "2027-07E"},
        "num_analysts": {"value": 42.0, "period": "2027-07E"},
        "recommendation_mean": {"value": 1.8, "period": "2027-07E"},
    }


def test_classify_trade_bias_long_from_recommendation(store, monkeypatch):
    _seed_metric(
        store, "NVDA", "recommendation_mean", 1.8,
        period="2027-07E", unit="score", period_type="estimate", source_type="estimates",
    )
    monkeypatch.setattr(
        data_tools, "get_sentiment_handler",
        lambda store, ticker, days=7: {"average_tone": None, "article_count": 0},
    )

    result = data_tools.classify_trade_bias_handler(store, ticker="nvda")

    assert result["ticker"] == "NVDA"
    assert result["bias"] == "long"
    assert result["evidence_status"] == "hit"
    assert result["web_search_allowed"] is False
    assert result["must_answer"] is True


def test_classify_trade_bias_short_from_recommendation(store, monkeypatch):
    _seed_metric(
        store, "NVDA", "recommendation_mean", 4.6,
        period="2027-07E", unit="score", period_type="estimate", source_type="estimates",
    )
    monkeypatch.setattr(
        data_tools, "get_sentiment_handler",
        lambda store, ticker, days=7: {"average_tone": None, "article_count": 0},
    )

    result = data_tools.classify_trade_bias_handler(store, ticker="nvda")

    assert result["bias"] == "short"
    assert result["evidence_status"] == "hit"


def test_classify_trade_bias_miss_without_evidence(store, monkeypatch):
    monkeypatch.setattr(
        data_tools, "get_sentiment_handler",
        lambda store, ticker, days=7: {"average_tone": None, "article_count": 0},
    )

    result = data_tools.classify_trade_bias_handler(store, ticker="ZZZZ")

    assert result["bias"] == "neutral"
    assert result["evidence_status"] == "miss"
    assert result["web_search_allowed"] is True


def test_check_freshness(store):
    store.mark_source_fresh("NVDA", "yfinance_fundamentals", 24)

    result = data_tools.check_freshness_handler(store, ticker="nvda")

    assert result["ticker"] == "NVDA"
    assert result["sources"]["yfinance_fundamentals"]["status"] == "fresh"
    assert result["overall"] in {"fresh", "partial"}
    assert isinstance(result["stale_sources"], list)


def test_all_read_tools_registered_write_false():
    importlib.reload(data_tools)
    expected = {
        "list_metrics",
        "query_facts",
        "get_fundamentals",
        "search_documents",
        "get_macro_snapshot",
        "get_sentiment",
        "get_guidance",
        "get_estimates",
        "get_price_targets",
        "classify_trade_bias",
        "check_freshness",
    }

    assert expected <= set(REGISTRY)
    assert all(REGISTRY[name].write is False for name in expected)
    schema_names = {entry["function"]["name"] for entry in openai_schema()}
    assert expected <= schema_names


def test_macro_snapshot_endpoint_shape(store, monkeypatch):
    _seed_metric(store, "MACRO", "GDP", 29000.0, unit="usd")
    _seed_metric(store, "MACRO", "CPIAUCSL", 314.2, unit="index")
    _seed_metric(store, "MACRO", "FEDFUNDS", 5.25, unit="percent")
    _seed_metric(store, "MACRO", "UNRATE", 4.0, unit="percent")
    _seed_metric(store, "MACRO", "DGS10", 4.35, unit="percent")
    _seed_metric(store, "MACRO", "T10Y2Y", -0.3, unit="percent")

    with TestClient(app) as client:
        monkeypatch.setattr(middleware_app, "store", store)
        response = client.get("/macro/snapshot")

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "gdp",
        "inflation_cpi",
        "fed_rate",
        "unemployment",
        "ten_year_treasury",
        "ten_two_spread",
        "timestamp",
    }
    data.pop("timestamp")
    assert data == {
        "gdp": 29000.0,
        "inflation_cpi": 314.2,
        "fed_rate": 5.25,
        "unemployment": 4.0,
        "ten_year_treasury": 4.35,
        "ten_two_spread": -0.3,
    }

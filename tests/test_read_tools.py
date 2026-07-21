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

from src.ingestion.records import ObservationRecord  # noqa: E402
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


def _seed_price_bar(store, ticker, market_date, metric_id, value, *, source_name="massive"):
    store.upsert_observation(ObservationRecord(
        observation_id=f"observation/{source_name}/{ticker}/{market_date}/{metric_id}",
        metric_id=metric_id,
        series_id=f"{ticker}:{metric_id}",
        value_text=str(value),
        value_numeric=value,
        unit="shares" if metric_id == "volume" else "usd",
        frequency="daily",
        period_start=market_date,
        period_end=market_date,
        vintage_at=None,
        as_of_at=market_date,
        # scope="global" (not "security") so the test doesn't need a
        # registered securities-table row; list_price_bars filters purely on
        # tickers_json, matching how the coverage tool treats provenance.
        scope="global",
        security_ids=(),
        tickers=(ticker,),
        sector=None,
        source_name=source_name,
        source_category="market_data",
        provider_record_id=f"{ticker}:{market_date}:{metric_id}",
        original_publisher=None,
        source_url="https://example.test/massive/bars",
        canonical_url=None,
        published_at=None,
        observed_at=None,
        accessed_at="2026-07-01T00:00:00Z",
        ingested_at="2026-07-01T00:00:00Z",
        license_label="provider_entitlement",
    ))


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


def test_get_price_history_pivots_bars_by_date_within_window(store):
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone.utc).date()
    day1 = (today - timedelta(days=2)).isoformat()
    day2 = (today - timedelta(days=1)).isoformat()
    _seed_price_bar(store, "NVDA", day1, "close", 120.5)
    _seed_price_bar(store, "NVDA", day1, "volume", 1000000)
    _seed_price_bar(store, "NVDA", day2, "close", 122.25)
    _seed_price_bar(store, "NVDA", day2, "volume", 1200000)
    # Outside the default 30-day window; must not appear in the result.
    _seed_price_bar(store, "NVDA", "2020-01-02", "close", 55.0)
    # A different ticker; must not leak into the NVDA result.
    _seed_price_bar(store, "AMD", day2, "close", 99.0)

    result = data_tools.get_price_history_handler(
        store, ticker="nvda", metrics=["close", "volume"], days=30
    )

    assert result["ticker"] == "NVDA"
    assert result["metrics"] == ["close", "volume"]
    assert result["bars"] == [
        {"date": day1, "close": 120.5, "volume": 1000000.0},
        {"date": day2, "close": 122.25, "volume": 1200000.0},
    ]


def test_get_price_history_defaults_to_close_and_rejects_unknown_metrics(store):
    result = data_tools.get_price_history_handler(store, ticker="nvda", metrics=["bogus"])

    assert result["ticker"] == "NVDA"
    assert "error" in result


def test_get_filing_overview_prefers_substantive_over_newer_form4(store, mock_chroma):
    # Regression: PLTR's filings table has only insider Form 4 rows plus older
    # substantive ones; picking the literal latest filing returns a Form 4 with
    # zero indexed sections. The tool must prefer the latest substantive type.
    store.register_filing(
        ticker="PLTR", filing_type="4", filing_date="2026-07-15", period="",
        accession="acc-4-1", source_url="https://sec.gov/acc-4-1",
    )
    store.register_filing(
        ticker="PLTR", filing_type="10-K", filing_date="2026-02-01", period="FY2025",
        accession="acc-10k-1", source_url="https://sec.gov/acc-10k-1",
    )
    mock_chroma.get_filing_section_families.return_value = []
    mock_chroma.get_document_family.return_value = []

    result = data_tools.get_filing_overview_handler(store, ticker="pltr")

    assert result["filing"]["filing_type"] == "10-K"
    assert result["filing"]["accession"] == "acc-10k-1"
    assert "coverage_note" not in result


def test_get_filing_overview_prefers_periodic_over_newer_8k(store, mock_chroma):
    # Regression: a June 8-K (event filing, metadata only) must not beat the
    # February 10-K for a default "break down the latest filing" ask; steer
    # with filing_type="8-K" to get the event filing explicitly.
    store.register_filing(
        ticker="PLTR", filing_type="8-K", filing_date="2026-06-09", period="",
        accession="acc-8k-1", source_url="https://sec.gov/acc-8k-1",
    )
    store.register_filing(
        ticker="PLTR", filing_type="10-K", filing_date="2026-02-17", period="FY2025",
        accession="acc-10k-2", source_url="https://sec.gov/acc-10k-2",
    )
    mock_chroma.get_filing_section_families.return_value = []
    mock_chroma.get_document_family.return_value = []

    result = data_tools.get_filing_overview_handler(store, ticker="pltr")
    assert result["filing"]["filing_type"] == "10-K"
    assert result["filing"]["accession"] == "acc-10k-2"

    explicit = data_tools.get_filing_overview_handler(
        store, ticker="pltr", filing_type="8-K"
    )
    assert explicit["filing"]["accession"] == "acc-8k-1"


def test_get_filing_overview_coverage_note_when_only_form4(store, mock_chroma):
    store.register_filing(
        ticker="PLTR2", filing_type="4", filing_date="2026-07-01", period="",
        accession="acc-4-1", source_url="https://sec.gov/acc-4-1",
    )
    store.register_filing(
        ticker="PLTR2", filing_type="4", filing_date="2026-07-10", period="",
        accession="acc-4-2", source_url="https://sec.gov/acc-4-2",
    )
    mock_chroma.get_filing_section_families.return_value = []
    mock_chroma.get_document_family.return_value = []

    result = data_tools.get_filing_overview_handler(store, ticker="pltr2")

    assert result["status"] == "metadata_only"
    assert result["filing"]["filing_type"] == "4"
    assert result["filing"]["accession"] == "acc-4-2"
    assert "10-K" in result["coverage_note"]
    assert {"filing_type": "4", "count": 2} in result["available_filing_types"]


def test_get_filing_overview_excerpt_fallback_from_chroma_chunks(store, mock_chroma):
    # No indexed sections (index_chunk_count effectively 0), but Chroma still
    # holds the filing's chunks under the sec:{accession}#{i} document family.
    store.register_filing(
        ticker="MSFT", filing_type="10-Q", filing_date="2026-04-01", period="2026-Q1",
        accession="acc-10q-1", source_url="https://sec.gov/acc-10q-1",
    )
    mock_chroma.get_filing_section_families.return_value = []
    mock_chroma.get_document_family.return_value = [
        {"id": "sec:acc-10q-1#0", "document": "First chunk about revenue.",
         "metadata": {"chunk_index": 0}},
        {"id": "sec:acc-10q-1#1", "document": "Second chunk about risk factors.",
         "metadata": {"chunk_index": 1}},
    ]

    result = data_tools.get_filing_overview_handler(store, ticker="msft")

    assert result["status"] == "found"
    assert result["excerpt_source"] == "chroma_raw_chunks"
    assert result["sections"] == [
        {"chunk_index": 0, "excerpt": "First chunk about revenue."},
        {"chunk_index": 1, "excerpt": "Second chunk about risk factors."},
    ]
    mock_chroma.get_document_family.assert_called_once_with(
        "sec:acc-10q-1", limit=data_tools._FILING_OVERVIEW_EXCERPT_CHUNKS, offset=0
    )


def test_get_filing_overview_excerpt_fallback_respects_budget(store, mock_chroma):
    store.register_filing(
        ticker="AMD", filing_type="10-K", filing_date="2026-01-01", period="FY2025",
        accession="acc-amd-1", source_url="https://sec.gov/acc-amd-1",
    )
    mock_chroma.get_filing_section_families.return_value = []
    big = "a" * 5000
    mock_chroma.get_document_family.return_value = [
        {"id": f"sec:acc-amd-1#{i}", "document": big, "metadata": {"chunk_index": i}}
        for i in range(20)
    ]

    result = data_tools.get_filing_overview_handler(store, ticker="amd")

    assert result["status"] == "found"
    assert result["excerpt_source"] == "chroma_raw_chunks"
    excerpt_chars = sum(len(s.get("excerpt", "")) for s in result["sections"])
    assert excerpt_chars <= data_tools._FILING_OVERVIEW_CHAR_BUDGET


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
        "get_price_history",
        "get_filing_overview",
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

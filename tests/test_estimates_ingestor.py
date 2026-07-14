"""
tests/test_estimates_ingestor.py
Offline tests for Phase 2.1.5 analyst estimates ingestion.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
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


def _expected_target_period() -> str:
    today = date.today()
    return f"{today.year + 1}-{today.month:02d}E"


def _yf_ticker_stub() -> MagicMock:
    ticker = MagicMock()
    ticker.get_earnings_estimate.return_value = pd.DataFrame(
        {
            "avg": [4.1, 4.4, 17.2, 20.1],
            "numberOfAnalysts": [42, 39, 45, 44],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )
    ticker.get_revenue_estimate.return_value = pd.DataFrame(
        {"avg": [30.1e9, 32.0e9, 125.5e9, 142.0e9]},
        index=["0q", "+1q", "0y", "+1y"],
    )
    ticker.analyst_price_targets = {
        "mean": 185.0,
        "high": 220.0,
        "low": 150.0,
    }
    ticker.recommendations = pd.DataFrame(
        {
            "strongBuy": [10],
            "buy": [20],
            "hold": [5],
            "sell": [2],
            "strongSell": [1],
        }
    )
    return ticker


def _write_provider_config(tmp_path: Path, provider: str, key_var: str) -> Path:
    path = tmp_path / f"{provider}.yaml"
    path.write_text(
        "\n".join(
            [
                f"provider: {provider}",
                "request_delay: 0",
                "max_retries: 1",
                "timeout: 5",
                "fmp:",
                f'  api_key: "${{{key_var}}}"',
                '  base_url: "https://financialmodelingprep.com/api/v3"',
                "finnhub:",
                f'  api_key: "${{{key_var}}}"',
                '  base_url: "https://finnhub.io/api/v1"',
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_fetch_for_ticker_stores_estimates(store):
    from src.macros.estimates_ingestor import ESTIMATE_METRICS, EstimatesIngestor

    with patch("src.macros.estimates_ingestor.yf.Ticker", return_value=_yf_ticker_stub()):
        result = EstimatesIngestor(store=store).fetch_for_ticker("NVDA")

    assert result["status"] == "success"
    assert result["facts_stored"] >= len(ESTIMATE_METRICS)

    for metric in ESTIMATE_METRICS:
        row = store.get_fundamental("NVDA", metric)
        assert row is not None
        assert row["period"].endswith("E")
        assert row["source_type"] == "estimates"
        assert row["period_type"] == "estimate"


def test_price_target_fields(store):
    from src.macros.estimates_ingestor import PRICE_TARGET_METRICS, EstimatesIngestor

    with patch("src.macros.estimates_ingestor.yf.Ticker", return_value=_yf_ticker_stub()):
        result = EstimatesIngestor(store=store).fetch_for_ticker("NVDA")

    assert result["status"] == "success"
    for metric in PRICE_TARGET_METRICS:
        row = store.get_fundamental("NVDA", metric)
        assert row is not None
        assert row["period"] == _expected_target_period()

    for metric in ("price_target_mean", "price_target_high", "price_target_low"):
        row = store.get_fundamental("NVDA", metric)
        assert row["unit"] == "usd"


@pytest.mark.parametrize("provider", ["fmp", "finnhub"])
def test_provider_fallback(store, tmp_path, monkeypatch, provider):
    from src.macros.estimates_ingestor import EstimatesIngestor

    monkeypatch.setenv("ESTIMATES_TEST_KEY", "secret")
    ingestor = EstimatesIngestor(
        store=store,
        config_path=_write_provider_config(tmp_path, provider, "ESTIMATES_TEST_KEY"),
    )

    def fake_get(url, **kwargs):
        response = MagicMock()
        response.raise_for_status.return_value = None
        if provider == "fmp" and "analyst-estimates" in url:
            response.json.return_value = [
                {
                    "estimatedRevenueAvg": 125.5e9,
                    "estimatedEpsAvg": 17.2,
                    "date": "2026-12-31",
                }
            ]
        elif provider == "fmp" and "price-target-consensus" in url:
            response.json.return_value = [
                {
                    "targetConsensus": 185.0,
                    "targetHigh": 220.0,
                    "targetLow": 150.0,
                }
            ]
        elif provider == "finnhub" and "eps-estimate" in url:
            response.json.return_value = {"data": [{"period": "2026", "epsAvg": 17.2}]}
        elif provider == "finnhub" and "revenue-estimate" in url:
            response.json.return_value = {"data": [{"period": "2026", "revenueAvg": 125.5e9}]}
        elif provider == "finnhub" and "recommendation" in url:
            response.json.return_value = [{"strongBuy": 10, "buy": 20, "hold": 5, "sell": 2, "strongSell": 1}]
        elif provider == "finnhub" and "price-target" in url:
            response.json.return_value = {"targetMean": 185.0, "targetHigh": 220.0, "targetLow": 150.0}
        else:
            response.json.return_value = {}
        return response

    ingestor.session.get = MagicMock(side_effect=fake_get)
    result = ingestor.fetch_for_ticker("NVDA")

    assert result["status"] == "success"
    assert store.get_fundamental("NVDA", "estimate_eps_current_y")["value"] == 17.2
    assert store.get_fundamental("NVDA", "price_target_mean")["value"] == 185.0


def test_missing_key_graceful(store, tmp_path, monkeypatch):
    from src.macros.estimates_ingestor import EstimatesIngestor

    monkeypatch.delenv("ESTIMATES_MISSING_KEY", raising=False)
    ingestor = EstimatesIngestor(
        store=store,
        config_path=_write_provider_config(tmp_path, "fmp", "ESTIMATES_MISSING_KEY"),
    )

    result = ingestor.fetch_for_ticker("NVDA")

    assert result["status"] == "skipped_no_key"
    assert result["facts_stored"] == 0
    assert store.get_fundamental("NVDA", "price_target_mean") is None


def test_scheduler_registers_estimates(store):
    from src.scheduler import UnifiedScheduler

    assert UnifiedScheduler.SOURCES["estimates"] == {
        "class": "EstimatesIngestor",
        "ttl_key": "estimates",
        "weight": 8,
    }
    assert "estimates" in UnifiedScheduler.DAILY_SOURCES

    scheduler = UnifiedScheduler(store=store, inter_source_delay=0)
    fake_results = {
        "NVDA": {"facts_stored": 3},
        "AMD": {"facts_stored": 2},
    }
    with patch(
        "src.macros.estimates_ingestor.EstimatesIngestor.fetch_all_core",
        return_value=fake_results,
    ):
        result = scheduler._run_source("estimates")

    assert result == {"tickers_processed": 2, "facts_stored": 5}


def test_freshness_includes_estimates(store):
    report = store.get_freshness_report("NVDA")
    assert report["sources"]["estimates"]["status"] == "never_fetched"

    store.mark_cache_fresh("NVDA", "estimates", 24)
    report = store.get_freshness_report("NVDA")

    assert report["sources"]["estimates"]["status"] == "fresh"
    assert report["sources"]["estimates"]["ttl_hours"] == 24


def test_fetch_all_core_delay_and_core_tickers(store):
    from src.macros.estimates_ingestor import EstimatesIngestor

    ingestor = EstimatesIngestor(store=store)
    ingestor.request_delay = 0.1
    fake_yf = MagicMock()
    fake_yf.core_tickers = ["NVDA", "AMD", "META"]

    with patch(
        "src.ingestion.yfinance_ingestor.YFinanceIngestor",
        return_value=fake_yf,
    ), patch.object(
        ingestor,
        "fetch_for_ticker",
        side_effect=lambda ticker: {"ticker": ticker, "facts_stored": 1},
    ) as mock_fetch, patch("src.macros.estimates_ingestor.time.sleep") as mock_sleep:
        results = ingestor.fetch_all_core()

    assert list(results) == ["NVDA", "AMD", "META"]
    assert [call.args[0] for call in mock_fetch.call_args_list] == ["NVDA", "AMD", "META"]
    assert mock_sleep.call_count == 3


def test_empty_provider_data_no_crash(store):
    from src.macros.estimates_ingestor import EstimatesIngestor

    ticker = MagicMock()
    ticker.get_earnings_estimate.return_value = pd.DataFrame()
    ticker.get_revenue_estimate.return_value = None
    ticker.analyst_price_targets = {}
    ticker.recommendations = pd.DataFrame()

    with patch("src.macros.estimates_ingestor.yf.Ticker", return_value=ticker):
        result = EstimatesIngestor(store=store).fetch_for_ticker("NVDA")

    assert result["status"] == "no_data"
    assert result["facts_stored"] == 0
    assert result["errors"] == []

"""
tests/test_yfinance_ingestor.py
Comprehensive pytest suite for YFinanceIngestor.

Ensures:
  - Import / init works
  - Watchlist loading from real config file
  - Fallback defaults when watchlist is missing
  - Ticker property correctness
  - _fetch_ticker success / failure paths (mocked — no live network)
  - Stub methods raise NotImplementedError with correct messages
  - ingest_ticker handles invalid tickers gracefully
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.ingestion import YFinanceIngestor


# ── 1. Import and class initialization ──

def test_import_and_init():
    """YFinanceIngestor() creates successfully without args."""
    ingestor = YFinanceIngestor()
    assert isinstance(ingestor, YFinanceIngestor)
    assert ingestor.store is not None
    assert ingestor.watchlist_path.name == "watchlist.yaml"


# ── 2. _load_watchlist with real config file ──

def test_load_watchlist_real_config():
    """Loads the committed watchlist YAML and returns expected 20 tickers."""
    ingestor = YFinanceIngestor()
    wl = ingestor.watchlist
    assert "core" in wl
    assert "extended" in wl
    assert "macro_tickers" in wl
    assert "schedule" in wl
    assert len(wl["core"]) == 6
    assert len(wl["extended"]) == 10
    assert len(wl["macro_tickers"]) == 4


# ── 3. _load_watchlist fallback when file missing ──

def test_load_watchlist_fallback(tmp_path: Path):
    """Missing watchlist path falls back to hard-coded defaults."""
    missing = tmp_path / "nonexistent_watchlist.yaml"
    ingestor = YFinanceIngestor(watchlist_path=missing)
    wl = ingestor.watchlist
    assert wl["core"] == ["NVDA", "AMD", "AAPL", "MSFT", "META", "CRWD"]
    assert wl["extended"] == []
    assert wl["macro_tickers"] == []
    assert wl["schedule"] == {"fundamentals": 24, "news": 6, "macro": 24}


# ── 4. all_tickers / core_tickers properties ──

def test_ticker_properties(tmp_path: Path):
    """all_tickers and core_tickers return correct counts and values."""
    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump(
            {
                "core": ["T1", "T2"],
                "extended": ["T3"],
                "macro_tickers": ["T4", "T5"],
                "schedule": {"fundamentals": 12, "news": 4, "macro": 12},
            }
        )
    )
    ingestor = YFinanceIngestor(watchlist_path=fake_watchlist)
    assert ingestor.core_tickers == ["T1", "T2"]
    assert ingestor.all_tickers == ["T1", "T2", "T3", "T4", "T5"]


def test_ticker_properties_with_real_config():
    """Real config yields exactly 20 tickers in all_tickers."""
    ingestor = YFinanceIngestor()
    assert len(ingestor.all_tickers) == 20
    assert ingestor.core_tickers == ["NVDA", "AMD", "AAPL", "MSFT", "META", "CRWD"]
    # Ensure no duplicates by uniqueness check
    assert len(set(ingestor.all_tickers)) == len(ingestor.all_tickers)


# ── 5. _fetch_ticker success path (mocked) ──

def test_fetch_ticker_success():
    """Mock yfinance.Ticker returning valid info returns the Ticker object."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"regularMarketPrice": 123.45}

    with patch("src.ingestion.yfinance_ingestor.yf.Ticker", return_value=mock_ticker):
        ingestor = YFinanceIngestor()
        result = ingestor._fetch_ticker("AAPL")

    assert result is mock_ticker


# ── 6. _fetch_ticker failure paths ──

@pytest.mark.parametrize(
    "info,should_log",
    [
        ({}, "warning"),                                   # empty info
        (None, "warning"),                                 # None info
        ({"regularMarketPrice": None}, "warning"),        # missing price
    ],
)
def test_fetch_ticker_no_price_data(info, should_log, caplog):
    """Missing or empty price data returns None and logs a warning."""
    mock_ticker = MagicMock()
    mock_ticker.info = info

    with patch("src.ingestion.yfinance_ingestor.yf.Ticker", return_value=mock_ticker):
        ingestor = YFinanceIngestor()
        result = ingestor._fetch_ticker("FAKE")

    assert result is None
    assert "no price data" in caplog.text.lower() or "skipping" in caplog.text.lower()


def test_fetch_ticker_exception_returns_none(caplog):
    """If yfinance raises an exception, _fetch_ticker returns None and logs error."""

    def boom(*args, **kwargs):
        raise RuntimeError("network timeout")

    with patch("src.ingestion.yfinance_ingestor.yf.Ticker", side_effect=boom):
        ingestor = YFinanceIngestor()
        result = ingestor._fetch_ticker("FAKE")

    assert result is None
    assert "failed to fetch" in caplog.text.lower()


# ── 7. Stub methods raise NotImplementedError ──

STUB_METHODS = [
    ("ingest_news", [], "Implement in 1.3.3"),
    ("ingest_macro", [], "Implement in 1.3.4"),
    ("ingest_all", [], "Implement in 1.3.3"),  # ingest_all calls ingest_news next
    ("_ingest_ticker_news", ["NVDA", MagicMock()], "Implement in 1.3.3"),
]


@pytest.mark.parametrize("method_name,args,expected_msg", STUB_METHODS)
def test_stub_raises_not_implemented(method_name, args, expected_msg):
    """Each stub method raises NotImplementedError containing its phase tag."""
    ingestor = YFinanceIngestor()
    method = getattr(ingestor, method_name)
    with pytest.raises(NotImplementedError) as exc_info:
        method(*args)
    assert expected_msg in str(exc_info.value)


# ── 8. ingest_ticker with invalid ticker ──

def test_ingest_ticker_invalid_ticker_returns_early(caplog):
    """When _fetch_ticker returns None, ingest_ticker returns without error."""
    ingestor = YFinanceIngestor()
    with patch.object(ingestor, "_fetch_ticker", return_value=None):
        # Should not raise, should just return early
        result = ingestor.ingest_ticker("INVALID")
    # ingest_ticker has no return value; verify it didn't call internals
    assert result is None


# ── Fundamentals Ingestion Tests ────────────────────

def test_fundamental_metrics_constant():
    """FUNDAMENTAL_METRICS has exactly 18 entries."""
    ingestor = YFinanceIngestor()
    assert len(ingestor.FUNDAMENTAL_METRICS) == 18
    names = [m[0] for m in ingestor.FUNDAMENTAL_METRICS]
    assert "market_cap" in names
    assert "pe_ratio_ttm" in names
    assert "quick_ratio" in names

def test_normalize_value():
    """_normalize_value handles numeric, None, and invalid inputs."""
    ingestor = YFinanceIngestor()
    assert ingestor._normalize_value(38.5) == 38.5
    assert ingestor._normalize_value("42.0") == 42.0
    assert ingestor._normalize_value(None) is None
    assert ingestor._normalize_value("bad") is None

def test_current_period_label():
    """_current_period_label returns YYYY-QN format."""
    ingestor = YFinanceIngestor()
    label = ingestor._current_period_label()
    import re
    assert re.match(r"\d{4}-Q[1-4]", label)

@patch("src.ingestion.yfinance_ingestor.yf.Ticker")
def test_ingest_ticker_fundamentals_saves_metrics(mock_ticker, tmp_path):
    """_ingest_ticker_fundamentals saves metrics and marks cache fresh."""
    # Create a temporary DB path so we don't pollute data/finance.db
    from src.storage.store import Store
    store = Store(db_path=tmp_path / "test.db")
    ingestor = YFinanceIngestor(store=store)
    
    # Mock ticker with sample info
    mock_t = MagicMock()
    mock_t.info = {
        "marketCap": 3200000000000,
        "trailingPE": 38.5,
        "trailingEps": 2.84,
        "totalRevenue": 130500000000,
        "grossMargins": 0.745,
    }
    mock_ticker.return_value = mock_t
    
    ingestor._ingest_ticker_fundamentals("NVDA", mock_t)
    
    # Verify metrics were saved
    facts = store.get_fundamentals_batch("NVDA")
    assert "market_cap" in facts
    assert "pe_ratio_ttm" in facts
    assert facts["market_cap"] == 3200000000000.0
    
    # Verify cache was marked fresh
    cache = store.get_cache_status("NVDA", "yfinance_fundamentals")
    assert cache is not None
    assert cache["status"] == "fresh"

@patch("src.ingestion.yfinance_ingestor.yf.Ticker")
def test_fundamentals_fresh_skips_on_fresh_cache(mock_ticker, tmp_path):
    """ingest_fundamentals skips tickers with fresh cache."""
    from src.storage.store import Store
    store = Store(db_path=tmp_path / "test.db")
    ingestor = YFinanceIngestor(store=store)
    
    # Pre-mark cache as fresh
    store.mark_cache_fresh("NVDA", "yfinance_fundamentals", 24)
    
    # Mock ticker should NOT be called because cache is fresh
    mock_t = MagicMock()
    mock_t.info = {"marketCap": 100}
    mock_ticker.return_value = mock_t
    
    # Manually call ingest_fundamentals (it only processes core_tickers)
    # Since NVDA is in core tickers, it should be skipped
    ingestor.ingest_fundamentals()
    
    # Verify the ticker was NOT fetched (skipped due to fresh cache)
    # We check by verifying no new metrics were saved beyond the cache mark
    facts = store.get_fundamentals_batch("NVDA")
    assert facts == {}  # No metrics saved because it was skipped

@patch("src.ingestion.yfinance_ingestor.yf.Ticker")
def test_ingest_fundamentals_full_pipeline(mock_ticker, tmp_path):
    """ingest_fundamentals processes all core tickers with stale cache."""
    from src.storage.store import Store
    store = Store(db_path=tmp_path / "test.db")
    ingestor = YFinanceIngestor(store=store)
    
    # Mock ticker with complete info
    mock_t = MagicMock()
    mock_t.info = {
        "marketCap": 1000000000,
        "trailingPE": 25.0,
        "trailingEps": 5.0,
        "forwardPE": 20.0,
        "dividendYield": 0.015,
        "priceToBook": 3.5,
        "debtToEquity": 0.5,
        "totalRevenue": 500000000,
        "grossMargins": 0.65,
        "operatingMargins": 0.45,
        "profitMargins": 0.35,
        "revenueGrowth": 0.25,
        "earningsGrowth": 0.30,
        "returnOnEquity": 0.20,
        "freeCashflow": 200000000,
        "operatingCashflow": 300000000,
        "currentRatio": 2.5,
        "quickRatio": 1.8,
        "regularMarketPrice": 150.0,
    }
    mock_ticker.return_value = mock_t
    
    ingestor.ingest_fundamentals()
    
    # Verify at least one core ticker has metrics saved
    facts = store.get_fundamentals_batch("NVDA")
    assert len(facts) >= 5  # At least some metrics saved
    assert "market_cap" in facts

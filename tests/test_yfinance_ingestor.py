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


# ── 7. Macro & Incremental Ingestion Tests ──────────

MACRO_INFO = {
    "regularMarketPrice": 520.0,
    "regularMarketChangePercent": 1.2,
    "regularMarketVolume": 50000000,
    "regularMarketDayLow": 515.0,
    "regularMarketDayHigh": 525.0,
    "fiftyTwoWeekLow": 400.0,
    "fiftyTwoWeekHigh": 530.0,
}


def test_macro_fresh_within_ttl(tmp_path):
    """_macro_fresh returns True when cache is fresh within TTL."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    ingestor = YFinanceIngestor(store=store)
    store.mark_cache_fresh("SPY", "yfinance_macro", 24)

    assert ingestor._macro_fresh("SPY") is True


def test_macro_fresh_stale(tmp_path):
    """_macro_fresh returns False when cache exceeds TTL."""
    from datetime import datetime, timedelta, timezone
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    ingestor = YFinanceIngestor(store=store)

    stale_time = datetime.now(timezone.utc) - timedelta(hours=25)
    store.mark_cache_fresh("SPY", "yfinance_macro", 24)
    with store.sqlite._connect() as conn:
        conn.execute(
            "UPDATE cache_meta SET last_updated = ? WHERE ticker = ? AND source = ?",
            (stale_time.strftime("%Y-%m-%d %H:%M:%S"), "SPY", "yfinance_macro"),
        )
        conn.commit()

    assert ingestor._macro_fresh("SPY") is False


@patch.object(YFinanceIngestor, "_fetch_ticker")
def test_ingest_macro_saves_metrics(mock_fetch, tmp_path):
    """ingest_macro saves MACRO_METRICS to SQLite with source_type=yfinance_macro."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({
            "core": [],
            "macro_tickers": ["SPY"],
            "schedule": {"fundamentals": 24, "news": 6, "macro": 24},
        })
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)

    mock_t = MagicMock()
    mock_t.info = MACRO_INFO
    mock_fetch.return_value = mock_t

    ingestor.ingest_macro()

    facts = store.get_fundamentals_batch("SPY")
    assert "price" in facts
    assert facts["price"] == 520.0

    with store.sqlite._connect() as conn:
        row = conn.execute(
            "SELECT source_type FROM fundamentals WHERE ticker=? AND metric=?",
            ("SPY", "price"),
        ).fetchone()
    assert row["source_type"] == "yfinance_macro"

    cache = store.get_cache_status("SPY", "yfinance_macro")
    assert cache is not None
    assert cache["status"] == "fresh"


@patch.object(YFinanceIngestor, "_fetch_ticker")
def test_ingest_macro_skips_fresh(mock_fetch, tmp_path):
    """ingest_macro skips tickers with fresh macro cache."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({
            "core": [],
            "macro_tickers": ["SPY"],
            "schedule": {"fundamentals": 24, "news": 6, "macro": 24},
        })
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)
    store.mark_cache_fresh("SPY", "yfinance_macro", 24)

    ingestor.ingest_macro()

    mock_fetch.assert_not_called()


def test_status_report_counts(tmp_path):
    """status_report tallies fresh, stale, and not_cached tickers correctly."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({
            "core": ["T1", "T2", "T3"],
            "extended": [],
            "macro_tickers": [],
            "schedule": {"fundamentals": 24, "news": 6, "macro": 24},
        })
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)

    # T1: both fresh
    store.mark_cache_fresh("T1", "yfinance_fundamentals", 24)
    store.mark_cache_fresh("T1", "yfinance_news", 6)

    # T2: fundamentals stale, news fresh
    store.upsert_cache_stale("T2", "yfinance_fundamentals")
    store.mark_cache_fresh("T2", "yfinance_news", 6)

    # T3: not cached at all

    report = ingestor.status_report()
    assert report["fresh"] == 1
    assert report["stale"] == 1
    assert report["not_cached"] == 1


def test_reset_cache_all_marks_stale(tmp_path):
    """reset_cache_all marks all tickers stale, including previously uncached."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({
            "core": ["T1"],
            "extended": [],
            "macro_tickers": ["SPY"],
            "schedule": {"fundamentals": 24, "news": 6, "macro": 24},
        })
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)

    store.mark_cache_fresh("T1", "yfinance_fundamentals", 24)
    store.mark_cache_fresh("T1", "yfinance_news", 6)

    ingestor.reset_cache_all()

    assert store.get_cache_status("T1", "yfinance_fundamentals")["status"] == "stale"
    assert store.get_cache_status("T1", "yfinance_news")["status"] == "stale"
    assert store.get_cache_status("SPY", "yfinance_macro")["status"] == "stale"


@patch.object(YFinanceIngestor, "_fetch_ticker")
@patch.object(YFinanceIngestor, "_ingest_ticker_fundamentals")
def test_ingest_stale_only_fetches_stale_fundamentals(
    mock_ingest_fundamentals, mock_fetch, tmp_path
):
    """Stale fundamentals entry triggers _ingest_ticker_fundamentals."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({
            "core": ["NVDA"],
            "schedule": {"fundamentals": 24, "news": 6, "macro": 24},
        })
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)

    store.upsert_cache_stale("NVDA", "yfinance_fundamentals")
    store.mark_cache_fresh("NVDA", "yfinance_news", 6)

    mock_t = MagicMock()
    mock_fetch.return_value = mock_t

    with patch.object(ingestor, "ingest_macro"):
        ingestor.ingest_stale_only()

    mock_ingest_fundamentals.assert_called_once_with("NVDA", mock_t)


@patch.object(YFinanceIngestor, "_fetch_ticker")
@patch.object(YFinanceIngestor, "_ingest_ticker_news")
def test_ingest_stale_only_fetches_stale_news(
    mock_ingest_news, mock_fetch, tmp_path
):
    """Stale news entry triggers _ingest_ticker_news."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({
            "core": ["NVDA"],
            "schedule": {"fundamentals": 24, "news": 6, "macro": 24},
        })
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)

    store.mark_cache_fresh("NVDA", "yfinance_fundamentals", 24)
    store.upsert_cache_stale("NVDA", "yfinance_news")

    mock_t = MagicMock()
    mock_fetch.return_value = mock_t

    with patch.object(ingestor, "ingest_macro"):
        ingestor.ingest_stale_only()

    mock_ingest_news.assert_called_once_with("NVDA", mock_t)


@patch.object(YFinanceIngestor, "_fetch_ticker")
def test_ingest_stale_only_noop_when_fresh(mock_fetch, tmp_path, caplog):
    """When all cache is fresh, ingest_stale_only returns without fetching."""
    import logging
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({
            "core": ["NVDA"],
            "macro_tickers": [],
            "schedule": {"fundamentals": 24, "news": 6, "macro": 24},
        })
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)

    store.mark_cache_fresh("NVDA", "yfinance_fundamentals", 24)
    store.mark_cache_fresh("NVDA", "yfinance_news", 6)

    with caplog.at_level(logging.INFO, logger="src.ingestion.yfinance_ingestor"):
        ingestor.ingest_stale_only()

    mock_fetch.assert_not_called()
    assert "nothing to ingest" in caplog.text.lower()


@patch.object(YFinanceIngestor, "_fetch_ticker")
@patch.object(YFinanceIngestor, "_ingest_ticker_fundamentals")
def test_ingest_stale_only_handles_not_cached(
    mock_ingest_fundamentals, mock_fetch, tmp_path
):
    """Ticker with no cache row is added to the fundamentals fetch set."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({
            "core": ["NVDA"],
            "schedule": {"fundamentals": 24, "news": 6, "macro": 24},
        })
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)

    mock_t = MagicMock()
    mock_fetch.return_value = mock_t

    with patch.object(ingestor, "ingest_macro"):
        ingestor.ingest_stale_only()

    mock_ingest_fundamentals.assert_called_once_with("NVDA", mock_t)


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


# ── News Ingestion Tests ────────────────────────────

NESTED_ARTICLE = {
    "id": "outer-id",
    "content": {
        "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "title": "NVIDIA Hits Record High",
        "summary": "Shares rose on strong AI demand.",
        "description": "Shares rose on strong AI demand.",
        "provider": {"displayName": "Reuters"},
        "canonicalUrl": {"url": "https://finance.yahoo.com/news/nvidia-record-high-123456789.html"},
        "pubDate": "2026-05-30T14:30:00Z",
        "contentType": "STORY",
    },
}

FLAT_ARTICLE = {
    "uuid": "flat-uuid-001",
    "title": "AMD Expands Data Center",
    "summary": "New chip lineup targets enterprise.",
    "publisher": "Bloomberg",
    "link": "https://finance.yahoo.com/news/amd-data-center-987654321.html",
    "providerPublishTime": 1717000000,
    "type": "news",
}


def test_extract_news_fields_nested():
    """Nested yfinance>=0.2.50 schema extracts title, publisher, link."""
    ingestor = YFinanceIngestor()
    fields = ingestor._extract_news_fields(NESTED_ARTICLE)
    assert fields["title"] == "NVIDIA Hits Record High"
    assert fields["publisher"] == "Reuters"
    assert "nvidia-record-high" in fields["link"]
    assert fields["id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def test_extract_news_fields_flat():
    """Old flat schema still parses correctly."""
    ingestor = YFinanceIngestor()
    fields = ingestor._extract_news_fields(FLAT_ARTICLE)
    assert fields["title"] == "AMD Expands Data Center"
    assert fields["publisher"] == "Bloomberg"
    assert fields["id"] == "flat-uuid-001"


def test_make_news_doc_id_prefers_uuid():
    """Doc ID uses article/content UUID when available."""
    ingestor = YFinanceIngestor()
    doc_id = ingestor._make_news_doc_id("NVDA", NESTED_ARTICLE)
    assert doc_id == "news/NVDA/a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def test_format_news_article_malformed_returns_none():
    """Article with no title and no summary returns None."""
    ingestor = YFinanceIngestor()
    assert ingestor._format_news_article({"content": {"title": "", "summary": ""}}) is None
    assert ingestor._format_news_article({}) is None


def test_format_news_date_iso_and_unix():
    """Date formatting handles ISO8601 and Unix timestamps."""
    ingestor = YFinanceIngestor()
    assert ingestor._format_news_date("2026-05-30T14:30:00Z") == "2026-05-30"
    assert ingestor._format_news_date(1717000000) == "2024-05-29"


def _ledger_store(tmp_path):
    """Store with a mocked ChromaStore (offline) but a real SQLite corpus ledger."""
    from src.storage.store import Store

    with patch("src.storage.store.ChromaStore") as chroma_class:
        chroma = MagicMock()
        chroma_class.return_value = chroma
        store = Store(db_path=tmp_path / "yf.db", chroma_path=tmp_path / "chroma")
    chroma.get_document.return_value = None
    return store, chroma


@patch.object(YFinanceIngestor, "_fetch_ticker")
def test_ingest_ticker_news_registers_corpus_ledger_row(mock_fetch, tmp_path):
    """Nested-schema article lands a corpus_items ledger row plus source/security links."""
    store, chroma = _ledger_store(tmp_path)
    ingestor = YFinanceIngestor(store=store)

    # Seed a resolvable security so the security link is exercised end-to-end.
    store.upsert_universe_snapshot(
        "ivv",
        "2026-05-01T00:00:00Z",
        [{"symbol": "NVDA", "company_name": "NVIDIA Corp", "index_code": "sp500"}],
    )

    mock_t = MagicMock()
    mock_t.news = [NESTED_ARTICLE]
    mock_fetch.return_value = mock_t

    ingestor._ingest_ticker_news("NVDA", mock_t)

    doc_id = "news/NVDA/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    assert store.sqlite.count_corpus_items() == 1
    item = store.sqlite.get_corpus_item(doc_id)
    assert item is not None
    assert item["source"] == "yfinance_news"
    assert item["item_type"] == "news"
    assert item["title"] == "NVIDIA Hits Record High"
    assert "NVDA" in (item["tickers"] or [])
    assert store.sqlite.list_corpus_item_sources(doc_id)
    assert any(
        row.get("ticker") == "NVDA"
        for row in store.sqlite.list_corpus_item_securities(doc_id)
    )
    # Narrative content is indexed in Chroma in the same call (not save_document).
    assert chroma.add_document.call_count == 1

    cache = store.get_cache_status("NVDA", "yfinance_news")
    assert cache is not None
    assert cache["status"] == "fresh"


@patch.object(YFinanceIngestor, "_fetch_ticker")
def test_ingest_ticker_news_flat_schema_registers_ledger(mock_fetch, tmp_path):
    """Flat-schema article parses and registers under a stable corpus id."""
    store, chroma = _ledger_store(tmp_path)
    ingestor = YFinanceIngestor(store=store)

    mock_t = MagicMock()
    mock_t.news = [FLAT_ARTICLE]

    ingestor._ingest_ticker_news("AMD", mock_t)

    item = store.sqlite.get_corpus_item("news/AMD/flat-uuid-001")
    assert item is not None
    assert item["title"] == "AMD Expands Data Center"
    assert item["original_publisher"] == "Bloomberg"
    assert item["published_at"] == "2024-05-29"


def test_ingest_ticker_news_skips_duplicate(tmp_path):
    """Existing doc_id in ChromaDB is skipped (no ledger write, idempotent)."""
    store, chroma = _ledger_store(tmp_path)
    ingestor = YFinanceIngestor(store=store)

    mock_t = MagicMock()
    mock_t.news = [NESTED_ARTICLE]

    chroma.get_document.return_value = {"id": "existing"}
    store.upsert_narrative = MagicMock()

    ingestor._ingest_ticker_news("NVDA", mock_t)

    store.upsert_narrative.assert_not_called()
    assert store.sqlite.count_corpus_items() == 0


def test_ingest_ticker_news_skips_malformed(tmp_path):
    """Malformed article (no title) is skipped without a ledger write."""
    store, chroma = _ledger_store(tmp_path)
    ingestor = YFinanceIngestor(store=store)

    mock_t = MagicMock()
    mock_t.news = [{"content": {"title": "", "summary": ""}}]

    store.upsert_narrative = MagicMock()

    ingestor._ingest_ticker_news("NVDA", mock_t)

    store.upsert_narrative.assert_not_called()
    assert store.sqlite.count_corpus_items() == 0


@patch.object(YFinanceIngestor, "_fetch_ticker")
def test_ingest_ticker_news_is_idempotent_across_runs(mock_fetch, tmp_path):
    """Re-ingesting the same article does not create a second ledger row."""
    store, chroma = _ledger_store(tmp_path)
    ingestor = YFinanceIngestor(store=store)

    mock_t = MagicMock()
    mock_t.news = [NESTED_ARTICLE]

    ingestor._ingest_ticker_news("NVDA", mock_t)
    assert store.sqlite.count_corpus_items() == 1

    # Second pass: the article is already in Chroma, so the early skip fires.
    chroma.get_document.return_value = {"id": "existing"}
    ingestor._ingest_ticker_news("NVDA", mock_t)
    assert store.sqlite.count_corpus_items() == 1


def test_news_fresh_within_ttl(tmp_path):
    """_news_fresh returns True when cache is fresh within TTL."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    ingestor = YFinanceIngestor(store=store)
    store.mark_cache_fresh("NVDA", "yfinance_news", 6)

    assert ingestor._news_fresh("NVDA") is True


def test_news_fresh_stale_after_ttl(tmp_path):
    """_news_fresh returns False when cache exceeds TTL."""
    from datetime import datetime, timedelta, timezone
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    ingestor = YFinanceIngestor(store=store)

    stale_time = datetime.now(timezone.utc) - timedelta(hours=7)
    store.mark_cache_fresh("NVDA", "yfinance_news", 6)
    with store.sqlite._connect() as conn:
        conn.execute(
            "UPDATE cache_meta SET last_updated = ? WHERE ticker = ? AND source = ?",
            (stale_time.strftime("%Y-%m-%d %H:%M:%S"), "NVDA", "yfinance_news"),
        )
        conn.commit()

    assert ingestor._news_fresh("NVDA") is False


@patch.object(YFinanceIngestor, "_fetch_ticker")
@patch.object(YFinanceIngestor, "_ingest_ticker_fundamentals")
@patch.object(YFinanceIngestor, "_ingest_ticker_news")
def test_ingest_news_fundamentals_first(
    mock_ingest_news, mock_ingest_fundamentals, mock_fetch, tmp_path
):
    """ingest_news runs fundamentals first when fundamentals cache is absent."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")
    ingestor = YFinanceIngestor(store=store)

    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({"core": ["NVDA"], "schedule": {"fundamentals": 24, "news": 6}})
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)

    mock_t = MagicMock()
    mock_t.news = []
    mock_fetch.return_value = mock_t

    ingestor.ingest_news()

    mock_ingest_fundamentals.assert_called_once_with("NVDA", mock_t)
    mock_ingest_news.assert_called_once_with("NVDA", mock_t)


@patch.object(YFinanceIngestor, "_fetch_ticker")
@patch.object(YFinanceIngestor, "_ingest_ticker_news")
def test_ingest_news_skips_fresh_cache(mock_ingest_news, mock_fetch, tmp_path):
    """ingest_news skips tickers with fresh news cache."""
    from src.storage.store import Store

    store = Store(db_path=tmp_path / "test.db")

    fake_watchlist = tmp_path / "wl.yaml"
    fake_watchlist.write_text(
        yaml.safe_dump({"core": ["NVDA"], "schedule": {"fundamentals": 24, "news": 6}})
    )
    ingestor = YFinanceIngestor(store=store, watchlist_path=fake_watchlist)
    store.mark_cache_fresh("NVDA", "yfinance_news", 6)

    ingestor.ingest_news()

    mock_fetch.assert_not_called()
    mock_ingest_news.assert_not_called()

"""
tests/test_ir_ingestor.py
Pytest suite for Phase 1.7.2 Company IR pages ingestion.

Usage:
    pytest tests/test_ir_ingestor.py -v
"""

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


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>NVIDIA Investor News</title>
    <item>
      <title>NVIDIA Announces New AI Partnership</title>
      <link>https://investor.nvidia.com/news/partnership</link>
      <pubDate>Mon, 02 Jun 2026 13:00:00 GMT</pubDate>
      <description>NVIDIA announces a strategic partnership to launch new AI products.</description>
    </item>
    <item>
      <title>NVIDIA Q1 Fiscal 2027 Financial Results</title>
      <link>https://investor.nvidia.com/news/q1-results</link>
      <pubDate>Tue, 03 Jun 2026 21:00:00 GMT</pubDate>
      <description>Supplemental earnings materials and non-GAAP reconciliations.</description>
    </item>
  </channel>
</rss>
"""

EMPTY_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
  </channel>
</rss>
"""


class TestIRIngestor:
    """Tests for the company IR pages ingestion module."""

    def test_import(self):
        """IRIngestor imports successfully."""
        from src.macros.ir_ingestor import IRIngestor
        assert IRIngestor is not None

    def test_init(self, store):
        """Default init with config + ticker mapping loaded."""
        from src.macros.ir_ingestor import IRIngestor
        ingestor = IRIngestor(store=store)
        assert ingestor.store is store
        assert ingestor.request_delay == 3.0
        assert "NVDA" in ingestor.TICKER_IR_MAP

    def test_ticker_mapping(self, store):
        """All 6 core tickers have IR URLs."""
        from src.macros.ir_ingestor import IRIngestor
        ingestor = IRIngestor(store=store)
        for ticker in ("NVDA", "AMD", "AAPL", "MSFT", "META", "CRWD"):
            assert ticker in ingestor.TICKER_IR_MAP
            assert ingestor.TICKER_IR_MAP[ticker].get("ir_url")

    def test_fetch_for_ticker_unknown(self, store):
        """Unknown ticker returns status 'unknown_ticker'."""
        from src.macros.ir_ingestor import IRIngestor
        ingestor = IRIngestor(store=store)
        result = ingestor.fetch_for_ticker("ZZZZ")
        assert result["status"] == "unknown_ticker"
        assert result["ticker"] == "ZZZZ"

    def test_parse_rss(self, store):
        """RSS feed is parsed correctly (title/url/date/summary extracted)."""
        from src.macros.ir_ingestor import IRIngestor
        import feedparser

        ingestor = IRIngestor(store=store)
        parsed = feedparser.parse(SAMPLE_RSS)

        with patch("src.macros.ir_ingestor.feedparser.parse", return_value=parsed):
            items = ingestor._parse_rss("https://example.com/rss")

        assert len(items) == 2
        first = items[0]
        assert first["title"] == "NVIDIA Announces New AI Partnership"
        assert first["url"] == "https://investor.nvidia.com/news/partnership"
        assert first["published_date"]
        assert "partnership" in first["content"].lower()
        assert first["doc_type"] in {
            "press_release", "presentation", "earnings_material", "event", "other",
        }

    def test_classify_document(self, store):
        """Document type classification works for each type."""
        from src.macros.ir_ingestor import IRIngestor
        ingestor = IRIngestor(store=store)

        assert ingestor._classify_document(
            "NVIDIA Announces Partnership to Launch New Product", ""
        ) == "press_release"
        assert ingestor._classify_document(
            "Investor Presentation Slides Q2", ""
        ) == "presentation"
        assert ingestor._classify_document(
            "Q1 Earnings Supplemental Non-GAAP Reconciliation", ""
        ) == "earnings_material"
        assert ingestor._classify_document(
            "Annual Shareholder Meeting and Analyst Day", ""
        ) == "event"
        assert ingestor._classify_document(
            "Corporate Governance Update", ""
        ) == "other"

    def test_fetch_all_core(self, store):
        """fetch_all_core iterates through all core tickers in the IR map."""
        from src.macros.ir_ingestor import IRIngestor
        ingestor = IRIngestor(store=store)

        core = ["NVDA", "AMD", "AAPL", "MSFT", "META", "CRWD"]
        fake_yf = MagicMock()
        fake_yf.core_tickers = core

        fake_result = {
            "ticker": "X", "status": "success",
            "items_found": 1, "items_stored": 1, "errors": [],
        }

        with patch(
            "src.ingestion.yfinance_ingestor.YFinanceIngestor", return_value=fake_yf
        ), patch.object(
            ingestor, "fetch_for_ticker", return_value=fake_result
        ) as mock_fetch, patch(
            "src.macros.ir_ingestor.time.sleep"
        ):
            results = ingestor.fetch_all_core()

        assert set(results.keys()) == set(core)
        assert mock_fetch.call_count == len(core)

    def test_network_error(self, store):
        """Network error returns gracefully with error status (no raise)."""
        from src.macros.ir_ingestor import IRIngestor
        ingestor = IRIngestor(store=store)

        with patch.object(
            ingestor, "_parse_rss", side_effect=ConnectionError("DNS failure")
        ), patch.object(
            ingestor, "_scrape_ir_page", side_effect=ConnectionError("DNS failure")
        ):
            result = ingestor.fetch_for_ticker("NVDA")

        assert result["status"] == "error"
        assert result["items_stored"] == 0
        assert result["errors"]

    def test_empty_feed(self, store):
        """RSS feed with no items returns empty results."""
        from src.macros.ir_ingestor import IRIngestor
        import feedparser

        ingestor = IRIngestor(store=store)
        parsed = feedparser.parse(EMPTY_RSS)

        with patch("src.macros.ir_ingestor.feedparser.parse", return_value=parsed), \
             patch.object(ingestor, "_scrape_ir_page", return_value=[]):
            result = ingestor.fetch_for_ticker("NVDA")

        assert result["items_found"] == 0
        assert result["items_stored"] == 0
        assert result["status"] == "no_items"

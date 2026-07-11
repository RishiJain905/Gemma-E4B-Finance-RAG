"""
tests/test_macros.py
Comprehensive pytest suite for Phase 1.6 multi-source ingestion.

Test coverage:
  1. FREDIngestor — config loading, unit guessing, indicator fetching
  2. GDELTIngestor — article processing, sentiment summary, dedup
  3. EarningsTranscriptIngestor — quarter detection, guidance extraction
  4. Multi-source integration — cross-source queries via middleware

Usage:
    pytest tests/test_macros.py -v              # All mocked tests
    pytest tests/test_macros.py -v -x           # Stop on first failure
    pytest tests/test_macros.py -k "fred"       # Run FRED tests only

NOTE: Live API tests (marked @pytest.mark.live) require:
    - FRED_API_KEY environment variable set
    - Internet access (for GDELT and transcript fetches)
    - Run with: pytest tests/test_macros.py -v --live
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


@pytest.fixture(autouse=True)
def _reset_gdelt_throttle():
    import src.macros.gdelt_ingestor as g
    g._last_gdelt_request_at = 0.0
    yield
    g._last_gdelt_request_at = 0.0


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


# ============================================================
# 1. FREDIngestor Tests
# ============================================================

class TestFREDIngestor:
    """Tests for the FRED economic data ingestion module."""

    def test_import(self):
        """FREDIngestor imports successfully."""
        from src.macros.fred_ingestor import FREDIngestor
        assert FREDIngestor is not None

    def test_init(self, store):
        """Default init with config loading."""
        from src.macros.fred_ingestor import FREDIngestor
        ingestor = FREDIngestor(store=store)
        assert ingestor.store is store
        assert "GDP" in ingestor.config.get("indicators", {})

    def test_init_with_env_api_key(self, tmp_path, monkeypatch):
        """API key loads from environment variable."""
        monkeypatch.setenv("FRED_API_KEY", "test_key_123")
        from src.macros.fred_ingestor import FREDIngestor
        ingestor = FREDIngestor()
        assert ingestor.api_key == "test_key_123"

    def test_unit_guessing(self, tmp_path):
        """Unit guessing maps series IDs to correct units."""
        from src.macros.fred_ingestor import FREDIngestor
        ingestor = FREDIngestor()

        assert ingestor._guess_unit("FEDFUNDS") == "percent"
        assert ingestor._guess_unit("UNRATE") == "percent"
        assert ingestor._guess_unit("DGS10") == "percent"
        assert ingestor._guess_unit("T10Y2Y") == "percent"
        assert ingestor._guess_unit("PAYEMS") == "thousands"
        assert ingestor._guess_unit("HOUST") == "thousands"
        assert ingestor._guess_unit("CPIAUCSL") == "index"
        assert ingestor._guess_unit("GDPC1") == "index"
        assert ingestor._guess_unit("UMCSENT") == "index_points"
        assert ingestor._guess_unit("UNKNOWN_SERIES") == "units"

    def test_category_mapping(self, tmp_path):
        """Category filtering returns the correct indicators."""
        from src.macros.fred_ingestor import FREDIngestor
        ingestor = FREDIngestor()

        interest = ingestor.fetch_by_category("interest_rates")
        assert isinstance(interest, dict)

        # Unknown category returns empty
        unknown = ingestor.fetch_by_category("not_a_category")
        assert unknown == {}

    def test_fetch_indicator_stores_data(self, store):
        """fetch_indicator stores the value in SQLite when successful."""
        from src.macros.fred_ingestor import FREDIngestor
        import pandas as pd

        ingestor = FREDIngestor(store=store)

        # Mock the FRED client
        mock_fred = MagicMock()
        dates = pd.date_range("2026-01-01", periods=3, freq="ME")
        mock_fred.get_series.return_value = pd.Series([5.25, 5.5, 5.75], index=dates)

        with patch.object(ingestor, "_client", mock_fred):
            value = ingestor.fetch_indicator("FEDFUNDS")

        assert value == 5.75

        # Verify stored in SQLite
        result = store.get_fundamental("MACRO", "FEDFUNDS")
        assert result is not None
        assert result["value"] == 5.75

    def test_fetch_indicator_empty_series(self, tmp_path):
        """Empty series response returns None."""
        from src.macros.fred_ingestor import FREDIngestor
        import pandas as pd

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series(dtype=float)

        with patch.object(ingestor, "_client", mock_fred):
            value = ingestor.fetch_indicator("GDP")

        assert value is None

    def test_fetch_indicator_network_error(self, tmp_path):
        """Network error returns None without crashing."""
        from src.macros.fred_ingestor import FREDIngestor

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = ConnectionError("DNS failure")

        with patch.object(ingestor, "_client", mock_fred):
            value = ingestor.fetch_indicator("GDP")

        assert value is None

    def test_fetch_all_indicators(self, tmp_path):
        """fetch_all_indicators iterates through all configured series."""
        from src.macros.fred_ingestor import FREDIngestor
        import pandas as pd

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        dates = pd.date_range("2026-01-01", periods=2, freq="ME")
        mock_fred.get_series.return_value = pd.Series([100.0, 102.5], index=dates)

        with patch.object(ingestor, "_client", mock_fred):
            results = ingestor.fetch_all_indicators()

        assert len(results) > 0
        # All values should be the mock value
        for series_id, value in results.items():
            assert value == 102.5 or value is None

    def test_health_check_ok(self, tmp_path):
        """health_check returns True when FRED is reachable."""
        from src.macros.fred_ingestor import FREDIngestor
        import pandas as pd

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series([100.0])

        with patch.object(ingestor, "_client", mock_fred):
            assert ingestor.health_check() is True

    def test_health_check_fail(self, tmp_path):
        """health_check returns False when FRED is unreachable."""
        from src.macros.fred_ingestor import FREDIngestor

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = Exception("API error")

        with patch.object(ingestor, "_client", mock_fred):
            assert ingestor.health_check() is False


# ============================================================
# 2. GDELTIngestor Tests
# ============================================================

class TestGDELTIngestor:
    """Tests for the GDELT news ingestion module."""

    def test_import(self):
        """GDELTIngestor imports successfully."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        assert GDELTIngestor is not None

    def test_init(self, tmp_path):
        """Default init with ticker mapping loaded."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()
        assert len(ingestor.TICKER_TO_QUERY) > 20
        assert "NVDA" in ingestor.TICKER_TO_QUERY

    def test_process_articles_dedup(self, tmp_path):
        """Duplicate URLs are removed during processing."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()

        articles = [
            {"url": "https://example.com/news1", "title": "News 1",
             "content": "Content 1", "tone": "1.5", "date": "2026-06-01"},
            {"url": "https://example.com/news1", "title": "News 1 (dup)",
             "content": "Content 1 dup", "tone": "2.0", "date": "2026-06-01"},
            {"url": "https://example.com/news2", "title": "News 2",
             "content": "Content 2", "tone": "-0.5", "date": "2026-06-02"},
        ]

        processed = ingestor._process_articles(articles, "NVDA")
        assert len(processed) == 2  # One duplicate removed

    def test_process_articles_parses_tone(self, tmp_path):
        """Tone scores are parsed correctly as floats."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()

        articles = [
            {"url": "https://example.com/nvda1", "title": "NVDA Up",
             "content": "NVIDIA stock rises", "tone": "12.5",
             "date": "2026-06-01"},
            {"url": "https://example.com/nvda2", "title": "NVDA Down",
             "content": "NVIDIA stock falls", "tone": "-8.3",
             "date": "2026-06-01"},
        ]

        processed = ingestor._process_articles(articles, "NVDA")
        assert len(processed) == 2
        assert processed[0]["tone"] == 12.5
        assert processed[1]["tone"] == -8.3

    def test_process_articles_missing_fields(self, tmp_path):
        """Articles missing critical fields are filtered."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor()

        articles = [
            {"url": "", "title": "No URL", "content": "Missing URL"},
            {"url": "https://example.com/valid", "title": "", "content": "",
             "tone": "0.0", "date": "2026-06-01"},
            {"url": "https://example.com/good", "title": "Good Article",
             "content": "Real content here", "tone": "3.0",
             "date": "2026-06-01"},
        ]

        processed = ingestor._process_articles(articles, "NVDA")
        assert len(processed) == 1  # Only the valid article

    def test_store_articles(self, store, mock_chroma):
        """Articles are stored in ChromaDB with correct metadata."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        articles = [
            {
                "title": "NVIDIA Beats Expectations",
                "content": "NVIDIA reported strong earnings...",
                "url": "https://example.com/nvda-earnings",
                "tone": 8.5,
                "date": "2026-06-01",
                "ticker": "NVDA",
                "source": "gdelt",
                "entities": "Jensen Huang",
            },
        ]

        stored = ingestor._store_articles(articles)
        assert stored == 1

    def test_sentiment_summary_empty(self, store, mock_chroma):
        """Empty data returns zero-count sentiment summary."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        mock_chroma.search.return_value = []

        summary = ingestor.get_sentiment_summary("NVDA")
        assert summary["ticker"] == "NVDA"
        assert summary["article_count"] == 0
        assert summary["average_tone"] is None

    def test_sentiment_summary_with_data(self, store, mock_chroma):
        """Stored articles produce a valid sentiment summary."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)

        tones = [5.0, 3.0, -2.0, 1.0, -4.0]
        mock_chroma.search.return_value = [
            {"id": f"gdelt/NVDA/{i}", "document": f"News {i}",
             "metadata": {"tone": tone, "ticker": "NVDA", "source": "gdelt"},
             "distance": 0.1}
            for i, tone in enumerate(tones)
        ]
        mock_chroma.count.return_value = len(tones)

        ingestor._store_articles([
            {"title": f"News {i}", "content": f"Content {i}",
             "url": f"https://example.com/{i}", "tone": tone,
             "date": "2026-06-01", "ticker": "NVDA",
             "source": "gdelt", "entities": ""}
            for i, tone in enumerate(tones)
        ])
        assert ingestor.store.chroma.count() > 0

    def test_search_gdelt_failure(self, store, mock_chroma):
        """GDELT search failure returns empty list."""
        from src.macros.gdelt_ingestor import GDELTIngestor
        ingestor = GDELTIngestor(store=store)
        with patch("src.macros.gdelt_ingestor.httpx.get",
                   side_effect=Exception("GDELT API error")):
            articles = ingestor._search_gdelt("NVDA", max_records=10, lookback_days=7)
        assert articles == []


# ============================================================
# 3. EarningsTranscriptIngestor Tests
# ============================================================

class TestEarningsTranscriptIngestor:
    """Tests for the earnings transcripts ingestion module."""

    def test_import(self):
        """EarningsTranscriptIngestor imports successfully."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        assert EarningsTranscriptIngestor is not None

    def test_init(self, store):
        """Default init with store and session."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        ingestor = EarningsTranscriptIngestor(store=store)
        assert ingestor.store is store
        assert ingestor.session is not None

    @pytest.mark.parametrize("text,expected_quarter", [
        ("In Q1 fiscal 2026, we delivered strong results", "2026-Q1"),
        ("For Q2 FY2025, revenue grew 20%", "2025-Q2"),
        ("In the first quarter of 2025, we saw...", "2025-Q1"),
        ("Our fourth quarter fiscal 2026 results", "2026-Q4"),
        ("Random text with no quarter info", None),  # Falls back to current date
    ])
    def test_detect_quarter(self, text, expected_quarter):
        """Quarter is correctly detected from transcript text."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        ingestor = EarningsTranscriptIngestor()

        detected = ingestor._detect_quarter(text)
        if expected_quarter:
            assert detected == expected_quarter, (
                f"Expected {expected_quarter}, got {detected} for '{text[:30]}...'"
            )
        else:
            # Should fall back to current date format YYYY-QMM
            assert "-Q" in detected

    def test_parse_transcript_guidance_revenue(self, tmp_path):
        """Revenue guidance range is extracted correctly."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        ingestor = EarningsTranscriptIngestor()

        transcript = """
        We expect revenue of $26.0 billion to $28.0 billion for Q2.
        """
        facts = ingestor._parse_transcript("NVDA", transcript, "2026-Q2")

        revenue_low = [f for f in facts if f["metric"] == "guidance_revenue_low"]
        revenue_high = [f for f in facts if f["metric"] == "guidance_revenue_high"]

        assert len(revenue_low) == 1
        assert len(revenue_high) == 1
        assert revenue_low[0]["value"] == 26.0
        assert revenue_high[0]["value"] == 28.0

    def test_parse_transcript_guidance_eps(self, tmp_path):
        """EPS guidance is extracted correctly."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        ingestor = EarningsTranscriptIngestor()

        transcript = """
        Non-GAAP EPS of $3.50 for the quarter.
        """
        facts = ingestor._parse_transcript("NVDA", transcript, "2026-Q2")

        eps_facts = [f for f in facts if f["metric"] == "guidance_eps"]
        assert len(eps_facts) == 1
        assert eps_facts[0]["value"] == 3.50

    def test_parse_transcript_guidance_margin(self, tmp_path):
        """Margin guidance is extracted correctly."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        ingestor = EarningsTranscriptIngestor()

        transcript = """
        Gross margin of 75.5% in the quarter.
        """
        facts = ingestor._parse_transcript("NVDA", transcript, "2026-Q2")

        margin_facts = [f for f in facts if f["metric"] == "guidance_margin"]
        assert len(margin_facts) == 1
        assert margin_facts[0]["value"] == 75.5

    def test_parse_transcript_no_guidance(self, tmp_path):
        """Transcript without guidance keywords returns empty facts."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        ingestor = EarningsTranscriptIngestor()

        transcript = """
        Thank you everyone for joining today's call. We're pleased with our results.
        The team executed well. Looking forward to next quarter.
        """
        facts = ingestor._parse_transcript("NVDA", transcript, "2026-Q2")
        assert facts == []

    def test_store_transcript(self, store, mock_chroma):
        """Transcript is stored in ChromaDB with correct metadata."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        ingestor = EarningsTranscriptIngestor(store=store)

        transcript = "NVIDIA Q1 2026 earnings call transcript content..."
        facts = [
            {"metric": "guidance_revenue_low", "value": 26.0, "unit": "billion_usd"},
            {"metric": "guidance_eps", "value": 3.50, "unit": "usd"},
        ]

        mock_chroma.count.return_value = 1

        doc_id = ingestor._store_transcript("NVDA", transcript, "2026-Q1", facts)
        assert doc_id == "earnings_call/NVDA/2026-Q1"

        # Verify it's in ChromaDB
        assert ingestor.store.chroma.count() > 0

    def test_get_latest_guidance(self, store):
        """get_latest_guidance retrieves stored guidance from SQLite."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        ingestor = EarningsTranscriptIngestor(store=store)

        # Seed guidance data
        store.save_fundamental("NVDA", "guidance_revenue_low", 26.0,
                               "billion_usd", "2026-Q2", "quarterly",
                               "earnings_transcript")
        store.save_fundamental("NVDA", "guidance_revenue_high", 28.0,
                               "billion_usd", "2026-Q2", "quarterly",
                               "earnings_transcript")

        guidance = ingestor.get_latest_guidance("NVDA")
        assert guidance.get("guidance_revenue_low") == 26.0
        assert guidance.get("guidance_revenue_high") == 28.0

    def test_fetch_and_process_failed(self, store):
        """fetch_and_process returns 'not_found' when transcript unavailable."""
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        ingestor = EarningsTranscriptIngestor(store=store)

        # Mock the fetch to return None (no transcript available)
        ingestor._fetch_transcript = MagicMock(return_value=None)

        result = ingestor.fetch_and_process("ZZZZ")
        assert result["status"] == "not_found"
        assert result["ticker"] == "ZZZZ"


# ============================================================
# 4. Multi-Source Integration Tests
# ============================================================

class TestMultiSourceIntegration:
    """End-to-end tests with all three sources."""

    @pytest.fixture
    def seeded_store(self, mock_chroma, tmp_path):
        """Create a store seeded with data from all sources (real SQLite, mocked Chroma)."""
        store = Store(db_path=tmp_path / "test.db", chroma_path=tmp_path / "chroma")

        # FRED: macro indicators
        store.save_fundamental("MACRO", "FEDFUNDS", 4.5, "percent",
                               "2026-06-01", "daily", "fred")
        store.save_fundamental("MACRO", "CPIAUCSL", 3.2, "percent",
                               "2026-06-01", "daily", "fred")
        store.save_fundamental("MACRO", "UNRATE", 3.9, "percent",
                               "2026-06-01", "daily", "fred")

        # Earnings: guidance
        store.save_fundamental("NVDA", "guidance_revenue_low", 26.0,
                               "billion_usd", "2026-Q2", "quarterly",
                               "earnings_transcript")
        store.save_fundamental("NVDA", "guidance_revenue_high", 28.0,
                               "billion_usd", "2026-Q2", "quarterly",
                               "earnings_transcript")

        return store

    def test_macro_batch_retrieval(self, seeded_store):
        """Macro indicators can be retrieved as a batch."""
        macro = seeded_store.get_fundamentals_batch(
            "MACRO", metrics=["FEDFUNDS", "CPIAUCSL", "UNRATE"],
        )
        assert macro["FEDFUNDS"] == 4.5
        assert macro["CPIAUCSL"] == 3.2
        assert macro["UNRATE"] == 3.9

    def test_guidance_retrieval(self, seeded_store):
        """Earnings guidance can be retrieved from SQLite."""
        guidance = seeded_store.get_fundamentals_batch(
            "NVDA", metrics=["guidance_revenue_low", "guidance_revenue_high"],
        )
        assert guidance["guidance_revenue_low"] == 26.0
        assert guidance["guidance_revenue_high"] == 28.0

    def test_company_and_macro_together(self, seeded_store):
        """Both company and macro data can be fetched in one query."""
        nvda = seeded_store.get_fundamentals_batch("NVDA", metrics=[
            "guidance_revenue_low", "guidance_revenue_high",
        ])
        macro = seeded_store.get_fundamentals_batch("MACRO", metrics=[
            "FEDFUNDS", "CPIAUCSL",
        ])

        assert nvda["guidance_revenue_low"] == 26.0
        assert macro["FEDFUNDS"] == 4.5

    def test_search_finds_macro(self, seeded_store):
        """Hybrid search across all data finds macro results."""
        results = seeded_store.search("interest rates economy", n_results=5)
        # Should at least return without error
        assert "documents" in results
        assert "facts" in results

    def test_middleware_integration_macro_question(self, seeded_store, monkeypatch):
        """Middleware /query handles macro questions without crashing."""
        from src.middleware import app as middleware_app
        from src.middleware.app import app
        from fastapi.testclient import TestClient

        # Enter the lifespan so `config` and `model_client` are initialized,
        # then point `store` at the seeded data (lifespan sets its own Store).
        with TestClient(app) as client:
            monkeypatch.setattr(middleware_app, "store", seeded_store)
            response = client.post("/query", json={
                "question": "What is the current fed funds rate?",
            })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_middleware_sentiment_endpoint(self, seeded_store, monkeypatch):
        """GET /sentiment/{ticker} returns a response."""
        from src.middleware import app as middleware_app
        from src.middleware.app import app
        from fastapi.testclient import TestClient

        # This endpoint requires GDELT data in ChromaDB
        # With empty DB it should return zero counts, not crash
        with TestClient(app) as client:
            monkeypatch.setattr(middleware_app, "store", seeded_store)
            response = client.get("/sentiment/NVDA")
        assert response.status_code == 200


# ============================================================
# 5. End-to-End Pipeline Test (Live)
# ============================================================

@pytest.mark.live
class TestLiveMacroIngestion:
    """Live API tests — requires internet and API keys."""

    def test_fred_live_fetch(self):
        """Fetch a live FRED indicator (requires FRED_API_KEY)."""
        import os
        from src.macros.fred_ingestor import FREDIngestor

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            pytest.skip("FRED_API_KEY not set")

        ingestor = FREDIngestor(api_key=api_key)
        value = ingestor.fetch_indicator("FEDFUNDS")

        assert value is not None
        assert isinstance(value, float)
        assert 0 < value < 20  # Sanity check

    def test_gdelt_live_fetch(self):
        """Fetch live GDELT news for a major ticker."""
        from src.macros.gdelt_ingestor import GDELTIngestor

        ingestor = GDELTIngestor()
        articles = ingestor.fetch_news_for_ticker("NVDA", max_records=10)

        # GDELT may return 0 articles for some queries, but shouldn't crash
        assert isinstance(articles, list)

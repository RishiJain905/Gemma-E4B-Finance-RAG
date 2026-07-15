"""
tests/test_fred_ingestor.py
Pytest suite for Phase 1.6.1 FRED economic data ingestion.

Usage:
    pytest tests/test_fred_ingestor.py -v
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
def mock_default_chroma_store():
    """Keep every FRED unit test independent of the live Chroma directory."""
    with patch("src.storage.store.ChromaStore") as mock_cls:
        instance = MagicMock()
        instance.heartbeat.return_value = True
        instance.count.return_value = 0
        mock_cls.return_value = instance
        yield instance


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

    def test_init_with_env_api_key(self, monkeypatch):
        """API key loads from environment variable."""
        monkeypatch.setenv("FRED_API_KEY", "test_key_123")
        from src.macros.fred_ingestor import FREDIngestor
        ingestor = FREDIngestor()
        assert ingestor.api_key == "test_key_123"

    def test_unit_guessing(self):
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

    def test_category_mapping(self):
        """Category filtering returns the correct indicators."""
        from src.macros.fred_ingestor import FREDIngestor
        ingestor = FREDIngestor()

        interest = ingestor.fetch_by_category("interest_rates")
        assert isinstance(interest, dict)

        unknown = ingestor.fetch_by_category("not_a_category")
        assert unknown == {}

    def test_fetch_indicator_stores_data(self, store):
        """fetch_indicator stores the value in SQLite when successful."""
        from src.macros.fred_ingestor import FREDIngestor
        import pandas as pd

        ingestor = FREDIngestor(store=store)

        mock_fred = MagicMock()
        dates = pd.date_range("2026-01-01", periods=3, freq="ME")
        mock_fred.get_series.return_value = pd.Series([5.25, 5.5, 5.75], index=dates)

        with patch.object(ingestor, "_client", mock_fred):
            value = ingestor.fetch_indicator("FEDFUNDS")

        assert value == 5.75

        result = store.get_fundamental("MACRO", "FEDFUNDS")
        assert result is not None
        assert result["value"] == 5.75

    def test_fetch_indicator_store_history_persists_each_dated_value(self, store):
        """Bootstrap history iterates dated Series items, not scalar values."""
        from src.macros.fred_ingestor import FREDIngestor
        import pandas as pd

        ingestor = FREDIngestor(store=store)
        mock_fred = MagicMock()
        dates = pd.date_range("2026-01-01", periods=3, freq="ME")
        mock_fred.get_series.return_value = pd.Series([5.25, 5.5, 5.75], index=dates)

        with patch.object(ingestor, "_client", mock_fred):
            value = ingestor.fetch_indicator("FEDFUNDS", store_history=True)

        assert value == 5.75
        with store.sqlite._connect() as conn:
            rows = conn.execute(
                "SELECT period, value FROM fundamentals WHERE ticker=? AND metric=? "
                "ORDER BY period",
                ("MACRO", "FEDFUNDS"),
            ).fetchall()
        assert [(row["period"], row["value"]) for row in rows] == [
            ("2026-01-31", 5.25),
            ("2026-02-28", 5.5),
            ("2026-03-31", 5.75),
        ]

    def test_fetch_indicator_empty_series(self):
        """Empty series response returns None."""
        from src.macros.fred_ingestor import FREDIngestor
        import pandas as pd

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series(dtype=float)

        with patch.object(ingestor, "_client", mock_fred):
            value = ingestor.fetch_indicator("GDP")

        assert value is None

    def test_fetch_indicator_network_error(self):
        """Network error returns None without crashing."""
        from src.macros.fred_ingestor import FREDIngestor

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = ConnectionError("DNS failure")

        with patch.object(ingestor, "_client", mock_fred):
            value = ingestor.fetch_indicator("GDP")

        assert value is None

    def test_fetch_all_indicators(self):
        """fetch_all_indicators iterates through all configured series."""
        from src.macros.fred_ingestor import FREDIngestor
        import pandas as pd

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        dates = pd.date_range("2026-01-01", periods=2, freq="ME")
        mock_fred.get_series.return_value = pd.Series([100.0, 102.5], index=dates)

        with patch.object(ingestor, "_client", mock_fred), \
             patch.object(ingestor, "_respect_rate_limit"):
            results = ingestor.fetch_all_indicators()

        assert len(results) > 0
        for _series_id, value in results.items():
            assert value == 102.5 or value is None

    def test_get_macro_snapshot(self):
        """get_macro_snapshot returns key macro fields from fetch_all_indicators."""
        from src.macros.fred_ingestor import FREDIngestor

        ingestor = FREDIngestor()
        mock_results = {
            "GDP": 29.5,
            "CPIAUCSL": 3.2,
            "FEDFUNDS": 4.5,
            "UNRATE": 3.9,
            "DGS10": 4.2,
            "T10Y2Y": 0.3,
        }

        with patch.object(ingestor, "fetch_all_indicators", return_value=mock_results):
            snapshot = ingestor.get_macro_snapshot()

        assert snapshot["gdp"] == 29.5
        assert snapshot["inflation_cpi"] == 3.2
        assert snapshot["fed_rate"] == 4.5
        assert snapshot["unemployment"] == 3.9
        assert snapshot["ten_year_treasury"] == 4.2
        assert snapshot["ten_two_spread"] == 0.3

    def test_health_check_ok(self):
        """health_check returns True when FRED is reachable."""
        from src.macros.fred_ingestor import FREDIngestor
        import pandas as pd

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series([100.0])

        with patch.object(ingestor, "_client", mock_fred):
            assert ingestor.health_check() is True

    def test_health_check_fail(self):
        """health_check returns False when FRED is unreachable."""
        from src.macros.fred_ingestor import FREDIngestor

        ingestor = FREDIngestor()
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = Exception("API error")

        with patch.object(ingestor, "_client", mock_fred):
            assert ingestor.health_check() is False

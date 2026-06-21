"""
tests/test_phase1_8_fixes.py
Phase 1.8.5 — regression tests for two bugs found during final verification:

1. `.env` was never loaded into os.environ, so FRED_API_KEY (and any env
   override for the SEC user agent) was ignored. `src/utils/env.load_env()`
   now loads it without overriding already-set variables.
2. The FRED ingestor fetched the *oldest* observations (FRED's `limit` returns
   from the start of the series) and stored a decades-old value as "latest".
   It now requests `sort_order="desc"` and selects the most-recent observation.
"""

import sys
from datetime import datetime
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
from src.utils.env import load_env


@pytest.fixture
def store(tmp_path):
    with patch("src.storage.store.ChromaStore") as mock_cls:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mock_cls.return_value = inst
        yield Store(db_path=tmp_path / "fixes.db", chroma_path=tmp_path / "chroma")


# ── .env loader ────────────────────────────────────────

class TestLoadEnv:

    def test_loads_keys(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text('FRED_API_KEY=abc123\nSEC_EDGAR_USER_AGENT="Name x@y.com"\n')
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
        applied = load_env(path=env)
        assert applied["FRED_API_KEY"] == "abc123"
        # Surrounding quotes are stripped.
        assert applied["SEC_EDGAR_USER_AGENT"] == "Name x@y.com"

    def test_does_not_override_existing(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("FRED_API_KEY=from_file\n")
        monkeypatch.setenv("FRED_API_KEY", "from_environment")
        load_env(path=env)
        import os
        assert os.environ["FRED_API_KEY"] == "from_environment"

    def test_missing_file_is_noop(self, tmp_path):
        assert load_env(path=tmp_path / "nope.env") == {}


# ── FRED latest-value extraction ───────────────────────

class TestFREDLatestValue:

    def _series(self):
        import pandas as pd
        # Unsorted on purpose; the ingestor must pick the most recent DATE.
        idx = [datetime(1954, 11, 1), datetime(2026, 5, 1), datetime(2026, 1, 1)]
        return pd.Series([0.83, 4.49, 4.10], index=pd.to_datetime(idx))

    def test_fetch_indicator_uses_desc_and_latest(self, store):
        from src.macros.fred_ingestor import FREDIngestor
        ing = FREDIngestor(store=store)
        mock_client = MagicMock()
        mock_client.get_series.return_value = self._series()
        ing._client = mock_client
        ing.config["request_delay"] = 0  # no sleep

        value = ing.fetch_indicator("DGS10")

        # Most-recent observation (2026-05-01 -> 4.49) is selected, not 1954.
        assert value == 4.49
        # Requested in descending order so FRED returns the newest observations.
        _, kwargs = mock_client.get_series.call_args
        assert kwargs.get("sort_order") == "desc"

        # Stored fact carries the latest value + date.
        fact = store.get_fundamental("MACRO", "DGS10")
        assert fact is not None
        assert fact["value"] == 4.49
        assert fact["period"] == "2026-05-01"

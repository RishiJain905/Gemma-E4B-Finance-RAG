# tests/test_query_facts_tool.py
# Offline tests for Phase 2.1.4.2 — query_facts & list_metrics analytical tools.

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.middleware.tools import REGISTRY, openai_schema, sanity  # noqa: E402
from src.middleware.tools.data_tools import (  # noqa: E402
    list_metrics_handler,
    query_facts_handler,
)
from src.storage.store import Store  # noqa: E402

TEST_ANALYTICS_CONFIG = {
    "excluded_symbols": ["TLT", "GLD"],
    "equity_only_metrics": ["forward_pe"],
    "metric_ranges": {"forward_pe": {"min": 0, "max": 500}},
}


@pytest.fixture(autouse=True)
def isolated_sanity_config(tmp_path, monkeypatch):
    """Point sanity at a test-controlled config so tests don't depend on the repo's analytics.yaml."""
    config_path = tmp_path / "analytics.yaml"
    config_path.write_text(yaml.safe_dump(TEST_ANALYTICS_CONFIG))
    monkeypatch.setattr(sanity, "_CONFIG_PATH", config_path)
    sanity._reset_cache()
    yield
    sanity._reset_cache()


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
    """Store with real SQLite (tmp_path) and mocked Chroma."""
    return Store(db_path=tmp_path / "test.db", chroma_path=tmp_path / "chroma")


def _seed_forward_pe(store, ticker, value, period="2026-Q1"):
    store.sqlite.upsert_fundamental(
        ticker=ticker, metric="forward_pe", value=value,
        unit="ratio", period=period, source_type="yfinance",
    )


# ── list_metrics ─────────────────────────────────────

def test_list_metrics(store):
    _seed_forward_pe(store, "NVDA", 16.5)
    _seed_forward_pe(store, "META", 15.9)
    store.sqlite.upsert_fundamental(
        ticker="NVDA", metric="revenue_ttm", value=1000.0,
        period="2026-Q1", source_type="yfinance",
    )

    result = list_metrics_handler(store)

    assert result["metrics"] == ["forward_pe", "revenue_ttm"]
    assert result["tickers"] == ["META", "NVDA"]


def test_list_metrics_scoped_to_ticker(store):
    _seed_forward_pe(store, "NVDA", 16.5)
    store.sqlite.upsert_fundamental(
        ticker="META", metric="revenue_ttm", value=1000.0,
        period="2026-Q1", source_type="yfinance",
    )

    result = list_metrics_handler(store, ticker="nvda")

    assert result["metrics"] == ["forward_pe"]
    assert result["tickers"] == ["META", "NVDA"]


# ── query_facts: ranking ─────────────────────────────

def test_query_facts_lowest(store):
    _seed_forward_pe(store, "META", 15.9)
    _seed_forward_pe(store, "NVDA", 16.5)
    _seed_forward_pe(store, "MSFT", 19.6)

    result = query_facts_handler(store, metric="forward_pe", order="asc", limit=1)

    assert result["metric"] == "forward_pe"
    assert len(result["results"]) == 1
    assert result["results"][0]["ticker"] == "META"
    assert result["results"][0]["value"] == pytest.approx(15.9)


def test_query_facts_threshold(store):
    _seed_forward_pe(store, "META", 15.9)
    _seed_forward_pe(store, "NVDA", 16.5)
    _seed_forward_pe(store, "TSLA", 173.6)

    result = query_facts_handler(store, metric="forward_pe", op="lt", value=20)

    tickers = {r["ticker"] for r in result["results"]}
    assert tickers == {"META", "NVDA"}


def test_query_facts_excludes_etfs(store):
    _seed_forward_pe(store, "META", 15.9)
    _seed_forward_pe(store, "TLT", -4288.0)
    _seed_forward_pe(store, "GLD", 12.0)

    result = query_facts_handler(store, metric="forward_pe", order="asc", limit=10)

    tickers = {r["ticker"] for r in result["results"]}
    assert "TLT" not in tickers
    assert "GLD" not in tickers
    assert tickers == {"META"}


def test_query_facts_drops_implausible(store):
    # Not in excluded_symbols — this proves the sane_range filter runs
    # independently of the ETF-exclude filter.
    _seed_forward_pe(store, "JUNK", -4288.0)
    _seed_forward_pe(store, "META", 15.9)

    result = query_facts_handler(store, metric="forward_pe", order="asc", limit=10)

    tickers = {r["ticker"] for r in result["results"]}
    assert "JUNK" not in tickers
    values = [r["value"] for r in result["results"]]
    assert all(0 <= v <= 500 for v in values)


def test_query_facts_latest_only(store):
    _seed_forward_pe(store, "NVDA", 30.0, period="2025-Q4")
    _seed_forward_pe(store, "NVDA", 16.5, period="2026-Q1")

    result = query_facts_handler(store, metric="forward_pe", tickers=["NVDA"], latest_only=True)

    assert len(result["results"]) == 1
    assert result["results"][0]["period"] == "2026-Q1"
    assert result["results"][0]["value"] == pytest.approx(16.5)


def test_invalid_metric(store):
    _seed_forward_pe(store, "NVDA", 16.5)

    result = query_facts_handler(store, metric="not_a_real_metric")

    assert "error" in result
    assert result["available_metrics"] == ["forward_pe"]


# ── Registration ─────────────────────────────────────

def test_tools_registered_write_false():
    assert "list_metrics" in REGISTRY
    assert "query_facts" in REGISTRY
    assert REGISTRY["list_metrics"].write is False
    assert REGISTRY["query_facts"].write is False


def test_tools_appear_in_openai_schema():
    names = {entry["function"]["name"] for entry in openai_schema()}
    assert {"list_metrics", "query_facts"} <= names


def test_query_facts_op_without_value(store):
    _seed_forward_pe(store, "NVDA", 16.5)

    result = query_facts_handler(store, metric="forward_pe", op="lt")

    assert "error" in result and "op and value" in result["error"]

    result = query_facts_handler(store, metric="forward_pe", value=20)

    assert "error" in result and "op and value" in result["error"]

"""
tests/test_full_pipeline_integration.py
Phase 1.8.3 — End-to-end pipeline integration with real external services.

These perform real ingestion (Yahoo Finance) into a temporary store and then
read the data back. They are marked ``slow`` + ``network`` and skip cleanly
when the upstream service or embedding endpoint is unavailable.

Usage:
    pytest tests/test_full_pipeline_integration.py -v -m "slow"
"""

import httpx
import pytest

from src.storage.store import Store

pytestmark = [pytest.mark.slow, pytest.mark.network, pytest.mark.integration]


@pytest.fixture
def pipeline_store(tmp_path):
    store = Store(
        db_path=tmp_path / "pipeline.db",
        chroma_path=tmp_path / "chroma",
        embedding_endpoint="http://127.0.0.1:8087/v1/embeddings",
    )
    yield store
    import shutil
    shutil.rmtree(tmp_path / "chroma", ignore_errors=True)


def test_ingest_then_query(pipeline_store):
    """Ingest one ticker's fundamentals from Yahoo Finance, then read it back."""
    from src.ingestion.yfinance_ingestor import YFinanceIngestor

    ingestor = YFinanceIngestor(store=pipeline_store)
    try:
        t = ingestor._fetch_ticker("NVDA")
        if t is None:
            pytest.skip("Yahoo Finance unreachable")
        ingestor._ingest_ticker_fundamentals("NVDA", t)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Yahoo Finance ingestion unavailable: {e}")

    # Cache freshness records the run for fundamentals.
    status = pipeline_store.get_cache_status("NVDA", "yfinance_fundamentals")
    assert status is not None

    # At least one fundamental fact was stored.
    facts = pipeline_store.sqlite.search_facts(ticker="NVDA", limit=5)
    assert len(facts) > 0


def test_ingest_then_search(pipeline_store):
    """Stored facts/documents are retrievable via hybrid search."""
    try:
        r = httpx.post("http://127.0.0.1:8087/v1/embeddings",
                       json={"model": "tracealchemy", "input": "ping"}, timeout=10)
        if r.status_code != 200:
            pytest.skip("embedding endpoint not available")
    except Exception:  # noqa: BLE001
        pytest.skip("embedding endpoint not available")

    pipeline_store.save_fundamental("NVDA", "total_revenue", 26.0, "usd", "2026-Q1")
    pipeline_store.save_document("pipeline/nvda/doc",
                                 "NVIDIA datacenter revenue grew significantly.",
                                 ticker="NVDA", source="test")

    results = pipeline_store.search("NVIDIA revenue", n_results=3)
    assert len(results["documents"]) > 0 or len(results["facts"]) > 0

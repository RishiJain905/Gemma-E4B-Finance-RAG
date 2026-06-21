"""
tests/test_phase1_8_coverage.py
Phase 1.8.3 — Additional coverage for file logging, scheduler source
dispatch, the degraded-answer formatter, and the macro/sentiment/guidance/
health endpoints.

Usage:
    pytest tests/test_phase1_8_coverage.py -v
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.scheduler import UnifiedScheduler
from src.storage.store import Store


@pytest.fixture
def store(tmp_path):
    with patch("src.storage.store.ChromaStore") as mock_cls:
        inst = MagicMock()
        inst.heartbeat.return_value = True
        inst.count.return_value = 0
        mock_cls.return_value = inst
        yield Store(db_path=tmp_path / "cov.db", chroma_path=tmp_path / "chroma")


# ════════════════════════════════════════════════════════
# File logging (src/utils/logging.py)
# ════════════════════════════════════════════════════════

class TestFileLogging:

    def test_setup_file_logging_creates_files(self, tmp_path):
        from src.utils.logging import setup_file_logging
        log_dir = tmp_path / "logs"
        main_h, err_h = setup_file_logging(str(log_dir), level=logging.DEBUG)
        try:
            logging.getLogger("test.logging").warning("a warning line")
            for h in (main_h, err_h):
                h.flush()
            assert (log_dir / "rag-system.log").exists()
            assert (log_dir / "rag-errors.log").exists()
            assert err_h.level == logging.WARNING
        finally:
            root = logging.getLogger()
            root.removeHandler(main_h)
            root.removeHandler(err_h)
            main_h.close()
            err_h.close()

    def test_ingestion_logger_formats(self, caplog):
        from src.utils.logging import IngestionLogger
        log = IngestionLogger("fred")
        with caplog.at_level(logging.INFO):
            log.info("Fetched GDP", series_id="GDP", value=28.5)
            log.warning("Rate limited", error="HTTP 429")
            log.error("Failed", series_id="GDP")
        text = caplog.text
        assert "source=fred" in text
        assert "series_id=GDP" in text
        assert "value=28.5" in text


# ════════════════════════════════════════════════════════
# Scheduler source dispatch (_run_source branches)
# ════════════════════════════════════════════════════════

class TestSchedulerDispatch:

    def test_dispatch_yfinance(self, store):
        sched = UnifiedScheduler(store=store)
        with patch("src.ingestion.yfinance_ingestor.YFinanceIngestor") as M:
            detail = sched._run_source("yfinance")
        assert detail == {"action": "ingest_all"}
        M.return_value.ingest_all.assert_called_once()

    def test_dispatch_sec_discovery(self, store):
        sched = UnifiedScheduler(store=store)
        with patch("src.sec.FilingScheduler") as M:
            M.return_value.run_discovery.return_value = {"discovered": 3}
            detail = sched._run_source("sec_filings", deep=False)
        assert detail == {"discovered": 3}

    def test_dispatch_sec_deep(self, store):
        sched = UnifiedScheduler(store=store)
        with patch("src.sec.FilingScheduler") as M:
            M.return_value.run_full_pipeline.return_value = {"pipeline": "ok"}
            detail = sched._run_source("sec_filings", deep=True)
        assert detail == {"pipeline": "ok"}

    def test_dispatch_fred(self, store):
        sched = UnifiedScheduler(store=store)
        with patch("src.macros.fred_ingestor.FREDIngestor") as M:
            M.return_value.fetch_all_indicators.return_value = {"GDP": 1.0, "CPI": None}
            detail = sched._run_source("fred")
        assert detail["indicators_fetched"] == 1
        assert detail["indicators_total"] == 2

    def test_dispatch_gdelt(self, store):
        sched = UnifiedScheduler(store=store)
        with patch("src.macros.gdelt_ingestor.GDELTIngestor") as M:
            M.return_value.fetch_and_store_all.return_value = {"NVDA": 5, "AMD": 3}
            detail = sched._run_source("gdelt")
        assert detail["articles_stored"] == 8
        assert detail["tickers"] == 2

    def test_dispatch_earnings(self, store):
        sched = UnifiedScheduler(store=store)
        with patch("src.macros.earnings_transcripts.EarningsTranscriptIngestor") as M:
            M.return_value.fetch_all_core.return_value = {"NVDA": {}, "AMD": {}}
            detail = sched._run_source("earnings_transcripts")
        assert detail["tickers_processed"] == 2

    def test_dispatch_ir_pages(self, store):
        sched = UnifiedScheduler(store=store)
        with patch("src.macros.ir_ingestor.IRIngestor") as M:
            M.return_value.fetch_all_core.return_value = {
                "NVDA": {"items_stored": 4}, "AMD": {"items_stored": 2},
            }
            detail = sched._run_source("ir_pages")
        assert detail["tickers_processed"] == 2
        assert detail["items_stored"] == 6

    def test_dispatch_unknown_source(self, store):
        sched = UnifiedScheduler(store=store)
        with pytest.raises(ValueError):
            sched._run_source("not_a_source")


# ════════════════════════════════════════════════════════
# Degraded answer formatter
# ════════════════════════════════════════════════════════

class TestDegradedAnswer:

    def test_format_with_facts_and_docs(self):
        from src.middleware.app import _format_degraded_answer
        retrieval = {
            "facts": [{"metric": "total_revenue", "value": 26.0, "period": "2026-Q1"}],
            "documents": [{"id": "sec/NVDA/10-Q", "metadata": {"source": "sec"}}],
        }
        out = _format_degraded_answer(retrieval, {})
        assert "Model unavailable" in out
        assert "total_revenue" in out
        assert "sec/NVDA/10-Q" in out

    def test_format_empty(self):
        from src.middleware.app import _format_degraded_answer
        out = _format_degraded_answer({"facts": [], "documents": []}, {})
        assert "No stored data found" in out


# ════════════════════════════════════════════════════════
# Endpoints (live store — needs embedding endpoint on :8087)
# ════════════════════════════════════════════════════════

def _embeddings_up() -> bool:
    try:
        r = httpx.post("http://127.0.0.1:8087/v1/embeddings",
                       json={"model": "tracealchemy", "input": "ping"}, timeout=10)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.skipif(not _embeddings_up(), reason="embedding endpoint not available")
class TestEndpoints:

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from src.middleware import app as middleware_app
        live = Store(
            db_path=tmp_path / "ep.db",
            chroma_path=tmp_path / "chroma",
            embedding_endpoint="http://127.0.0.1:8087/v1/embeddings",
        )
        with TestClient(middleware_app.app) as c:
            monkeypatch.setattr(middleware_app, "store", live)
            yield c, live
        import shutil
        shutil.rmtree(tmp_path / "chroma", ignore_errors=True)

    def test_health_enhanced(self, client):
        c, store = client
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "scheduler" in data
        assert "freshness" in data
        assert set(["NVDA", "AMD"]).issubset(data["freshness"].keys())

    def test_macro_snapshot(self, client):
        c, store = client
        store.save_fundamental("MACRO", "GDP", 28.5, "trillion_usd", "2026-Q1",
                               source_type="fred")
        resp = c.get("/macro/snapshot")
        assert resp.status_code == 200
        assert resp.json()["gdp"] == 28.5

    def test_sentiment(self, client):
        c, store = client
        resp = c.get("/sentiment/NVDA?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "NVDA"
        assert data["article_count"] == 0  # none stored

    def test_guidance_not_found(self, client):
        c, store = client
        resp = c.get("/guidance/NVDA")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"

    def test_guidance_found(self, client):
        c, store = client
        store.save_fundamental("NVDA", "guidance_revenue_low", 28.0, "usd", "2026-Q2")
        resp = c.get("/guidance/NVDA")
        assert resp.status_code == 200
        assert resp.json()["status"] == "found"

    def test_refresh_unknown_source(self, client):
        c, store = client
        resp = c.post("/refresh/NVDA", json={"sources": ["unknown_source"]})
        assert resp.status_code == 200
        assert resp.json()["refreshed"] == []

    def test_freshness_endpoint(self, client):
        c, store = client
        store.mark_source_fresh("NVDA", "yfinance_fundamentals", 24)
        resp = c.get("/freshness/NVDA")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "NVDA"
        assert "sources" in data

    @pytest.mark.slow
    def test_real_query_end_to_end(self, client):
        """Exercise the real model-call path (no _call_model mock)."""
        c, store = client
        store.save_fundamental("NVDA", "total_revenue", 26.0, "usd", "2026-Q1")
        store.save_document("test/nvda/doc", "NVIDIA reported $26B revenue in Q1 2026.",
                            ticker="NVDA", source="sec")
        resp = c.post("/query", json={"question": "What is NVDA's revenue?",
                                      "refresh": False, "max_tokens": 64})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_available"] is True
        # Model output content is nondeterministic; assert the path returns a
        # string answer and citation list rather than specific text.
        assert isinstance(data["answer"], str)
        assert isinstance(data["citations"], list)

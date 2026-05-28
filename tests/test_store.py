# tests/test_store.py
# Mocked unit tests for unified Store facade — Phase 1.2.4

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


@pytest.fixture
def fully_mocked_store():
    """Store with both backends mocked at construction."""
    with patch("src.storage.store.SQLiteStore") as mock_sqlite_cls, \
         patch("src.storage.store.ChromaStore") as mock_chroma_cls:
        sqlite = MagicMock()
        chroma = MagicMock()
        mock_sqlite_cls.return_value = sqlite
        mock_chroma_cls.return_value = chroma
        s = Store()
        s.sqlite = sqlite
        s.chroma = chroma
        yield s, sqlite, chroma


# ── Initialization ───────────────────────────────────

def test_store_init(tmp_path: Path):
    with patch("src.storage.store.ChromaStore") as mock_chroma_cls, \
         patch("src.storage.store.SQLiteStore") as mock_sqlite_cls:
        Store(db_path=tmp_path / "test.db", chroma_path=tmp_path / "chroma")
        mock_sqlite_cls.assert_called_once_with(db_path=tmp_path / "test.db")
        mock_chroma_cls.assert_called_once()
        call_kwargs = mock_chroma_cls.call_args.kwargs
        assert call_kwargs["persist_directory"] == tmp_path / "chroma"
        assert call_kwargs["collection_name"] == "tracealchemy_docs"


# ── Health ───────────────────────────────────────────

def test_heartbeat(store, mock_chroma):
    mock_chroma.heartbeat.return_value = True
    mock_chroma.count.return_value = 7

    health = store.heartbeat()

    assert health == {"sqlite": True, "chroma": True, "chroma_doc_count": 7}
    mock_chroma.heartbeat.assert_called_once()
    mock_chroma.count.assert_called_once()


# ── Fundamentals ─────────────────────────────────────

def test_save_and_get_fundamental(store):
    assert store.save_fundamental(
        "NVDA", "revenue_q1_2026", 26.0, "usd", "2026-Q1"
    ) is True

    fact = store.get_fundamental("NVDA", "revenue_q1_2026", "2026-Q1")
    assert fact is not None
    assert fact["value"] == 26.0
    assert fact["metric"] == "revenue_q1_2026"


# ── Documents ────────────────────────────────────────

def test_save_document(store, mock_chroma):
    doc_id = store.save_document(
        document_id="test/nvda-earnings",
        text="NVIDIA reported record revenue.",
        ticker="NVDA",
        source="earnings_call",
        date="2026-05-15",
    )

    assert doc_id == "test/nvda-earnings"
    mock_chroma.add_document.assert_called_once_with(
        document_id="test/nvda-earnings",
        text="NVIDIA reported record revenue.",
        ticker="NVDA",
        source="earnings_call",
        date="2026-05-15",
        metadata=None,
    )


# ── Hybrid search ────────────────────────────────────

def test_search_hybrid(fully_mocked_store):
    s, sqlite, chroma = fully_mocked_store

    filtered_doc = {"id": "nvda-1", "document": "NVDA datacenter", "distance": 0.1}
    broad_doc = {"id": "macro-1", "document": "Fed rates", "distance": 0.3}
    chroma.search.side_effect = [
        [filtered_doc],
        [broad_doc],
    ]
    sqlite.search_facts.return_value = [
        {"metric": "revenue_q1", "value": 26.0, "ticker": "NVDA"}
    ]

    query = "What was NVDA datacenter revenue?"
    results = s.search(query, n_results=4)

    assert results["ticker"] == "NVDA"
    assert len(results["documents"]) == 2
    assert results["documents"][0]["id"] == "nvda-1"
    assert len(results["facts"]) == 1
    assert chroma.search.call_count == 2
    chroma.search.assert_any_call(
        query=query,
        n_results=4,
        filter_dict={"ticker": "NVDA"},
    )
    sqlite.search_facts.assert_called_once_with(ticker="NVDA", limit=4)


def test_search_by_ticker(fully_mocked_store):
    s, _sqlite, _chroma = fully_mocked_store

    with patch.object(s, "search", return_value={"documents": [], "facts": [], "ticker": "AMD"}) as mock_search:
        s.search_by_ticker("What is happening?", "AMD", n_results=3)
        mock_search.assert_called_once_with(
            query="What is happening?",
            n_results=3,
            ticker="AMD",
        )


def test_detect_ticker_nvda():
    assert Store._detect_ticker("What is NVDA doing?") == "NVDA"


def test_detect_ticker_none():
    assert Store._detect_ticker("What is the outlook for interest rates?") is None


# ── Filing pipeline ──────────────────────────────────

def test_process_filing(store, mock_chroma):
    store.register_filing(
        "NVDA", "10-Q", "2026-05-15", "2026-Q1",
        "0001045810-26-000001", "http://sec.gov/example",
    )

    filing_record = {
        "ticker": "NVDA",
        "filing_type": "10-Q",
        "filing_date": "2026-05-15",
        "period": "2026-Q1",
        "accession": "0001045810-26-000001",
        "source_url": "http://sec.gov/example",
        "source_type": "sec",
    }
    extracted_facts = [
        {"metric": "revenue", "value": 26.0, "unit": "usd", "period": "2026-Q1"},
    ]

    store.process_filing(
        filing_record,
        extracted_text="NVIDIA Q1 summary text.",
        extracted_facts=extracted_facts,
    )

    expected_doc_id = "sec/NVDA/10-Q-2026-Q1"
    mock_chroma.add_document.assert_called_once()
    assert mock_chroma.add_document.call_args.kwargs["document_id"] == expected_doc_id

    fact = store.get_fundamental("NVDA", "revenue", "2026-Q1")
    assert fact["value"] == 26.0

    with store.sqlite._connect() as conn:
        row = conn.execute(
            "SELECT status, summary_embedding_id FROM filings WHERE accession=?",
            (filing_record["accession"],),
        ).fetchone()
    assert row["status"] == "parsed"
    assert row["summary_embedding_id"] == expected_doc_id


def test_register_filing_delegate(fully_mocked_store):
    s, sqlite, _chroma = fully_mocked_store
    sqlite.register_filing.return_value = True

    result = s.register_filing(
        "AAPL", "10-K", "2025-09-30", "2025-FY",
        "0000320193-25-000123", "http://sec.gov",
    )

    assert result is True
    sqlite.register_filing.assert_called_once_with(
        "AAPL", "10-K", "2025-09-30", "2025-FY",
        "0000320193-25-000123", "http://sec.gov",
    )


# ── Cache delegates ──────────────────────────────────

def test_cache_delegates(fully_mocked_store):
    s, sqlite, _chroma = fully_mocked_store
    sqlite.get_stale_cache_entries.return_value = [{"ticker": "NVDA"}]

    s.mark_cache_fresh("NVDA", "yfinance", ttl_hours=12)
    s.mark_cache_stale("NVDA", "yfinance", error="timeout")
    entries = s.get_stale_entries(limit=5)

    sqlite.mark_cache_fresh.assert_called_once_with("NVDA", "yfinance", 12)
    sqlite.mark_cache_stale.assert_called_once_with("NVDA", "yfinance", "timeout")
    sqlite.get_stale_cache_entries.assert_called_once_with(5)
    assert entries == [{"ticker": "NVDA"}]


# ── Reset ────────────────────────────────────────────

def test_reset(store, mock_chroma, tmp_path: Path):
    store.save_fundamental("TEST", "metric_a", 1.0, period="2026-Q1")
    store.register_filing(
        "TEST", "10-K", "2026-01-01", "2026-FY",
        "acc-reset-1", "http://example.com",
    )

    store.reset()

    mock_chroma.reset_collection.assert_called_once()
    with store.sqlite._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
    assert count == 0

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

from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash
from src.ingestion.records import NarrativeRecord
from src.storage.store import Store
from src.sec.filing_sections import FilingSection


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

def test_heartbeat_uses_indexed_count_without_loading_chroma(fully_mocked_store):
    store, sqlite, chroma = fully_mocked_store
    chroma.heartbeat.return_value = True
    chroma.corpus_revision.return_value = 12
    chroma.count.side_effect = AssertionError("health must not load the vector index")
    sqlite.get_lexical_index_state.return_value = {
        "indexed_revision": 12,
        "row_count": 7,
        "rebuild_cursor": 0,
    }

    health = store.heartbeat()

    assert health == {"sqlite": True, "chroma": True, "chroma_doc_count": 7}
    chroma.heartbeat.assert_called_once()
    chroma.count.assert_not_called()


# ── Fundamentals ─────────────────────────────────────

def test_heartbeat_hides_count_when_corpus_revisions_differ(fully_mocked_store):
    store, sqlite, chroma = fully_mocked_store
    chroma.heartbeat.return_value = True
    chroma.corpus_revision.return_value = 13
    chroma.count.side_effect = AssertionError("health must not load the vector index")
    sqlite.get_lexical_index_state.return_value = {
        "indexed_revision": 12,
        "row_count": 7,
        "rebuild_cursor": 0,
    }

    health = store.heartbeat()

    assert health == {"sqlite": True, "chroma": True, "chroma_doc_count": None}
    chroma.count.assert_not_called()


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


def test_add_filing_sections_routes_through_structural_chunker(store, mock_chroma):
    section = FilingSection(
        accession="ACC-1", ticker="NVDA", form="10-Q",
        filing_date="2026-05-15", report_period="2026-03-31",
        section_key="item_2", section_heading="Item 2. MD&A", section_index=0,
        text="Item 2. MD&A\nRevenue increased.", source_url="https://sec.example",
        parsed_path="parsed/ACC-1.txt",
    )
    mock_chroma.count_filing_section_chunks.side_effect = [0, 1]

    result = store.add_filing_sections([section])

    mock_chroma.delete_filing_section_family.assert_not_called()
    mock_chroma.add_document.assert_called_once()
    call = mock_chroma.add_document.call_args.kwargs
    assert call["document_id"] == section.document_id
    assert call["text"].startswith("Item 2. MD&A")
    assert call["source"] == "sec_filing"
    assert call["replace_family"] is True
    assert call["metadata"] == {
        "accession": "ACC-1", "as_of_at": "2026-03-31",
        "authority_tier": "direct_sec",
        "content_hash": content_hash(section.text),
        "corpus_item_id": "sec:ACC-1:item_2",
        "document_family_id": "sec:ACC-1:item_2",
        "evidence_authority": "direct_sec", "filing_date": "2026-05-15",
        "form": "10-Q", "item": "item_2", "item_type": "sec_filing",
        "normalization_version": NORMALIZATION_VERSION,
        "parent_id": "sec:ACC-1:item_2", "parsed_path": "parsed/ACC-1.txt",
        "provider_record_id": "ACC-1", "published_at": "2026-05-15",
        "report_period": "2026-03-31", "section_heading": "Item 2. MD&A",
        "section_index": 0, "section_key": "item_2",
        "source_category": "regulatory_filing", "source_name": "sec",
        "source_url": "https://sec.example", "tickers": "NVDA",
    }
    assert result == {
        "sections_written": 1, "chunks_written": 1,
        "replacements": 0, "skipped": 0,
    }


def test_reprocessing_filing_section_replaces_family_without_duplicate(store, mock_chroma):
    section = FilingSection(
        accession="ACC-1", ticker="NVDA", form="10-Q",
        filing_date="2026-05-15", report_period="2026-03-31",
        section_key="item_2", section_heading="Item 2. MD&A", section_index=0,
        text="Item 2. MD&A\nUpdated revenue.", source_url="https://sec.example",
        parsed_path="parsed/ACC-1.txt",
    )
    mock_chroma.count_filing_section_chunks.side_effect = [1, 1]

    result = store.add_filing_sections([section])

    mock_chroma.delete_filing_section_family.assert_not_called()
    assert mock_chroma.add_document.call_args.kwargs["document_id"] == section.document_id
    assert mock_chroma.add_document.call_args.kwargs["replace_family"] is True
    assert result["replacements"] == 1
    assert result["sections_written"] == 1


def test_section_read_apis_delegate_with_bounds(fully_mocked_store):
    store, _sqlite, chroma = fully_mocked_store
    chroma.get_section_chunks.return_value = [{"id": "x"}]
    chroma.get_adjacent_sections.return_value = [{"id": "y"}]
    chroma.count_filing_sections.return_value = 4

    assert store.get_section_chunks("parent", limit=5, offset=2) == [{"id": "x"}]
    assert store.get_adjacent_sections("ACC", 3, before=1, after=2) == [{"id": "y"}]
    assert store.count_filing_sections("ACC") == 4
    store.delete_filing_section_family("parent")

    chroma.get_section_chunks.assert_called_once_with("parent", limit=5, offset=2)
    chroma.get_adjacent_sections.assert_called_once_with("ACC", 3, before=1, after=2)
    chroma.count_filing_sections.assert_called_once_with("ACC")
    chroma.delete_filing_section_family.assert_called_once_with("parent")


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


# ── Data Revision (2.2.6.2) ──────────────────────────

def test_retrieval_revision_delegates(fully_mocked_store):
    s, sqlite, _chroma = fully_mocked_store
    sqlite.get_store_revision.return_value = 7
    sqlite.bump_store_revision.return_value = 8

    assert s.retrieval_revision() == 7
    assert s.bump_retrieval_revision("ingest") == 8
    sqlite.get_store_revision.assert_called_once_with()
    sqlite.bump_store_revision.assert_called_once_with("ingest")


def test_facade_mutations_bump_revision_before_writing(store):
    """Every model-visible facade write advances the revision (real SQLite)."""
    r0 = store.retrieval_revision()
    store.save_fundamental("NVDA", "revenue", 26.0, period="2026-Q1")
    r1 = store.retrieval_revision()
    assert r1 > r0
    store.save_document("sec/NVDA/10-K", "body", ticker="NVDA")
    assert store.retrieval_revision() > r1


def test_bump_failure_never_crashes_mutation(store, monkeypatch):
    """A revision-store hiccup must not crash the write it precedes (fail-soft)."""
    monkeypatch.setattr(
        store.sqlite, "bump_store_revision",
        MagicMock(side_effect=RuntimeError("locked")))
    # Still writes the fact; only the (best-effort) revision bump is skipped.
    assert store.save_fundamental("NVDA", "eps", 2.5, period="2026-Q1") is True


# ── Reset ────────────────────────────────────────────

def test_reset(store, mock_chroma, tmp_path: Path):
    store.save_fundamental("TEST", "metric_a", 1.0, period="2026-Q1")
    store.upsert_universe_snapshot(
        "ivv",
        "2026-07-01T00:00:00Z",
        [{
            "symbol": "TEST", "company_name": "Test Incorporated",
            "source": "ivv", "index_code": "sp500",
        }],
    )
    store.register_filing(
        "TEST", "10-K", "2026-01-01", "2026-FY",
        "acc-reset-1", "http://example.com",
    )

    store.reset()

    mock_chroma.reset_collection.assert_called_once()
    with store.sqlite._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
        security_count = conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
    assert count == 0
    assert security_count == 0


def test_universe_facade_methods_delegate_without_extra_revision_bump(fully_mocked_store):
    store, sqlite, _chroma = fully_mocked_store
    sqlite.list_securities.return_value = [{"ticker": "AAPL"}]
    sqlite.get_security.return_value = {"ticker": "AAPL"}
    sqlite.resolve_security.return_value = {"ticker": "AAPL"}
    sqlite.list_memberships.return_value = [{"index_code": "sp500"}]
    sqlite.upsert_universe_snapshot.return_value = {"changed": False}
    sqlite.list_universe_errors.return_value = []

    assert store.list_securities(index="sp500", limit=5, offset=1) == [{"ticker": "AAPL"}]
    assert store.get_security("AAPL") == {"ticker": "AAPL"}
    assert store.resolve_security("AAPL", provider="sec", as_of="2026-01-01") == {
        "ticker": "AAPL"
    }
    assert store.list_memberships(index_code="sp500", active=True) == [
        {"index_code": "sp500"}
    ]
    assert store.upsert_universe_snapshot("ivv", "2026-01-01", []) == {"changed": False}
    assert store.list_universe_errors("run-1") == []

    sqlite.list_securities.assert_called_once_with(
        index="sp500", active=True, sector=None, limit=5, offset=1,
    )
    sqlite.resolve_security.assert_called_once_with(
        "AAPL", provider="sec", as_of="2026-01-01",
    )
    sqlite.upsert_universe_snapshot.assert_called_once_with("ivv", "2026-01-01", [])
    sqlite.bump_store_revision.assert_not_called()


def test_structured_record_facade_methods_delegate(fully_mocked_store):
    store, sqlite, _chroma = fully_mocked_store
    observation = MagicMock()
    event = MagicMock()
    sqlite.upsert_observation_record.return_value = {"created": True}
    sqlite.upsert_event_record.return_value = {"created": True}

    assert store.upsert_observation(observation) == {"created": True}
    assert store.upsert_event(event) == {"created": True}
    sqlite.upsert_observation_record.assert_called_once_with(observation)
    sqlite.upsert_event_record.assert_called_once_with(event)


def test_numeric_narrative_is_rejected_before_either_store_is_mutated(store, mock_chroma):
    body = "ACME close was 42.00 USD."
    record = NarrativeRecord(
        corpus_item_id="bad-observation",
        source_name="massive",
        source_category="market_data",
        provider_record_id="bar-1",
        original_publisher=None,
        item_type="observation",
        title="ACME daily close",
        body=body,
        published_at="2026-07-14T00:00:00Z",
        observed_at="2026-07-14T00:00:00Z",
        accessed_at="2026-07-14T00:00:00Z",
        ingested_at="2026-07-14T00:00:00Z",
        source_url="https://example.test/bar/1",
        canonical_url=None,
        license_label="provider_entitlement",
        normalization_version=NORMALIZATION_VERSION,
        content_hash=content_hash(body),
        document_family="market_observation",
    )

    with pytest.raises(ValueError, match="structured-only"):
        store.upsert_narrative(record)

    assert store.sqlite.count_corpus_items() == 0
    mock_chroma.add_document.assert_not_called()


def test_news_indexing_uses_provider_summary_and_complete_family_facets(store, mock_chroma):
    store.upsert_universe_snapshot(
        "ivv", "2026-07-14T00:00:00Z", [{
            "symbol": "ACME", "company_name": "Acme Corporation",
            "source": "ivv", "index_code": "sp500", "exchange": "NYSE",
            "sector": "Industrials", "industry": "Machinery",
        }],
    )
    security = store.resolve_security("ACME")
    body = "Headline\n\nProvider summary\n\nPublisher name that must not be indexed"
    record = NarrativeRecord(
        corpus_item_id="news-1",
        source_name="finnhub",
        source_category="news_vendor",
        provider_record_id="provider-1",
        original_publisher="Publisher",
        item_type="news",
        title="Headline",
        body=body,
        summary="Provider summary",
        published_at="2026-07-14T12:00:00Z",
        effective_at="2026-07-15T00:00:00Z",
        as_of_at="2026-07-14T00:00:00Z",
        observed_at="2026-07-14T12:01:00Z",
        accessed_at="2026-07-14T12:02:00Z",
        ingested_at="2026-07-14T12:03:00Z",
        source_url="https://example.test/news/1",
        canonical_url="https://publisher.test/news/1",
        license_label="provider_entitlement",
        normalization_version=NORMALIZATION_VERSION,
        content_hash=content_hash(body),
        document_family="company_news",
        security_ids=(security["security_id"],),
        tickers=("ACME",),
        evidence_authority="provider",
    )

    store.upsert_narrative(record)

    call = mock_chroma.add_document.call_args.kwargs
    assert call["text"] == "Headline\n\nProvider summary"
    assert call["replace_family"] is True
    assert call["document_id"] == "news-1"
    assert call["metadata"] == {
        "authority_tier": "provider",
        "canonical_url": "https://publisher.test/news/1",
        "content_hash": record.content_hash,
        "corpus_item_id": "news-1",
        "document_family": "company_news",
        "document_family_id": "news-1",
        "effective_at": "2026-07-15T00:00:00Z",
        "evidence_authority": "provider",
        "index_memberships": "sp500",
        "industries": "Machinery",
        "item_type": "news",
        "license_label": "provider_entitlement",
        "normalization_version": NORMALIZATION_VERSION,
        "original_publisher": "Publisher",
        "provider_record_id": "provider-1",
        "published_at": "2026-07-14T12:00:00Z",
        "as_of_at": "2026-07-14T00:00:00Z",
        "security_ids": security["security_id"],
        "sectors": "Industrials",
        "source_category": "news_vendor",
        "source_name": "finnhub",
        "tickers": "ACME",
    }


def test_corpus_accounting_delegates_to_sqlite_only(fully_mocked_store):
    store, sqlite, chroma = fully_mocked_store
    sqlite.get_corpus_accounting.return_value = [{
        "key": "news_vendor", "count": 2, "approximate_bytes": 512,
    }]

    result = store.get_corpus_accounting(
        "source_category", indexing_state="indexed", limit=10, offset=2,
    )

    assert result == [{"key": "news_vendor", "count": 2, "approximate_bytes": 512}]
    sqlite.get_corpus_accounting.assert_called_once_with(
        "source_category", source_category=None, source=None, item_type=None,
        event_type=None, security=None, sector=None, industry=None, index=None,
        coverage_tier=None, year=None, month=None, indexing_state="indexed",
        limit=10, offset=2,
    )
    chroma.iter_documents.assert_not_called()


# ── Narrative inventory counts: SQLite lexical ledger + Chroma fallback (2.3.7.6) ──

def test_store_source_counts_use_sqlite_lexical_ledger(store, mock_chroma):
    """Narrative source/ticker counts come from the indexed SQLite lexical
    ledger, never a full Chroma metadata scan, when FTS5 is available."""
    if not store.sqlite.fts5_available():
        pytest.skip("FTS5 unavailable")
    store.save_documents_batch(
        ["d1", "d2", "d3"],
        ["alpha body", "beta body", "gamma body"],
        [
            {"source": "sec", "ticker": "NVDA"},
            {"source": "sec", "ticker": "NVDA"},
            {"source": "gdelt", "ticker": "AMD"},
        ],
    )
    # The Chroma scan must not be consulted when the ledger can answer.
    mock_chroma.get_source_counts.side_effect = AssertionError("scanned Chroma")
    mock_chroma.get_ticker_counts.side_effect = AssertionError("scanned Chroma")

    src = {row["source"]: row["count"] for row in store.get_source_counts(limit=100)}
    assert src == {"sec": 2, "gdelt": 1}

    tickers = {row["ticker"]: row for row in store.get_ticker_counts(limit=100)}
    assert tickers["NVDA"]["record_count"] == 2
    assert tickers["AMD"]["record_count"] == 1
    # Response shape is unchanged from the legacy Chroma-scan merge.
    assert set(tickers["NVDA"]) == {
        "ticker", "record_count", "sources", "source_counts", "company_name",
    }


def test_store_source_counts_fall_back_to_chroma_without_lexical_ledger(
    store, mock_chroma, monkeypatch,
):
    """An FTS5-less runtime keeps the legacy Chroma metadata-scan count basis."""
    monkeypatch.setattr(store.sqlite, "lexical_counts_available", lambda: False)
    mock_chroma.get_source_counts.return_value = [{"source": "legacy", "count": 5}]
    mock_chroma.get_ticker_counts.return_value = [
        {"ticker": "NVDA", "record_count": 7, "sources": ["legacy"],
         "company_name": None},
    ]

    src = {row["source"]: row["count"] for row in store.get_source_counts(limit=100)}
    assert src.get("legacy") == 5

    tickers = {row["ticker"]: row for row in store.get_ticker_counts(limit=100)}
    assert tickers["NVDA"]["record_count"] == 7

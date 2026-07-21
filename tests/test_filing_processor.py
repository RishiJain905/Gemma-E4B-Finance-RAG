"""
tests/test_filing_processor.py
Test suite for FilingProcessor — the SEC filing pipeline orchestrator.

Two layers:
  1. Deterministic unit tests (no network) using mocked fetcher/parser/store
     to exercise pure orchestration logic: process_pending_filings,
     process_ticker, _process_single_filing, discovery delegation, and the
     combined discover+process pipeline. Plus a status_report test on a REAL
     temp SQLite store seeded with register_filing + mark_filing_parsed.
  2. One LIVE end-to-end test that hits EDGAR + the TraceAlchemy model on
     port 8087 + ChromaDB embeddings, isolated via a temp Store. It is
     skip-guarded so the suite stays green when those endpoints are offline.
"""

import socket
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import httpx
import pytest

from src.sec import FilingProcessor
from src.storage.store import Store


# ── Helpers / fixtures ──────────────────────────────

def _make_filing(ticker="AAPL", filing_type="10-K", accession="0000320193-25-000001",
                 period="2024", filing_date="2025-01-15",
                 source_url="https://www.sec.gov/Archives/edgar/data/320193/x/a.htm"):
    return {
        "ticker": ticker,
        "filing_type": filing_type,
        "filing_date": filing_date,
        "period": period,
        "accession": accession,
        "source_url": source_url,
    }


@pytest.fixture
def mock_processor():
    """A FilingProcessor with fully mocked store/fetcher/parser (no network)."""
    store = MagicMock()
    fetcher = MagicMock()
    parser = MagicMock()
    proc = FilingProcessor(store=store, fetcher=fetcher, parser=parser)
    # These tests exercise the legacy whole-document orchestration; the section
    # index path (default-on since 2026-07-20) is covered separately. Tests that
    # want it set proc.index_filing_text = True explicitly.
    proc.index_filing_text = False
    return proc


# ── 1. process_pending_filings orchestration ────────

def test_process_pending_filings_empty(mock_processor):
    """No unprocessed filings -> zeroed result with empty errors."""
    mock_processor.store.sqlite.get_unprocessed_filings.return_value = []

    result = mock_processor.process_pending_filings()

    assert result == {"processed": 0, "failed": 0, "errors": []}


def test_process_pending_filings_enabled_reports_zero_index_counts(mock_processor):
    mock_processor.index_filing_text = True
    mock_processor.store.sqlite.get_unprocessed_filings.return_value = []

    result = mock_processor.process_pending_filings()

    assert result == {
        "processed": 0, "failed": 0, "errors": [],
        "sections_written": 0, "chunks_written": 0, "replacements": 0,
        "skipped": 0, "index_pending": 0,
    }


def test_process_pending_filings_mixed(mock_processor):
    """Mixed success / no-facts / exception yields correct counts + errors."""
    filings = [
        _make_filing(ticker="AAPL", filing_type="10-K", accession="ACC-1"),
        _make_filing(ticker="MSFT", filing_type="10-Q", accession="ACC-2"),
        _make_filing(ticker="NVDA", filing_type="10-K", accession="ACC-3"),
    ]
    mock_processor.store.sqlite.get_unprocessed_filings.return_value = filings

    with patch.object(
        mock_processor, "_process_single_filing",
        side_effect=[True, False, RuntimeError("boom")],
    ):
        result = mock_processor.process_pending_filings()

    assert result["processed"] == 1
    assert result["failed"] == 2
    assert len(result["errors"]) == 2
    # No-facts path: descriptive "returned no facts" error string
    assert any("ACC-2" in e and "no facts" in e for e in result["errors"])
    # Exception path: surfaces the exception message
    assert any("ACC-3" in e and "boom" in e for e in result["errors"])


# ── 2. process_ticker filtering (case-insensitive) ──

def test_process_ticker_case_insensitive(mock_processor):
    """process_ticker filters get_unprocessed_filings to the requested ticker."""
    filings = [
        _make_filing(ticker="AAPL", accession="A-1"),
        _make_filing(ticker="MSFT", accession="M-1"),
        _make_filing(ticker="AAPL", accession="A-2"),
    ]
    mock_processor.store.sqlite.get_unprocessed_filings.return_value = filings

    processed_accessions = []

    def _fake_single(filing):
        processed_accessions.append(filing["accession"])
        return True

    with patch.object(mock_processor, "_process_single_filing", side_effect=_fake_single):
        # lowercase ticker should still match the uppercase rows
        result = mock_processor.process_ticker("aapl")

    assert result["processed"] == 2
    assert result["failed"] == 0
    assert set(processed_accessions) == {"A-1", "A-2"}


def test_process_ticker_no_match(mock_processor):
    """A ticker with no unprocessed filings returns a zeroed result."""
    mock_processor.store.sqlite.get_unprocessed_filings.return_value = [
        _make_filing(ticker="AAPL", accession="A-1"),
    ]
    with patch.object(mock_processor, "_process_single_filing") as single:
        result = mock_processor.process_ticker("TSLA")

    assert result == {"processed": 0, "failed": 0, "errors": []}
    single.assert_not_called()


# ── 3. _process_single_filing paths ─────────────────

def test_process_single_filing_download_none(mock_processor):
    """Download returns None -> False and mark_cache_stale is called."""
    filing = _make_filing(ticker="AAPL", filing_type="10-K", accession="ACC-X")
    mock_processor.fetcher.download_filing_text.return_value = None

    result = mock_processor._process_single_filing(filing)

    assert result is False
    mock_processor.store.sqlite.mark_cache_stale.assert_called_once()
    args, kwargs = mock_processor.store.sqlite.mark_cache_stale.call_args
    assert args[0] == "AAPL"
    assert args[1] == "sec_10-K_text"
    # Parser/store should not have been invoked
    mock_processor.parser.extract_facts_from_filing.assert_not_called()
    mock_processor.store.process_filing.assert_not_called()


def test_process_single_filing_empty_facts(mock_processor):
    """Text downloads but parser yields no facts -> False, no store write."""
    filing = _make_filing(ticker="AAPL", filing_type="10-Q", accession="ACC-Y")
    mock_processor.fetcher.download_filing_text.return_value = "some filing text"
    mock_processor.parser.extract_facts_from_filing.return_value = []

    result = mock_processor._process_single_filing(filing)

    assert result is False
    mock_processor.store.process_filing.assert_not_called()


def test_process_single_filing_happy_path(mock_processor):
    """Happy path: stores with source_type=sec_<lower> and the full text, True."""
    long_text = "X" * 9000
    facts = [{"metric": "total_revenue", "value": 100.0, "unit": "billion_usd"}]
    filing = _make_filing(ticker="AAPL", filing_type="10-K", accession="ACC-Z", period="2024")
    mock_processor.fetcher.download_filing_text.return_value = long_text
    mock_processor.parser.extract_facts_from_filing.return_value = facts

    result = mock_processor._process_single_filing(filing)

    assert result is True
    mock_processor.store.process_filing.assert_called_once()
    kwargs = mock_processor.store.process_filing.call_args.kwargs
    assert kwargs["filing_record"]["source_type"] == "sec_10-k"
    # Full text is now passed through; ChromaStore handles chunking for embedding.
    assert kwargs["extracted_text"] == long_text
    assert kwargs["extracted_facts"] == facts
    # Parser was called with the filing's fields
    parse_kwargs = mock_processor.parser.extract_facts_from_filing.call_args.kwargs
    assert parse_kwargs["ticker"] == "AAPL"
    assert parse_kwargs["filing_type"] == "10-K"
    assert parse_kwargs["period"] == "2024"


def test_enabled_section_indexing_persists_artifact_then_marks_parsed(tmp_path):
    store = MagicMock()
    store.add_filing_sections.return_value = {
        "sections_written": 2, "chunks_written": 3, "replacements": 0, "skipped": 0,
    }
    store.count_filing_sections.return_value = 2
    fetcher = MagicMock()
    fetcher.download_filing_text.return_value = (
        "Item 1. Business\nWe sell products.\n"
        "Item 7. Management's Discussion and Analysis\nRevenue was $42.\n"
    )
    parser = MagicMock()
    parser.extract_facts_from_filing.return_value = [
        {"metric": "revenue", "value": 42.0, "unit": "usd"}
    ]
    processor = FilingProcessor(
        store=store, fetcher=fetcher, parser=parser,
        sec_config={
            "index_filing_text": True,
            "max_sections_per_filing": 100,
            "max_section_chars": 1_000_000,
            "index_forms": ["10-K", "10-Q", "8-K"],
        },
        parsed_dir=tmp_path / "parsed",
    )

    assert processor._process_single_filing(_make_filing(accession="ACC-INDEX")) is True

    sections = store.add_filing_sections.call_args.args[0]
    assert len(sections) == 2
    artifact = tmp_path / "parsed" / "ACC-INDEX.txt"
    assert artifact.read_text(encoding="utf-8").startswith("Item 1. Business")
    assert all(section.parsed_path == str(artifact) for section in sections)
    store.process_filing.assert_called_once()
    assert store.process_filing.call_args.kwargs["index_document"] is False
    assert store.process_filing.call_args.kwargs["mark_parsed"] is False
    store.sqlite.mark_filing_parsed.assert_called_once_with(
        "ACC-INDEX", embedding_id="sec:ACC-INDEX", file_path=str(artifact),
        section_count=2, chunk_count=3,
    )


def test_chroma_failure_records_retryable_index_pending(tmp_path):
    store = MagicMock()
    store.add_filing_sections.side_effect = RuntimeError("Chroma unavailable")
    fetcher = MagicMock()
    fetcher.download_filing_text.return_value = "Item 1. Business\nUsable filing prose."
    parser = MagicMock()
    parser.extract_facts_from_filing.return_value = [
        {"metric": "revenue", "value": 42.0, "unit": "usd"}
    ]
    processor = FilingProcessor(
        store=store, fetcher=fetcher, parser=parser,
        sec_config={"index_filing_text": True, "max_sections_per_filing": 100,
                    "max_section_chars": 1_000_000, "index_forms": ["10-K"]},
        parsed_dir=tmp_path / "parsed",
    )

    assert processor._process_single_filing(_make_filing(accession="ACC-PENDING")) is False

    artifact = tmp_path / "parsed" / "ACC-PENDING.txt"
    assert artifact.exists()
    store.sqlite.mark_filing_index_pending.assert_called_once_with(
        "ACC-PENDING", file_path=str(artifact), error="Chroma unavailable",
    )
    store.sqlite.mark_filing_parsed.assert_not_called()


def test_index_pending_rows_remain_retryable(tmp_path):
    store = Store(db_path=tmp_path / "pending.db", chroma_path=tmp_path / "chroma")
    store.register_filing("AAPL", "10-K", "2025-01-15", "2024", "ACC-P", "url")
    store.sqlite.mark_filing_index_pending(
        "ACC-P", file_path=str(tmp_path / "ACC-P.txt"), error="offline",
    )

    rows = store.sqlite.get_unprocessed_filings()

    assert [row["accession"] for row in rows] == ["ACC-P"]
    assert rows[0]["status"] == "index_pending"
    assert rows[0]["index_error"] == "offline"


def _indexing_processor(store, fetcher, parser, tmp_path, *, index_forms=("10-K",)):
    """A deep-index FilingProcessor (index_filing_text on) with a mocked store."""
    return FilingProcessor(
        store=store, fetcher=fetcher, parser=parser,
        sec_config={
            "index_filing_text": True,
            "max_sections_per_filing": 100,
            "max_section_chars": 1_000_000,
            "index_forms": list(index_forms),
        },
        parsed_dir=tmp_path / "parsed",
    )


def test_zero_facts_with_sections_indexes_and_marks_parsed(tmp_path):
    """Decoupling: no facts but usable sections -> success, sections indexed,
    filing marked parsed (the tonight's-backfill failure case)."""
    store = MagicMock()
    store.add_filing_sections.return_value = {
        "sections_written": 2, "chunks_written": 3, "replacements": 0, "skipped": 0,
    }
    store.count_filing_sections.return_value = 2
    fetcher = MagicMock()
    fetcher.download_filing_text.return_value = (
        "Item 1. Business\nWe sell products.\n"
        "Item 7. Management's Discussion and Analysis\nRevenue grew a lot.\n"
    )
    parser = MagicMock()
    parser.extract_facts_from_filing.return_value = []  # zero facts
    processor = _indexing_processor(store, fetcher, parser, tmp_path)

    assert processor._process_single_filing(_make_filing(accession="ACC-NOFACTS")) is True

    store.add_filing_sections.assert_called_once()
    artifact = tmp_path / "parsed" / "ACC-NOFACTS.txt"
    store.sqlite.mark_filing_parsed.assert_called_once_with(
        "ACC-NOFACTS", embedding_id="sec:ACC-NOFACTS", file_path=str(artifact),
        section_count=2, chunk_count=3,
    )
    # Facts stream empty: process_filing still runs for the section path but
    # stores no facts and does not index the whole document.
    kwargs = store.process_filing.call_args.kwargs
    assert kwargs["extracted_facts"] == []
    assert kwargs["index_document"] is False
    assert kwargs["mark_parsed"] is False
    store.sqlite.mark_filing_index_pending.assert_not_called()


def test_facts_without_sections_stored_legacy_style(tmp_path):
    """Decoupling: facts but no usable sections -> facts stored legacy-style,
    success, no index_pending."""
    store = MagicMock()
    fetcher = MagicMock()
    # No recognizable SEC item/all-caps headings -> zero usable sections.
    fetcher.download_filing_text.return_value = (
        "Plain narrative prose with no filing structure at all."
    )
    parser = MagicMock()
    parser.extract_facts_from_filing.return_value = [
        {"metric": "revenue", "value": 42.0, "unit": "usd"}
    ]
    processor = _indexing_processor(store, fetcher, parser, tmp_path)

    assert processor._process_single_filing(_make_filing(accession="ACC-FACTSONLY")) is True

    # Legacy-style storage: process_filing with defaults (whole-doc + parsed).
    store.process_filing.assert_called_once()
    kwargs = store.process_filing.call_args.kwargs
    assert kwargs["extracted_facts"][0]["metric"] == "revenue"
    assert "index_document" not in kwargs  # defaults -> legacy whole-doc index
    assert "mark_parsed" not in kwargs
    store.add_filing_sections.assert_not_called()
    store.sqlite.mark_filing_index_pending.assert_not_called()
    store.sqlite.mark_filing_parsed.assert_not_called()


def test_zero_facts_zero_sections_marks_index_pending(tmp_path):
    """Decoupling: neither facts nor usable sections -> failure, retryable
    index_pending, nothing stored."""
    store = MagicMock()
    fetcher = MagicMock()
    fetcher.download_filing_text.return_value = (
        "Plain prose without any filing structure."
    )
    parser = MagicMock()
    parser.extract_facts_from_filing.return_value = []
    processor = _indexing_processor(store, fetcher, parser, tmp_path)

    assert processor._process_single_filing(_make_filing(accession="ACC-EMPTY")) is False

    artifact = tmp_path / "parsed" / "ACC-EMPTY.txt"
    store.sqlite.mark_filing_index_pending.assert_called_once()
    call = store.sqlite.mark_filing_index_pending.call_args
    assert call.args[0] == "ACC-EMPTY"
    assert call.kwargs["file_path"] == str(artifact)
    store.process_filing.assert_not_called()
    store.sqlite.mark_filing_parsed.assert_not_called()


def test_fact_extraction_failure_still_indexes_sections(tmp_path):
    """A parser exception on the deep path is fail-soft: section indexing
    still proceeds and the filing is stored on its sections alone."""
    store = MagicMock()
    store.add_filing_sections.return_value = {
        "sections_written": 1, "chunks_written": 1, "replacements": 0, "skipped": 0,
    }
    store.count_filing_sections.return_value = 1
    fetcher = MagicMock()
    fetcher.download_filing_text.return_value = "Item 1. Business\nUsable filing prose."
    parser = MagicMock()
    parser.extract_facts_from_filing.side_effect = RuntimeError("parser blew up")
    processor = _indexing_processor(store, fetcher, parser, tmp_path)

    assert processor._process_single_filing(_make_filing(accession="ACC-PARSEFAIL")) is True

    store.add_filing_sections.assert_called_once()
    store.sqlite.mark_filing_parsed.assert_called_once()
    assert store.process_filing.call_args.kwargs["extracted_facts"] == []


def test_split_filing_sections_handles_20f_structure():
    """20-F item headings split into usable sections (no new parser needed)."""
    from src.sec.filing_sections import split_filing_sections

    text = (
        "Item 3. Key Information\n"
        "Selected financial data for the foreign private issuer follows.\n"
        "Item 5. Operating and Financial Review and Prospects\n"
        "Revenue increased year over year across all reportable segments.\n"
        "Item 18. Financial Statements\n"
        "The audited consolidated statements are included in this report.\n"
    )
    sections = split_filing_sections(
        text,
        {"accession": "acc-20f", "ticker": "NBIS", "filing_type": "20-F",
         "filing_date": "2026-04-30", "period": "2025"},
    )

    assert len(sections) >= 1
    assert all(section.form == "20-F" for section in sections)
    assert any(section.section_key.startswith("item_") for section in sections)


def test_legacy_path_zero_facts_stores_nothing(mock_processor):
    """Flag off: the fact gate is preserved byte-for-byte — zero facts returns
    False and touches neither the legacy nor the section-index store paths."""
    mock_processor.index_filing_text = False
    filing = _make_filing(ticker="AAPL", filing_type="10-K", accession="ACC-LEG")
    mock_processor.fetcher.download_filing_text.return_value = "Item 1. Business\nText."
    mock_processor.parser.extract_facts_from_filing.return_value = []

    assert mock_processor._process_single_filing(filing) is False

    mock_processor.store.process_filing.assert_not_called()
    mock_processor.store.add_filing_sections.assert_not_called()
    mock_processor.store.sqlite.mark_filing_index_pending.assert_not_called()


# ── 4. Discovery delegation ─────────────────────────

def test_discover_new_filings_delegates(mock_processor):
    """discover_new_filings delegates to fetcher.register_discovered_filings."""
    mock_processor.fetcher.register_discovered_filings.return_value = 3

    count = mock_processor.discover_new_filings("NVDA")

    assert count == 3
    mock_processor.fetcher.register_discovered_filings.assert_called_once_with("NVDA")


def test_discover_all_core_tickers_delegates(mock_processor):
    """discover_all_core_tickers delegates to fetcher.register_all_core_tickers."""
    mock_processor.fetcher.register_all_core_tickers.return_value = {"NVDA": 2, "AAPL": 1}

    result = mock_processor.discover_all_core_tickers()

    assert result == {"NVDA": 2, "AAPL": 1}
    mock_processor.fetcher.register_all_core_tickers.assert_called_once_with()


# ── 5. Combined pipeline composition ────────────────

def test_discover_and_process_all_composition(mock_processor):
    """discover_and_process_all composes discovery + processing(limit=50)."""
    with patch.object(
        mock_processor, "discover_all_core_tickers", return_value={"NVDA": 1}
    ) as disc, patch.object(
        mock_processor, "process_pending_filings",
        return_value={"processed": 1, "failed": 0, "errors": []},
    ) as proc:
        result = mock_processor.discover_and_process_all()

    disc.assert_called_once_with()
    proc.assert_called_once_with(limit=50)
    assert result == {
        "discovery": {"NVDA": 1},
        "processing": {"processed": 1, "failed": 0, "errors": []},
    }


def test_discover_and_process_ticker_composition(mock_processor):
    """discover_and_process_ticker composes per-ticker discovery + processing."""
    with patch.object(
        mock_processor, "discover_new_filings", return_value=2
    ) as disc, patch.object(
        mock_processor, "process_ticker",
        return_value={"processed": 2, "failed": 0, "errors": []},
    ) as proc:
        result = mock_processor.discover_and_process_ticker("AAPL")

    disc.assert_called_once_with("AAPL")
    proc.assert_called_once_with("AAPL")
    assert result == {
        "ticker": "AAPL",
        "new_filings_discovered": 2,
        "processing": {"processed": 2, "failed": 0, "errors": []},
    }


# ── 6. status_report on a real temp SQLite store ────

def test_status_report_real_sqlite(tmp_path):
    """status_report aggregates unprocessed/parsed counts per ticker."""
    store = Store(
        db_path=tmp_path / "x.db",
        chroma_path=tmp_path / "chroma",
    )
    # Seed filings directly via the SQLite layer (no network).
    store.register_filing("AAPL", "10-K", "2025-01-15", "2024", "AAPL-1", "u1")
    store.register_filing("AAPL", "10-Q", "2025-04-15", "2025-Q1", "AAPL-2", "u2")
    store.register_filing("MSFT", "10-K", "2025-02-01", "2024", "MSFT-1", "u3")
    # Mark one AAPL filing as parsed.
    store.sqlite.mark_filing_parsed("AAPL-1", embedding_id="emb-1")

    proc = FilingProcessor(store=store, fetcher=MagicMock(), parser=MagicMock())
    report = proc.status_report()

    assert report["total_unprocessed"] == 2  # AAPL-2 + MSFT-1
    assert report["total_parsed"] == 1       # AAPL-1
    assert report["filings_by_ticker"]["AAPL"] == {"unprocessed": 1, "parsed": 1}
    assert report["filings_by_ticker"]["MSFT"] == {"unprocessed": 1, "parsed": 0}


def test_status_report_includes_persisted_index_counts(tmp_path):
    store = Store(db_path=tmp_path / "status.db", chroma_path=tmp_path / "chroma")
    store.register_filing("AAPL", "10-K", "2025-01-15", "2024", "A-1", "u1")
    store.register_filing("AAPL", "10-Q", "2025-04-15", "2025-Q1", "A-2", "u2")
    store.sqlite.mark_filing_parsed(
        "A-1", embedding_id="sec:A-1", file_path="a.txt", section_count=3, chunk_count=8,
    )
    store.sqlite.mark_filing_index_pending("A-2", file_path="b.txt", error="offline")

    report = FilingProcessor(store=store, fetcher=MagicMock(), parser=MagicMock()).status_report()

    assert report["total_index_pending"] == 1
    assert report["indexed_sections"] == 3
    assert report["indexed_chunks"] == 8
    assert report["filings_by_ticker"]["AAPL"] == {
        "unprocessed": 0, "index_pending": 1, "parsed": 1,
    }


# ── 7. LIVE end-to-end (skip-guarded) ───────────────

EDGAR_HOST = "www.sec.gov"
MODEL_ENDPOINT = "http://127.0.0.1:8087/v1/embeddings"


def _endpoint_reachable(url: str, timeout: float = 3.0) -> bool:
    """TCP-connect check for a host:port derived from a URL."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _model_alive(timeout: float = 5.0) -> bool:
    """Confirm the llama-server embeddings endpoint actually responds."""
    try:
        resp = httpx.post(
            MODEL_ENDPOINT,
            json={"input": "ping", "model": "tracealchemy"},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.live
def test_live_end_to_end_pipeline(tmp_path):
    """Live: register AAPL filings then process one through the full pipeline."""
    if not _endpoint_reachable(f"https://{EDGAR_HOST}"):
        pytest.skip("SEC EDGAR is unreachable")
    if not _model_alive():
        pytest.skip("TraceAlchemy model endpoint (port 8087) is unreachable")

    store = Store(
        db_path=tmp_path / "finance.db",
        chroma_path=tmp_path / "chroma",
    )
    processor = FilingProcessor(store=store)

    # Discover + register AAPL filings (live EDGAR).
    new_count = processor.fetcher.register_discovered_filings("AAPL")
    if new_count == 0:
        pytest.skip("No AAPL filings discovered from EDGAR")

    # Process a single filing end-to-end (download -> parse -> store).
    result = processor.process_pending_filings(limit=1)

    assert result["processed"] >= 1, (
        f"expected at least one filing processed end-to-end: {result}"
    )

    # Facts persisted to the fundamentals table.
    with store.sqlite._connect() as conn:
        fact_count = conn.execute(
            "SELECT COUNT(*) AS n FROM fundamentals WHERE ticker = ?", ("AAPL",)
        ).fetchone()["n"]
        parsed_count = conn.execute(
            "SELECT COUNT(*) AS n FROM filings WHERE ticker = ? AND status = 'parsed'",
            ("AAPL",),
        ).fetchone()["n"]

    assert fact_count >= 1, "expected at least one fundamental fact persisted"
    assert parsed_count >= 1, "expected at least one filing marked parsed"

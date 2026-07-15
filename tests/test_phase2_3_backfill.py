"""tests/test_phase2_3_backfill.py
Offline tests for resumable Phase 2.3 identity and corpus metadata backfill.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.storage.phase2_3_backfill import Phase23Backfill
from src.storage.store import Store


@pytest.fixture
def legacy_store(tmp_path):
    documents = [
        {
            "id": "sec/AAPL/10-K#0",
            "metadata": {
                "document_family_id": "sec/AAPL/10-K",
                "ticker": "AAPL",
                "source": "10-K",
                "date": "2025-11-01",
                "source_url": "https://www.sec.gov/aapl-10k",
            },
        },
        {
            "id": "sec/AAPL/10-K#1",
            "metadata": {
                "document_family_id": "sec/AAPL/10-K",
                "ticker": "AAPL",
                "source": "10-K",
                "date": "2025-11-01",
                "source_url": "https://www.sec.gov/aapl-10k",
            },
        },
        {
            "id": "news/UNKNOWN/item",
            "metadata": {"ticker": "UNKNOWN", "source": "legacy_news"},
        },
        {
            "id": "news/multi/item",
            "metadata": {"tickers": "AAPL,MSFT", "source": "legacy_news"},
        },
    ]
    chroma = MagicMock()
    chroma.iter_document_metadata.side_effect = lambda limit, offset: documents[
        offset : offset + limit
    ]
    chroma.count.return_value = len(documents)
    chroma.heartbeat.return_value = True

    with patch("src.storage.store.ChromaStore", return_value=chroma):
        store = Store(db_path=tmp_path / "legacy.db", chroma_path=tmp_path / "chroma")

    with store.sqlite._connect() as conn:
        conn.execute(
            "INSERT INTO fundamentals "
            "(ticker, metric, value, period, source_type) VALUES ('AAPL', 'revenue', 1, '2025', 'sec')"
        )
        conn.execute(
            "INSERT INTO fundamentals "
            "(ticker, metric, value, period, source_type) VALUES ('MSFT', 'revenue', 2, '2025', 'sec')"
        )
        conn.execute(
            """INSERT INTO sec_companyfacts (
                ticker, cik, taxonomy, concept, value_text, value_numeric, unit,
                period_end, period_kind, form, filed_at, accession, source_url,
                source_accessed_at
            ) VALUES (
                'AAPL', '0000320193', 'us-gaap', 'Revenue', '1', 1, 'USD',
                '2025-09-30', 'annual', '10-K', '2025-11-01', 'acc-1',
                'https://www.sec.gov/companyfacts', '2025-11-01'
            )"""
        )
        conn.execute(
            """INSERT INTO filings (
                ticker, filing_type, filing_date, period, accession, source_url,
                status, summary_embedding_id
            ) VALUES (
                'AAPL', '10-K', '2025-11-01', '2025-FY', 'acc-1',
                'https://www.sec.gov/aapl-10k', 'parsed', 'sec/AAPL/10-K'
            )"""
        )
        conn.execute(
            "INSERT INTO cache_meta (ticker, source, status) VALUES ('AAPL', 'sec', 'fresh')"
        )
        conn.commit()
    return store, chroma


def test_backfill_interrupts_and_resumes_without_duplicates_or_embedding(legacy_store):
    store, chroma = legacy_store
    before_revision = store.retrieval_revision()
    before_counts = store.sqlite.get_ticker_counts(limit=20)

    first = Phase23Backfill(store, batch_size=1).run(max_batches=2)
    assert first["complete"] is False
    assert store.retrieval_revision() > before_revision

    result = first
    for _ in range(30):
        result = Phase23Backfill(store, batch_size=1).run(max_batches=2)
        if result["complete"]:
            break
    assert result["complete"] is True

    rerun = Phase23Backfill(store, batch_size=1).run()
    assert rerun["inserted"] == 0
    assert store.sqlite.get_ticker_counts(limit=20) == before_counts
    chroma.add_document.assert_not_called()
    chroma.add_documents_batch.assert_not_called()

    with store.sqlite._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0] >= 2
        assert conn.execute(
            "SELECT COUNT(*) FROM securities WHERE normalized_ticker='NVDA'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT cik FROM securities WHERE normalized_ticker='AAPL'"
        ).fetchone()[0] == "0000320193"
        for table in ("fundamentals", "sec_companyfacts", "filings", "cache_meta"):
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE security_id IS NOT NULL"
            ).fetchone()[0] >= 1
        assert conn.execute(
            "SELECT COUNT(*) FROM corpus_items WHERE document_family_id='sec/AAPL/10-K'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM corpus_item_securities cis "
            "JOIN securities s ON s.security_id=cis.security_id "
            "WHERE s.normalized_ticker='AAPL'"
        ).fetchone()[0] >= 1
        assert conn.execute(
            "SELECT COUNT(*) FROM corpus_item_securities cis "
            "JOIN corpus_items ci ON ci.corpus_item_id=cis.corpus_item_id "
            "WHERE ci.document_family_id='news/multi/item'"
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM identity_reconciliation_errors "
            "WHERE identifier='UNKNOWN' AND issue_type='orphan'"
        ).fetchone()[0] == 1


def test_backfill_records_ambiguous_cik_instead_of_guessing(tmp_path):
    chroma = MagicMock()
    chroma.iter_document_metadata.return_value = []
    chroma.count.return_value = 0
    with patch("src.storage.store.ChromaStore", return_value=chroma):
        store = Store(db_path=tmp_path / "ambiguous.db", chroma_path=tmp_path / "chroma")
    with store.sqlite._connect() as conn:
        for row_id, cik in enumerate(("0000000001", "0000000002"), start=1):
            conn.execute(
                """INSERT INTO sec_companyfacts (
                    ticker, cik, taxonomy, concept, value_text, value_numeric, unit,
                    period_end, period_kind, form, filed_at, accession, source_url,
                    source_accessed_at
                ) VALUES ('DUPE', ?, 'us-gaap', 'Revenue', '1', 1, 'USD',
                    '2025-12-31', 'annual', '10-K', '2026-01-01', ?, 'https://sec', '2026-01-01')""",
                (cik, f"acc-{row_id}"),
            )
        conn.commit()

    result = Phase23Backfill(store, batch_size=10).run()

    assert result["complete"] is True
    with store.sqlite._connect() as conn:
        assert conn.execute(
            "SELECT cik FROM securities WHERE normalized_ticker='DUPE'"
        ).fetchone()[0] is None
        assert conn.execute(
            "SELECT COUNT(*) FROM identity_reconciliation_errors "
            "WHERE identifier='DUPE' AND issue_type='ambiguous'"
        ).fetchone()[0] == 1

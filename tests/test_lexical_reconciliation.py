"""Offline tests for bounded lexical rebuild and reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.sqlite_store import SQLiteStore


def _row(chunk_id: str, text: str) -> dict:
    return {
        "id": chunk_id,
        "document": text,
        "metadata": {
            "document_family_id": chunk_id.split("#", 1)[0],
            "title": chunk_id,
            "ticker": "ACME",
            "source_category": "company_news",
            "source_name": "gdelt",
            "item_type": "news",
            "published_at": "2026-07-01",
        },
    }


def test_rebuild_is_bounded_resumable_and_idempotent(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "lexical.db")
    if not store.fts5_available():
        pytest.skip("test runtime has no SQLite FTS5")
    rows = [_row(f"family-{index}#0", f"document {index}") for index in range(5)]
    target_revision = store.bump_store_revision("seed")

    first = store.rebuild_lexical_index(
        lambda offset, limit: rows[offset: offset + limit],
        batch_size=2,
        target_revision=target_revision,
        max_batches=1,
    )
    second = store.rebuild_lexical_index(
        lambda offset, limit: rows[offset: offset + limit],
        batch_size=2,
        target_revision=target_revision,
    )
    third = store.rebuild_lexical_index(
        lambda offset, limit: rows[offset: offset + limit],
        batch_size=2,
        target_revision=target_revision,
    )

    assert first == {"status": "in_progress", "processed": 2, "cursor": 2}
    assert second["status"] == "completed"
    assert second["row_count"] == 5
    assert third["status"] == "completed"
    assert third["row_count"] == 5
    assert store.get_lexical_index_state()["indexed_revision"] == target_revision


def test_reconciliation_reports_and_repairs_missing_duplicate_stale_and_orphan(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "lexical.db")
    if not store.fts5_available():
        pytest.skip("test runtime has no SQLite FTS5")
    expected = [_row("a#0", "alpha current"), _row("b#0", "beta current")]
    revision = store.replace_lexical_families({
        "a": [_row("a#0", "alpha stale")],
        "orphan": [_row("orphan#0", "orphan")],
    })
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO corpus_fts (title, body, ticker, source_category, item_type, chunk_id, family_id) "
            "SELECT title, body, ticker, source_category, item_type, chunk_id, family_id "
            "FROM corpus_fts WHERE chunk_id='a#0'"
        )
        conn.commit()

    report = store.reconcile_lexical_index(iter(expected), repair=False, revision=revision)
    repaired = store.reconcile_lexical_index(iter(expected), repair=True, revision=revision)
    clean = store.reconcile_lexical_index(iter(expected), repair=False, revision=revision)

    assert report["counts"] == {"missing": 1, "duplicate": 1, "stale": 1, "orphan": 1}
    assert repaired["repaired"] is True
    assert clean["counts"] == {"missing": 0, "duplicate": 0, "stale": 0, "orphan": 0}

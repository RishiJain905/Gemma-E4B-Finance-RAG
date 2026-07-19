"""tests/test_watch_scheduler.py
Offline tests for scripts/watch_scheduler.py -- the read-only scheduler
health monitor. Seeds a real, migrated temp SQLite store (the same schema
the scheduler writes to) and drives the pure collector/render functions
plus the --once CLI path; never touches the live stack, network, or :8087.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import scripts.watch_scheduler as ws
from src.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "watch_test.db")


def _seed_dead_letter(store: SQLiteStore, count: int) -> None:
    with store._connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dead_letter (
                source TEXT NOT NULL,
                item_key TEXT NOT NULL,
                error TEXT,
                failed_at TEXT,
                retry_count INTEGER DEFAULT 0,
                last_error TEXT,
                PRIMARY KEY (source, item_key)
            )
            """
        )
        for i in range(count):
            conn.execute(
                "INSERT INTO dead_letter (source, item_key, error, failed_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                ("gdelt", f"item-{i}", "boom"),
            )
        conn.commit()


def _seed_corpus_items(store: SQLiteStore, count: int) -> None:
    with store._connect() as conn:
        for i in range(count):
            conn.execute(
                """
                INSERT INTO corpus_items (
                    corpus_item_id, source, source_category, item_type, title,
                    normalized_headline, language, accessed_at, ingested_at,
                    source_url, content_hash, document_family, indexing_status,
                    license_label, normalization_version, evidence_authority
                ) VALUES (?, 'yfinance', 'news', 'news', ?, ?, 'en',
                          datetime('now'), datetime('now'), 'https://example.test',
                          ?, 'news', 'indexed', 'public', '1', 'primary')
                """,
                (f"item-{i}", f"Headline {i}", f"headline {i}", f"hash-{i}"),
            )
        conn.commit()


def _seed_filings(store: SQLiteStore, *, parsed: int, unprocessed: int) -> None:
    with store._connect() as conn:
        for i in range(parsed):
            conn.execute(
                "INSERT INTO filings (ticker, filing_type, accession, status) "
                "VALUES ('NVDA', '10-K', ?, 'parsed')",
                (f"parsed-{i}",),
            )
        for i in range(unprocessed):
            conn.execute(
                "INSERT INTO filings (ticker, filing_type, accession, status) "
                "VALUES ('NVDA', '10-K', ?, 'unprocessed')",
                (f"unproc-{i}",),
            )
        conn.commit()


def _complete_run(store: SQLiteStore, *, sources: list[dict]) -> str:
    """Persist one finished scheduler run with the given per-source outcomes."""
    names = [s["source"] for s in sources]
    run_id = store.start_scheduler_run(
        "daily", policy_revision="p1", config_revision="c1",
        requested_sources=names,
    )
    for s in sources:
        store.record_scheduler_source_summary({
            "run_id": run_id,
            "source": s["source"],
            "status": s["status"],
            "items": s.get("items", 0),
            "error_class": s.get("error_class"),
            "error_message": s.get("error_message"),
        })
    if all(s["status"] == "success" for s in sources):
        overall = "success"
    elif any(s["status"] == "success" for s in sources):
        overall = "partial"
    else:
        overall = "error"
    store.complete_scheduler_run(run_id, status=overall)
    return run_id


# ── --once end-to-end ──────────────────────────────────────────────────

def test_once_snapshot_renders_all_sections(store, capsys):
    _complete_run(store, sources=[
        {"source": "finnhub", "status": "success", "items": 12},
        {"source": "gdelt", "status": "skipped"},
    ])
    _seed_filings(store, parsed=3, unprocessed=2)
    _seed_corpus_items(store, 4)
    store.mark_cache_fresh("SCHEDULER", "unified:finnhub", 24)

    rc = ws.main(["--db", str(store.db_path), "--once"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Scheduler Watch" in out
    assert "LAST COMPLETED RUN" in out
    assert "FRESHNESS" in out
    assert "QUEUES & STORES" in out
    assert "Ctrl+C to exit" in out
    assert "ALL GOOD" in out  # zero errors in this run
    assert "finnhub" in out
    assert "corpus_items=4" in out
    assert "3 parsed / 2 unprocessed" in out


def test_error_run_shows_red_verdict_and_error_detail(store, capsys):
    _complete_run(store, sources=[
        {"source": "finnhub", "status": "success", "items": 5},
        {
            "source": "sec_filings", "status": "error",
            "error_class": "rate_limit", "error_message": "HTTP 429 from EDGAR",
        },
    ])

    ws.main(["--db", str(store.db_path), "--once"])
    out = capsys.readouterr().out

    assert "1 SOURCE ERRORS" in out
    assert "sec_filings" in out
    assert "rate_limit" in out
    assert "1 success / 0 skipped / 1 error" in out


def test_no_completed_run_yet(store, capsys):
    ws.main(["--db", str(store.db_path), "--once"])
    out = capsys.readouterr().out

    assert "LAST COMPLETED RUN" in out
    assert "none recorded yet" in out


def test_missing_db_shows_waiting_grace(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.db"
    rc = ws.main(["--db", str(missing), "--once"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "waiting for database" in out
    assert "Scheduler Watch" in out  # header still renders


def test_dead_letter_count_surfaces_in_queues(store, capsys):
    _seed_dead_letter(store, 3)

    ws.main(["--db", str(store.db_path), "--once"])
    out = capsys.readouterr().out

    assert "dead_letter=3" in out


# ── Section collectors (direct, no CLI) ─────────────────────────────────

def test_freshness_staleness_math(store):
    store.mark_cache_fresh("SCHEDULER", "unified:finnhub", 24)
    store.upsert_cache_stale("SCHEDULER", "unified:gdelt", "boom")
    store.upsert_cache_stale("SCHEDULER", "unified:estimates", "boom")
    store.mark_cache_fresh("SCHEDULER", "unified:massive", 24)
    store.mark_cache_fresh("SCHEDULER", "unified:massive_news", 1)
    # sec_filings never fetched (no cache_meta row) but is known via a run
    _complete_run(store, sources=[{"source": "sec_filings", "status": "skipped"}])

    snapshot = ws.collect_snapshot(store.db_path)
    by_name = {e["name"]: e for e in snapshot.freshness}

    assert by_name["finnhub"]["state"] == "fresh"
    assert by_name["finnhub"]["age_hours"] is not None
    assert by_name["finnhub"]["ttl_hours"] == pytest.approx(24.0, abs=0.2)
    assert by_name["gdelt"]["state"] == "disabled"
    assert by_name["estimates"]["state"] == "stale"
    assert by_name["sec_filings"]["state"] == "never"
    assert by_name["massive"]["ttl_hours"] == pytest.approx(24.0, abs=0.2)
    assert by_name["massive_news"]["ttl_hours"] == pytest.approx(1.0, abs=0.2)

    rendered = ws.render_freshness(snapshot)
    assert "gdelt" in rendered and "state=disabled" in rendered
    assert "massive" in rendered and "massive_news" in rendered

    # stale-first ordering: stale, configured-disabled, never, then fresh
    states = [e["state"] for e in snapshot.freshness]
    assert (
        states.index("stale")
        < states.index("disabled")
        < states.index("never")
        < states.index("fresh")
    )


def _write_fake_chroma(db_path, revision):
    """Create a minimal chroma.sqlite3 publishing one corpus_revision."""
    import sqlite3 as _sq

    chroma_dir = db_path.parent / "chroma"
    chroma_dir.mkdir(exist_ok=True)
    conn = _sq.connect(chroma_dir / "chroma.sqlite3")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS collection_metadata "
        "(key TEXT, int_value INTEGER, str_value TEXT)"
    )
    conn.execute("DELETE FROM collection_metadata WHERE key='corpus_revision'")
    if revision is not None:
        conn.execute(
            "INSERT INTO collection_metadata (key, int_value) VALUES ('corpus_revision', ?)",
            (int(revision),),
        )
    conn.commit()
    conn.close()


def test_store_revision_ahead_is_not_a_fusion_failure(store, capsys):
    """Structured-only writes advance store_revision without touching the
    index; fusion health is indexed vs the CHROMA-published revision."""
    store.bump_store_revision("test")
    store.bump_store_revision("test")
    _write_fake_chroma(store.db_path, 0)  # matches indexed_revision=0

    ws.main(["--db", str(store.db_path), "--once"])
    out = capsys.readouterr().out

    assert "rev 0/2" in out
    assert "chroma rev 0 ok" in out
    assert "fusion OFF" not in out


def test_chroma_revision_drift_flags_fusion_off(store, capsys):
    _write_fake_chroma(store.db_path, 7)  # indexed_revision is 0 -> drift

    ws.main(["--db", str(store.db_path), "--once"])
    out = capsys.readouterr().out

    assert "chroma rev 7 DRIFT" in out
    assert "BM25 fusion OFF" in out


def test_unpublished_chroma_revision_flags_fusion_off(store, capsys):
    """The 2026-07-17 live bug: revision never published to Chroma silently
    disabled BM25 fusion. The monitor must make that state loud."""
    ws.main(["--db", str(store.db_path), "--once"])
    out = capsys.readouterr().out

    assert "UNPUBLISHED" in out
    assert "BM25 fusion OFF" in out


def test_revision_match_is_not_flagged(store, capsys):
    ws.main(["--db", str(store.db_path), "--once"])
    out = capsys.readouterr().out

    assert "MISMATCH" not in out
    assert "rev 0/0" in out


def test_active_run_panel_classifies_sources(store, capsys):
    started_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    store.start_scheduler_run(
        "daily", policy_revision="p1", config_revision="c1",
        requested_sources=["finnhub", "gdelt", "massive"],
        started_at=started_at,
    )
    # finnhub finished successfully after the run started -> "success"
    store.mark_cache_fresh("SCHEDULER", "unified:finnhub", 24)
    # gdelt was already fresh before the run started -> predicted "skip"
    store.mark_cache_fresh("SCHEDULER", "unified:gdelt", 24)
    with store._connect() as conn:
        conn.execute(
            "UPDATE cache_meta SET last_updated=datetime('now', '-2 hours') "
            "WHERE ticker='SCHEDULER' AND source='unified:gdelt'"
        )
        conn.commit()
    # massive has not been touched at all -> pending

    ws.main(["--db", str(store.db_path), "--once"])
    out = capsys.readouterr().out

    assert "ACTIVE RUN" in out
    assert "finnhub" in out and "done" in out
    assert "fresh, TTL" in out
    assert "massive" in out

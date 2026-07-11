# Test suite for SQLiteStore — Phase 1.2.2

from pathlib import Path

import pytest

from src.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    db = tmp_path / "test.db"
    return SQLiteStore(db)


# ── Initialization ──────────────────────────────────

def test_init_creates_db(tmp_path: Path):
    db = tmp_path / "test.db"
    store = SQLiteStore(db)
    assert store.db_path.exists()

    with store._connect() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"fundamentals", "filings", "cache_meta", "ingestion_log"} <= tables


def test_wal_mode(store: SQLiteStore):
    with store._connect() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# ── Fundamentals ────────────────────────────────────

def test_upsert_fundamental_insert(store: SQLiteStore):
    store.upsert_fundamental(
        ticker="AAPL",
        metric="revenue",
        value=100.0,
        unit="usd",
        period="2026-Q1",
        source_type="sec",
        source_url="http://example.com/sec",
    )
    row = store.get_fundamental("AAPL", "revenue", "2026-Q1")
    assert row is not None
    assert row["value"] == 100.0
    assert row["unit"] == "usd"
    assert row["source_type"] == "sec"


def test_upsert_fundamental_update(store: SQLiteStore):
    store.upsert_fundamental("AAPL", "revenue", 100.0, period="2026-Q1")
    store.upsert_fundamental("AAPL", "revenue", 200.0, period="2026-Q1")
    row = store.get_fundamental("AAPL", "revenue", "2026-Q1")
    assert row["value"] == 200.0


def test_get_fundamental_with_period(store: SQLiteStore):
    store.upsert_fundamental("AAPL", "revenue", 100.0, period="2026-Q1")
    store.upsert_fundamental("AAPL", "revenue", 200.0, period="2026-Q2")

    q1 = store.get_fundamental("AAPL", "revenue", "2026-Q1")
    q2 = store.get_fundamental("AAPL", "revenue", "2026-Q2")
    assert q1["value"] == 100.0
    assert q2["value"] == 200.0


def test_get_fundamental_latest(store: SQLiteStore):
    store.upsert_fundamental("AAPL", "revenue", 100.0, period="2026-Q1")
    store.upsert_fundamental("AAPL", "revenue", 200.0, period="2026-Q2")

    latest = store.get_fundamental("AAPL", "revenue")
    assert latest["value"] == 200.0
    assert latest["period"] == "2026-Q2"


def test_get_fundamentals_batch(store: SQLiteStore):
    store.upsert_fundamental("AAPL", "revenue", 100.0, period="2026-Q1")
    store.upsert_fundamental("AAPL", "revenue", 110.0, period="2026-Q2")
    store.upsert_fundamental("AAPL", "eps", 2.5, period="2026-Q1")
    store.upsert_fundamental("AAPL", "eps", 2.8, period="2026-Q2")

    batch = store.get_fundamentals_batch("AAPL", ["revenue", "eps"])
    assert batch == {"revenue": 110.0, "eps": 2.8}

    all_batch = store.get_fundamentals_batch("AAPL")
    assert all_batch == {"revenue": 110.0, "eps": 2.8}


# ── Filings ───────────────────────────────────────

def test_register_filing_new(store: SQLiteStore):
    result = store.register_filing(
        "AAPL", "10-K", "2025-09-30", "2025-FY", "0000320193-25-000123", "http://sec.gov"
    )
    assert result is True


def test_register_filing_duplicate(store: SQLiteStore):
    store.register_filing(
        "AAPL", "10-K", "2025-09-30", "2025-FY", "0000320193-25-000123", "http://sec.gov"
    )
    result = store.register_filing(
        "AAPL", "10-K", "2025-09-30", "2025-FY", "0000320193-25-000123", "http://sec.gov"
    )
    assert result is False


def test_mark_filing_parsed(store: SQLiteStore):
    acc = "0000320193-25-000124"
    store.register_filing("AAPL", "10-Q", "2025-12-31", "2026-Q1", acc, "http://sec.gov")
    store.mark_filing_parsed(acc, embedding_id="emb-123")

    with store._connect() as conn:
        row = conn.execute(
            "SELECT * FROM filings WHERE accession=?", (acc,)
        ).fetchone()

    assert row["status"] == "parsed"
    assert row["parsed_at"] is not None
    assert row["summary_embedding_id"] == "emb-123"


def test_get_unprocessed_filings(store: SQLiteStore):
    store.register_filing("AAPL", "10-K", "2025-09-30", "2025-FY", "acc-1", "http://sec.gov")
    store.register_filing("AAPL", "10-Q", "2025-12-31", "2026-Q1", "acc-2", "http://sec.gov")
    store.mark_filing_parsed("acc-1")

    unprocessed = store.get_unprocessed_filings(limit=10)
    accs = {r["accession"] for r in unprocessed}
    assert accs == {"acc-2"}


# ── Cache Management ──────────────────────────────

def test_mark_cache_fresh(store: SQLiteStore):
    store.mark_cache_fresh("AAPL", "yfinance", ttl_hours=24)
    entry = store.get_cache_status("AAPL", "yfinance")
    assert entry is not None
    assert entry["status"] == "fresh"
    assert entry["next_scheduled_update"] > entry["last_updated"]


def test_mark_cache_stale(store: SQLiteStore):
    store.mark_cache_fresh("AAPL", "yfinance")
    store.mark_cache_stale("AAPL", "yfinance", error="connection timeout")
    entry = store.get_cache_status("AAPL", "yfinance")
    assert entry["status"] == "stale"
    assert entry["error_message"] == "connection timeout"


def test_get_stale_cache_entries(store: SQLiteStore):
    # Insert an entry with next_scheduled_update in the past
    with store._connect() as conn:
        conn.execute(
            """
            INSERT INTO cache_meta (ticker, source, last_updated, next_scheduled_update, status)
            VALUES (?, ?, datetime('now'), datetime('now', '-1 hours'), 'fresh')
            ON CONFLICT(ticker, source, metric_scope) DO UPDATE SET
                next_scheduled_update = datetime('now', '-1 hours'),
                status = 'fresh'
            """,
            ("NVDA", "yfinance"),
        )
        conn.commit()

    stale = store.get_stale_cache_entries(limit=20)
    tickers = {r["ticker"] for r in stale}
    assert "NVDA" in tickers


def test_mark_cache_stale_appears_in_stale_entries(store: SQLiteStore):
    """mark_cache_stale makes entry visible even when next_scheduled_update is in the future."""
    store.mark_cache_fresh("AAPL", "yfinance_fundamentals", ttl_hours=24)
    entry = store.get_cache_status("AAPL", "yfinance_fundamentals")
    assert entry["next_scheduled_update"] > entry["last_updated"]

    store.mark_cache_stale("AAPL", "yfinance_fundamentals")
    stale = store.get_stale_cache_entries(limit=20)
    tickers = {r["ticker"] for r in stale}
    assert "AAPL" in tickers


def test_upsert_cache_stale_creates_row(store: SQLiteStore):
    """upsert_cache_stale creates a stale row for tickers with no prior cache entry."""
    assert store.get_cache_status("NEWCO", "yfinance_fundamentals") is None
    store.upsert_cache_stale("NEWCO", "yfinance_fundamentals")
    entry = store.get_cache_status("NEWCO", "yfinance_fundamentals")
    assert entry is not None
    assert entry["status"] == "stale"
    stale = store.get_stale_cache_entries(limit=20)
    assert "NEWCO" in {r["ticker"] for r in stale}


# ── Ingestion Log ─────────────────────────────────

def test_log_ingestion_start(store: SQLiteStore):
    run_id = store.log_ingestion_start()
    assert isinstance(run_id, str)
    assert len(run_id) > 0

    with store._connect() as conn:
        row = conn.execute(
            "SELECT * FROM ingestion_log WHERE run_id=?", (run_id,)
        ).fetchone()
    assert row is not None
    assert row["status"] == "started"


def test_log_ingestion_complete(store: SQLiteStore):
    run_id = store.log_ingestion_start()
    store.log_ingestion_complete(run_id, status="completed", items=5, new=2, updated=3)

    with store._connect() as conn:
        row = conn.execute(
            "SELECT * FROM ingestion_log WHERE run_id=?", (run_id,)
        ).fetchone()
    assert row["status"] == "completed"
    assert row["items_processed"] == 5
    assert row["items_new"] == 2
    assert row["items_updated"] == 3
    assert row["duration_seconds"] is not None
    assert isinstance(row["duration_seconds"], float)


# ── Query Support ─────────────────────────────────

def test_search_facts(store: SQLiteStore):
    store.upsert_fundamental(
        "AAPL", "revenue", 100.0, period="2026-Q1", source_type="sec"
    )
    store.upsert_fundamental(
        "AAPL", "gross_margin", 0.45, period="2026-Q1", source_type="sec"
    )
    store.upsert_fundamental(
        "NVDA", "revenue", 26.0, period="2026-Q1", source_type="yfinance"
    )

    results = store.search_facts(ticker="AAPL")
    assert len(results) >= 1
    assert all(r["ticker"] == "AAPL" for r in results)

    rev_results = store.search_facts(ticker="AAPL", metric="rev")
    assert any("rev" in r["metric"] for r in rev_results)

    sec_results = store.search_facts(source="sec")
    assert all(r["source_type"] == "sec" for r in sec_results)


def test_search_facts_empty(store: SQLiteStore):
    results = store.search_facts(ticker="NOEXIST")
    assert results == []


# ── Schema ────────────────────────────────────────

def test_inline_schema(store: SQLiteStore):
    schema = store._inline_schema()
    assert isinstance(schema, str)
    assert "CREATE TABLE IF NOT EXISTS fundamentals" in schema
    assert "CREATE TABLE IF NOT EXISTS filings" in schema
    assert "CREATE TABLE IF NOT EXISTS cache_meta" in schema
    assert "CREATE TABLE IF NOT EXISTS ingestion_log" in schema

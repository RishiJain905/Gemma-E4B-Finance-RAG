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
    assert {
        "fundamentals", "sec_companyfacts", "filings", "cache_meta", "ingestion_log",
    } <= tables


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


# ── Corpus explorer inventory reads (2.2.7.2) ─────────────────────────────

def test_corpus_inventory_reads_are_filtered_and_bounded(store: SQLiteStore):
    store.upsert_fundamental(
        "NVDA", "revenue", 26.0, unit="usd", period="2026-Q1",
        source_type="sec", source_url="https://sec.example/revenue",
    )
    store.upsert_fundamental(
        "AMD", "revenue", 5.8, unit="usd", period="2026-Q1",
        source_type="yfinance",
    )
    store.register_filing(
        "NVDA", "10-Q", "2026-05-15", "2026-Q1", "ACC-1",
        "https://sec.example/ACC-1",
    )
    store.mark_filing_parsed(
        "ACC-1", file_path=r"C:\private\ACC-1.txt", section_count=2,
        chunk_count=3,
    )
    store.mark_cache_fresh("NVDA", "sec_filings")

    assert store.get_source_counts(limit=10, offset=0)
    tickers = store.get_ticker_counts(limit=10, offset=0)
    assert {row["ticker"] for row in tickers} >= {"NVDA", "AMD"}

    metrics = store.search_corpus_metrics(
        query="revenue", ticker="NVDA", unit="usd", limit=10, offset=0,
    )
    assert metrics[0]["metric"] == "revenue"
    assert metrics[0]["ticker"] == "NVDA"

    filings = store.list_filings(ticker="NVDA", limit=10, offset=0)
    assert filings[0]["accession"] == "ACC-1"
    assert "file_path" not in filings[0]
    assert store.get_filing("ACC-1")["index_chunk_count"] == 3
    assert store.list_freshness(ticker="NVDA", limit=10, offset=0)


def test_corpus_inventory_rejects_invalid_pages(store: SQLiteStore):
    with pytest.raises(ValueError):
        store.get_source_counts(limit=0)
    with pytest.raises(ValueError):
        store.list_filings(limit=SQLiteStore.MAX_INVENTORY_LIMIT + 1)
    with pytest.raises(ValueError):
        store.search_corpus_metrics(limit=1, offset=-1)


# ── Store Revision (2.2.6.2) ──────────────────────

def test_store_revision_starts_at_zero(store: SQLiteStore):
    assert store.get_store_revision() == 0


def test_bump_store_revision_is_monotonic(store: SQLiteStore):
    assert store.bump_store_revision("first") == 1
    assert store.bump_store_revision("second") == 2
    assert store.get_store_revision() == 2


def test_store_revision_persists_across_connections(tmp_path: Path):
    db = tmp_path / "rev.db"
    SQLiteStore(db).bump_store_revision("write")
    # A fresh SQLiteStore over the same file re-runs the schema (INSERT OR IGNORE
    # keeps the row) and must read the persisted revision, not reset it.
    assert SQLiteStore(db).get_store_revision() == 1


# ── Schema ────────────────────────────────────────

def test_inline_schema(store: SQLiteStore):
    schema = store._inline_schema()
    assert isinstance(schema, str)
    assert "CREATE TABLE IF NOT EXISTS fundamentals" in schema
    assert "CREATE TABLE IF NOT EXISTS sec_companyfacts" in schema
    assert "CREATE TABLE IF NOT EXISTS filings" in schema
    assert "CREATE TABLE IF NOT EXISTS cache_meta" in schema
    assert "CREATE TABLE IF NOT EXISTS ingestion_log" in schema
    assert "CREATE TABLE IF NOT EXISTS store_revision" in schema
    assert "CREATE TABLE IF NOT EXISTS securities" in schema
    assert "CREATE TABLE IF NOT EXISTS security_aliases" in schema
    assert "CREATE TABLE IF NOT EXISTS security_memberships" in schema


def test_init_creates_store_revision_table(tmp_path: Path):
    s = SQLiteStore(tmp_path / "test.db")
    with s._connect() as conn:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    assert "store_revision" in tables


# -- Security universe (2.3.1.1) ---------------------------------------------

def test_init_creates_universe_tables_and_indexes(store: SQLiteStore):
    with store._connect() as conn:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        indexes = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    assert {
        "securities", "security_aliases", "security_memberships", "universe_errors",
    } <= tables
    assert {
        "idx_securities_active_ticker", "idx_securities_cik",
        "idx_security_aliases_lookup", "idx_memberships_active_index",
        "idx_securities_sector", "idx_securities_last_seen",
    } <= indexes


def test_universe_reads_filter_and_resolve_aliases(store: SQLiteStore):
    result = store.upsert_universe_snapshot(
        "ivv",
        "2026-07-01T00:00:00Z",
        [
            {
                "symbol": "BRK.B", "company_name": "Berkshire Hathaway Class B",
                "source": "ivv", "index_code": "sp500", "exchange": "NYSE",
                "cik": "0001067983", "sector": "Financials",
                "source_url": "https://example.test/ivv",
            },
            {
                "symbol": "MSFT", "company_name": "Microsoft Corporation",
                "source": "ivv", "index_code": "sp500", "exchange": "NASDAQ",
                "cik": None, "sector": "Technology",
                "source_url": "https://example.test/ivv",
            },
        ],
    )

    assert result["changed"] is True
    assert [row["ticker"] for row in store.list_securities(
        index="sp500", active=True, sector="Financials", limit=10, offset=0,
    )] == ["BRK-B"]
    security = store.get_security("BRK.B")
    assert security["ticker"] == "BRK-B"
    assert store.get_security(security["security_id"])["security_id"] == security["security_id"]
    assert store.resolve_security("BRK.B", provider="ivv")["security_id"] == security["security_id"]
    assert store.list_memberships(security_id=security["security_id"], active=True)


def test_list_tickers_uses_canonical_active_symbols_without_aliases(store: SQLiteStore):
    store.upsert_fundamental("LEGACY", "revenue", 1.0, period="2026-Q1")
    store.upsert_universe_snapshot(
        "ivv",
        "2026-07-01T00:00:00Z",
        [{
            "symbol": "BRK.B", "company_name": "Berkshire Hathaway Class B",
            "source": "ivv", "index_code": "sp500", "exchange": "NYSE",
            "source_url": "https://example.test/ivv",
        }],
    )
    with store._connect() as conn:
        security_id = conn.execute(
            "SELECT security_id FROM securities WHERE ticker='BRK-B'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO security_aliases "
            "(security_id, alias, normalized_alias, alias_type, source, valid_from) "
            "VALUES (?, 'BF.B', 'BF-B', 'former_ticker', 'ivv', '2020-01-01')",
            (security_id,),
        )
        conn.commit()

    assert store.list_tickers() == ["BRK-B"]

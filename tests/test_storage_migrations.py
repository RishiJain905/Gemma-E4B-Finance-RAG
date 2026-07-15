"""tests/test_storage_migrations.py
Offline tests for ordered Phase 2.3 SQLite schema migrations.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml
from unittest.mock import MagicMock, patch

from src.scheduler.source_registry import SourceRegistry
from src.storage.migrations import LATEST_SCHEMA_VERSION, apply_migrations, schema_signature
from src.storage.sqlite_store import SQLiteStore
from src.storage.store import Store


PHASE_22_SCHEMA = """
CREATE TABLE fundamentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL,
    metric TEXT NOT NULL, value REAL, unit TEXT DEFAULT 'usd', period TEXT,
    period_type TEXT DEFAULT 'quarterly', source_type TEXT NOT NULL,
    source_url TEXT, source_accessed_at TEXT, ingested_at TEXT,
    UNIQUE(ticker, metric, period)
);
CREATE TABLE sec_companyfacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, cik TEXT NOT NULL,
    taxonomy TEXT NOT NULL, concept TEXT NOT NULL, label TEXT, description TEXT,
    value_text TEXT NOT NULL, value_numeric REAL NOT NULL, unit TEXT NOT NULL,
    period_start TEXT NOT NULL DEFAULT '', period_end TEXT NOT NULL,
    period_kind TEXT NOT NULL, fiscal_year INTEGER, fiscal_period TEXT,
    form TEXT NOT NULL, filed_at TEXT NOT NULL, accession TEXT NOT NULL,
    frame TEXT NOT NULL DEFAULT '', source_url TEXT NOT NULL,
    source_accessed_at TEXT NOT NULL, ingested_at TEXT NOT NULL
);
CREATE TABLE filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL,
    filing_type TEXT NOT NULL, filing_date TEXT, period TEXT,
    accession TEXT UNIQUE, source_url TEXT, file_path TEXT,
    status TEXT DEFAULT 'unprocessed', parsed_at TEXT,
    summary_embedding_id TEXT, ingested_at TEXT
);
CREATE TABLE cache_meta (
    ticker TEXT NOT NULL, source TEXT NOT NULL, metric_scope TEXT DEFAULT 'all',
    last_updated TEXT, next_scheduled_update TEXT, status TEXT DEFAULT 'fresh',
    error_message TEXT, PRIMARY KEY (ticker, source, metric_scope)
);
CREATE TABLE store_revision (
    id INTEGER PRIMARY KEY CHECK (id = 1), revision INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO store_revision (id, revision) VALUES (1, 0);
CREATE TABLE ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ticker TEXT,
    source TEXT NOT NULL, status TEXT NOT NULL, items_processed INTEGER DEFAULT 0,
    items_new INTEGER DEFAULT 0, items_updated INTEGER DEFAULT 0,
    error_message TEXT, started_at TEXT, completed_at TEXT, duration_seconds REAL
);
"""


def _migration_versions(db_path: Path) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        return [
            int(row[0])
            for row in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]


def test_phase22_database_migrates_additively_and_twice_is_a_noop(tmp_path):
    db_path = tmp_path / "phase22.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(PHASE_22_SCHEMA)
        conn.execute(
            "INSERT INTO fundamentals "
            "(ticker, metric, value, period, source_type) VALUES ('AAPL', 'revenue', 1, '2025', 'sec')"
        )
        conn.commit()
        first = apply_migrations(conn)
        before = schema_signature(conn)
        total_changes = conn.total_changes
        second = apply_migrations(conn)

        assert first == list(range(1, LATEST_SCHEMA_VERSION + 1))
        assert second == []
        assert schema_signature(conn) == before
        assert conn.total_changes == total_changes
        assert conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0] == 1
        assert "security_id" in {
            row[1] for row in conn.execute("PRAGMA table_info(fundamentals)")
        }
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 5


def test_canonical_and_inline_new_database_schemas_converge(tmp_path, monkeypatch):
    canonical_path = tmp_path / "canonical.db"
    inline_path = tmp_path / "inline.db"

    SQLiteStore(canonical_path)
    monkeypatch.setattr(SQLiteStore, "SCHEMA_SQL", tmp_path / "missing-schema.sql")
    SQLiteStore(inline_path)

    with sqlite3.connect(canonical_path) as canonical, sqlite3.connect(inline_path) as inline:
        assert schema_signature(canonical) == schema_signature(inline)
    assert _migration_versions(canonical_path) == list(range(1, 6))
    assert _migration_versions(inline_path) == list(range(1, 6))


def test_store_reopens_migrated_database_without_schema_changes(tmp_path):
    db_path = tmp_path / "reopen.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(PHASE_22_SCHEMA)

    SQLiteStore(db_path)
    with sqlite3.connect(db_path) as conn:
        expected = schema_signature(conn)
    for _ in range(3):
        SQLiteStore(db_path)
        with sqlite3.connect(db_path) as conn:
            assert schema_signature(conn) == expected


def test_store_reset_recreates_migrated_tables(tmp_path):
    chroma = MagicMock()
    with patch("src.storage.store.ChromaStore", return_value=chroma):
        store = Store(db_path=tmp_path / "reset.db", chroma_path=tmp_path / "chroma")
        store.reset()

    with store.sqlite._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 5
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='securities'"
        ).fetchone() is not None


def test_phase23_switches_default_off_and_status_never_emits_secret_values():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "universe.yaml").open(encoding="utf-8") as handle:
        universe = yaml.safe_load(handle)
    with (root / "configs" / "sources.yaml").open(encoding="utf-8") as handle:
        sources = yaml.safe_load(handle)
    with (root / "configs" / "middleware.yaml").open(encoding="utf-8") as handle:
        middleware = yaml.safe_load(handle)

    switches = {
        "universe_refresh": universe["feature_flags"]["universe_refresh"],
        **sources["feature_flags"],
        "retrieval_facets_ranking": middleware["enable_phase2_3_retrieval"],
        "corpus_explorer_projection": middleware["enable_phase2_3_corpus_projection"],
    }
    assert switches == {
        "universe_refresh": False,
        "sec_broad_events": False,
        "company_news": False,
        "grouped_market_data": False,
        "official_feeds": False,
        "sector_feeds": False,
        "retrieval_facets_ranking": False,
        "corpus_explorer_projection": False,
    }

    secret = "do-not-print-this-secret"
    status = SourceRegistry.load(
        root / "configs" / "sources.yaml",
        environ={"FINNHUB_API_KEY": secret},
    ).status()
    assert status["finnhub"]["configured"] is True
    assert status["massive"]["configured"] is False
    assert secret not in json.dumps(status)

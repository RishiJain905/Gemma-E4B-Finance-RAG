"""
src/storage/sqlite_store.py
SQLite storage layer for structured financial data.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.ingestion.normalization import (
    content_hash,
    normalize_canonical_url,
    normalize_headline,
    syndicated_news_key,
)
from src.ingestion.records import EventRecord, NarrativeRecord, ObservationRecord

logger = logging.getLogger(__name__)


class SQLiteStore:
    """Handles all SQLite operations for structured financial data."""

    MAX_INVENTORY_LIMIT = 200
    MAX_INVENTORY_OFFSET = 10_000
    MAX_MAINTENANCE_LIMIT = 10_000

    SCHEMA_SQL = Path(__file__).parent.parent.parent / "docs/phase1.2/schema.sql"
    DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data/finance.db"

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Connection ────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Return a new SQLite connection configured for WAL mode and row access."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        """Create all tables and indexes if they don't exist."""
        if self.SCHEMA_SQL.exists():
            with open(self.SCHEMA_SQL, encoding="utf-8") as f:
                sql = f.read()
        else:
            sql = self._inline_schema()

        with self._connect() as conn:
            conn.executescript(sql)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS sec_daily_indexes (
                    index_date TEXT PRIMARY KEY,
                    source_url TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('processed')),
                    registered_count INTEGER NOT NULL DEFAULT 0,
                    processed_at TEXT NOT NULL DEFAULT (datetime('now'))
                )"""
            )
            existing = {
                row[1] for row in conn.execute("PRAGMA table_info(filings)").fetchall()
            }
            migrations = {
                "index_error": "ALTER TABLE filings ADD COLUMN index_error TEXT",
                "index_section_count": (
                    "ALTER TABLE filings ADD COLUMN index_section_count INTEGER DEFAULT 0"
                ),
                "index_chunk_count": (
                    "ALTER TABLE filings ADD COLUMN index_chunk_count INTEGER DEFAULT 0"
                ),
                "cik": "ALTER TABLE filings ADD COLUMN cik TEXT",
                "primary_document": "ALTER TABLE filings ADD COLUMN primary_document TEXT",
                "discovery_scope": (
                    "ALTER TABLE filings ADD COLUMN discovery_scope TEXT DEFAULT 'deep'"
                ),
                "items_json": "ALTER TABLE filings ADD COLUMN items_json TEXT DEFAULT '[]'",
                "exhibits_json": "ALTER TABLE filings ADD COLUMN exhibits_json TEXT DEFAULT '[]'",
            }
            for column, statement in migrations.items():
                if column not in existing:
                    conn.execute(statement)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS source_cursors (
                    source TEXT NOT NULL,
                    partition_key TEXT NOT NULL,
                    cursor_value TEXT,
                    cursor_type TEXT NOT NULL DEFAULT 'none',
                    overlap_value TEXT,
                    last_successful_run_id TEXT,
                    last_successful_at TEXT,
                    version TEXT NOT NULL DEFAULT '1',
                    status TEXT NOT NULL DEFAULT 'unknown',
                    error_class TEXT,
                    error_message TEXT,
                    retry_after REAL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source, partition_key)
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source_cursors_status "
                "ON source_cursors(source, status, updated_at)"
            )
            cursor_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(source_cursors)").fetchall()
            }
            if "last_successful_at" not in cursor_columns:
                conn.execute(
                    "ALTER TABLE source_cursors ADD COLUMN last_successful_at TEXT"
                )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS source_budget_usage (
                    source TEXT NOT NULL,
                    window_kind TEXT NOT NULL CHECK (window_kind IN ('minute', 'day')),
                    window_start TEXT NOT NULL,
                    attempted_requests INTEGER NOT NULL DEFAULT 0,
                    successful_requests INTEGER NOT NULL DEFAULT 0,
                    provider_remaining INTEGER,
                    provider_reset TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source, window_kind, window_start)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS scheduler_runs (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    policy_revision TEXT NOT NULL,
                    config_revision TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_seconds REAL,
                    status TEXT NOT NULL DEFAULT 'running',
                    requested_sources_json TEXT NOT NULL DEFAULT '[]',
                    completed_sources_json TEXT NOT NULL DEFAULT '[]',
                    skipped_sources_json TEXT NOT NULL DEFAULT '[]',
                    failed_sources_json TEXT NOT NULL DEFAULT '[]',
                    terminal_error_class TEXT,
                    terminal_error_message TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS scheduler_run_sources (
                    run_id TEXT NOT NULL REFERENCES scheduler_runs(run_id) ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    policy_revision TEXT NOT NULL,
                    config_revision TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    duration_seconds REAL,
                    status TEXT NOT NULL,
                    requested INTEGER NOT NULL DEFAULT 1,
                    completed INTEGER NOT NULL DEFAULT 0,
                    skipped INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    partitions INTEGER NOT NULL DEFAULT 0,
                    items INTEGER NOT NULL DEFAULT 0,
                    requests INTEGER NOT NULL DEFAULT 0,
                    new_items INTEGER NOT NULL DEFAULT 0,
                    updated_items INTEGER NOT NULL DEFAULT 0,
                    duplicates INTEGER NOT NULL DEFAULT 0,
                    cursor_before_json TEXT,
                    cursor_after_json TEXT,
                    quota_remaining INTEGER,
                    freshness TEXT,
                    last_success TEXT,
                    next_due TEXT,
                    cooldown_reset TEXT,
                    error_class TEXT,
                    error_message TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (run_id, source)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS bootstrap_partitions (
                    run_id TEXT NOT NULL REFERENCES scheduler_runs(run_id) ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    partition_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    items INTEGER NOT NULL DEFAULT 0,
                    new_items INTEGER NOT NULL DEFAULT 0,
                    updated_items INTEGER NOT NULL DEFAULT 0,
                    duplicates INTEGER NOT NULL DEFAULT 0,
                    error_class TEXT,
                    error_message TEXT,
                    PRIMARY KEY (run_id, source, partition_key)
                )"""
            )
            bootstrap_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(bootstrap_partitions)"
                ).fetchall()
            }
            for column in ("new_items", "updated_items", "duplicates"):
                if column not in bootstrap_columns:
                    conn.execute(
                        f"ALTER TABLE bootstrap_partitions ADD COLUMN {column} "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scheduler_runs_started "
                "ON scheduler_runs(started_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scheduler_run_sources_source "
                "ON scheduler_run_sources(source, ended_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bootstrap_partitions_status "
                "ON bootstrap_partitions(source, status, run_id)"
            )
            corpus_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(corpus_items)").fetchall()
            }
            corpus_migrations = {
                "document_family_id": "ALTER TABLE corpus_items ADD COLUMN document_family_id TEXT",
                "narrative_bytes": (
                    "ALTER TABLE corpus_items ADD COLUMN narrative_bytes INTEGER NOT NULL DEFAULT 0"
                ),
                "metadata_bytes": (
                    "ALTER TABLE corpus_items ADD COLUMN metadata_bytes INTEGER NOT NULL DEFAULT 0"
                ),
                "is_tombstone": (
                    "ALTER TABLE corpus_items ADD COLUMN is_tombstone INTEGER NOT NULL DEFAULT 0"
                ),
                "retired_at": "ALTER TABLE corpus_items ADD COLUMN retired_at TEXT",
                "retention_reason": "ALTER TABLE corpus_items ADD COLUMN retention_reason TEXT",
            }
            for column, statement in corpus_migrations.items():
                if column not in corpus_columns:
                    conn.execute(statement)
            conn.execute(
                "UPDATE corpus_items SET document_family_id=corpus_item_id "
                "WHERE document_family_id IS NULL OR document_family_id=''"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_items_document_family "
                "ON corpus_items(document_family_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_corpus_items_retention "
                "ON corpus_items(item_type, is_tombstone, indexing_status, published_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_corpus_observations_market_key "
                "ON corpus_observations(source_name, metric_id, period_end, tickers_json)"
            )
            # Official revisions share an agency record id across vintages.  The
            # migration is intentionally additive: existing rows remain intact,
            # while metric + provider id + vintage becomes the observation key.
            conn.execute("DROP INDEX IF EXISTS idx_corpus_observations_provider")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_observations_provider "
                "ON corpus_observations(source_name, metric_id, provider_record_id, vintage_at) "
                "WHERE provider_record_id IS NOT NULL AND provider_record_id <> ''"
            )
            conn.commit()

    @staticmethod
    def _inline_schema() -> str:
        """Returns the full DDL as a string — used when schema.sql doesn't exist."""
        return """
-- ── Financial Fundamentals ──────────────────────────────
CREATE TABLE IF NOT EXISTS fundamentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    unit TEXT DEFAULT 'usd',
    period TEXT,
    period_type TEXT DEFAULT 'quarterly',
    source_type TEXT NOT NULL,
    source_url TEXT,
    source_accessed_at TEXT,
    ingested_at TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, metric, period)
);

-- â”€â”€ SEC CompanyFacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CREATE TABLE IF NOT EXISTS sec_companyfacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    cik TEXT NOT NULL,
    taxonomy TEXT NOT NULL,
    concept TEXT NOT NULL,
    label TEXT,
    description TEXT,
    value_text TEXT NOT NULL,
    value_numeric REAL NOT NULL,
    unit TEXT NOT NULL,
    period_start TEXT NOT NULL DEFAULT '',
    period_end TEXT NOT NULL,
    period_kind TEXT NOT NULL,
    fiscal_year INTEGER,
    fiscal_period TEXT,
    form TEXT NOT NULL,
    filed_at TEXT NOT NULL,
    accession TEXT NOT NULL,
    frame TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL,
    source_accessed_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (
        ticker, taxonomy, concept, unit, period_start,
        period_end, accession, frame
    )
);

-- ── Filing Index ────────────────────────────────────
CREATE TABLE IF NOT EXISTS filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    filing_type TEXT NOT NULL,
    filing_date TEXT,
    period TEXT,
    accession TEXT UNIQUE,
    source_url TEXT,
    file_path TEXT,
    status TEXT DEFAULT 'unprocessed',
    parsed_at TEXT,
    summary_embedding_id TEXT,
    index_error TEXT,
    index_section_count INTEGER DEFAULT 0,
    index_chunk_count INTEGER DEFAULT 0,
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ── Cache Freshness ────────────────────────────────
CREATE TABLE IF NOT EXISTS cache_meta (
    ticker TEXT NOT NULL,
    source TEXT NOT NULL,
    metric_scope TEXT DEFAULT 'all',
    last_updated TEXT,
    next_scheduled_update TEXT,
    status TEXT DEFAULT 'fresh',
    error_message TEXT,
    PRIMARY KEY (ticker, source, metric_scope)
);

-- ── Store Revision (2.2.6.2) ───────────────────────
CREATE TABLE IF NOT EXISTS store_revision (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    revision INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO store_revision (id, revision) VALUES (1, 0);

-- -- Incremental source cursors (2.3.2/2.3.4 forward-compatible seam) -----
CREATE TABLE IF NOT EXISTS source_cursors (
    source TEXT NOT NULL,
    partition_key TEXT NOT NULL,
    cursor_value TEXT,
    cursor_type TEXT NOT NULL DEFAULT 'none',
    overlap_value TEXT,
    last_successful_run_id TEXT,
    last_successful_at TEXT,
    version TEXT NOT NULL DEFAULT '1',
    status TEXT NOT NULL DEFAULT 'unknown',
    error_class TEXT,
    error_message TEXT,
    retry_after REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source, partition_key)
);
CREATE INDEX IF NOT EXISTS idx_source_cursors_status
    ON source_cursors(source, status, updated_at);

-- -- Provider budget usage (separate from freshness and cursors) -----------
CREATE TABLE IF NOT EXISTS source_budget_usage (
    source TEXT NOT NULL,
    window_kind TEXT NOT NULL CHECK (window_kind IN ('minute', 'day')),
    window_start TEXT NOT NULL,
    attempted_requests INTEGER NOT NULL DEFAULT 0,
    successful_requests INTEGER NOT NULL DEFAULT 0,
    provider_remaining INTEGER,
    provider_reset TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source, window_kind, window_start)
);

CREATE TABLE IF NOT EXISTS scheduler_runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    policy_revision TEXT NOT NULL,
    config_revision TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_seconds REAL,
    status TEXT NOT NULL DEFAULT 'running',
    requested_sources_json TEXT NOT NULL DEFAULT '[]',
    completed_sources_json TEXT NOT NULL DEFAULT '[]',
    skipped_sources_json TEXT NOT NULL DEFAULT '[]',
    failed_sources_json TEXT NOT NULL DEFAULT '[]',
    terminal_error_class TEXT,
    terminal_error_message TEXT
);

CREATE TABLE IF NOT EXISTS scheduler_run_sources (
    run_id TEXT NOT NULL REFERENCES scheduler_runs(run_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    mode TEXT NOT NULL,
    policy_revision TEXT NOT NULL,
    config_revision TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    duration_seconds REAL,
    status TEXT NOT NULL,
    requested INTEGER NOT NULL DEFAULT 1,
    completed INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    partitions INTEGER NOT NULL DEFAULT 0,
    items INTEGER NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 0,
    new_items INTEGER NOT NULL DEFAULT 0,
    updated_items INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    cursor_before_json TEXT,
    cursor_after_json TEXT,
    quota_remaining INTEGER,
    freshness TEXT,
    last_success TEXT,
    next_due TEXT,
    cooldown_reset TEXT,
    error_class TEXT,
    error_message TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, source)
);

CREATE TABLE IF NOT EXISTS bootstrap_partitions (
    run_id TEXT NOT NULL REFERENCES scheduler_runs(run_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    partition_key TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    items INTEGER NOT NULL DEFAULT 0,
    new_items INTEGER NOT NULL DEFAULT 0,
    updated_items INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    error_class TEXT,
    error_message TEXT,
    PRIMARY KEY (run_id, source, partition_key)
);
CREATE INDEX IF NOT EXISTS idx_scheduler_runs_started
    ON scheduler_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_scheduler_run_sources_source
    ON scheduler_run_sources(source, ended_at DESC);
CREATE INDEX IF NOT EXISTS idx_bootstrap_partitions_status
    ON bootstrap_partitions(source, status, run_id);

-- ── Ingestion Log ──────────────────────────────────
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ticker TEXT,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    items_processed INTEGER DEFAULT 0,
    items_new INTEGER DEFAULT 0,
    items_updated INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_seconds REAL
);

-- ── Indexes for Query Performance ─────────────────
CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker ON fundamentals(ticker);
CREATE INDEX IF NOT EXISTS idx_fundamentals_metric ON fundamentals(metric);
CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_metric ON fundamentals(ticker, metric);
CREATE INDEX IF NOT EXISTS idx_fundamentals_period ON fundamentals(period);
CREATE INDEX IF NOT EXISTS idx_sec_companyfacts_ticker_concept_period
    ON sec_companyfacts(ticker, concept, period_end);
CREATE INDEX IF NOT EXISTS idx_sec_companyfacts_ticker_filed_at
    ON sec_companyfacts(ticker, filed_at);
CREATE INDEX IF NOT EXISTS idx_sec_companyfacts_accession
    ON sec_companyfacts(accession);
CREATE INDEX IF NOT EXISTS idx_sec_companyfacts_ticker_kind_period
    ON sec_companyfacts(ticker, period_kind, period_end);
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_status ON filings(status);
CREATE INDEX IF NOT EXISTS idx_cache_meta_status ON cache_meta(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_run ON ingestion_log(run_id);
-- -- Security Universe (2.3.1.1) ---------------------------------------------
CREATE TABLE IF NOT EXISTS securities (
    security_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    normalized_ticker TEXT NOT NULL,
    company_name TEXT NOT NULL,
    exchange TEXT NOT NULL DEFAULT '',
    cik TEXT,
    security_type TEXT NOT NULL DEFAULT 'common_stock',
    share_class TEXT,
    sector TEXT,
    industry TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(normalized_ticker, exchange)
);

CREATE TABLE IF NOT EXISTS security_aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id TEXT NOT NULL REFERENCES securities(security_id),
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    alias_type TEXT NOT NULL CHECK (
        alias_type IN (
            'ticker', 'vendor_symbol', 'former_ticker', 'issuer_alias',
            'manufacturer', 'recipient_uei'
        )
    ),
    provider TEXT,
    valid_from TEXT,
    valid_to TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS security_memberships (
    membership_id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id TEXT NOT NULL REFERENCES securities(security_id),
    index_code TEXT NOT NULL CHECK (index_code IN ('sp500', 'nasdaq100')),
    effective_from TEXT NOT NULL,
    effective_to TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    source TEXT NOT NULL,
    source_url TEXT,
    observed_at TEXT NOT NULL,
    UNIQUE(security_id, index_code, effective_from)
);

CREATE TABLE IF NOT EXISTS universe_errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    symbol TEXT,
    error_code TEXT NOT NULL,
    message TEXT NOT NULL,
    payload TEXT,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_securities_active_ticker
    ON securities(active, normalized_ticker);
CREATE INDEX IF NOT EXISTS idx_securities_cik ON securities(cik);
CREATE INDEX IF NOT EXISTS idx_securities_sector ON securities(sector);
CREATE INDEX IF NOT EXISTS idx_securities_last_seen ON securities(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_security_aliases_lookup
    ON security_aliases(normalized_alias, provider, valid_from, valid_to);
CREATE UNIQUE INDEX IF NOT EXISTS idx_security_aliases_active_global
    ON security_aliases(normalized_alias)
    WHERE valid_to IS NULL AND provider IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_security_aliases_active_provider
    ON security_aliases(normalized_alias, provider)
    WHERE valid_to IS NULL AND provider IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memberships_active_index
    ON security_memberships(index_code, active, security_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memberships_one_active
    ON security_memberships(security_id, index_code)
    WHERE active = 1;
CREATE INDEX IF NOT EXISTS idx_universe_errors_run ON universe_errors(run_id);

-- -- Corpus metadata ledger (2.3.3.1) ----------------------------------------
CREATE TABLE IF NOT EXISTS corpus_items (
    corpus_item_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_category TEXT NOT NULL,
    provider_record_id TEXT,
    original_publisher TEXT,
    item_type TEXT NOT NULL,
    event_type TEXT,
    title TEXT NOT NULL,
    normalized_headline TEXT NOT NULL,
    syndicated_key TEXT,
    summary TEXT CHECK (summary IS NULL OR length(summary) <= 4000),
    language TEXT NOT NULL,
    published_at TEXT,
    effective_at TEXT,
    as_of_at TEXT,
    observed_at TEXT,
    accessed_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    source_url TEXT NOT NULL,
    canonical_url TEXT,
    tickers_json TEXT NOT NULL DEFAULT '[]',
    index_codes_json TEXT NOT NULL DEFAULT '[]',
    sectors_json TEXT NOT NULL DEFAULT '[]',
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    document_family TEXT NOT NULL,
    indexing_status TEXT NOT NULL CHECK (
        indexing_status IN ('pending', 'indexed', 'error', 'not_applicable')
    ),
    index_error TEXT,
    license_label TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    evidence_authority TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS corpus_item_sources (
    corpus_item_id TEXT NOT NULL REFERENCES corpus_items(corpus_item_id) ON DELETE CASCADE,
    source_key TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_category TEXT NOT NULL,
    provider_record_id TEXT,
    original_publisher TEXT,
    source_url TEXT NOT NULL,
    canonical_url TEXT,
    published_at TEXT,
    observed_at TEXT,
    accessed_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    license_label TEXT NOT NULL,
    evidence_authority TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (corpus_item_id, source_key)
);

CREATE TABLE IF NOT EXISTS corpus_item_securities (
    corpus_item_id TEXT NOT NULL REFERENCES corpus_items(corpus_item_id) ON DELETE CASCADE,
    security_id TEXT NOT NULL REFERENCES securities(security_id),
    ticker TEXT,
    PRIMARY KEY (corpus_item_id, security_id)
);

CREATE TABLE IF NOT EXISTS corpus_observations (
    observation_id TEXT PRIMARY KEY,
    metric_id TEXT NOT NULL,
    series_id TEXT,
    value_text TEXT NOT NULL,
    value_numeric REAL,
    unit TEXT NOT NULL,
    frequency TEXT NOT NULL,
    period_start TEXT,
    period_end TEXT NOT NULL,
    vintage_at TEXT,
    as_of_at TEXT,
    scope TEXT NOT NULL CHECK (scope IN ('security', 'sector', 'global')),
    tickers_json TEXT NOT NULL DEFAULT '[]',
    sector TEXT,
    source_name TEXT NOT NULL,
    source_category TEXT NOT NULL,
    provider_record_id TEXT,
    original_publisher TEXT,
    source_url TEXT NOT NULL,
    canonical_url TEXT,
    published_at TEXT,
    observed_at TEXT,
    accessed_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    license_label TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    evidence_authority TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS observation_securities (
    observation_id TEXT NOT NULL REFERENCES corpus_observations(observation_id) ON DELETE CASCADE,
    security_id TEXT NOT NULL REFERENCES securities(security_id),
    PRIMARY KEY (observation_id, security_id)
);

CREATE TABLE IF NOT EXISTS corpus_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    effective_at TEXT,
    announced_at TEXT,
    status TEXT NOT NULL,
    amount REAL,
    currency TEXT,
    rate REAL,
    ratio REAL,
    action_date TEXT,
    classifier_version TEXT,
    explanation TEXT CHECK (explanation IS NULL OR length(explanation) <= 4000),
    source_name TEXT NOT NULL,
    source_category TEXT NOT NULL,
    provider_record_id TEXT,
    original_publisher TEXT,
    source_url TEXT NOT NULL,
    canonical_url TEXT,
    published_at TEXT,
    observed_at TEXT,
    accessed_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    license_label TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    evidence_authority TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS event_securities (
    event_id TEXT NOT NULL REFERENCES corpus_events(event_id) ON DELETE CASCADE,
    security_id TEXT NOT NULL REFERENCES securities(security_id),
    PRIMARY KEY (event_id, security_id)
);

CREATE TABLE IF NOT EXISTS event_corpus_items (
    event_id TEXT NOT NULL REFERENCES corpus_events(event_id) ON DELETE CASCADE,
    corpus_item_id TEXT NOT NULL REFERENCES corpus_items(corpus_item_id),
    PRIMARY KEY (event_id, corpus_item_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_items_provider_identity
    ON corpus_items(source, provider_record_id)
    WHERE provider_record_id IS NOT NULL AND provider_record_id <> '';
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_items_url_identity
    ON corpus_items(source, canonical_url, published_at)
    WHERE provider_record_id IS NULL AND canonical_url IS NOT NULL
        AND published_at IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_items_hash_identity
    ON corpus_items(source, content_hash)
    WHERE provider_record_id IS NULL AND canonical_url IS NULL;
CREATE INDEX IF NOT EXISTS idx_corpus_items_canonical_url
    ON corpus_items(canonical_url, published_at);
CREATE INDEX IF NOT EXISTS idx_corpus_items_content_hash
    ON corpus_items(content_hash);
CREATE INDEX IF NOT EXISTS idx_corpus_items_headline_window
    ON corpus_items(syndicated_key)
    WHERE syndicated_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_corpus_items_indexing_status
    ON corpus_items(indexing_status, updated_at);
CREATE INDEX IF NOT EXISTS idx_corpus_item_sources_source
    ON corpus_item_sources(source_name, provider_record_id);
CREATE INDEX IF NOT EXISTS idx_corpus_item_securities_security
    ON corpus_item_securities(security_id, corpus_item_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_observations_provider
    ON corpus_observations(source_name, metric_id, provider_record_id, vintage_at)
    WHERE provider_record_id IS NOT NULL AND provider_record_id <> '';
CREATE INDEX IF NOT EXISTS idx_corpus_observations_metric_period
    ON corpus_observations(metric_id, period_end, vintage_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_events_provider
    ON corpus_events(source_name, provider_record_id)
    WHERE provider_record_id IS NOT NULL AND provider_record_id <> '';
CREATE INDEX IF NOT EXISTS idx_corpus_events_type_effective
    ON corpus_events(event_type, effective_at);


"""

    # ── Fundamentals CRUD ─────────────────────────────

    def upsert_fundamental(self, ticker: str, metric: str, value: float,
                           unit: str = "usd", period: str = None,
                           period_type: str = "quarterly",
                           source_type: str = "yfinance",
                           source_url: str = None) -> int:
        """Insert or update a single financial metric."""
        sql = """
        INSERT INTO fundamentals (ticker, metric, value, unit, period, period_type, source_type, source_url, source_accessed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(ticker, metric, period) DO UPDATE SET
            value = excluded.value,
            source_url = excluded.source_url,
            source_accessed_at = excluded.source_accessed_at
        """
        with self._connect() as conn:
            conn.execute(sql, (ticker, metric, value, unit, period, period_type, source_type, source_url))
            conn.commit()
            return conn.total_changes

    def get_fundamental(self, ticker: str, metric: str,
                        period: str = None) -> Optional[dict]:
        """Get the latest value for a ticker+metric, optionally by period."""
        if period:
            sql = (
                "SELECT * FROM fundamentals WHERE ticker=? AND metric=? AND period=? "
                "ORDER BY ingested_at DESC LIMIT 1"
            )
            params = (ticker, metric, period)
        else:
            sql = (
                "SELECT * FROM fundamentals WHERE ticker=? AND metric=? "
                "ORDER BY period DESC, ingested_at DESC LIMIT 1"
            )
            params = (ticker, metric)

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def get_fundamentals_batch(self, ticker: str,
                               metrics: list[str] = None) -> dict:
        """Get multiple metrics for a ticker at once. Returns {metric: value}."""
        if metrics:
            placeholders = ",".join("?" for _ in metrics)
            sql = f"""
                SELECT f_outer.metric, f_outer.value FROM fundamentals AS f_outer
                WHERE f_outer.ticker=? AND f_outer.metric IN ({placeholders})
                  AND f_outer.period = (
                      SELECT MAX(f_inner.period) FROM fundamentals AS f_inner
                      WHERE f_inner.ticker = f_outer.ticker AND f_inner.metric = f_outer.metric
                  )
                GROUP BY f_outer.metric
            """
            params = [ticker] + metrics
        else:
            sql = """
                SELECT f_outer.metric, f_outer.value FROM fundamentals AS f_outer
                WHERE f_outer.ticker=?
                  AND f_outer.period = (
                      SELECT MAX(f_inner.period) FROM fundamentals AS f_inner
                      WHERE f_inner.ticker = f_outer.ticker AND f_inner.metric = f_outer.metric
                  )
                GROUP BY f_outer.metric
            """
            params = [ticker]

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return {row["metric"]: row["value"] for row in rows}

    # â”€â”€ SEC CompanyFacts CRUD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def upsert_sec_companyfacts(self, rows: list[dict]) -> dict:
        """Insert or refresh CompanyFacts observations in one transaction."""
        if not rows:
            return {"rows_received": 0, "rows_written": 0}
        sql = """
        INSERT INTO sec_companyfacts (
            ticker, cik, taxonomy, concept, label, description,
            value_text, value_numeric, unit, period_start, period_end,
            period_kind, fiscal_year, fiscal_period, form, filed_at,
            accession, frame, source_url, source_accessed_at
        ) VALUES (
            :ticker, :cik, :taxonomy, :concept, :label, :description,
            :value_text, :value_numeric, :unit, :period_start, :period_end,
            :period_kind, :fiscal_year, :fiscal_period, :form, :filed_at,
            :accession, :frame, :source_url, :source_accessed_at
        )
        ON CONFLICT(
            ticker, taxonomy, concept, unit, period_start,
            period_end, accession, frame
        ) DO UPDATE SET
            cik = excluded.cik,
            label = excluded.label,
            description = excluded.description,
            value_text = excluded.value_text,
            value_numeric = excluded.value_numeric,
            period_kind = excluded.period_kind,
            fiscal_year = excluded.fiscal_year,
            fiscal_period = excluded.fiscal_period,
            form = excluded.form,
            filed_at = excluded.filed_at,
            source_url = excluded.source_url,
            source_accessed_at = excluded.source_accessed_at,
            ingested_at = datetime('now')
        """
        normalized_rows = [
            {**row, "period_start": row.get("period_start") or "", "frame": row.get("frame") or ""}
            for row in rows
        ]
        with self._connect() as conn:
            conn.executemany(sql, normalized_rows)
            conn.commit()
        return {"rows_received": len(rows), "rows_written": len(rows)}

    @staticmethod
    def _validate_as_of(as_of: Optional[str]) -> str:
        """Validate an ISO date before constructing or executing SQL."""
        value = as_of or datetime.now(timezone.utc).date().isoformat()
        if not isinstance(value, str):
            raise ValueError("as_of must be an ISO date in YYYY-MM-DD format")
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("as_of must be an ISO date in YYYY-MM-DD format") from exc
        if parsed.isoformat() != value:
            raise ValueError("as_of must be an ISO date in YYYY-MM-DD format")
        return value

    def query_sec_companyfacts(
        self,
        ticker: str,
        concepts: list[str],
        *,
        units: Optional[list[str]] = None,
        period_kinds: Optional[list[str]] = None,
        period_end: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> list[dict]:
        """Return provenance-preserving CompanyFacts rows eligible as of a date."""
        cutoff = self._validate_as_of(as_of)
        if not concepts or units == [] or period_kinds == []:
            return []

        conditions = ["ticker = ?", "filed_at <= ?"]
        params: list = [ticker.upper(), cutoff]
        concept_placeholders = ",".join("?" for _ in concepts)
        conditions.append(f"concept IN ({concept_placeholders})")
        params.extend(concepts)
        if units is not None:
            placeholders = ",".join("?" for _ in units)
            conditions.append(f"unit IN ({placeholders})")
            params.extend(units)
        if period_kinds is not None:
            placeholders = ",".join("?" for _ in period_kinds)
            conditions.append(f"period_kind IN ({placeholders})")
            params.extend(period_kinds)
        if period_end is not None:
            conditions.append("period_end = ?")
            params.append(period_end)

        sql = f"""
            SELECT * FROM sec_companyfacts
            WHERE {' AND '.join(conditions)}
            ORDER BY period_end DESC, concept ASC, filed_at DESC,
                     accession DESC, source_accessed_at DESC, id ASC
        """
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def count_sec_companyfacts(self, ticker: Optional[str] = None) -> int:
        """Count stored CompanyFacts observations, optionally for one ticker."""
        sql = "SELECT COUNT(*) FROM sec_companyfacts"
        params: tuple = ()
        if ticker is not None:
            sql += " WHERE ticker = ?"
            params = (ticker.upper(),)
        with self._connect() as conn:
            return int(conn.execute(sql, params).fetchone()[0])

    # ── Filing Tracking ───────────────────────────────

    def register_filing(self, ticker: str, filing_type: str,
                        filing_date: str, period: str,
                        accession: str, source_url: str, *,
                        cik: Optional[str] = None,
                        primary_document: Optional[str] = None,
                        discovery_scope: str = "deep",
                        items: Optional[list[str]] = None,
                        exhibits: Optional[list[dict]] = None) -> bool:
        """Register a filing as processed. Returns True if new, False if duplicate."""
        sql = """
        INSERT OR IGNORE INTO filings
            (ticker, filing_type, filing_date, period, accession, source_url, status,
             cik, primary_document, discovery_scope, items_json, exhibits_json)
        VALUES (?, ?, ?, ?, ?, ?, 'unprocessed', ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, (
                ticker, filing_type, filing_date, period, accession, source_url,
                cik, primary_document, discovery_scope,
                json.dumps(items or []), json.dumps(exhibits or []),
            ))
            conn.commit()
            return cursor.rowcount > 0

    def register_sec_daily_index(
        self, index_date: str, source_url: str, filings: list[dict],
    ) -> dict[str, object]:
        """Atomically register an SEC daily-index batch and its cursor checkpoint."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT status FROM sec_daily_indexes WHERE index_date=?", (index_date,),
            ).fetchone()
            if existing:
                return {"registered": 0, "replayed": True}
            try:
                conn.execute("BEGIN")
                registered = self._insert_filing_rows(conn, filings)
                conn.execute(
                    "INSERT INTO sec_daily_indexes "
                    "(index_date, source_url, status, registered_count) VALUES (?, ?, 'processed', ?)",
                    (index_date, source_url, registered),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"registered": registered, "replayed": False}

    @staticmethod
    def _insert_filing_rows(conn: sqlite3.Connection, filings: list[dict]) -> int:
        """Insert accession-unique filing rows within the caller's transaction."""
        registered = 0
        for filing in filings:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO filings (
                    ticker, filing_type, filing_date, period, accession,
                    source_url, status, cik, primary_document, discovery_scope,
                    items_json, exhibits_json
                ) VALUES (?, ?, ?, ?, ?, ?, 'unprocessed', ?, ?, ?, ?, ?)""",
                (
                    filing["ticker"], filing["filing_type"], filing.get("filing_date", ""),
                    filing.get("period", ""), filing["accession"], filing["source_url"],
                    filing.get("cik"), filing.get("primary_document"),
                    filing.get("discovery_scope", "broad"),
                    json.dumps(filing.get("items") or []),
                    json.dumps(filing.get("exhibits") or []),
                ),
            )
            registered += int(cursor.rowcount > 0)
        return registered

    def register_sec_filings(self, filings: list[dict]) -> int:
        """Atomically register an accession-unique SEC filing batch."""
        with self._connect() as conn:
            try:
                conn.execute("BEGIN")
                registered = self._insert_filing_rows(conn, filings)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return registered

    def get_sec_daily_index_status(self, index_date: str) -> Optional[str]:
        """Return the durable status for one SEC daily index date."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM sec_daily_indexes WHERE index_date=?", (index_date,),
            ).fetchone()
        return str(row[0]) if row else None

    def get_sec_daily_index_cursor(self) -> Optional[str]:
        """Return the latest completely registered SEC daily-index date."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(index_date) FROM sec_daily_indexes WHERE status='processed'"
            ).fetchone()
        return str(row[0]) if row and row[0] else None

    def count_filings(self) -> int:
        """Return the number of accession-unique registered SEC filings."""
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0])

    def mark_filing_parsed(
        self,
        accession: str,
        embedding_id: str = None,
        file_path: str = None,
        section_count: int = 0,
        chunk_count: int = 0,
    ):
        """Mark a filing as successfully parsed by TraceAlchemy."""
        sql = """
        UPDATE filings SET status='parsed', parsed_at=datetime('now'),
            summary_embedding_id=?, file_path=COALESCE(?, file_path), index_error=NULL,
            index_section_count=?, index_chunk_count=?
        WHERE accession=?
        """
        with self._connect() as conn:
            conn.execute(
                sql, (embedding_id, file_path, section_count, chunk_count, accession),
            )
            conn.commit()

    def mark_filing_index_pending(
        self, accession: str, *, file_path: str, error: str,
    ) -> None:
        """Record a durable parsed artifact whose vector indexing must retry."""
        sql = """
        UPDATE filings SET status='index_pending', file_path=?, index_error=?
        WHERE accession=?
        """
        with self._connect() as conn:
            conn.execute(sql, (file_path, error, accession))
            conn.commit()

    def get_unprocessed_filings(self, limit: int = 10) -> list[dict]:
        """Get filings that haven't been parsed yet."""
        sql = (
            "SELECT * FROM filings WHERE status IN ('unprocessed', 'index_pending') "
            "ORDER BY filing_date DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ── Cache Management ──────────────────────────────

    def get_cache_status(self, ticker: str, source: str) -> Optional[dict]:
        """Get cache metadata for a ticker+source."""
        sql = "SELECT * FROM cache_meta WHERE ticker=? AND source=?"
        with self._connect() as conn:
            row = conn.execute(sql, (ticker, source)).fetchone()
            return dict(row) if row else None

    def mark_cache_fresh(self, ticker: str, source: str,
                         ttl_hours: int = 24):
        """Update cache metadata after a successful ingestion."""
        sql = f"""
        INSERT INTO cache_meta (ticker, source, last_updated, next_scheduled_update, status)
        VALUES (?, ?, datetime('now'), datetime('now', '+{ttl_hours} hours'), 'fresh')
        ON CONFLICT(ticker, source, metric_scope) DO UPDATE SET
            last_updated = datetime('now'),
            next_scheduled_update = datetime('now', '+{ttl_hours} hours'),
            status = 'fresh',
            error_message = NULL
        """
        with self._connect() as conn:
            conn.execute(sql, (ticker, source))
            conn.commit()

    def mark_cache_stale(self, ticker: str, source: str, error: str = None):
        """Mark cache metadata as stale, optionally with an error message."""
        sql = """
        UPDATE cache_meta SET
            status = 'stale',
            error_message = ?,
            next_scheduled_update = datetime('now', '-1 hour')
        WHERE ticker=? AND source=?
        """
        with self._connect() as conn:
            conn.execute(sql, (error, ticker, source))
            conn.commit()

    def upsert_cache_stale(self, ticker: str, source: str, error: str = None):
        """Insert or update a stale cache row (for tickers without prior cache entries)."""
        sql = """
        INSERT INTO cache_meta (ticker, source, last_updated, next_scheduled_update, status, error_message)
        VALUES (?, ?, datetime('now'), datetime('now', '-1 hour'), 'stale', ?)
        ON CONFLICT(ticker, source, metric_scope) DO UPDATE SET
            status = 'stale',
            next_scheduled_update = datetime('now', '-1 hour'),
            error_message = excluded.error_message
        """
        with self._connect() as conn:
            conn.execute(sql, (ticker, source, error))
            conn.commit()

    def get_stale_cache_entries(self, limit: int = 20) -> list[dict]:
        """Find entries past their scheduled update time or explicitly marked stale."""
        sql = (
            "SELECT * FROM cache_meta "
            "WHERE (next_scheduled_update < datetime('now') OR status = 'stale') "
            "AND status != 'fetching' LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ── Ingestion Logging ─────────────────────────────

    def log_ingestion_start(self) -> str:
        """Start a new ingestion run. Returns run_id."""
        run_id = str(uuid.uuid4())
        sql = (
            "INSERT INTO ingestion_log (run_id, source, status, started_at) "
            "VALUES (?, 'system', 'started', datetime('now'))"
        )
        with self._connect() as conn:
            conn.execute(sql, (run_id,))
            conn.commit()
        return run_id

    def log_ingestion_complete(self, run_id: str, status: str = "completed",
                               items: int = 0, new: int = 0, updated: int = 0):
        """Mark an ingestion run as complete with stats."""
        sql = """
        UPDATE ingestion_log SET
            status=?, items_processed=?, items_new=?, items_updated=?,
            completed_at=datetime('now'),
            duration_seconds = julianday(datetime('now')) - julianday(started_at)
        WHERE run_id=?
        """
        with self._connect() as conn:
            conn.execute(sql, (status, items, new, updated, run_id))
            conn.commit()

    # -- Scheduler run status (2.3.4.3) -------------------------------------

    @staticmethod
    def _decode_json_value(value: object, default: object) -> object:
        """Decode one persisted JSON value while keeping status reads fail-soft."""
        if value in (None, ""):
            return default
        try:
            decoded = json.loads(str(value))
        except (TypeError, ValueError):
            return default
        return decoded

    def start_scheduler_run(
        self,
        mode: str,
        *,
        policy_revision: str,
        config_revision: str,
        requested_sources: list[str],
        run_id: Optional[str] = None,
        started_at: Optional[str] = None,
        bootstrap_manifest: Optional[dict[str, list[str]]] = None,
    ) -> str:
        """Persist one scheduler run header before any source work begins."""
        run_id = str(run_id or uuid.uuid4())
        started = str(started_at or datetime.now(timezone.utc).isoformat())
        requested = [str(source) for source in dict.fromkeys(requested_sources)]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO scheduler_runs (
                    run_id, mode, policy_revision, config_revision, started_at,
                    status, requested_sources_json
                ) VALUES (?, ?, ?, ?, ?, 'running', ?)""",
                (
                    run_id,
                    str(mode),
                    str(policy_revision),
                    str(config_revision),
                    started,
                    json.dumps(requested, sort_keys=True),
                ),
            )
            if bootstrap_manifest:
                conn.executemany(
                    "INSERT OR IGNORE INTO bootstrap_partitions "
                    "(run_id, source, partition_key, status) VALUES (?, ?, ?, 'pending')",
                    [
                        (run_id, str(source), str(partition))
                        for source, partitions in bootstrap_manifest.items()
                        for partition in partitions
                    ],
                )
            conn.commit()
        return run_id

    def record_scheduler_source_summary(self, summary: dict) -> None:
        """Upsert one bounded per-source summary for a scheduler run."""
        run_id = str(summary.get("run_id") or "").strip()
        source = str(summary.get("source") or "").strip()
        if not run_id or not source:
            raise ValueError("run_id and source are required")

        def integer(name: str) -> int:
            try:
                return max(int(summary.get(name) or 0), 0)
            except (TypeError, ValueError):
                return 0

        details = summary.get("details")
        details_json = json.dumps(details if isinstance(details, dict) else {}, default=str)
        values = (
            run_id,
            source,
            str(summary.get("mode") or "incremental"),
            str(summary.get("policy_revision") or "unknown"),
            str(summary.get("config_revision") or "unknown"),
            summary.get("started_at"),
            summary.get("ended_at"),
            summary.get("duration_seconds"),
            str(summary.get("status") or "error"),
            integer("requested"),
            integer("completed"),
            integer("skipped"),
            integer("failed"),
            integer("partitions"),
            integer("items"),
            integer("requests"),
            integer("new_items"),
            integer("updated_items"),
            integer("duplicates"),
            json.dumps(summary.get("cursor_before"), default=str)
            if summary.get("cursor_before") is not None else None,
            json.dumps(summary.get("cursor_after"), default=str)
            if summary.get("cursor_after") is not None else None,
            summary.get("quota_remaining"),
            summary.get("freshness"),
            summary.get("last_success"),
            summary.get("next_due"),
            summary.get("cooldown_reset"),
            summary.get("error_class"),
            summary.get("error_message"),
            details_json,
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO scheduler_run_sources (
                    run_id, source, mode, policy_revision, config_revision,
                    started_at, ended_at, duration_seconds, status, requested,
                    completed, skipped, failed, partitions, items, requests,
                    new_items, updated_items, duplicates, cursor_before_json,
                    cursor_after_json, quota_remaining, freshness, last_success,
                    next_due, cooldown_reset, error_class, error_message,
                    details_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id, source) DO UPDATE SET
                    mode=excluded.mode,
                    policy_revision=excluded.policy_revision,
                    config_revision=excluded.config_revision,
                    started_at=excluded.started_at,
                    ended_at=excluded.ended_at,
                    duration_seconds=excluded.duration_seconds,
                    status=excluded.status,
                    requested=excluded.requested,
                    completed=excluded.completed,
                    skipped=excluded.skipped,
                    failed=excluded.failed,
                    partitions=excluded.partitions,
                    items=excluded.items,
                    requests=excluded.requests,
                    new_items=excluded.new_items,
                    updated_items=excluded.updated_items,
                    duplicates=excluded.duplicates,
                    cursor_before_json=excluded.cursor_before_json,
                    cursor_after_json=excluded.cursor_after_json,
                    quota_remaining=excluded.quota_remaining,
                    freshness=excluded.freshness,
                    last_success=excluded.last_success,
                    next_due=excluded.next_due,
                    cooldown_reset=excluded.cooldown_reset,
                    error_class=excluded.error_class,
                    error_message=excluded.error_message,
                    details_json=excluded.details_json""",
                values,
            )
            conn.commit()

    def complete_scheduler_run(
        self,
        run_id: str,
        *,
        status: str,
        ended_at: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        completed_sources: Optional[list[str]] = None,
        skipped_sources: Optional[list[str]] = None,
        failed_sources: Optional[list[str]] = None,
        terminal_error_class: Optional[str] = None,
        terminal_error_message: Optional[str] = None,
    ) -> None:
        """Close a scheduler run and retain only bounded redacted terminal state."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE scheduler_runs SET
                    ended_at=?, duration_seconds=?, status=?,
                    completed_sources_json=?, skipped_sources_json=?,
                    failed_sources_json=?, terminal_error_class=?,
                    terminal_error_message=?
                WHERE run_id=?""",
                (
                    ended_at or datetime.now(timezone.utc).isoformat(),
                    duration_seconds,
                    str(status),
                    json.dumps(list(completed_sources or []), sort_keys=True),
                    json.dumps(list(skipped_sources or []), sort_keys=True),
                    json.dumps(list(failed_sources or []), sort_keys=True),
                    terminal_error_class,
                    str(terminal_error_message)[:2_000]
                    if terminal_error_message else None,
                    str(run_id),
                ),
            )
            conn.commit()

    def record_bootstrap_partition(
        self,
        run_id: str,
        source: str,
        partition_key: str,
        *,
        status: str,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        attempts: int = 0,
        items: int = 0,
        new_items: int = 0,
        updated_items: int = 0,
        duplicates: int = 0,
        error_class: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Checkpoint one bootstrap partition after its writes complete."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO bootstrap_partitions (
                    run_id, source, partition_key, status, started_at, ended_at,
                    attempts, items, new_items, updated_items, duplicates,
                    error_class, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, source, partition_key) DO UPDATE SET
                    status=excluded.status,
                    started_at=COALESCE(excluded.started_at, bootstrap_partitions.started_at),
                    ended_at=excluded.ended_at,
                    attempts=excluded.attempts,
                    items=excluded.items,
                    new_items=excluded.new_items,
                    updated_items=excluded.updated_items,
                    duplicates=excluded.duplicates,
                    error_class=excluded.error_class,
                    error_message=excluded.error_message""",
                (
                    str(run_id), str(source), str(partition_key), str(status),
                    started_at, ended_at, max(int(attempts), 0), max(int(items), 0),
                    max(int(new_items), 0), max(int(updated_items), 0),
                    max(int(duplicates), 0),
                    error_class,
                    str(error_message)[:2_000] if error_message else None,
                ),
            )
            conn.commit()

    def list_bootstrap_partitions(
        self, run_id: str, source: Optional[str] = None,
    ) -> list[dict]:
        """Return durable bootstrap partition checkpoints for one run."""
        where = ["run_id = ?"]
        params: list[object] = [str(run_id)]
        if source is not None:
            where.append("source = ?")
            params.append(str(source))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM bootstrap_partitions WHERE "
                + " AND ".join(where)
                + " ORDER BY source, partition_key",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_resumable_bootstrap_run(self, source: Optional[str] = None) -> Optional[dict]:
        """Return the newest unfinished bootstrap run, optionally source-scoped."""
        params: list[object] = []
        where = ["mode = 'bootstrap'", "status IN ('running', 'error', 'partial', 'interrupted')"]
        if source is not None:
            where.append(
                "requested_sources_json LIKE ?"
            )
            params.append(f'%"{str(source)}"%')
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scheduler_runs WHERE " + " AND ".join(where)
                + " ORDER BY started_at DESC LIMIT 1",
                params,
            ).fetchone()
        return dict(row) if row else None

    def list_scheduler_runs(
        self, *, limit: int = 20, source: Optional[str] = None,
    ) -> list[dict]:
        """Return bounded scheduler run headers with their source summaries."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= self.MAX_INVENTORY_LIMIT:
            raise ValueError(f"limit must be between 1 and {self.MAX_INVENTORY_LIMIT}")
        params: list[object] = []
        where = ""
        if source is not None:
            where = (
                "WHERE EXISTS (SELECT 1 FROM scheduler_run_sources rs "
                "WHERE rs.run_id = scheduler_runs.run_id AND rs.source = ?)"
            )
            params.append(str(source))
        params.append(limit)
        with self._connect() as conn:
            runs = conn.execute(
                "SELECT * FROM scheduler_runs " + where
                + " ORDER BY started_at DESC LIMIT ?", params,
            ).fetchall()
            result = []
            for row in runs:
                run = dict(row)
                for name in (
                    "requested_sources_json", "completed_sources_json",
                    "skipped_sources_json", "failed_sources_json",
                ):
                    run[name.removesuffix("_json")] = self._decode_json_value(
                        run.pop(name), []
                    )
                sources = conn.execute(
                    "SELECT * FROM scheduler_run_sources WHERE run_id=? "
                    "ORDER BY source", (run["run_id"],),
                ).fetchall()
                decoded_sources = []
                for source_row in sources:
                    source_value = dict(source_row)
                    for name in ("cursor_before_json", "cursor_after_json", "details_json"):
                        source_value[name.removesuffix("_json")] = self._decode_json_value(
                            source_value.pop(name), {} if name == "details_json" else None
                        )
                    decoded_sources.append(source_value)
                run["sources"] = decoded_sources
                result.append(run)
        return result

    def prune_scheduler_history(self, max_runs: int = 100) -> int:
        """Delete oldest completed scheduler summaries beyond the configured bound."""
        if not isinstance(max_runs, int) or max_runs < 1:
            raise ValueError("max_runs must be a positive integer")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM scheduler_runs "
                "WHERE status NOT IN ('running', 'interrupted') "
                "AND NOT (mode='bootstrap' AND status IN ('error', 'partial')) "
                "ORDER BY started_at DESC "
                "LIMIT -1 OFFSET ?", (max_runs,),
            ).fetchall()
            ids = [str(row[0]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                cursor = conn.execute(
                    f"DELETE FROM scheduler_runs WHERE run_id IN ({placeholders})",
                    ids,
                )
                conn.commit()
                return int(cursor.rowcount)
        return 0

    # ── Store Revision (2.2.6.2) ───────────────────────

    def get_store_revision(self) -> int:
        """Return the current monotonic data revision (0 if never bumped)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT revision FROM store_revision WHERE id=1"
            ).fetchone()
        return int(row[0]) if row else 0

    def bump_store_revision(self, reason: Optional[str] = None) -> int:
        """Increment the data revision and return the new value.

        Called BEFORE any model-visible mutation begins so the versioned
        retrieval cache keyed on this revision can never serve pre-mutation
        evidence. ``reason`` is diagnostic only (logged, never persisted).
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO store_revision (id, revision, updated_at) "
                "VALUES (1, 1, datetime('now')) "
                "ON CONFLICT(id) DO UPDATE SET "
                "    revision = revision + 1, updated_at = datetime('now')"
            )
            conn.commit()
            row = conn.execute(
                "SELECT revision FROM store_revision WHERE id=1"
            ).fetchone()
        revision = int(row[0]) if row else 0
        logger.debug("Store revision bumped to %d (%s)", revision, reason or "")
        return revision

    # -- Normalized corpus records (2.3.3.1) ---------------------------------

    @staticmethod
    def _json_value(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _bump_revision_in_transaction(conn: sqlite3.Connection) -> int:
        conn.execute(
            "INSERT INTO store_revision (id, revision, updated_at) "
            "VALUES (1, 1, datetime('now')) ON CONFLICT(id) DO UPDATE SET "
            "revision=revision+1, updated_at=datetime('now')"
        )
        return int(conn.execute(
            "SELECT revision FROM store_revision WHERE id=1"
        ).fetchone()[0])

    @staticmethod
    def _source_key(record: NarrativeRecord, canonical_url: Optional[str]) -> str:
        identity = (
            f"provider:{record.provider_record_id}"
            if record.provider_record_id
            else f"url:{canonical_url or record.source_url}"
        )
        return content_hash(f"{record.source_name}\n{identity}")

    @staticmethod
    def _decode_corpus_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        for name in ("tickers_json", "index_codes_json", "sectors_json", "metadata_json"):
            result[name.removesuffix("_json")] = json.loads(result.pop(name))
        result["is_tombstone"] = bool(result.get("is_tombstone", 0))
        return result

    @classmethod
    def _corpus_metadata_bytes(cls, record: NarrativeRecord) -> int:
        """Estimate retained SQLite metadata bytes without storing the narrative body."""
        values = (
            record.corpus_item_id,
            record.source_name,
            record.source_category,
            record.provider_record_id,
            record.original_publisher,
            record.item_type,
            record.event_type,
            record.title,
            record.summary,
            record.published_at,
            record.effective_at,
            record.as_of_at,
            record.source_url,
            record.canonical_url,
            cls._json_value(record.tickers),
            cls._json_value(record.index_codes),
            cls._json_value(record.sectors),
            cls._json_value(dict(record.metadata)),
        )
        return sum(len(str(value).encode("utf-8")) for value in values if value is not None)

    def _find_corpus_duplicate(
        self,
        conn: sqlite3.Connection,
        record: NarrativeRecord,
        canonical_url: Optional[str],
        news_key: Optional[str],
    ) -> tuple[Optional[sqlite3.Row], Optional[str]]:
        lookups = []
        if record.provider_record_id:
            lookups.append((
                "provider_identity", "source=? AND provider_record_id=?",
                (record.source_name, record.provider_record_id),
            ))
        if canonical_url and record.published_at:
            lookups.append((
                "canonical_url", "canonical_url=? AND published_at=?",
                (canonical_url, record.published_at),
            ))
        lookups.append(("content_hash", "content_hash=?", (record.content_hash,)))
        if news_key:
            lookups.append((
                "syndicated_headline",
                "syndicated_key=? AND source<>?",
                (news_key, record.source_name),
            ))
        for layer, predicate, params in lookups:
            row = conn.execute(
                f"SELECT * FROM corpus_items WHERE {predicate} "
                "ORDER BY created_at, corpus_item_id LIMIT 1", params,
            ).fetchone()
            if row:
                return row, layer
        return None, None

    def upsert_narrative_record(self, record: NarrativeRecord) -> dict:
        """Atomically upsert narrative metadata, provenance, and security links."""
        canonical_url = normalize_canonical_url(record.canonical_url)
        headline = normalize_headline(record.title)
        news_key = (
            syndicated_news_key(record.title, record.published_at)
            if record.item_type == "news" and record.published_at else None
        )
        metadata_json = self._json_value(dict(record.metadata))
        initial_status = (
            "not_applicable" if record.indexing_status == "not_applicable" else "pending"
        )
        with self._connect() as conn:
            existing, layer = self._find_corpus_duplicate(
                conn, record, canonical_url, news_key,
            )
            created = existing is None
            item_id = record.corpus_item_id if created else str(existing["corpus_item_id"])
            content_changed = bool(
                existing is not None
                and existing["content_hash"] != record.content_hash
                and (
                    layer == "provider_identity"
                    or (
                        layer == "canonical_url"
                        and existing["source"] == record.source_name
                    )
                )
            )
            previous_status = None if existing is None else str(existing["indexing_status"])
            authority_promoted = bool(
                existing is not None
                and record.evidence_authority == "direct_sec"
                and existing["evidence_authority"] != "direct_sec"
            )
            common_values = (
                record.event_type, record.title, headline, news_key, record.summary,
                record.language, record.published_at, record.effective_at,
                record.as_of_at, record.observed_at, record.accessed_at,
                record.ingested_at, record.source_url, canonical_url,
                self._json_value(record.tickers), self._json_value(record.index_codes),
                self._json_value(record.sectors), record.content_hash, metadata_json,
                record.document_family, initial_status, record.license_label,
                record.normalization_version, record.evidence_authority,
            )
            if created:
                conn.execute(
                    """INSERT INTO corpus_items (
                        corpus_item_id, source, source_category, provider_record_id,
                        original_publisher, item_type, event_type, title,
                        normalized_headline, syndicated_key, summary, language,
                        published_at, effective_at, as_of_at, observed_at, accessed_at,
                        ingested_at, source_url, canonical_url, tickers_json,
                        index_codes_json, sectors_json, content_hash, metadata_json,
                        document_family, indexing_status, license_label,
                        normalization_version, evidence_authority
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        item_id, record.source_name, record.source_category,
                        record.provider_record_id, record.original_publisher,
                        record.item_type, *common_values,
                    ),
                )
            elif authority_promoted:
                conn.execute(
                    """UPDATE corpus_items SET
                        source=?, source_category=?, provider_record_id=?,
                        original_publisher=?, item_type=?, event_type=?, title=?,
                        normalized_headline=?, syndicated_key=?, summary=?, language=?,
                        published_at=?, effective_at=?, as_of_at=?, observed_at=?,
                        accessed_at=?, ingested_at=?, source_url=?, canonical_url=?,
                        tickers_json=?, index_codes_json=?, sectors_json=?, content_hash=?,
                        metadata_json=?, document_family=?, indexing_status=?, index_error=NULL,
                        license_label=?, normalization_version=?, evidence_authority=?,
                        updated_at=datetime('now')
                    WHERE corpus_item_id=?""",
                    (
                        record.source_name, record.source_category,
                        record.provider_record_id, record.original_publisher,
                        record.item_type, *common_values, item_id,
                    ),
                )
            elif content_changed:
                conn.execute(
                    """UPDATE corpus_items SET
                        event_type=?, title=?, normalized_headline=?, syndicated_key=?,
                        summary=?, language=?, published_at=?, effective_at=?, as_of_at=?,
                        observed_at=?, accessed_at=?, ingested_at=?, source_url=?,
                        canonical_url=?, tickers_json=?, index_codes_json=?, sectors_json=?,
                        content_hash=?, metadata_json=?, document_family=?,
                        indexing_status=?, index_error=NULL, license_label=?,
                        normalization_version=?, evidence_authority=?, updated_at=datetime('now')
                    WHERE corpus_item_id=?""",
                    (*common_values, item_id),
                )
            else:
                conn.execute(
                    "UPDATE corpus_items SET accessed_at=MAX(accessed_at, ?), "
                    "ingested_at=MAX(ingested_at, ?), updated_at=datetime('now') "
                    "WHERE corpus_item_id=?",
                    (record.accessed_at, record.ingested_at, item_id),
                )
            if created or content_changed or authority_promoted:
                conn.execute(
                    "UPDATE corpus_items SET document_family_id=?, narrative_bytes=?, "
                    "metadata_bytes=?, is_tombstone=0, retired_at=NULL, retention_reason=NULL "
                    "WHERE corpus_item_id=?",
                    (
                        item_id,
                        len(record.body.encode("utf-8")),
                        self._corpus_metadata_bytes(record),
                        item_id,
                    ),
                )
            conn.execute(
                """INSERT INTO corpus_item_sources (
                    corpus_item_id, source_key, source_name, source_category,
                    provider_record_id, original_publisher, source_url, canonical_url,
                    published_at, observed_at, accessed_at, ingested_at, license_label,
                    evidence_authority, metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(corpus_item_id, source_key) DO UPDATE SET
                    source_category=excluded.source_category,
                    original_publisher=excluded.original_publisher,
                    source_url=excluded.source_url, canonical_url=excluded.canonical_url,
                    published_at=excluded.published_at, observed_at=excluded.observed_at,
                    accessed_at=MAX(corpus_item_sources.accessed_at, excluded.accessed_at),
                    ingested_at=MAX(corpus_item_sources.ingested_at, excluded.ingested_at),
                    license_label=excluded.license_label,
                    evidence_authority=excluded.evidence_authority,
                    metadata_json=excluded.metadata_json""",
                (
                    item_id, self._source_key(record, canonical_url), record.source_name,
                    record.source_category, record.provider_record_id,
                    record.original_publisher, record.source_url, canonical_url,
                    record.published_at, record.observed_at, record.accessed_at,
                    record.ingested_at, record.license_label,
                    record.evidence_authority, metadata_json,
                ),
            )
            for index, security_id in enumerate(record.security_ids):
                ticker = record.tickers[index] if index < len(record.tickers) else None
                conn.execute(
                    "INSERT OR IGNORE INTO corpus_item_securities "
                    "(corpus_item_id, security_id, ticker) VALUES (?, ?, ?)",
                    (item_id, security_id, ticker),
                )
            revision = self._bump_revision_in_transaction(conn)
            conn.commit()
        needs_index = initial_status != "not_applicable" and (
            created or content_changed or authority_promoted
            or previous_status in {"pending", "error"}
        )
        return {
            "corpus_item_id": item_id,
            "created": created,
            "deduplicated": not created,
            "deduplication_layer": layer,
            "content_changed": content_changed,
            "authority_promoted": authority_promoted,
            "needs_index": needs_index,
            "indexing_status": initial_status if created or content_changed else previous_status,
            "document_family_id": item_id,
            "revision": revision,
        }

    def set_corpus_index_status(
        self, corpus_item_id: str, status: str, error: Optional[str] = None,
    ) -> None:
        if status not in {"pending", "indexed", "error", "not_applicable"}:
            raise ValueError("invalid corpus indexing status")
        with self._connect() as conn:
            conn.execute(
                "UPDATE corpus_items SET indexing_status=?, index_error=?, "
                "updated_at=datetime('now') WHERE corpus_item_id=?",
                (status, error[:2_000] if error else None, corpus_item_id),
            )
            conn.commit()

    def get_corpus_item(self, corpus_item_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM corpus_items WHERE corpus_item_id=?", (corpus_item_id,),
            ).fetchone()
        return self._decode_corpus_row(row) if row else None

    def count_corpus_items(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM corpus_items").fetchone()[0])

    def list_corpus_item_sources(self, corpus_item_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM corpus_item_sources WHERE corpus_item_id=? "
                "ORDER BY source_name, source_key", (corpus_item_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_corpus_item_securities(self, corpus_item_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM corpus_item_securities WHERE corpus_item_id=? "
                "ORDER BY security_id", (corpus_item_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_retryable_corpus_items(
        self,
        limit: int = 100,
        *,
        source: Optional[str] = None,
        security: Optional[str] = None,
        item_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[dict]:
        """List pending/error narrative metadata with bounded repair filters."""
        if limit < 1 or limit > self.MAX_INVENTORY_LIMIT:
            raise ValueError(f"limit must be between 1 and {self.MAX_INVENTORY_LIMIT}")
        conditions = ["ci.indexing_status IN ('pending', 'error')", "ci.is_tombstone=0"]
        params: list[object] = []
        if source:
            conditions.append("ci.source = ?")
            params.append(str(source))
        if item_id:
            conditions.append("(ci.corpus_item_id = ? OR ci.provider_record_id = ?)")
            params.extend([str(item_id), str(item_id)])
        if date_from:
            conditions.append("COALESCE(ci.published_at, ci.effective_at, ci.observed_at) >= ?")
            params.append(str(date_from))
        if date_to:
            conditions.append("COALESCE(ci.published_at, ci.effective_at, ci.observed_at) <= ?")
            params.append(str(date_to))
        if run_id:
            conditions.append(
                "(ci.metadata_json LIKE ? OR EXISTS ("
                "SELECT 1 FROM scheduler_runs sr WHERE sr.run_id=? "
                "AND datetime(ci.ingested_at) >= datetime(sr.started_at) "
                "AND datetime(ci.ingested_at) <= datetime(COALESCE(sr.ended_at, 'now'))"
                ") OR EXISTS ("
                "SELECT 1 FROM ingestion_log il WHERE il.run_id=? "
                "AND datetime(ci.ingested_at) >= datetime(il.started_at) "
                "AND datetime(ci.ingested_at) <= datetime(COALESCE(il.completed_at, 'now'))"
                "))"
            )
            normalized_run_id = str(run_id).strip()
            params.extend([f"%{normalized_run_id}%", normalized_run_id, normalized_run_id])
        if security:
            conditions.append(
                "EXISTS (SELECT 1 FROM corpus_item_securities cis "
                "LEFT JOIN securities s ON s.security_id = cis.security_id "
                "WHERE cis.corpus_item_id = ci.corpus_item_id "
                "AND (cis.security_id = ? OR cis.ticker = ? OR s.ticker = ?))"
            )
            normalized = str(security).strip().upper()
            params.extend([str(security), normalized, normalized])
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ci.* FROM corpus_items ci WHERE "
                + " AND ".join(conditions)
                + " ORDER BY ci.updated_at, ci.corpus_item_id LIMIT ?",
                params,
            ).fetchall()
        return [self._decode_corpus_row(row) for row in rows]

    def list_retryable_filings(
        self,
        limit: int = 100,
        *,
        security: Optional[str] = None,
        item_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[dict]:
        """List stored SEC artifacts whose vector indexing can be retried."""
        if limit < 1 or limit > self.MAX_INVENTORY_LIMIT:
            raise ValueError(f"limit must be between 1 and {self.MAX_INVENTORY_LIMIT}")
        conditions = ["status = 'index_pending'"]
        params: list[object] = []
        if item_id:
            conditions.append("accession = ?")
            params.append(str(item_id))
        if security:
            conditions.append("ticker = ?")
            params.append(str(security).strip().upper())
        if date_from:
            conditions.append("filing_date >= ?")
            params.append(str(date_from))
        if date_to:
            conditions.append("filing_date <= ?")
            params.append(str(date_to))
        if run_id:
            conditions.append(
                "(EXISTS ("
                "SELECT 1 FROM scheduler_runs sr WHERE sr.run_id=? "
                "AND datetime(filings.ingested_at) >= datetime(sr.started_at) "
                "AND datetime(filings.ingested_at) <= datetime(COALESCE(sr.ended_at, 'now'))"
                ") OR EXISTS ("
                "SELECT 1 FROM ingestion_log il WHERE il.run_id=? "
                "AND datetime(filings.ingested_at) >= datetime(il.started_at) "
                "AND datetime(filings.ingested_at) <= datetime(COALESCE(il.completed_at, 'now'))"
                "))"
            )
            normalized_run_id = str(run_id).strip()
            params.extend([normalized_run_id, normalized_run_id])
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM filings WHERE " + " AND ".join(conditions)
                + " ORDER BY filing_date, accession LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_retryable_filing(self, accession: str) -> Optional[dict]:
        """Read one internal SEC repair row, including its local artifact path."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM filings WHERE accession=? AND status='index_pending'",
                (str(accession),),
            ).fetchone()
        return dict(row) if row else None

    def list_news_retention_candidates(
        self,
        cutoff: str,
        *,
        limit: int,
        document_family_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return a bounded set of indexed company-news families older than cutoff."""
        if limit < 1 or limit > self.MAX_MAINTENANCE_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {self.MAX_MAINTENANCE_LIMIT}"
            )
        ids = list(dict.fromkeys(str(value) for value in (document_family_ids or []) if value))
        conditions = [
            "item_type='news'", "is_tombstone=0", "indexing_status='indexed'",
            "published_at IS NOT NULL", "published_at < ?",
        ]
        params: list[object] = [cutoff]
        if document_family_ids is not None and not ids:
            conditions.append("0=1")
        elif ids:
            placeholders = ",".join("?" for _ in ids)
            conditions.append(f"document_family_id IN ({placeholders})")
            params.extend(ids)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT corpus_item_id, document_family_id, published_at "
                "FROM corpus_items WHERE " + " AND ".join(conditions)
                + " ORDER BY published_at, corpus_item_id LIMIT ?",
                [*params, limit],
            ).fetchall()
        return [dict(row) for row in rows]

    def expire_news_narratives(
        self,
        corpus_item_ids: list[str],
        *,
        retired_at: str,
        reason: str,
    ) -> int:
        """Turn selected news rows into tombstones and bump revision once."""
        item_ids = list(dict.fromkeys(corpus_item_ids))
        if not item_ids:
            return 0
        if len(item_ids) > self.MAX_MAINTENANCE_LIMIT:
            raise ValueError("too many corpus items for one retention run")
        placeholders = ",".join("?" for _ in item_ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE corpus_items SET is_tombstone=1, narrative_bytes=0, "
                f"indexing_status='not_applicable', index_error=NULL, retired_at=?, "
                f"retention_reason=?, updated_at=datetime('now') WHERE corpus_item_id IN "
                f"({placeholders}) AND item_type='news' AND is_tombstone=0 "
                f"AND indexing_status='indexed'",
                (retired_at, reason, *item_ids),
            )
            changed = int(cursor.rowcount)
            if changed:
                self._bump_revision_in_transaction(conn)
            conn.commit()
        return changed

    def get_corpus_accounting(
        self,
        group_by: str,
        *,
        source_category: Optional[str] = None,
        source: Optional[str] = None,
        item_type: Optional[str] = None,
        security: Optional[str] = None,
        year: Optional[str] = None,
        month: Optional[str] = None,
        indexing_state: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Return bounded corpus counts and byte estimates from SQLite metadata."""
        self._validate_inventory_page(limit, offset)
        include_security = group_by == "security" or security is not None
        narrative_join = (
            "LEFT JOIN corpus_item_securities link "
            "ON link.corpus_item_id=ci.corpus_item_id"
            if include_security else ""
        )
        observation_join = (
            "LEFT JOIN observation_securities link "
            "ON link.observation_id=observation.observation_id "
            "LEFT JOIN securities security ON security.security_id=link.security_id"
            if include_security else ""
        )
        event_join = (
            "LEFT JOIN event_securities link ON link.event_id=event.event_id "
            "LEFT JOIN securities security ON security.security_id=link.security_id"
            if include_security else ""
        )
        narrative_security = "link.security_id" if include_security else "NULL"
        narrative_ticker = "link.ticker" if include_security else "NULL"
        structured_security = "link.security_id" if include_security else "NULL"
        structured_ticker = "security.ticker" if include_security else "NULL"
        cte = f"""
            WITH accounting AS (
                SELECT ci.source_category, ci.source, ci.item_type,
                       {narrative_security} AS security_id,
                       {narrative_ticker} AS ticker,
                       COALESCE(ci.published_at, ci.effective_at, ci.as_of_at,
                                ci.ingested_at) AS occurred_at,
                       ci.indexing_status AS indexing_state,
                       ci.metadata_bytes + ci.narrative_bytes AS approximate_bytes
                FROM corpus_items ci {narrative_join}
                UNION ALL
                SELECT observation.source_category, observation.source_name,
                       'observation', {structured_security}, {structured_ticker},
                       COALESCE(observation.published_at, observation.as_of_at,
                                observation.period_end, observation.ingested_at),
                       'not_applicable',
                       length(CAST(COALESCE(observation.metric_id, '') AS BLOB)) +
                       length(CAST(COALESCE(observation.value_text, '') AS BLOB)) +
                       length(CAST(COALESCE(observation.unit, '') AS BLOB)) +
                       length(CAST(COALESCE(observation.metadata_json, '') AS BLOB))
                FROM corpus_observations observation {observation_join}
                UNION ALL
                SELECT event.source_category, event.source_name, 'event',
                       {structured_security}, {structured_ticker},
                       COALESCE(event.published_at, event.effective_at,
                                event.announced_at, event.ingested_at),
                       'not_applicable',
                       length(CAST(COALESCE(event.event_type, '') AS BLOB)) +
                       length(CAST(COALESCE(event.explanation, '') AS BLOB)) +
                       length(CAST(COALESCE(event.metadata_json, '') AS BLOB))
                FROM corpus_events event {event_join}
            )
        """
        date_value = "a.occurred_at"
        dimensions = {
            "source_category": "a.source_category",
            "source": "a.source",
            "item_type": "a.item_type",
            "security": "COALESCE(a.security_id, 'unlinked')",
            "year": f"substr({date_value}, 1, 4)",
            "month": f"substr({date_value}, 1, 7)",
            "indexing_state": "a.indexing_state",
        }
        expression = dimensions.get(group_by)
        if expression is None:
            raise ValueError(f"group_by must be one of {sorted(dimensions)}")

        predicates: list[str] = []
        params: list[object] = []
        filters = (
            ("a.source_category", source_category),
            ("a.source", source),
            ("a.item_type", item_type),
            (f"substr({date_value}, 1, 4)", year),
            (f"substr({date_value}, 1, 7)", month),
            ("a.indexing_state", indexing_state),
        )
        for column, value in filters:
            if value is not None:
                predicates.append(f"{column}=?")
                params.append(value)
        if security is not None:
            predicates.append("(a.security_id=? OR a.ticker=?)")
            params.extend((security, security.upper()))
        where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        sql = (
            cte +
            f"SELECT {expression} AS key, COUNT(*) AS count, "
            f"COALESCE(SUM(a.approximate_bytes), 0) AS approximate_bytes "
            f"FROM accounting a {where} GROUP BY {expression} "
            f"ORDER BY count DESC, key LIMIT ? OFFSET ?"
        )
        params.extend((limit, offset))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "key": row["key"] or "unknown",
                "count": int(row["count"]),
                "approximate_bytes": int(row["approximate_bytes"]),
            }
            for row in rows
        ]

    def upsert_observation_record(self, record: ObservationRecord) -> dict:
        """Atomically upsert one structured observation and its security links."""
        values = {
            "metric_id": record.metric_id,
            "series_id": record.series_id,
            "value_text": record.value_text,
            "value_numeric": record.value_numeric,
            "unit": record.unit,
            "frequency": record.frequency,
            "period_start": record.period_start,
            "period_end": record.period_end,
            "vintage_at": record.vintage_at,
            "as_of_at": record.as_of_at,
            "scope": record.scope,
            "tickers_json": self._json_value(record.tickers),
            "sector": record.sector,
            "source_name": record.source_name,
            "source_category": record.source_category,
            "provider_record_id": record.provider_record_id,
            "original_publisher": record.original_publisher,
            "source_url": record.source_url,
            "canonical_url": normalize_canonical_url(record.canonical_url),
            "published_at": record.published_at,
            "observed_at": record.observed_at,
            "accessed_at": record.accessed_at,
            "ingested_at": record.ingested_at,
            "license_label": record.license_label,
            "normalization_version": record.normalization_version,
            "evidence_authority": record.evidence_authority,
            "metadata_json": self._json_value(dict(record.metadata)),
        }
        with self._connect() as conn:
            existing = None
            if record.source_category == "market_data" and record.tickers:
                existing = conn.execute(
                    "SELECT * FROM corpus_observations WHERE source_name=? "
                    "AND metric_id=? AND period_end=? AND tickers_json=? "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (
                        record.source_name,
                        record.metric_id,
                        record.period_end,
                        values["tickers_json"],
                    ),
                ).fetchone()
            if record.provider_record_id:
                existing = existing or conn.execute(
                    "SELECT * FROM corpus_observations WHERE source_name=? "
                    "AND metric_id=? AND provider_record_id=? "
                    "AND vintage_at IS ?",
                    (
                        record.source_name,
                        record.metric_id,
                        record.provider_record_id,
                        record.vintage_at,
                    ),
                ).fetchone()
            if existing is None and not record.provider_record_id:
                existing = conn.execute(
                    "SELECT * FROM corpus_observations WHERE observation_id=?",
                    (record.observation_id,),
                ).fetchone()
            observation_id = (
                record.observation_id if existing is None else existing["observation_id"]
            )
            linked = {
                row[0] for row in conn.execute(
                    "SELECT security_id FROM observation_securities WHERE observation_id=?",
                    (observation_id,),
                ).fetchall()
            }
            changed = existing is None or any(
                existing[key] != value for key, value in values.items()
            )
            changed = changed or linked != set(record.security_ids)
            if not changed:
                return {
                    "observation_id": observation_id,
                    "created": False,
                    "changed": False,
                    "revision": self.get_store_revision(),
                }
            columns = ", ".join(values)
            placeholders = ", ".join("?" for _ in values)
            updates = ", ".join(f"{name}=excluded.{name}" for name in values)
            conn.execute(
                f"INSERT INTO corpus_observations (observation_id, {columns}) "
                f"VALUES (?, {placeholders}) ON CONFLICT(observation_id) DO UPDATE SET "
                f"{updates}, updated_at=datetime('now')",
                (observation_id, *values.values()),
            )
            conn.execute(
                "DELETE FROM observation_securities WHERE observation_id=?",
                (observation_id,),
            )
            conn.executemany(
                "INSERT INTO observation_securities (observation_id, security_id) "
                "VALUES (?, ?)",
                ((observation_id, security_id) for security_id in record.security_ids),
            )
            revision = self._bump_revision_in_transaction(conn)
            conn.commit()
        return {
            "observation_id": observation_id,
            "created": existing is None,
            "changed": True,
            "revision": revision,
        }

    def count_observations(self) -> int:
        with self._connect() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM corpus_observations"
            ).fetchone()[0])

    def list_observations(
        self,
        *,
        source_name: Optional[str] = None,
        metric_id: Optional[str] = None,
        period_end: Optional[str] = None,
        vintage_at: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Return bounded structured observations, including every vintage."""
        self._validate_inventory_page(limit, offset)
        conditions: list[str] = []
        params: list[object] = []
        for column, value in (
            ("source_name", source_name),
            ("metric_id", metric_id),
            ("period_end", period_end),
            ("vintage_at", vintage_at),
        ):
            if value is not None:
                conditions.append(f"{column} = ?")
                params.append(str(value))
        where = " AND ".join(conditions) or "1=1"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM corpus_observations WHERE {where} "
                "ORDER BY period_end DESC, vintage_at DESC, metric_id, observation_id "
                "LIMIT ? OFFSET ?",
                params,
            ).fetchall()
            observations: list[dict] = []
            for row in rows:
                item = dict(row)
                item["metadata"] = json.loads(item.pop("metadata_json"))
                item["tickers"] = json.loads(item.pop("tickers_json"))
                item["security_ids"] = [
                    linked[0]
                    for linked in conn.execute(
                        "SELECT security_id FROM observation_securities "
                        "WHERE observation_id=? ORDER BY security_id",
                        (item["observation_id"],),
                    ).fetchall()
                ]
                observations.append(item)
        return observations

    # -- Incremental source cursors -----------------------------------------

    def get_source_cursor_state(
        self, source: str, partition_key: str,
    ) -> Optional[dict]:
        """Return one cursor/status row without conflating it with freshness."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM source_cursors WHERE source=? AND partition_key=?",
                (str(source), str(partition_key)),
            ).fetchone()
        return dict(row) if row else None

    def get_source_cursor(self, source: str, partition_key: str) -> Optional[str]:
        """Return the last committed cursor value for one source partition."""
        state = self.get_source_cursor_state(source, partition_key)
        return str(state["cursor_value"]) if state and state.get("cursor_value") is not None else None

    def set_source_cursor(
        self,
        source: str,
        partition_key: str,
        cursor_value: Optional[str],
        *,
        cursor_type: str = "none",
        overlap_value: Optional[str] = None,
        last_successful_run_id: Optional[str] = None,
        version: str = "1",
        status: str = "success",
        error_class: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> dict:
        """Upsert a committed cursor and its bounded source status metadata."""
        source = str(source or "").strip()
        partition_key = str(partition_key or "").strip()
        cursor_type = str(cursor_type or "none").strip()
        status = str(status or "unknown").strip()
        if not source or not partition_key:
            raise ValueError("source and partition_key are required")
        if cursor_value is not None:
            cursor_value = str(cursor_value)
        if overlap_value is not None:
            overlap_value = str(overlap_value)
        if error_message is not None:
            error_message = str(error_message)[:2_000]
        updated_at = datetime.now(timezone.utc).isoformat()
        last_successful_at = (
            updated_at if status in {"success", "ok", "partial"} else None
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO source_cursors (
                    source, partition_key, cursor_value, cursor_type,
                    overlap_value, last_successful_run_id, last_successful_at,
                    version, status,
                    error_class, error_message, retry_after, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, partition_key) DO UPDATE SET
                    cursor_value=excluded.cursor_value,
                    cursor_type=excluded.cursor_type,
                    overlap_value=excluded.overlap_value,
                    last_successful_run_id=excluded.last_successful_run_id,
                    last_successful_at=CASE
                        WHEN excluded.status IN ('success', 'ok', 'partial')
                        THEN excluded.updated_at
                        ELSE source_cursors.last_successful_at
                    END,
                    version=excluded.version,
                    status=excluded.status,
                    error_class=excluded.error_class,
                    error_message=excluded.error_message,
                    retry_after=excluded.retry_after,
                    updated_at=excluded.updated_at""",
                (
                    source,
                    partition_key,
                    cursor_value,
                    cursor_type,
                    overlap_value,
                    last_successful_run_id,
                    last_successful_at,
                    str(version or "1"),
                    status,
                    error_class,
                    error_message,
                    retry_after,
                    updated_at,
                ),
            )
            conn.commit()
        return self.get_source_cursor_state(source, partition_key) or {}

    def set_source_cursors(self, updates: list[dict]) -> list[dict]:
        """Atomically upsert multiple source cursor partitions."""
        normalized = [self._normalize_source_cursor_update(update) for update in updates]
        if not normalized:
            return []
        with self._connect() as conn:
            for update in normalized:
                conn.execute(
                    """INSERT INTO source_cursors (
                        source, partition_key, cursor_value, cursor_type,
                        overlap_value, last_successful_run_id, last_successful_at,
                        version, status,
                        error_class, error_message, retry_after, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, partition_key) DO UPDATE SET
                        cursor_value=excluded.cursor_value,
                        cursor_type=excluded.cursor_type,
                        overlap_value=excluded.overlap_value,
                        last_successful_run_id=excluded.last_successful_run_id,
                        last_successful_at=CASE
                            WHEN excluded.status IN ('success', 'ok', 'partial')
                            THEN excluded.updated_at
                            ELSE source_cursors.last_successful_at
                        END,
                        version=excluded.version,
                        status=excluded.status,
                        error_class=excluded.error_class,
                        error_message=excluded.error_message,
                        retry_after=excluded.retry_after,
                        updated_at=excluded.updated_at""",
                    (
                        update["source"],
                        update["partition_key"],
                        update["cursor_value"],
                        update["cursor_type"],
                        update["overlap_value"],
                        update["last_successful_run_id"],
                        (
                            update["updated_at"]
                            if update["status"] in {"success", "ok", "partial"}
                            else None
                        ),
                        update["version"],
                        update["status"],
                        update["error_class"],
                        update["error_message"],
                        update["retry_after"],
                        update["updated_at"],
                    ),
                )
            conn.commit()
        return [
            self.get_source_cursor_state(update["source"], update["partition_key"])
            or {}
            for update in normalized
        ]

    @staticmethod
    def _normalize_source_cursor_update(update: dict) -> dict:
        """Validate and normalize one bulk cursor update before its transaction."""
        source = str(update.get("source") or "").strip()
        partition_key = str(update.get("partition_key") or "").strip()
        if not source or not partition_key:
            raise ValueError("source and partition_key are required")
        cursor_value = update.get("cursor_value")
        overlap_value = update.get("overlap_value")
        error_message = update.get("error_message")
        return {
            "source": source,
            "partition_key": partition_key,
            "cursor_value": str(cursor_value) if cursor_value is not None else None,
            "cursor_type": str(update.get("cursor_type") or "none").strip(),
            "overlap_value": (
                str(overlap_value) if overlap_value is not None else None
            ),
            "last_successful_run_id": update.get("last_successful_run_id"),
            "version": str(update.get("version") or "1"),
            "status": str(update.get("status") or "success").strip(),
            "error_class": update.get("error_class"),
            "error_message": (
                str(error_message)[:2_000] if error_message is not None else None
            ),
            "retry_after": update.get("retry_after"),
            "updated_at": str(
                update.get("updated_at") or datetime.now(timezone.utc).isoformat()
            ),
        }

    def list_source_cursor_states(self, source: str) -> list[dict]:
        """Return cursor states for one source ordered by oldest update first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM source_cursors WHERE source=? "
                "ORDER BY updated_at, partition_key",
                (str(source),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_source_budget_usage(
        self,
        source: str,
        *,
        day_start: str,
        minute_start: str,
    ) -> dict:
        """Load persisted provider usage for the current day and minute windows."""
        with self._connect() as conn:
            rows = {
                str(row["window_kind"]): dict(row)
                for row in conn.execute(
                    "SELECT * FROM source_budget_usage WHERE source=? AND "
                    "((window_kind='day' AND window_start=?) OR "
                    "(window_kind='minute' AND window_start=?))",
                    (str(source), str(day_start), str(minute_start)),
                ).fetchall()
            }
        day = rows.get("day", {})
        minute = rows.get("minute", {})
        return {
            "day_requests": int(day.get("attempted_requests") or 0),
            "minute_requests": int(minute.get("attempted_requests") or 0),
            "provider_remaining": day.get("provider_remaining"),
            "provider_reset": day.get("provider_reset"),
        }

    def record_source_budget_usage(
        self,
        source: str,
        *,
        day_start: str,
        minute_start: str,
        attempted_requests: int,
        successful_requests: int,
        provider_remaining: Optional[int] = None,
        provider_reset: Optional[str] = None,
    ) -> None:
        """Atomically add one run's quota counters to day and minute windows."""
        attempted_requests = max(int(attempted_requests), 0)
        successful_requests = max(int(successful_requests), 0)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for kind, window_start in (
                ("day", str(day_start)),
                ("minute", str(minute_start)),
            ):
                conn.execute(
                    """INSERT INTO source_budget_usage (
                        source, window_kind, window_start, attempted_requests,
                        successful_requests, provider_remaining, provider_reset,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, window_kind, window_start) DO UPDATE SET
                        attempted_requests=(
                            source_budget_usage.attempted_requests
                            + excluded.attempted_requests
                        ),
                        successful_requests=(
                            source_budget_usage.successful_requests
                            + excluded.successful_requests
                        ),
                        provider_remaining=COALESCE(
                            excluded.provider_remaining,
                            source_budget_usage.provider_remaining
                        ),
                        provider_reset=COALESCE(
                            excluded.provider_reset,
                            source_budget_usage.provider_reset
                        ),
                        updated_at=excluded.updated_at""",
                    (
                        str(source),
                        kind,
                        window_start,
                        attempted_requests,
                        successful_requests,
                        provider_remaining if kind == "day" else None,
                        provider_reset if kind == "day" else None,
                        now,
                    ),
                )
            conn.commit()

    def set_source_status(
        self,
        source: str,
        partition_key: str,
        status: str,
        *,
        error_class: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> dict:
        """Persist a source capability state while retaining its last cursor."""
        previous = self.get_source_cursor_state(source, partition_key) or {}
        return self.set_source_cursor(
            source,
            partition_key,
            previous.get("cursor_value"),
            cursor_type=str(previous.get("cursor_type") or "none"),
            overlap_value=previous.get("overlap_value"),
            last_successful_run_id=previous.get("last_successful_run_id"),
            version=str(previous.get("version") or "1"),
            status=status,
            error_class=error_class,
            error_message=error_message,
            retry_after=retry_after,
        )

    def upsert_event_record(self, record: EventRecord) -> dict:
        """Atomically upsert one event and all item/security relationships."""
        values = {
            "event_type": record.event_type,
            "effective_at": record.effective_at,
            "announced_at": record.announced_at,
            "status": record.status,
            "amount": record.amount,
            "currency": record.currency,
            "rate": record.rate,
            "ratio": record.ratio,
            "action_date": record.action_date,
            "classifier_version": record.classifier_version,
            "explanation": record.explanation,
            "source_name": record.source_name,
            "source_category": record.source_category,
            "provider_record_id": record.provider_record_id,
            "original_publisher": record.original_publisher,
            "source_url": record.source_url,
            "canonical_url": normalize_canonical_url(record.canonical_url),
            "published_at": record.published_at,
            "observed_at": record.observed_at,
            "accessed_at": record.accessed_at,
            "ingested_at": record.ingested_at,
            "license_label": record.license_label,
            "normalization_version": record.normalization_version,
            "evidence_authority": record.evidence_authority,
            "metadata_json": self._json_value(dict(record.metadata)),
        }
        with self._connect() as conn:
            existing = None
            if record.provider_record_id:
                existing = conn.execute(
                    "SELECT * FROM corpus_events WHERE source_name=? AND provider_record_id=?",
                    (record.source_name, record.provider_record_id),
                ).fetchone()
            if existing is None:
                existing = conn.execute(
                    "SELECT * FROM corpus_events WHERE event_id=?", (record.event_id,),
                ).fetchone()
            event_id = record.event_id if existing is None else existing["event_id"]
            old_securities = {
                row[0] for row in conn.execute(
                    "SELECT security_id FROM event_securities WHERE event_id=?",
                    (event_id,),
                ).fetchall()
            }
            old_items = {
                row[0] for row in conn.execute(
                    "SELECT corpus_item_id FROM event_corpus_items WHERE event_id=?",
                    (event_id,),
                ).fetchall()
            }
            changed = existing is None or any(
                existing[key] != value for key, value in values.items()
            )
            changed = changed or old_securities != set(record.security_ids)
            changed = changed or old_items != set(record.source_corpus_item_ids)
            if not changed:
                return {
                    "event_id": event_id,
                    "created": False,
                    "changed": False,
                    "revision": self.get_store_revision(),
                }
            columns = ", ".join(values)
            placeholders = ", ".join("?" for _ in values)
            updates = ", ".join(f"{name}=excluded.{name}" for name in values)
            conn.execute(
                f"INSERT INTO corpus_events (event_id, {columns}) VALUES (?, {placeholders}) "
                f"ON CONFLICT(event_id) DO UPDATE SET {updates}, updated_at=datetime('now')",
                (event_id, *values.values()),
            )
            conn.execute("DELETE FROM event_securities WHERE event_id=?", (event_id,))
            conn.executemany(
                "INSERT INTO event_securities (event_id, security_id) VALUES (?, ?)",
                ((event_id, security_id) for security_id in record.security_ids),
            )
            conn.execute("DELETE FROM event_corpus_items WHERE event_id=?", (event_id,))
            conn.executemany(
                "INSERT INTO event_corpus_items (event_id, corpus_item_id) VALUES (?, ?)",
                ((event_id, item_id) for item_id in record.source_corpus_item_ids),
            )
            revision = self._bump_revision_in_transaction(conn)
            conn.commit()
        return {
            "event_id": event_id,
            "created": existing is None,
            "changed": True,
            "revision": revision,
        }

    def get_event(self, event_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM corpus_events WHERE event_id=?", (event_id,),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["metadata"] = json.loads(result.pop("metadata_json"))
            result["security_ids"] = [
                linked[0] for linked in conn.execute(
                    "SELECT security_id FROM event_securities WHERE event_id=? "
                    "ORDER BY security_id", (event_id,),
                ).fetchall()
            ]
            result["source_corpus_item_ids"] = [
                linked[0] for linked in conn.execute(
                    "SELECT corpus_item_id FROM event_corpus_items WHERE event_id=? "
                    "ORDER BY corpus_item_id", (event_id,),
                ).fetchall()
            ]
        return result

    def count_events(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM corpus_events").fetchone()[0])

    # ── Query Support (for middleware) ─────────────────

    # -- Security Universe (2.3.1.1) -----------------------------------------

    @staticmethod
    def _normalize_universe_symbol(symbol: object) -> str:
        """Normalize dot/slash share-class spellings to the canonical dash form."""
        import re

        value = str(symbol or "").strip().upper()
        value = re.sub(r"[./]", "-", value)
        return re.sub(r"\s+", "", value)

    @staticmethod
    def _normalize_exchange(exchange: object) -> str:
        """Normalize common exchange labels used by the three providers."""
        value = str(exchange or "").strip().upper()
        if value.startswith("NASDAQ"):
            return "NASDAQ"
        if value.startswith("NEW YORK STOCK EXCHANGE") or value == "NYSE":
            return "NYSE"
        return value

    @staticmethod
    def _normalize_company_identity(company_name: object) -> str:
        """Return a conservative comparison form for CIK-only identity matches."""
        import re

        value = re.sub(r"[^A-Z0-9]+", " ", str(company_name or "").upper())
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _share_class_parts(symbol: object) -> tuple[str, Optional[str]]:
        """Split explicit dash-suffixed share classes such as BRK-A and BRK-B."""
        import re

        value = str(symbol or "")
        match = re.fullmatch(r"(.+)-([A-Z])", value)
        return (match.group(1), match.group(2)) if match else (value, None)

    @classmethod
    def _symbols_look_like_share_classes(cls, first: object, second: object) -> bool:
        """Recognize common listed share-class symbol pairs conservatively."""
        first_value = str(first or "")
        second_value = str(second or "")
        if first_value == second_value:
            return False
        first_base, first_class = cls._share_class_parts(first_value)
        second_base, second_class = cls._share_class_parts(second_value)
        if first_base == second_base and (first_class or second_class):
            return True
        shorter, longer = sorted((first_value, second_value), key=len)
        return len(shorter) >= 2 and len(longer) == len(shorter) + 1 and longer.startswith(
            shorter
        )

    @staticmethod
    def _normalize_cik(cik: object) -> Optional[str]:
        """Return a zero-padded SEC CIK, or None when the provider omitted it."""
        value = str(cik or "").strip()
        if not value:
            return None
        if not value.isdigit():
            raise ValueError(f"invalid CIK: {value}")
        return value.zfill(10)

    @staticmethod
    def _normalize_observed_at(observed_at: object) -> tuple[str, str]:
        """Validate an ISO observation timestamp and return it plus its date."""
        value = str(observed_at or "").strip()
        if not value:
            raise ValueError("observed_at is required")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("observed_at must be an ISO date or timestamp") from exc
        return value, value[:10]

    @staticmethod
    def _universe_row_dict(row: object) -> dict:
        """Accept provider dataclasses or plain dictionaries at the Store seam."""
        if isinstance(row, dict):
            return dict(row)
        as_dict = getattr(row, "as_dict", None)
        if callable(as_dict):
            return dict(as_dict())
        raise TypeError("universe rows must be dictionaries or UniverseRecord values")

    @staticmethod
    def _record_universe_error(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        source: str,
        row: dict,
        error_code: str,
        message: str,
        observed_at: str,
    ) -> None:
        """Persist one bounded reconciliation error inside the snapshot transaction."""
        conn.execute(
            """
            INSERT INTO universe_errors (
                run_id, source, symbol, error_code, message, payload, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                source,
                str(row.get("symbol") or "")[:32],
                error_code,
                message[:500],
                json.dumps(row, sort_keys=True, default=str)[:4000],
                observed_at,
            ),
        )

    @staticmethod
    def _security_candidates(rows: list[sqlite3.Row]) -> list[dict]:
        """Deduplicate joined identity candidates by security id."""
        candidates: dict[str, dict] = {}
        for row in rows:
            item = dict(row)
            candidates[item["security_id"]] = item
        return list(candidates.values())

    def _find_universe_security(
        self,
        conn: sqlite3.Connection,
        row: dict,
        *,
        observed_date: str,
        multi_symbol_ciks: set[str],
    ) -> tuple[Optional[dict], Optional[str]]:
        """Return one safe identity match, or an explicit ambiguity code."""
        normalized = row["normalized_symbol"]
        exchange = row["normalized_exchange"]
        cik = row["cik"]
        direct = self._security_candidates(
            conn.execute(
                "SELECT * FROM securities WHERE normalized_ticker = ?",
                (normalized,),
            ).fetchall()
        )
        if len(direct) > 1 and exchange:
            direct = [item for item in direct if item["exchange"] == exchange]
        if len(direct) == 1:
            candidate = direct[0]
            conflicting_exchange = bool(
                exchange
                and candidate["exchange"]
                and candidate["exchange"] != exchange
                and (not cik or candidate.get("cik") != cik)
            )
            if not conflicting_exchange:
                return candidate, None
            direct = []
        if len(direct) > 1:
            return None, "ambiguous_symbol"

        aliases = self._security_candidates(
            conn.execute(
                """
                SELECT s.*
                FROM security_aliases a
                JOIN securities s ON s.security_id = a.security_id
                WHERE a.normalized_alias = ?
                  AND (a.provider = ? OR a.provider IS NULL)
                  AND (a.valid_from IS NULL OR a.valid_from <= ?)
                  AND (a.valid_to IS NULL OR a.valid_to >= ?)
                ORDER BY CASE WHEN a.provider = ? THEN 0 ELSE 1 END
                """,
                (
                    normalized,
                    row["source"],
                    observed_date,
                    observed_date,
                    row["source"],
                ),
            ).fetchall()
        )
        if exchange:
            aliases = [
                candidate
                for candidate in aliases
                if not candidate["exchange"]
                or candidate["exchange"] == exchange
                or (cik and candidate.get("cik") == cik)
            ]
        if len(aliases) == 1:
            return aliases[0], None
        if len(aliases) > 1:
            return None, "ambiguous_alias"

        if not cik or cik in multi_symbol_ciks:
            return None, None
        cik_rows = self._security_candidates(
            conn.execute(
                "SELECT * FROM securities WHERE cik = ? AND active = 1", (cik,)
            ).fetchall()
        )
        if len(cik_rows) > 1 and exchange:
            exchange_rows = [item for item in cik_rows if item["exchange"] == exchange]
            if exchange_rows:
                cik_rows = exchange_rows
        if len(cik_rows) == 1:
            candidate = cik_rows[0]
            different_share_classes = self._symbols_look_like_share_classes(
                candidate["normalized_ticker"], row["normalized_symbol"]
            )
            same_name = self._normalize_company_identity(
                candidate["company_name"]
            ) == self._normalize_company_identity(row["company_name"])
            same_share_class = bool(
                row.get("share_class")
                and row.get("share_class") == candidate.get("share_class")
            )
            if not different_share_classes and (same_name or same_share_class):
                return candidate, None
            return None, None
        if len(cik_rows) > 1:
            return None, "ambiguous_cik"
        return None, None

    @staticmethod
    def _ensure_universe_alias(
        conn: sqlite3.Connection,
        *,
        security_id: str,
        alias: str,
        normalized_alias: str,
        alias_type: str,
        provider: Optional[str],
        source: str,
        valid_from: Optional[str],
        valid_to: Optional[str] = None,
    ) -> bool:
        """Insert one alias when the same scoped mapping is not already present."""
        if valid_to is None:
            existing = conn.execute(
                """
                SELECT 1 FROM security_aliases
                WHERE security_id = ? AND normalized_alias = ?
                  AND alias_type = ? AND provider IS ? AND valid_to IS NULL
                """,
                (security_id, normalized_alias, alias_type, provider),
            ).fetchone()
            if not existing:
                conflicting = conn.execute(
                    """
                    SELECT 1 FROM security_aliases
                    WHERE security_id <> ? AND normalized_alias = ?
                      AND provider IS ? AND valid_to IS NULL
                    """,
                    (security_id, normalized_alias, provider),
                ).fetchone()
                if conflicting:
                    return False
        else:
            existing = conn.execute(
                """
                SELECT 1 FROM security_aliases
                WHERE security_id = ? AND normalized_alias = ?
                  AND alias_type = ? AND provider IS ?
                  AND valid_from IS ? AND valid_to IS ?
                """,
                (
                    security_id,
                    normalized_alias,
                    alias_type,
                    provider,
                    valid_from,
                    valid_to,
                ),
            ).fetchone()
        if existing:
            return False
        conn.execute(
            """
            INSERT INTO security_aliases (
                security_id, alias, normalized_alias, alias_type,
                provider, valid_from, valid_to, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                security_id,
                alias,
                normalized_alias,
                alias_type,
                provider,
                valid_from,
                valid_to,
                source,
            ),
        )
        return True

    def upsert_universe_snapshot(
        self,
        source: str,
        observed_at: str,
        rows: list[object],
    ) -> dict:
        """Reconcile one validated provider snapshot in a single transaction."""
        observed_at, observed_date = self._normalize_observed_at(observed_at)
        source = str(source or "").strip().lower()
        if not source:
            raise ValueError("source is required")
        normalized_rows = []
        for value in rows:
            row = self._universe_row_dict(value)
            symbol = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
            company_name = str(row.get("company_name") or row.get("name") or "").strip()
            normalized_symbol = self._normalize_universe_symbol(symbol)
            if not normalized_symbol or not company_name:
                raise ValueError("every universe row requires symbol and company_name")
            index_code = row.get("index_code")
            if index_code is not None and index_code not in {"sp500", "nasdaq100"}:
                raise ValueError(f"unsupported index_code: {index_code}")
            normalized_rows.append(
                {
                    **row,
                    "symbol": symbol,
                    "company_name": company_name,
                    "source": source,
                    "index_code": index_code,
                    "normalized_symbol": normalized_symbol,
                    "normalized_exchange": self._normalize_exchange(row.get("exchange")),
                    "cik": self._normalize_cik(row.get("cik")),
                    "security_type": str(row.get("security_type") or "common_stock"),
                }
            )
        symbols = [row["normalized_symbol"] for row in normalized_rows]
        if len(symbols) != len(set(symbols)):
            raise ValueError("snapshot contains duplicate normalized symbols")

        cik_symbols: dict[str, set[str]] = {}
        for row in normalized_rows:
            if row["cik"]:
                cik_symbols.setdefault(row["cik"], set()).add(row["normalized_symbol"])
        multi_symbol_ciks = {
            cik for cik, cik_tickers in cik_symbols.items() if len(cik_tickers) > 1
        }
        run_id = str(uuid.uuid4())
        counts = {
            "securities_created": 0,
            "securities_updated": 0,
            "memberships_opened": 0,
            "memberships_closed": 0,
            "aliases_created": 0,
            "errors": 0,
        }
        changed = False
        resolved_by_index: dict[str, set[str]] = {}
        index_errors: set[str] = set()

        with self._connect() as conn:
            for row in normalized_rows:
                security, error_code = self._find_universe_security(
                    conn,
                    row,
                    observed_date=observed_date,
                    multi_symbol_ciks=multi_symbol_ciks,
                )
                if error_code:
                    self._record_universe_error(
                        conn,
                        run_id=run_id,
                        source=source,
                        row=row,
                        error_code=error_code,
                        message=f"No safe identity match for {row['symbol']}: {error_code}",
                        observed_at=observed_at,
                    )
                    counts["errors"] += 1
                    if row["index_code"]:
                        index_errors.add(row["index_code"])
                    continue

                if security is None:
                    security_id = str(uuid.uuid4())
                    conn.execute(
                        """
                        INSERT INTO securities (
                            security_id, ticker, normalized_ticker, company_name,
                            exchange, cik, security_type, share_class, sector,
                            industry, active, first_seen_at, last_seen_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                        """,
                        (
                            security_id,
                            row["normalized_symbol"],
                            row["normalized_symbol"],
                            row["company_name"],
                            row["normalized_exchange"],
                            row["cik"],
                            row["security_type"],
                            row.get("share_class"),
                            row.get("sector"),
                            row.get("industry"),
                            observed_at,
                            observed_at,
                            observed_at,
                        ),
                    )
                    security = dict(
                        conn.execute(
                            "SELECT * FROM securities WHERE security_id = ?", (security_id,)
                        ).fetchone()
                    )
                    counts["securities_created"] += 1
                    changed = True
                else:
                    security_id = security["security_id"]
                    if row["cik"] and security.get("cik") not in {None, row["cik"]}:
                        self._record_universe_error(
                            conn,
                            run_id=run_id,
                            source=source,
                            row=row,
                            error_code="identity_conflict",
                            message=(
                                f"CIK {row['cik']} conflicts with stored CIK {security['cik']}"
                            ),
                            observed_at=observed_at,
                        )
                        counts["errors"] += 1
                        if row["index_code"]:
                            index_errors.add(row["index_code"])
                        continue

                    updates = {
                        "company_name": row["company_name"],
                        "exchange": row["normalized_exchange"] or security["exchange"],
                        "cik": row["cik"] or security["cik"],
                        "security_type": row["security_type"],
                        "share_class": row.get("share_class") or security["share_class"],
                        "sector": row.get("sector") or security["sector"],
                        "industry": row.get("industry") or security["industry"],
                    }
                    renamed = security["normalized_ticker"] != row["normalized_symbol"]
                    if renamed:
                        old_ticker = security["ticker"]
                        old_normalized = security["normalized_ticker"]
                        conn.execute(
                            """
                            UPDATE security_aliases SET valid_to = ?
                            WHERE security_id = ? AND normalized_alias = ?
                              AND valid_to IS NULL
                            """,
                            (observed_date, security_id, old_normalized),
                        )
                        if self._ensure_universe_alias(
                            conn,
                            security_id=security_id,
                            alias=old_ticker,
                            normalized_alias=old_normalized,
                            alias_type="former_ticker",
                            provider=None,
                            source=source,
                            valid_from=str(security["first_seen_at"])[:10],
                            valid_to=observed_date,
                        ):
                            counts["aliases_created"] += 1
                        updates["ticker"] = row["normalized_symbol"]
                        updates["normalized_ticker"] = row["normalized_symbol"]
                    metadata_changed = renamed or any(
                        security.get(key) != value for key, value in updates.items()
                    )
                    if metadata_changed:
                        updates["last_seen_at"] = observed_at
                        updates["updated_at"] = observed_at
                        assignments = ", ".join(f"{key} = ?" for key in updates)
                        conn.execute(
                            f"UPDATE securities SET {assignments} WHERE security_id = ?",
                            (*updates.values(), security_id),
                        )
                        counts["securities_updated"] += 1
                        changed = True

                if self._ensure_universe_alias(
                    conn,
                    security_id=security_id,
                    alias=row["symbol"],
                    normalized_alias=row["normalized_symbol"],
                    alias_type="vendor_symbol",
                    provider=source,
                    source=source,
                    valid_from=observed_date,
                ):
                    counts["aliases_created"] += 1
                    changed = True

                index_code = row["index_code"]
                if not index_code:
                    continue
                resolved_by_index.setdefault(index_code, set()).add(security_id)
                active_membership = conn.execute(
                    """
                    SELECT membership_id FROM security_memberships
                    WHERE security_id = ? AND index_code = ? AND active = 1
                    """,
                    (security_id, index_code),
                ).fetchone()
                if not active_membership:
                    conn.execute(
                        """
                        INSERT INTO security_memberships (
                            security_id, index_code, effective_from, active,
                            source, source_url, observed_at
                        ) VALUES (?, ?, ?, 1, ?, ?, ?)
                        """,
                        (
                            security_id,
                            index_code,
                            observed_date,
                            source,
                            row.get("source_url"),
                            observed_at,
                        ),
                    )
                    counts["memberships_opened"] += 1
                    changed = True

            for index_code, resolved_ids in resolved_by_index.items():
                if index_code in index_errors:
                    continue
                active_rows = conn.execute(
                    """
                    SELECT membership_id, security_id FROM security_memberships
                    WHERE index_code = ? AND active = 1
                    """,
                    (index_code,),
                ).fetchall()
                for membership in active_rows:
                    if membership["security_id"] in resolved_ids:
                        continue
                    conn.execute(
                        """
                        UPDATE security_memberships
                        SET active = 0, effective_to = ?
                        WHERE membership_id = ?
                        """,
                        (observed_date, membership["membership_id"]),
                    )
                    counts["memberships_closed"] += 1
                    changed = True

            if changed:
                conn.execute(
                    """
                    INSERT INTO store_revision (id, revision, updated_at)
                    VALUES (1, 1, datetime('now'))
                    ON CONFLICT(id) DO UPDATE SET
                        revision = revision + 1, updated_at = datetime('now')
                    """
                )
            revision_row = conn.execute(
                "SELECT revision FROM store_revision WHERE id = 1"
            ).fetchone()
            revision = int(revision_row[0]) if revision_row else 0

        return {
            "run_id": run_id,
            "source": source,
            "changed": changed,
            **counts,
            "revision": revision,
        }

    def list_securities(
        self,
        index: Optional[str] = None,
        active: Optional[bool] = True,
        sector: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List bounded canonical securities with optional membership filters."""
        self._validate_inventory_page(limit, offset)
        conditions = []
        params: list = []
        join = ""
        if index is not None:
            if index not in {"sp500", "nasdaq100"}:
                raise ValueError(f"unsupported index: {index}")
            join = "JOIN security_memberships m ON m.security_id = s.security_id"
            conditions.append("m.index_code = ?")
            params.append(index)
            if active is not None:
                conditions.append("m.active = ?")
                params.append(int(active))
        elif active is not None:
            conditions.append("s.active = ?")
            params.append(int(active))
        if sector is not None:
            conditions.append("s.sector = ?")
            params.append(sector)
        where = " AND ".join(conditions) or "1=1"
        sql = f"""
            SELECT DISTINCT s.* FROM securities s {join}
            WHERE {where}
            ORDER BY s.ticker
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def get_security(self, ticker_or_id: str) -> Optional[dict]:
        """Return one canonical security by opaque id or normalized ticker."""
        value = str(ticker_or_id or "").strip()
        if not value:
            return None
        normalized = self._normalize_universe_symbol(value)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM securities
                WHERE security_id = ? OR normalized_ticker = ?
                ORDER BY active DESC, updated_at DESC
                LIMIT 1
                """,
                (value, normalized),
            ).fetchone()
        return dict(row) if row else None

    def resolve_security(
        self,
        symbol: str,
        provider: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve a canonical or aliased symbol, optionally at a historical date."""
        normalized = self._normalize_universe_symbol(symbol)
        if not normalized:
            return None
        as_of_date = self._validate_as_of(as_of) if as_of is not None else None
        with self._connect() as conn:
            if as_of_date is None:
                direct = conn.execute(
                    """
                    SELECT * FROM securities WHERE normalized_ticker = ?
                    ORDER BY active DESC, updated_at DESC
                    """,
                    (normalized,),
                ).fetchall()
            else:
                direct_conditions = [
                    "s.normalized_ticker = ?",
                    "a.normalized_alias = ?",
                    "(a.valid_from IS NULL OR a.valid_from <= ?)",
                    "(a.valid_to IS NULL OR a.valid_to >= ?)",
                ]
                direct_params: list = [
                    normalized,
                    normalized,
                    as_of_date,
                    as_of_date,
                ]
                if provider is not None:
                    direct_conditions.append("(a.provider = ? OR a.provider IS NULL)")
                    direct_params.append(provider.lower())
                direct = conn.execute(
                    f"""
                    SELECT s.* FROM securities s
                    JOIN security_aliases a ON a.security_id = s.security_id
                    WHERE {' AND '.join(direct_conditions)}
                    ORDER BY s.active DESC, s.updated_at DESC
                    """,
                    direct_params,
                ).fetchall()
            direct_candidates = self._security_candidates(direct)
            if len(direct_candidates) == 1:
                return direct_candidates[0]

            conditions = ["a.normalized_alias = ?"]
            params: list = [normalized]
            if provider is not None:
                conditions.append("(a.provider = ? OR a.provider IS NULL)")
                params.append(provider.lower())
            if as_of_date is not None:
                conditions.extend(
                    [
                        "(a.valid_from IS NULL OR a.valid_from <= ?)",
                        "(a.valid_to IS NULL OR a.valid_to >= ?)",
                    ]
                )
                params.extend([as_of_date, as_of_date])
            else:
                conditions.append("a.valid_to IS NULL")
            order = ""
            if provider is not None:
                order = "ORDER BY CASE WHEN a.provider = ? THEN 0 ELSE 1 END"
                params.append(provider.lower())
            alias_rows = conn.execute(
                f"""
                SELECT s.* FROM security_aliases a
                JOIN securities s ON s.security_id = a.security_id
                WHERE {' AND '.join(conditions)}
                {order}
                """,
                params,
            ).fetchall()
        candidates = self._security_candidates(alias_rows)
        return candidates[0] if len(candidates) == 1 else None

    def resolve_exact_security(self, identifier: str) -> Optional[dict]:
        """Resolve one exact ticker, registry alias, or company-name identifier.

        Company names are compared after case/whitespace/punctuation normalization
        only; no substring or fuzzy matching is performed.  Multiple exact
        candidates intentionally return ``None`` so an official event remains
        unattached for review.
        """
        value = str(identifier or "").strip()
        if not value:
            return None
        symbol_key = self._normalize_universe_symbol(value)
        company_key = self._normalize_company_identity(value)
        with self._connect() as conn:
            candidates: dict[str, dict] = {}
            for row in conn.execute(
                "SELECT * FROM securities WHERE normalized_ticker=? OR normalized_ticker=?",
                (symbol_key, value.upper()),
            ).fetchall():
                candidates[str(row["security_id"])] = dict(row)
            for row in conn.execute("SELECT * FROM securities").fetchall():
                if self._normalize_company_identity(row["company_name"]) == company_key:
                    candidates[str(row["security_id"])] = dict(row)
            aliases = conn.execute(
                "SELECT a.alias, a.normalized_alias, s.* FROM security_aliases a "
                "JOIN securities s ON s.security_id=a.security_id "
                "WHERE a.valid_to IS NULL"
            ).fetchall()
            for row in aliases:
                if (
                    self._normalize_universe_symbol(row["alias"]) == symbol_key
                    or self._normalize_company_identity(row["alias"]) == company_key
                ):
                    candidates[str(row["security_id"])] = {
                        key: row[key] for key in row.keys() if key not in {"alias", "normalized_alias"}
                    }
        return next(iter(candidates.values())) if len(candidates) == 1 else None

    def register_security_alias(
        self,
        security_id: str,
        alias: str,
        *,
        alias_type: str = "issuer_alias",
        provider: Optional[str] = None,
        source: str = "registry",
    ) -> bool:
        """Register one exact issuer/manufacturer/UEI identity in the registry."""
        allowed = {"issuer_alias", "manufacturer", "recipient_uei", "vendor_symbol"}
        alias_type = str(alias_type or "issuer_alias").strip().lower()
        value = str(alias or "").strip()
        if alias_type not in allowed:
            raise ValueError(f"unsupported security alias type: {alias_type}")
        if not value or not security_id:
            raise ValueError("security_id and alias are required")
        normalized = self._normalize_company_identity(value)
        with self._connect() as conn:
            if not conn.execute(
                "SELECT 1 FROM securities WHERE security_id=?", (security_id,)
            ).fetchone():
                raise ValueError(f"unknown security_id: {security_id}")
            existing = conn.execute(
                "SELECT 1 FROM security_aliases WHERE security_id=? "
                "AND alias=? AND alias_type=? AND provider IS ? AND valid_to IS NULL",
                (security_id, value, alias_type, provider),
            ).fetchone()
            if existing:
                return False
            conflicting = conn.execute(
                "SELECT 1 FROM security_aliases WHERE security_id<>? "
                "AND normalized_alias=? AND provider IS ? AND valid_to IS NULL",
                (security_id, normalized, provider),
            ).fetchone()
            if conflicting:
                # Preserve the exact ambiguity for review rather than selecting a
                # ticker.  The resolver will return None for multiple candidates.
                return False
            conn.execute(
                "INSERT INTO security_aliases ("
                "security_id, alias, normalized_alias, alias_type, provider, source"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (security_id, value, normalized, alias_type, provider, source),
            )
            conn.commit()
        return True

    def list_memberships(
        self,
        security_id: Optional[str] = None,
        index_code: Optional[str] = None,
        active: Optional[bool] = None,
    ) -> list[dict]:
        """List current or historical index membership rows."""
        conditions = []
        params: list = []
        if security_id is not None:
            conditions.append("m.security_id = ?")
            params.append(security_id)
        if index_code is not None:
            if index_code not in {"sp500", "nasdaq100"}:
                raise ValueError(f"unsupported index_code: {index_code}")
            conditions.append("m.index_code = ?")
            params.append(index_code)
        if active is not None:
            conditions.append("m.active = ?")
            params.append(int(active))
        where = " AND ".join(conditions) or "1=1"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT m.*, s.ticker, s.company_name
                FROM security_memberships m
                JOIN securities s ON s.security_id = m.security_id
                WHERE {where}
                ORDER BY m.index_code, m.effective_from, s.ticker
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def list_universe_errors(self, run_id: str) -> list[dict]:
        """Return reconciliation errors for one universe refresh run."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT error_id, run_id, source, symbol, error_code,
                       message, payload, observed_at, created_at
                FROM universe_errors WHERE run_id = ? ORDER BY error_id
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_metrics(self, ticker: Optional[str] = None) -> list[str]:
        """Return the distinct metric names present in `fundamentals`, optionally scoped to a ticker."""
        if ticker:
            sql = "SELECT DISTINCT metric FROM fundamentals WHERE ticker=? ORDER BY metric"
            params = (ticker,)
        else:
            sql = "SELECT DISTINCT metric FROM fundamentals ORDER BY metric"
            params = ()
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [row["metric"] for row in rows]

    def list_tickers(self) -> list[str]:
        """Return active canonical tickers, falling back to legacy facts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker FROM securities WHERE active = 1 ORDER BY ticker"
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    "SELECT DISTINCT ticker FROM fundamentals ORDER BY ticker"
                ).fetchall()
            return [row["ticker"] for row in rows]

    _ORDER_SQL = {"asc": "ASC", "desc": "DESC"}
    _OP_SQL = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">=", "eq": "=", "ne": "!="}

    def query_metric(self, metric: str, tickers: Optional[list[str]] = None,
                     order: str = "asc", limit: int = 10, op: Optional[str] = None,
                     value: Optional[float] = None, latest_only: bool = True,
                     exclude: Optional[list[str]] = None,
                     sane_range: Optional[tuple] = None) -> list[dict]:
        """Rank/filter `fundamentals` rows for one metric via parameterized SQL.

        `order` and `op` are only ever used to select whitelisted SQL fragments
        (`_ORDER_SQL` / `_OP_SQL`) — caller-supplied strings are never
        interpolated directly into the query.
        """
        order_sql = self._ORDER_SQL.get(order, "ASC")
        limit = max(1, min(int(limit), 100))

        conditions = ["f_outer.metric=?"]
        params: list = [metric]

        if latest_only:
            conditions.append(
                "f_outer.period = (SELECT MAX(f_inner.period) FROM fundamentals AS f_inner "
                "WHERE f_inner.ticker=f_outer.ticker AND f_inner.metric=f_outer.metric)"
            )

        if tickers:
            placeholders = ",".join("?" for _ in tickers)
            conditions.append(f"f_outer.ticker IN ({placeholders})")
            params.extend(t.upper() for t in tickers)

        if exclude:
            placeholders = ",".join("?" for _ in exclude)
            conditions.append(f"f_outer.ticker NOT IN ({placeholders})")
            params.extend(exclude)

        op_sql = self._OP_SQL.get(op) if op else None
        if op_sql and value is not None:
            conditions.append(f"f_outer.value {op_sql} ?")
            params.append(value)

        if sane_range:
            low, high = sane_range
            if low is not None:
                conditions.append("f_outer.value >= ?")
                params.append(low)
            if high is not None:
                conditions.append("f_outer.value <= ?")
                params.append(high)

        conditions.append("f_outer.value IS NOT NULL")

        sql = f"""
            SELECT f_outer.ticker, f_outer.value, f_outer.period, f_outer.unit
            FROM fundamentals AS f_outer
            WHERE {' AND '.join(conditions)}
            ORDER BY f_outer.value {order_sql}
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def search_facts(self, ticker: str = None, metric: str = None,
                     source: str = None, limit: int = 10) -> list[dict]:
        """Flexible fact search — used by the middleware."""
        conditions = []
        params = []
        if ticker:
            conditions.append("ticker=?")
            params.append(ticker)
        if metric:
            conditions.append("metric LIKE ?")
            params.append(f"%{metric}%")
        if source:
            conditions.append("source_type=?")
            params.append(source)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM fundamentals WHERE {where} ORDER BY ingested_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    # ── Corpus explorer inventory reads (2.2.7.2) ─────────────────────────

    @classmethod
    def _validate_inventory_page(cls, limit: int, offset: int = 0) -> None:
        """Reject unbounded inventory pages before executing SQL."""
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if not 1 <= limit <= cls.MAX_INVENTORY_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {cls.MAX_INVENTORY_LIMIT}")
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise ValueError("offset must be an integer")
        if not 0 <= offset <= cls.MAX_INVENTORY_OFFSET:
            raise ValueError(
                f"offset must be between 0 and {cls.MAX_INVENTORY_OFFSET}")

    def get_source_counts(self, *, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return bounded counts by stored logical source, without document bodies."""
        self._validate_inventory_page(limit, offset)
        sql = """
            WITH inventory(source) AS (
                SELECT source_type FROM fundamentals
                UNION ALL SELECT 'sec_companyfacts' FROM sec_companyfacts
                UNION ALL SELECT 'sec_filings' FROM filings
                UNION ALL
                    SELECT source FROM cache_meta WHERE ticker <> 'SCHEDULER'
            )
            SELECT source, COUNT(*) AS count
            FROM inventory
            WHERE source IS NOT NULL AND source <> ''
            GROUP BY source
            ORDER BY source
            LIMIT ? OFFSET ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit, offset)).fetchall()
        return [{"source": row["source"], "count": int(row["count"])} for row in rows]

    def get_ticker_counts(self, *, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return bounded ticker coverage counts and source memberships."""
        self._validate_inventory_page(limit, offset)
        sql = """
            WITH inventory(ticker, source) AS (
                SELECT ticker, source_type FROM fundamentals
                UNION ALL SELECT ticker, 'sec_companyfacts' FROM sec_companyfacts
                UNION ALL SELECT ticker, 'sec_filings' FROM filings
                UNION ALL
                    SELECT ticker, source FROM cache_meta WHERE ticker <> 'SCHEDULER'
            ), grouped AS (
                SELECT ticker, source, COUNT(*) AS source_count
                FROM inventory
                WHERE ticker IS NOT NULL AND ticker <> ''
                GROUP BY ticker, source
            )
            SELECT ticker, SUM(source_count) AS record_count,
                   GROUP_CONCAT(source || ':' || source_count) AS source_counts
            FROM grouped
            GROUP BY ticker
            ORDER BY ticker
            LIMIT ? OFFSET ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit, offset)).fetchall()

        result = []
        for row in rows:
            source_counts = {}
            for item in str(row["source_counts"] or "").split(","):
                if ":" not in item:
                    continue
                source, count = item.rsplit(":", 1)
                try:
                    source_counts[source] = int(count)
                except ValueError:
                    continue
            result.append({
                "ticker": row["ticker"].upper(),
                "record_count": int(row["record_count"] or 0),
                "sources": sorted(source_counts),
                "source_counts": source_counts,
            })
        return result

    def search_corpus_metrics(
        self,
        query: Optional[str] = None,
        *,
        ticker: Optional[str] = None,
        unit: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Search canonical metrics and SEC concepts with safe SQL filters."""
        self._validate_inventory_page(limit, offset)
        query_text = str(query or "").strip()
        conditions = []
        params: list = []
        if query_text:
            pattern = f"%{query_text}%"
            conditions.append("(metric LIKE ? OR concept LIKE ? OR label LIKE ?)")
            params.extend([pattern, pattern, pattern])
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper())
        if unit:
            conditions.append("unit = ?")
            params.append(unit)
        where = " AND ".join(conditions) or "1=1"
        sql = f"""
            WITH metric_rows AS (
                SELECT metric AS metric, NULL AS concept, NULL AS label,
                       ticker, unit, source_type AS source, COUNT(*) AS observation_count
                FROM fundamentals
                GROUP BY metric, ticker, unit, source_type
                UNION ALL
                SELECT concept AS metric, concept, label,
                       ticker, unit, 'sec_companyfacts' AS source,
                       COUNT(*) AS observation_count
                FROM sec_companyfacts
                GROUP BY concept, label, ticker, unit
            )
            SELECT metric, concept, label, ticker, unit, source, observation_count
            FROM metric_rows
            WHERE {where}
            ORDER BY metric, ticker, unit, source
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "metric": row["metric"],
                "concept": row["concept"],
                "label": row["label"],
                "ticker": row["ticker"].upper(),
                "unit": row["unit"],
                "source": row["source"],
                "observation_count": int(row["observation_count"] or 0),
            }
            for row in rows
        ]

    def search_corpus_facts(
        self,
        query: Optional[str] = None,
        *,
        ticker: Optional[str] = None,
        unit: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Return bounded fact metadata for corpus search and detail reads."""
        self._validate_inventory_page(limit, offset)
        query_text = str(query or "").strip()
        conditions = []
        params: list = []
        if query_text:
            pattern = f"%{query_text}%"
            conditions.append("(metric LIKE ? OR concept LIKE ? OR ticker LIKE ?)")
            params.extend([pattern, pattern, pattern])
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper())
        if unit:
            conditions.append("unit = ?")
            params.append(unit)
        where = " AND ".join(conditions) or "1=1"
        sql = f"""
            WITH facts AS (
                SELECT 'fundamental' AS record_kind, id AS record_id,
                       ticker, metric, NULL AS concept, value,
                       CAST(value AS TEXT) AS value_text, unit, period,
                       ingested_at AS as_of, source_type AS source, source_url,
                       NULL AS accession, NULL AS filed_at
                FROM fundamentals
                UNION ALL
                SELECT 'companyfact' AS record_kind, id AS record_id,
                       ticker, concept AS metric, concept, value_numeric AS value,
                       value_text, unit, period_end AS period, ingested_at AS as_of,
                       'sec_companyfacts' AS source, source_url, accession, filed_at
                FROM sec_companyfacts
            )
            SELECT record_kind, record_id, ticker, metric, concept, value,
                   value_text, unit, period, as_of, source, source_url,
                   accession, filed_at
            FROM facts
            WHERE {where}
            ORDER BY ticker, metric, period DESC, record_kind, record_id
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "record_kind": row["record_kind"],
                "record_id": int(row["record_id"]),
                "ticker": row["ticker"].upper(),
                "metric": row["metric"],
                "concept": row["concept"],
                "value": row["value"],
                "value_text": row["value_text"],
                "unit": row["unit"],
                "period": row["period"],
                "as_of": row["as_of"],
                "source": row["source"],
                "source_url": row["source_url"],
                "accession": row["accession"],
                "filed_at": row["filed_at"],
            }
            for row in rows
        ]

    def get_corpus_fact(self, record_kind: str, record_id: int) -> Optional[dict]:
        """Read one bounded fact record by an allowlisted internal kind/id."""
        if record_kind not in {"fundamental", "companyfact"}:
            raise ValueError("unsupported fact kind")
        try:
            record_id = int(record_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("record_id must be an integer") from exc
        if record_id < 1:
            raise ValueError("record_id must be positive")
        if record_kind == "fundamental":
            sql = """
                SELECT id, ticker, metric, value, CAST(value AS TEXT) AS value_text,
                       unit, period, ingested_at AS as_of, source_type AS source,
                       source_url, NULL AS accession, NULL AS filed_at
                FROM fundamentals WHERE id = ?
            """
        else:
            sql = """
                SELECT id, ticker, concept AS metric, value_numeric AS value,
                       value_text, unit, period_end AS period, ingested_at AS as_of,
                       'sec_companyfacts' AS source, source_url, accession, filed_at
                FROM sec_companyfacts WHERE id = ?
            """
        with self._connect() as conn:
            row = conn.execute(sql, (record_id,)).fetchone()
        if row is None:
            return None
        return {
            "record_kind": record_kind,
            "record_id": int(row["id"]),
            "ticker": row["ticker"].upper(),
            "metric": row["metric"],
            "value": row["value"],
            "value_text": row["value_text"],
            "unit": row["unit"],
            "period": row["period"],
            "as_of": row["as_of"],
            "source": row["source"],
            "source_url": row["source_url"],
            "accession": row["accession"],
            "filed_at": row["filed_at"],
        }

    def list_filings(
        self,
        *,
        query: Optional[str] = None,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List bounded filing metadata without returning local file paths."""
        self._validate_inventory_page(limit, offset)
        conditions = []
        params: list = []
        if query:
            pattern = f"%{str(query).strip()}%"
            conditions.append(
                "(accession LIKE ? OR ticker LIKE ? OR filing_type LIKE ? OR period LIKE ?)"
            )
            params.extend([pattern] * 4)
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper())
        if filing_type:
            conditions.append("filing_type = ?")
            params.append(filing_type)
        if date_from:
            conditions.append("filing_date >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("filing_date <= ?")
            params.append(date_to)
        where = " AND ".join(conditions) or "1=1"
        sql = f"""
            SELECT id, ticker, filing_type, filing_date, period, accession,
                   source_url, status, parsed_at, summary_embedding_id,
                   index_error, index_section_count, index_chunk_count, ingested_at
            FROM filings
            WHERE {where}
            ORDER BY filing_date DESC, accession DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_filing(self, accession: str) -> Optional[dict]:
        """Read one filing's safe metadata by accession."""
        sql = """
            SELECT id, ticker, filing_type, filing_date, period, accession,
                   source_url, status, parsed_at, summary_embedding_id,
                   index_error, index_section_count, index_chunk_count, ingested_at
            FROM filings WHERE accession = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (accession,)).fetchone()
        return dict(row) if row else None

    def list_freshness(
        self,
        *,
        ticker: Optional[str] = None,
        source: Optional[str] = None,
        include_scheduler: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Return bounded cache freshness rows, optionally including scheduler rows."""
        self._validate_inventory_page(limit, offset)
        conditions = []
        params: list = []
        if not include_scheduler:
            conditions.append("ticker <> 'SCHEDULER'")
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper())
        if source:
            conditions.append("source = ?")
            params.append(source)
        where = " AND ".join(conditions) or "1=1"
        sql = f"""
            SELECT ticker, source, metric_scope, last_updated,
                   next_scheduled_update, status, error_message
            FROM cache_meta
            WHERE {where}
            ORDER BY ticker, source, metric_scope
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def list_scheduler_sources(self, *, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return scheduler cadence rows without invoking the scheduler."""
        self._validate_inventory_page(limit, offset)
        sql = """
            SELECT source, last_updated AS last_run, next_scheduled_update,
                   status, error_message
            FROM cache_meta
            WHERE ticker = 'SCHEDULER' AND source LIKE 'unified:%'
            ORDER BY source
            LIMIT ? OFFSET ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit, offset)).fetchall()
        return [dict(row) for row in rows]

    # Descriptive aliases keep the explorer seam discoverable without exposing
    # a second implementation or an arbitrary SQL interface.
    search_metrics = search_corpus_metrics
    search_facts_for_corpus = search_corpus_facts
    list_filing_inventory = list_filings
    get_freshness_summaries = list_freshness

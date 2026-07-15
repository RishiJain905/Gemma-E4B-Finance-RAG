-- Phase 2.3 cursors, bounded runs, circuit/cooldown state, and backfill progress.
CREATE TABLE IF NOT EXISTS sec_daily_indexes (
    index_date TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processed')),
    registered_count INTEGER NOT NULL DEFAULT 0,
    processed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

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

CREATE TABLE IF NOT EXISTS source_circuit_state (
    source TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'closed',
    failure_count INTEGER NOT NULL DEFAULT 0,
    opened_at TEXT,
    cooldown_until TEXT,
    error_class TEXT,
    error_message TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS phase2_3_backfill_progress (
    stage TEXT PRIMARY KEY,
    cursor_value TEXT,
    completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
    rows_processed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);


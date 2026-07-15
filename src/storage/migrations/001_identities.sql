-- Phase 2.3 canonical securities, aliases, memberships, and review errors.
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
    alias_type TEXT NOT NULL CHECK (alias_type IN (
        'ticker', 'vendor_symbol', 'former_ticker', 'issuer_alias',
        'manufacturer', 'recipient_uei'
    )),
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

CREATE TABLE IF NOT EXISTS identity_reconciliation_errors (
    error_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    legacy_table TEXT NOT NULL,
    legacy_row_id TEXT,
    identifier TEXT,
    issue_type TEXT NOT NULL CHECK (issue_type IN ('orphan', 'ambiguous', 'conflict')),
    candidates_json TEXT NOT NULL DEFAULT '[]',
    details_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);


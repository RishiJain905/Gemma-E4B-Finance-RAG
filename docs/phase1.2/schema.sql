-- ============================================================
-- Schema: finance.db
-- Purpose: Structured financial fact storage for Gemma-E4B-Finance-RAG
-- Date: 2026-05-27
-- ============================================================

-- ── Financial Fundamentals ──────────────────────────────
-- Stores individual financial metrics per ticker per period.
-- This is the primary data store for quantitative facts.
CREATE TABLE IF NOT EXISTS fundamentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,                            -- 'NVDA', 'AMD', etc.
    metric TEXT NOT NULL,                            -- 'revenue_q1_2026', 'pe_ratio_ttm', 'gross_margin'
    value REAL,                                      -- The numeric value
    unit TEXT DEFAULT 'usd',                         -- 'usd', 'percent', 'ratio', 'shares'
    period TEXT,                                     -- '2026-Q1', '2025-FY', '2026-05-15'
    period_type TEXT DEFAULT 'quarterly',            -- 'quarterly', 'annual', 'ttm', 'daily', 'point_in_time'
    source_type TEXT NOT NULL,                       -- 'sec', 'yfinance', 'fred', 'earnings_call'
    source_url TEXT,                                 -- Direct URL to the source document
    source_accessed_at TEXT,                         -- ISO timestamp when we fetched this
    ingested_at TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, metric, period)                   -- One fact per ticker per metric per period
);

-- â”€â”€ SEC CompanyFacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
-- Tracks which SEC filings / documents have been processed.
-- Prevents re-processing the same document on subsequent runs.
CREATE TABLE IF NOT EXISTS filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    filing_type TEXT NOT NULL,                       -- '10-K', '10-Q', '8-K', 'earnings_call', 'press_release'
    filing_date TEXT,                                -- Date of filing
    period TEXT,                                     -- Period covered (for 10-Q: '2026-Q1')
    accession TEXT UNIQUE,                           -- SEC accession number (or doc hash for other sources)
    source_url TEXT,                                 -- EDGAR URL or source link
    file_path TEXT,                                  -- Local cached copy path
    status TEXT DEFAULT 'unprocessed',               -- 'unprocessed', 'index_pending', 'parsed', 'failed'
    parsed_at TEXT,                                  -- When TraceAlchemy finished parsing
    summary_embedding_id TEXT,                       -- Link to ChromaDB embedding
    index_error TEXT,                                -- Retryable filing-text index failure reason
    index_section_count INTEGER DEFAULT 0,           -- Verified section parents written
    index_chunk_count INTEGER DEFAULT 0,             -- Verified child chunks written
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ── Cache Freshness ────────────────────────────────
-- Tracks when each ticker's data was last updated per source.
-- The middleware checks this before serving cached data.
CREATE TABLE IF NOT EXISTS cache_meta (
    ticker TEXT NOT NULL,
    source TEXT NOT NULL,                            -- 'sec', 'yfinance', 'fred', 'gdelt'
    metric_scope TEXT DEFAULT 'all',                 -- 'all', 'fundamentals', 'filings', 'news'
    last_updated TEXT,                               -- When we last fetched this
    next_scheduled_update TEXT,                      -- When we should fetch again
    status TEXT DEFAULT 'fresh',                     -- 'fresh', 'stale', 'fetching', 'error'
    error_message TEXT,
    PRIMARY KEY (ticker, source, metric_scope)
);

-- ── Store Revision (2.2.6.2) ───────────────────────
-- Single-row monotonic data revision. Every Store facade mutation that can
-- change model-visible facts/documents bumps `revision` BEFORE the mutation
-- begins, so the versioned retrieval cache (src/middleware/retrieval_cache.py)
-- can key on it and never serve evidence from before an ingestion write. A
-- failed mutation may leave the revision advanced (an extra cache miss) but can
-- never leave a stale cache entry valid. Same-count document replacements bump
-- the revision even though Chroma's document count is unchanged, so count alone
-- is never used for invalidation.
CREATE TABLE IF NOT EXISTS store_revision (
    id INTEGER PRIMARY KEY CHECK (id = 1),           -- single row, always id=1
    revision INTEGER NOT NULL DEFAULT 0,             -- monotonically increasing
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO store_revision (id, revision) VALUES (1, 0);

-- ── Ingestion Log ──────────────────────────────────
-- Audit trail of every ingestion run.
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,                            -- UUID for this ingestion run
    ticker TEXT,
    source TEXT NOT NULL,                            -- 'sec', 'yfinance', 'fred', 'gdelt'
    status TEXT NOT NULL,                            -- 'started', 'completed', 'failed', 'skipped'
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
        alias_type IN ('ticker', 'vendor_symbol', 'former_ticker')
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

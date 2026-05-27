"""
src/storage/sqlite_store.py
SQLite storage layer for structured financial data.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class SQLiteStore:
    """Handles all SQLite operations for structured financial data."""

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
            with open(self.SCHEMA_SQL) as f:
                sql = f.read()
        else:
            sql = self._inline_schema()

        with self._connect() as conn:
            conn.executescript(sql)
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
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_status ON filings(status);
CREATE INDEX IF NOT EXISTS idx_cache_meta_status ON cache_meta(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_run ON ingestion_log(run_id);
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

    # ── Filing Tracking ───────────────────────────────

    def register_filing(self, ticker: str, filing_type: str,
                        filing_date: str, period: str,
                        accession: str, source_url: str) -> bool:
        """Register a filing as processed. Returns True if new, False if duplicate."""
        sql = """
        INSERT OR IGNORE INTO filings
            (ticker, filing_type, filing_date, period, accession, source_url, status)
        VALUES (?, ?, ?, ?, ?, ?, 'unprocessed')
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, (ticker, filing_type, filing_date, period, accession, source_url))
            conn.commit()
            return cursor.rowcount > 0

    def mark_filing_parsed(self, accession: str, embedding_id: str = None):
        """Mark a filing as successfully parsed by TraceAlchemy."""
        sql = """
        UPDATE filings SET status='parsed', parsed_at=datetime('now'), summary_embedding_id=?
        WHERE accession=?
        """
        with self._connect() as conn:
            conn.execute(sql, (embedding_id, accession))
            conn.commit()

    def get_unprocessed_filings(self, limit: int = 10) -> list[dict]:
        """Get filings that haven't been parsed yet."""
        sql = (
            "SELECT * FROM filings WHERE status='unprocessed' "
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
        UPDATE cache_meta SET status='stale', error_message=?
        WHERE ticker=? AND source=?
        """
        with self._connect() as conn:
            conn.execute(sql, (error, ticker, source))
            conn.commit()

    def get_stale_cache_entries(self, limit: int = 20) -> list[dict]:
        """Find entries past their scheduled update time."""
        sql = (
            "SELECT * FROM cache_meta WHERE next_scheduled_update < datetime('now') "
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

    # ── Query Support (for middleware) ─────────────────

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

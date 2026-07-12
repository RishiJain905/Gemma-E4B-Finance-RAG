"""
src/storage/sqlite_store.py
SQLite storage layer for structured financial data.
"""

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
            with open(self.SCHEMA_SQL, encoding="utf-8") as f:
                sql = f.read()
        else:
            sql = self._inline_schema()

        with self._connect() as conn:
            conn.executescript(sql)
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
            }
            for column, statement in migrations.items():
                if column not in existing:
                    conn.execute(statement)
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

    # ── Query Support (for middleware) ─────────────────

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
        """Return the distinct tickers present in `fundamentals`."""
        sql = "SELECT DISTINCT ticker FROM fundamentals ORDER BY ticker"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
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

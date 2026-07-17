"""src/storage/migrations.py
Ordered standard-library SQLite migrations for the finance store.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).with_name("migrations")
MIGRATION_FILE_RE = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")


@dataclass(frozen=True)
class Migration:
    """One immutable numbered SQL migration."""

    version: int
    name: str
    path: Path
    sql: str
    checksum: str


# SQLite has no portable ``ADD COLUMN IF NOT EXISTS``. These additive columns
# are associated with their owning migration and checked through table_info
# inside the same transaction as the SQL file.
ADDITIVE_COLUMNS: dict[int, tuple[tuple[str, str, str], ...]] = {
    1: (
        ("fundamentals", "security_id", "TEXT REFERENCES securities(security_id)"),
        ("sec_companyfacts", "security_id", "TEXT REFERENCES securities(security_id)"),
        ("filings", "security_id", "TEXT REFERENCES securities(security_id)"),
        ("cache_meta", "security_id", "TEXT REFERENCES securities(security_id)"),
    ),
    2: (
        ("corpus_items", "document_family_id", "TEXT"),
        ("corpus_items", "narrative_bytes", "INTEGER NOT NULL DEFAULT 0"),
        ("corpus_items", "metadata_bytes", "INTEGER NOT NULL DEFAULT 0"),
        ("corpus_items", "is_tombstone", "INTEGER NOT NULL DEFAULT 0"),
        ("corpus_items", "retired_at", "TEXT"),
        ("corpus_items", "retention_reason", "TEXT"),
    ),
    4: (
        ("filings", "index_error", "TEXT"),
        ("filings", "index_section_count", "INTEGER DEFAULT 0"),
        ("filings", "index_chunk_count", "INTEGER DEFAULT 0"),
        ("filings", "cik", "TEXT"),
        ("filings", "primary_document", "TEXT"),
        ("filings", "discovery_scope", "TEXT DEFAULT 'deep'"),
        ("filings", "items_json", "TEXT DEFAULT '[]'"),
        ("filings", "exhibits_json", "TEXT DEFAULT '[]'"),
        ("source_cursors", "last_successful_at", "TEXT"),
        ("bootstrap_partitions", "new_items", "INTEGER NOT NULL DEFAULT 0"),
        ("bootstrap_partitions", "updated_items", "INTEGER NOT NULL DEFAULT 0"),
        ("bootstrap_partitions", "duplicates", "INTEGER NOT NULL DEFAULT 0"),
    ),
}


def discover_migrations(directory: Optional[Path] = None) -> tuple[Migration, ...]:
    """Load ordered migration files and reject gaps or duplicate versions."""
    root = Path(directory) if directory is not None else MIGRATIONS_DIR
    migrations: list[Migration] = []
    for path in sorted(root.glob("*.sql")):
        match = MIGRATION_FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid migration filename: {path.name}")
        sql = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=int(match.group(1)),
                name=match.group(2),
                path=path,
                sql=sql,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    versions = [item.version for item in migrations]
    expected = list(range(1, len(migrations) + 1))
    if versions != expected:
        raise ValueError(f"migration versions must be contiguous: {versions}")
    return tuple(migrations)


MIGRATIONS = discover_migrations()
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version if MIGRATIONS else 0


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def fts5_available(conn: sqlite3.Connection) -> bool:
    """Probe FTS5 without leaving schema objects behind."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(value)")
        conn.execute("DROP TABLE temp.__fts5_probe")
        return True
    except sqlite3.DatabaseError:
        return False


def _ensure_columns(conn: sqlite3.Connection, version: int) -> None:
    for table, column, definition in ADDITIVE_COLUMNS.get(version, ()):
        if not _table_exists(conn, table):
            continue
        existing = {
            str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')
        }
        if column not in existing:
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {definition}')


def _statements(sql: str) -> Iterable[str]:
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                yield statement
            buffer = ""
    if buffer.strip():
        raise ValueError("incomplete SQL statement in migration")


def apply_migrations(
    conn: sqlite3.Connection,
    *,
    directory: Optional[Path] = None,
) -> list[int]:
    """Apply pending migrations in order and return applied version numbers."""
    migrations = discover_migrations(directory) if directory is not None else MIGRATIONS
    conn.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    applied_rows = conn.execute(
        "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    applied = {int(row[0]): (str(row[1]), str(row[2])) for row in applied_rows}
    known = {migration.version: migration for migration in migrations}
    unknown = sorted(set(applied) - set(known))
    if unknown:
        raise RuntimeError(f"database has unknown migration versions: {unknown}")
    for version, (name, checksum) in applied.items():
        migration = known[version]
        if (name, checksum) != (migration.name, migration.checksum):
            raise RuntimeError(f"migration {version:03d} checksum/name mismatch")

    completed: list[int] = []
    for migration in migrations:
        if migration.version in applied:
            continue
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_columns(conn, migration.version)
            supports_fts5 = fts5_available(conn)
            for statement in _statements(migration.sql):
                if "USING fts5" in statement and not supports_fts5:
                    logger.warning(
                        "SQLite FTS5 is unavailable; lexical index will run degraded"
                    )
                    continue
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, checksum) VALUES (?, ?, ?)",
                (migration.version, migration.name, migration.checksum),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        logger.info("Applied SQLite migration %03d_%s", migration.version, migration.name)
        completed.append(migration.version)
    return completed


def schema_signature(conn: sqlite3.Connection) -> tuple[tuple[object, ...], ...]:
    """Return a formatting-independent signature of tables, columns, and indexes."""
    signature: list[tuple[object, ...]] = []
    tables = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    for table in tables:
        signature.append(("table", table))
        for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall():
            # Column order can differ when an existing table gains an additive
            # column; name/type/default/nullability/key semantics must converge.
            signature.append(("column", table, *tuple(row)[1:]))
        for row in conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall():
            signature.append(("foreign_key", table, *tuple(row)))
        for index_row in conn.execute(f'PRAGMA index_list("{table}")').fetchall():
            index_name = str(index_row[1])
            signature.append(("index", table, *tuple(index_row[1:])))
            for column_row in conn.execute(
                f'PRAGMA index_info("{index_name}")'
            ).fetchall():
                signature.append(
                    ("index_column", index_name, column_row[0], column_row[2])
                )
    return tuple(sorted(signature, key=repr))

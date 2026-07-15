"""src/storage/phase2_3_backfill.py
Resumable metadata-only backfill for Phase 2.3 identities and corpus links.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import yaml

logger = logging.getLogger(__name__)


class Phase23Backfill:
    """Backfill legacy rows in bounded, independently committed stages."""

    STAGES = (
        "securities",
        "fundamentals",
        "companyfacts",
        "filings",
        "freshness",
        "chroma_documents",
    )

    def __init__(
        self,
        store: object,
        *,
        batch_size: int = 100,
        coverage_path: Optional[Path] = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.store = store
        self.sqlite = store.sqlite
        self.chroma = store.chroma
        self.batch_size = min(batch_size, 1000)
        self.coverage_path = coverage_path or (
            Path(__file__).resolve().parents[2] / "configs" / "coverage.yaml"
        )

    def _deep_tickers(self) -> set[str]:
        """Load the preserved core/deep policy without reading credentials."""
        try:
            with self.coverage_path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            values = (config.get("deep") or {}).get("tickers") or []
        except (OSError, TypeError, yaml.YAMLError):
            logger.warning("Could not read deep coverage policy", exc_info=True)
            return set()
        return {
            self._normalized_ticker(value)
            for value in values
            if self._normalized_ticker(value)
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalized_ticker(value: object) -> str:
        return str(value or "").strip().upper().replace(".", "-")

    @staticmethod
    def _stable_id(prefix: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
        return f"{prefix}-{digest}"

    @staticmethod
    def _bump_revision(conn: sqlite3.Connection) -> None:
        conn.execute(
            """INSERT INTO store_revision (id, revision, updated_at)
            VALUES (1, 1, datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                revision=revision + 1, updated_at=datetime('now')"""
        )

    @staticmethod
    def _progress(conn: sqlite3.Connection, stage: str) -> tuple[str, bool]:
        row = conn.execute(
            "SELECT cursor_value, completed FROM phase2_3_backfill_progress WHERE stage=?",
            (stage,),
        ).fetchone()
        if row is None:
            return "", False
        return str(row[0] or ""), bool(row[1])

    def _finish_batch(
        self,
        conn: sqlite3.Connection,
        stage: str,
        cursor: str,
        processed: int,
        completed: bool,
    ) -> None:
        conn.execute(
            """INSERT INTO phase2_3_backfill_progress (
                stage, cursor_value, completed, rows_processed, updated_at
            ) VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(stage) DO UPDATE SET
                cursor_value=excluded.cursor_value,
                completed=excluded.completed,
                rows_processed=phase2_3_backfill_progress.rows_processed + excluded.rows_processed,
                updated_at=datetime('now')""",
            (stage, cursor or None, int(completed), processed),
        )
        self._bump_revision(conn)

    def _record_error(
        self,
        conn: sqlite3.Connection,
        *,
        stage: str,
        legacy_table: str,
        legacy_row_id: object,
        identifier: object,
        issue_type: str,
        candidates: Optional[list[str]] = None,
        details: Optional[dict] = None,
    ) -> bool:
        stable_key = "|".join(
            (stage, legacy_table, str(legacy_row_id or ""), str(identifier or ""), issue_type)
        )
        cursor = conn.execute(
            """INSERT OR IGNORE INTO identity_reconciliation_errors (
                error_id, stage, legacy_table, legacy_row_id, identifier,
                issue_type, candidates_json, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._stable_id("reconcile", stable_key),
                stage,
                legacy_table,
                str(legacy_row_id) if legacy_row_id is not None else None,
                str(identifier) if identifier is not None else None,
                issue_type,
                json.dumps(candidates or [], sort_keys=True),
                json.dumps(details or {}, sort_keys=True),
            ),
        )
        return cursor.rowcount > 0

    def _resolve_security(
        self,
        conn: sqlite3.Connection,
        ticker: object,
        *,
        stage: str,
        table: str,
        row_id: object,
    ) -> tuple[Optional[str], int]:
        normalized = self._normalized_ticker(ticker)
        if not normalized:
            inserted = self._record_error(
                conn,
                stage=stage,
                legacy_table=table,
                legacy_row_id=row_id,
                identifier=ticker,
                issue_type="orphan",
                details={"reason": "missing ticker"},
            )
            return None, int(inserted)
        rows = conn.execute(
            "SELECT security_id FROM securities WHERE normalized_ticker=? ORDER BY security_id",
            (normalized,),
        ).fetchall()
        if len(rows) == 1:
            return str(rows[0][0]), 0
        issue_type = "ambiguous" if rows else "orphan"
        inserted = self._record_error(
            conn,
            stage=stage,
            legacy_table=table,
            legacy_row_id=row_id,
            identifier=normalized,
            issue_type=issue_type,
            candidates=[str(row[0]) for row in rows],
            details={"reason": "ticker did not resolve to exactly one security"},
        )
        return None, int(inserted)

    def _security_batch(self, conn: sqlite3.Connection, cursor: str) -> dict[str, object]:
        legacy_rows = conn.execute(
            """SELECT ticker FROM (
                SELECT ticker FROM fundamentals
                UNION SELECT ticker FROM sec_companyfacts
                UNION SELECT ticker FROM filings
                UNION SELECT ticker FROM cache_meta WHERE ticker <> 'SCHEDULER'
            ) GROUP BY ticker""",
        ).fetchall()
        candidates = {
            self._normalized_ticker(row[0]) for row in legacy_rows
            if self._normalized_ticker(row[0])
        }
        candidates.update(self._deep_tickers())
        rows = [(ticker,) for ticker in sorted(candidates) if ticker > cursor][
            : self.batch_size
        ]
        inserted = 0
        errors = 0
        last_cursor = cursor
        now = self._now()
        for row in rows:
            raw_ticker = str(row[0] or "")
            last_cursor = raw_ticker
            ticker = self._normalized_ticker(raw_ticker)
            if not ticker:
                errors += int(
                    self._record_error(
                        conn,
                        stage="securities",
                        legacy_table="legacy_tickers",
                        legacy_row_id=raw_ticker,
                        identifier=raw_ticker,
                        issue_type="orphan",
                        details={"reason": "empty canonical ticker"},
                    )
                )
                continue
            cik_rows = conn.execute(
                """SELECT DISTINCT cik FROM (
                    SELECT cik FROM sec_companyfacts WHERE upper(ticker)=? AND cik IS NOT NULL
                    UNION SELECT cik FROM filings WHERE upper(ticker)=? AND cik IS NOT NULL
                ) WHERE cik <> '' ORDER BY cik""",
                (raw_ticker.upper(), raw_ticker.upper()),
            ).fetchall()
            ciks = [str(item[0]).zfill(10) for item in cik_rows]
            cik = ciks[0] if len(ciks) == 1 else None
            if len(ciks) > 1:
                errors += int(
                    self._record_error(
                        conn,
                        stage="securities",
                        legacy_table="sec_companyfacts",
                        legacy_row_id=ticker,
                        identifier=ticker,
                        issue_type="ambiguous",
                        candidates=ciks,
                        details={"reason": "multiple SEC CIKs for one legacy ticker"},
                    )
                )

            candidates = conn.execute(
                "SELECT security_id, cik FROM securities WHERE normalized_ticker=?",
                (ticker,),
            ).fetchall()
            if len(candidates) > 1:
                errors += int(
                    self._record_error(
                        conn,
                        stage="securities",
                        legacy_table="securities",
                        legacy_row_id=ticker,
                        identifier=ticker,
                        issue_type="ambiguous",
                        candidates=[str(item[0]) for item in candidates],
                    )
                )
                continue
            if candidates:
                security_id = str(candidates[0][0])
                stored_cik = str(candidates[0][1] or "") or None
                if cik and stored_cik and cik != stored_cik:
                    errors += int(
                        self._record_error(
                            conn,
                            stage="securities",
                            legacy_table="securities",
                            legacy_row_id=security_id,
                            identifier=ticker,
                            issue_type="conflict",
                            candidates=[stored_cik, cik],
                        )
                    )
                    cik = None
                elif cik and not stored_cik:
                    conn.execute(
                        "UPDATE securities SET cik=?, updated_at=? WHERE security_id=?",
                        (cik, now, security_id),
                    )
            else:
                security_id = self._stable_id("legacy-security", ticker)
                conn.execute(
                    """INSERT INTO securities (
                        security_id, ticker, normalized_ticker, company_name,
                        exchange, cik, first_seen_at, last_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, '', ?, ?, ?, ?)""",
                    (security_id, ticker, ticker, ticker, cik, now, now, now),
                )
                inserted += 1
            conn.execute(
                """INSERT OR IGNORE INTO security_aliases (
                    security_id, alias, normalized_alias, alias_type, source
                ) VALUES (?, ?, ?, 'ticker', 'phase2_3_backfill')""",
                (security_id, ticker, ticker),
            )
        return {
            "cursor": last_cursor,
            "processed": len(rows),
            "inserted": inserted,
            "errors": errors,
            "completed": len(rows) < self.batch_size,
        }

    def _link_batch(
        self,
        conn: sqlite3.Connection,
        *,
        stage: str,
        table: str,
        cursor: str,
    ) -> dict[str, object]:
        last_id = int(cursor or 0)
        rows = conn.execute(
            f"SELECT rowid, ticker FROM {table} WHERE rowid>? ORDER BY rowid LIMIT ?",
            (last_id, self.batch_size),
        ).fetchall()
        errors = 0
        linked = 0
        for row in rows:
            row_id = int(row[0])
            last_id = row_id
            security_id, new_errors = self._resolve_security(
                conn, row[1], stage=stage, table=table, row_id=row_id
            )
            errors += new_errors
            if security_id:
                conn.execute(
                    f"UPDATE {table} SET security_id=? WHERE rowid=?",
                    (security_id, row_id),
                )
                linked += 1
        return {
            "cursor": str(last_id),
            "processed": len(rows),
            "inserted": linked,
            "errors": errors,
            "completed": len(rows) < self.batch_size,
        }

    @staticmethod
    def _source_category(source: str) -> str:
        lowered = source.lower()
        if any(form in lowered for form in ("10-k", "10-q", "8-k", "sec")):
            return "sec_filing"
        if "news" in lowered or "gdelt" in lowered:
            return "company_news"
        if "transcript" in lowered:
            return "earnings_transcript"
        if lowered.startswith("ir"):
            return "investor_relations"
        return "legacy_document"

    def _metadata_tickers(self, metadata: dict) -> list[str]:
        """Normalize legacy scalar/list/comma-separated Chroma ticker metadata."""
        value = metadata.get("ticker")
        if value is None:
            value = metadata.get("tickers")
        if isinstance(value, (list, tuple, set)):
            raw_values = value
        else:
            raw_values = str(value or "").split(",")
        return sorted(
            {
                normalized
                for item in raw_values
                if (normalized := self._normalized_ticker(item))
            }
        )

    def _ensure_corpus_item(
        self,
        conn: sqlite3.Connection,
        *,
        item_id: str,
        family_id: str,
        source: str,
        item_type: str,
        title: str,
        published_at: Optional[str],
        source_url: str,
        ticker: Optional[str],
        metadata: dict,
        provider_record_id: Optional[str] = None,
        tickers: Optional[list[str]] = None,
    ) -> tuple[str, bool]:
        existing = conn.execute(
            "SELECT corpus_item_id FROM corpus_items WHERE document_family_id=?",
            (family_id,),
        ).fetchone()
        if existing:
            return str(existing[0]), False
        now = self._now()
        metadata_json = json.dumps(metadata, sort_keys=True, default=str)
        digest_source = str(metadata.get("content_hash") or "") or (
            family_id + "|" + metadata_json
        )
        content_digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
        item_tickers = tickers if tickers is not None else ([ticker] if ticker else [])
        conn.execute(
            """INSERT OR IGNORE INTO corpus_items (
                corpus_item_id, source, source_category, provider_record_id,
                item_type, title, normalized_headline, language, published_at,
                accessed_at, ingested_at, source_url, tickers_json, content_hash,
                metadata_json, document_family, document_family_id,
                indexing_status, license_label, normalization_version,
                evidence_authority, metadata_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'en', ?, ?, ?, ?, ?, ?, ?, ?, ?,
                'indexed', 'legacy-unknown', 'phase2_3_backfill_v1', 'legacy', ?)""",
            (
                item_id,
                source,
                self._source_category(source),
                provider_record_id,
                item_type,
                title,
                " ".join(title.lower().split()),
                published_at,
                now,
                now,
                source_url,
                json.dumps(item_tickers),
                content_digest,
                metadata_json,
                source,
                family_id,
                len(metadata_json.encode("utf-8")),
            ),
        )
        row = conn.execute(
            "SELECT corpus_item_id FROM corpus_items WHERE document_family_id=?",
            (family_id,),
        ).fetchone()
        resolved_item_id = str(row[0]) if row else item_id
        conn.execute(
            """INSERT OR IGNORE INTO corpus_item_sources (
                corpus_item_id, source_key, source_name, source_category,
                provider_record_id, source_url, accessed_at, ingested_at,
                license_label, evidence_authority, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'legacy-unknown', 'legacy', ?)""",
            (
                resolved_item_id,
                self._stable_id("source", source_url + "|" + family_id),
                source,
                self._source_category(source),
                provider_record_id,
                source_url,
                now,
                now,
                metadata_json,
            ),
        )
        return resolved_item_id, True

    def _filing_batch(self, conn: sqlite3.Connection, cursor: str) -> dict[str, object]:
        last_id = int(cursor or 0)
        rows = conn.execute(
            "SELECT rowid, * FROM filings WHERE rowid>? ORDER BY rowid LIMIT ?",
            (last_id, self.batch_size),
        ).fetchall()
        inserted = 0
        errors = 0
        for row in rows:
            data = dict(row)
            row_id = int(data["id"])
            last_id = row_id
            security_id, new_errors = self._resolve_security(
                conn,
                data.get("ticker"),
                stage="filings",
                table="filings",
                row_id=row_id,
            )
            errors += new_errors
            if security_id:
                conn.execute(
                    "UPDATE filings SET security_id=? WHERE rowid=?",
                    (security_id, row_id),
                )
            family_id = str(data.get("summary_embedding_id") or f"legacy-filing-{row_id}")
            accession = str(data.get("accession") or row_id)
            source_url = str(data.get("source_url") or f"legacy-filing://{quote(accession)}")
            item_id, created = self._ensure_corpus_item(
                conn,
                item_id=self._stable_id("legacy-filing", accession),
                family_id=family_id,
                source="sec_filings",
                item_type="sec_filing",
                title=f"{data.get('ticker') or 'Unknown'} {data.get('filing_type') or 'filing'}",
                published_at=data.get("filing_date"),
                source_url=source_url,
                ticker=self._normalized_ticker(data.get("ticker")) or None,
                metadata={
                    "legacy_table": "filings",
                    "legacy_id": row_id,
                    "accession": data.get("accession"),
                    "filing_type": data.get("filing_type"),
                },
                provider_record_id=str(data.get("accession") or "") or None,
            )
            inserted += int(created)
            if security_id:
                conn.execute(
                    """INSERT OR IGNORE INTO corpus_item_securities
                    (corpus_item_id, security_id, ticker) VALUES (?, ?, ?)""",
                    (item_id, security_id, self._normalized_ticker(data.get("ticker"))),
                )
        return {
            "cursor": str(last_id),
            "processed": len(rows),
            "inserted": inserted,
            "errors": errors,
            "completed": len(rows) < self.batch_size,
        }

    def _chroma_batch(self, conn: sqlite3.Connection, cursor: str) -> dict[str, object]:
        offset = int(cursor or 0)
        rows = self.chroma.iter_document_metadata(limit=self.batch_size, offset=offset)
        inserted = 0
        errors = 0
        for row in rows:
            document_id = str(row.get("id") or "")
            metadata = dict(row.get("metadata") or {})
            family_id = str(
                metadata.get("document_family_id")
                or metadata.get("corpus_item_id")
                or metadata.get("parent_id")
                or document_id.split("#", 1)[0]
            )
            source = str(metadata.get("source") or "legacy_chroma")
            tickers = self._metadata_tickers(metadata)
            source_url = str(
                metadata.get("source_url")
                or metadata.get("canonical_url")
                or f"legacy-chroma://{quote(family_id, safe='')}"
            )
            item_id, created = self._ensure_corpus_item(
                conn,
                item_id=self._stable_id("legacy-chroma", family_id),
                family_id=family_id,
                source=source,
                item_type=str(metadata.get("item_type") or "legacy_document"),
                title=str(metadata.get("title") or family_id),
                published_at=metadata.get("published_at") or metadata.get("date"),
                source_url=source_url,
                ticker=tickers[0] if len(tickers) == 1 else None,
                metadata={"legacy_document_id": document_id, **metadata},
                provider_record_id=(
                    str(metadata.get("provider_record_id"))
                    if metadata.get("provider_record_id") is not None
                    else None
                ),
                tickers=tickers,
            )
            inserted += int(created)
            for ticker in tickers:
                security_id, new_errors = self._resolve_security(
                    conn,
                    ticker,
                    stage="chroma_documents",
                    table="chroma",
                    row_id=family_id,
                )
                errors += new_errors
                if security_id:
                    conn.execute(
                        """INSERT OR IGNORE INTO corpus_item_securities
                        (corpus_item_id, security_id, ticker) VALUES (?, ?, ?)""",
                        (item_id, security_id, ticker),
                    )
        return {
            "cursor": str(offset + len(rows)),
            "processed": len(rows),
            "inserted": inserted,
            "errors": errors,
            "completed": len(rows) < self.batch_size,
        }

    def _run_stage_batch(self, stage: str, cursor: str) -> dict[str, object]:
        with self.sqlite._connect() as conn:
            if stage == "securities":
                result = self._security_batch(conn, cursor)
            elif stage == "fundamentals":
                result = self._link_batch(
                    conn, stage=stage, table="fundamentals", cursor=cursor
                )
            elif stage == "companyfacts":
                result = self._link_batch(
                    conn, stage=stage, table="sec_companyfacts", cursor=cursor
                )
            elif stage == "filings":
                result = self._filing_batch(conn, cursor)
            elif stage == "freshness":
                result = self._link_batch(
                    conn, stage=stage, table="cache_meta", cursor=cursor
                )
            elif stage == "chroma_documents":
                result = self._chroma_batch(conn, cursor)
            else:
                raise ValueError(f"unknown backfill stage: {stage}")
            self._finish_batch(
                conn,
                stage,
                str(result["cursor"]),
                int(result["processed"]),
                bool(result["completed"]),
            )
            conn.commit()
        return result

    def run(self, *, max_batches: Optional[int] = None) -> dict[str, object]:
        """Run pending bounded batches; stop cleanly after ``max_batches``."""
        if max_batches is not None and (
            not isinstance(max_batches, int)
            or isinstance(max_batches, bool)
            or max_batches < 1
        ):
            raise ValueError("max_batches must be a positive integer")
        totals = {"batches": 0, "processed": 0, "inserted": 0, "errors": 0}
        for stage in self.STAGES:
            while True:
                with self.sqlite._connect() as conn:
                    cursor, completed = self._progress(conn, stage)
                if completed:
                    break
                if max_batches is not None and totals["batches"] >= max_batches:
                    break
                result = self._run_stage_batch(stage, cursor)
                totals["batches"] += 1
                totals["processed"] += int(result["processed"])
                totals["inserted"] += int(result["inserted"])
                totals["errors"] += int(result["errors"])
                if result["completed"]:
                    break
            if max_batches is not None and totals["batches"] >= max_batches:
                break

        with self.sqlite._connect() as conn:
            incomplete = conn.execute(
                "SELECT COUNT(*) FROM phase2_3_backfill_progress WHERE completed=0"
            ).fetchone()[0]
            recorded = conn.execute(
                "SELECT COUNT(*) FROM phase2_3_backfill_progress"
            ).fetchone()[0]
            revision = conn.execute(
                "SELECT revision FROM store_revision WHERE id=1"
            ).fetchone()[0]
        totals["complete"] = recorded == len(self.STAGES) and incomplete == 0
        totals["revision"] = int(revision)
        return totals

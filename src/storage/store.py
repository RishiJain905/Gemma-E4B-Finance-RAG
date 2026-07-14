"""
src/storage/store.py
Unified storage facade over SQLiteStore + ChromaStore.

Usage:
    store = Store()

    # Save structured facts
    store.save_fundamental("NVDA", "revenue_q1", 26.0, "usd", "2026-Q1")

    # Save a document with embedding
    doc_id = store.save_document("NVDA 10-Q", "NVIDIA reported...",
                                   ticker="NVDA", source="sec")

    # Search everything
    results = store.search("What is NVDA's revenue?")
    # Returns: {"facts": [...], "documents": [...]}
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from src.ingestion.records import EventRecord, NarrativeRecord, ObservationRecord

from .chroma_store import ChromaStore
from .sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


def _companyfacts_cutoff(as_of: Optional[str]) -> str:
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


def _candidate_sort_key(row: dict, concept_order: dict[str, int]) -> tuple:
    return (
        concept_order.get(row.get("concept", ""), len(concept_order)),
        -int(str(row.get("filed_at", "0000-00-00")).replace("-", "") or 0),
        str(row.get("accession", "")),
        str(row.get("source_accessed_at", "")),
        str(row.get("value_text", "")),
    )


def select_companyfacts(
    rows: list[dict],
    metric_rules: dict,
    metrics: list[str],
    *,
    periods: Optional[list[str]] = None,
    as_of: Optional[str] = None,
) -> list[dict]:
    """Pure, stable canonical selection over raw CompanyFacts candidates."""
    cutoff = _companyfacts_cutoff(as_of)
    requested_periods = set(periods) if periods is not None else None
    results: list[dict] = []

    for metric_position, metric in enumerate(metrics):
        rule = metric_rules.get(metric)
        if not isinstance(rule, dict):
            continue
        concepts = list(rule.get("concepts", []))
        concept_order = {concept: index for index, concept in enumerate(concepts)}
        allowed_units = set(rule.get("units", []))
        allowed_kinds = set(rule.get("period_kinds", []))
        eligible = [
            row
            for row in rows
            if row.get("concept") in concept_order
            and (not allowed_units or row.get("unit") in allowed_units)
            and (not allowed_kinds or row.get("period_kind") in allowed_kinds)
            and (requested_periods is None or row.get("period_end") in requested_periods)
            and str(row.get("filed_at", "")) <= cutoff
        ]

        groups: dict[tuple[str, str], list[dict]] = {}
        for row in eligible:
            groups.setdefault((row["unit"], row["period_end"]), []).append(row)

        for (unit, period_end), candidates in groups.items():
            newest_by_concept: dict[str, dict] = {}
            for row in sorted(candidates, key=lambda item: _candidate_sort_key(item, concept_order)):
                newest_by_concept.setdefault(row["concept"], row)
            selected = min(
                newest_by_concept.values(),
                key=lambda item: _candidate_sort_key(item, concept_order),
            )

            alternatives = []
            seen_values = {Decimal(selected["value_text"])}
            for candidate in sorted(
                candidates,
                key=lambda item: _candidate_sort_key(item, concept_order),
            ):
                value_text = candidate["value_text"]
                decimal_value = Decimal(value_text)
                if decimal_value in seen_values:
                    continue
                seen_values.add(decimal_value)
                alternatives.append(
                    {
                        "value_text": value_text,
                        "taxonomy": candidate["taxonomy"],
                        "concept": candidate["concept"],
                        "accession": candidate["accession"],
                        "form": candidate["form"],
                        "filed_at": candidate["filed_at"],
                        "source_url": candidate["source_url"],
                    }
                )

            results.append(
                {
                    "ticker": selected["ticker"],
                    "metric": metric,
                    "value": selected["value_numeric"],
                    "value_text": selected["value_text"],
                    "unit": unit,
                    "period": period_end,
                    "period_start": selected["period_start"],
                    "period_type": selected["period_kind"],
                    "source_type": "sec_companyfacts",
                    "source_url": selected["source_url"],
                    "source_accessed_at": selected["source_accessed_at"],
                    "taxonomy": selected["taxonomy"],
                    "concept": selected["concept"],
                    "accession": selected["accession"],
                    "form": selected["form"],
                    "filed_at": selected["filed_at"],
                    "as_of": cutoff,
                    "conflict": bool(alternatives),
                    "alternatives": alternatives,
                    "_metric_position": metric_position,
                }
            )

    results.sort(key=lambda item: (item["unit"], item["concept"]))
    results.sort(key=lambda item: item["period"], reverse=True)
    results.sort(key=lambda item: item["_metric_position"])
    for result in results:
        result.pop("_metric_position")
    return results


def _companyfacts_alt_value(value_text: str):
    """Best-effort numeric value for a CompanyFacts alternative observation."""
    try:
        return float(Decimal(str(value_text)))
    except Exception:  # noqa: BLE001 - keep the raw text if it will not parse
        return value_text


def companyfacts_rows_to_evidence(rows: list[dict]) -> list[dict]:
    """Project canonical CompanyFacts rows into structured fact-evidence rows.

    Each selected observation becomes one authoritative fact-evidence row
    (``source_type="sec_companyfacts"``) carrying full provenance (taxonomy /
    concept / accession / filed_at / as_of / source_url). Every distinct
    ``alternatives`` value is emitted as its OWN separate evidence row flagged
    ``conflict=True`` — conflicting filed values are never averaged or collapsed,
    so the evidence grader can disclose them. Order is preserved (primary row
    first, then its alternatives).
    """
    evidence: list[dict] = []
    for row in rows or []:
        primary = {
            "metric": row.get("metric"),
            "value": row.get("value"),
            "value_text": row.get("value_text"),
            "ticker": row.get("ticker"),
            "period": row.get("period"),
            "period_start": row.get("period_start"),
            "period_type": row.get("period_type"),
            "unit": row.get("unit"),
            "source_type": "sec_companyfacts",
            "source_url": row.get("source_url"),
            "as_of": row.get("as_of"),
            "taxonomy": row.get("taxonomy"),
            "concept": row.get("concept"),
            "accession": row.get("accession"),
            "form": row.get("form"),
            "filed_at": row.get("filed_at"),
            "conflict": bool(row.get("conflict")),
        }
        if row.get("conflict"):
            primary["conflict_reason"] = "companyfacts_multiple_filed_values"
        evidence.append(primary)
        for alt in row.get("alternatives") or []:
            evidence.append({
                "metric": row.get("metric"),
                "value": _companyfacts_alt_value(alt.get("value_text")),
                "value_text": alt.get("value_text"),
                "ticker": row.get("ticker"),
                "period": row.get("period"),
                "period_type": row.get("period_type"),
                "unit": row.get("unit"),
                "source_type": "sec_companyfacts",
                "source_url": alt.get("source_url"),
                "as_of": row.get("as_of"),
                "taxonomy": alt.get("taxonomy"),
                "concept": alt.get("concept"),
                "accession": alt.get("accession"),
                "form": alt.get("form"),
                "filed_at": alt.get("filed_at"),
                "conflict": True,
                "conflict_reason": "companyfacts_alternative_value",
            })
    return evidence


class Store:
    """
    Unified storage layer combining structured (SQLite) and
    semantic (ChromaDB) storage.
    """

    COMPANYFACTS_CONFIG_PATH = (
        Path(__file__).parent.parent.parent / "configs/sec_companyfacts.yaml"
    )
    MAX_CORPUS_PAGE_LIMIT = 200
    MAX_CORPUS_OFFSET = 10_000

    def __init__(self,
                 db_path: Optional[Path] = None,
                 chroma_path: Optional[Path] = None,
                 collection_name: str = "tracealchemy_docs",
                 embedding_endpoint: str = "http://127.0.0.1:8087/v1/embeddings",
                 embedding_cache_size: int = 256):

        self.sqlite = SQLiteStore(db_path=db_path)
        self.chroma = ChromaStore(
            persist_directory=chroma_path,
            collection_name=collection_name,
            embedding_endpoint=embedding_endpoint,
            embedding_cache_size=embedding_cache_size,
        )

    # ── Health ─────────────────────────────────────────

    def heartbeat(self) -> dict:
        """Check both storage backends are responsive."""
        return {
            "sqlite": self._check_sqlite(),
            "chroma": self.chroma.heartbeat(),
            "chroma_doc_count": self.chroma.count(),
        }

    def _check_sqlite(self) -> bool:
        try:
            with self.sqlite._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("SQLite heartbeat failed: %s", e)
            return False

    # ── Data Revision (2.2.6.2) ────────────────────────
    #
    # A single cross-process monotonic integer that the versioned retrieval
    # cache (src/middleware/retrieval_cache.py) keys on. Every facade mutation
    # below that can change model-visible facts/documents calls ``_bump_revision``
    # BEFORE the mutation begins, so a cache entry captured at revision N can
    # never be served after any ingestion write advanced the revision. A failed
    # mutation may leave the revision advanced (an extra cache miss) but never
    # leaves a stale entry valid. Direct SQLite writers that bypass this facade
    # (e.g. the SEC CompanyFacts ingestor calling ``sqlite.upsert_sec_companyfacts``
    # directly) MUST call :meth:`bump_retrieval_revision` themselves — it is the
    # one documented helper for that.

    def retrieval_revision(self) -> int:
        """Return the current monotonic data revision (0 when never bumped)."""
        return self.sqlite.get_store_revision()

    def bump_retrieval_revision(self, reason: str = "") -> int:
        """Advance the data revision and return the new value (documented helper)."""
        return self.sqlite.bump_store_revision(reason)

    def _bump_revision(self, reason: str) -> None:
        """Bump the revision before a model-visible mutation begins.

        Best-effort so a revision-store hiccup never crashes ingestion, but it
        runs BEFORE the mutation: if it fails, the worst case is that a cache
        entry that predates this mutation is *not* invalidated — which is why the
        retrieval cache also fails soft to a miss on any revision-read error.
        """
        try:
            self.sqlite.bump_store_revision(reason)
        except Exception:  # noqa: BLE001 - revision bookkeeping must never crash a write
            logger.warning("Failed to bump store revision (%s)", reason, exc_info=True)

    # ─── Structured Facts (SQLite) ────────────────────

    def save_fundamental(self, ticker: str, metric: str, value: float,
                         unit: str = "usd", period: str = None,
                         period_type: str = "quarterly",
                         source_type: str = "yfinance",
                         source_url: str = None) -> bool:
        """Save or update a single financial metric."""
        self._bump_revision("save_fundamental")
        return bool(self.sqlite.upsert_fundamental(
            ticker, metric, value, unit, period,
            period_type, source_type, source_url
        ))

    def get_fundamental(self, ticker: str, metric: str,
                        period: str = None) -> Optional[dict]:
        """Get the latest value for a metric."""
        return self.sqlite.get_fundamental(ticker, metric, period)

    def get_fundamentals_batch(self, ticker: str,
                               metrics: list[str] = None) -> dict:
        """Get multiple metrics for a ticker at once."""
        return self.sqlite.get_fundamentals_batch(ticker, metrics)

    def get_companyfacts(
        self,
        ticker: str,
        metrics: list[str],
        *,
        periods: Optional[list[str]] = None,
        as_of: Optional[str] = None,
    ) -> list[dict]:
        """Project raw SEC observations into configured canonical metrics."""
        import yaml

        with open(self.COMPANYFACTS_CONFIG_PATH, encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not config.get("enabled", False):
            return []

        cutoff = _companyfacts_cutoff(as_of)
        metric_rules = config.get("metrics", {}) or {}
        requested_rules = [metric_rules[name] for name in metrics if name in metric_rules]
        concepts = list(
            dict.fromkeys(
                concept
                for rule in requested_rules
                for concept in rule.get("concepts", [])
            )
        )
        if not concepts:
            return []
        raw_rows = self.sqlite.query_sec_companyfacts(
            ticker,
            concepts,
            as_of=cutoff,
        )
        return select_companyfacts(
            raw_rows,
            metric_rules,
            metrics,
            periods=periods,
            as_of=cutoff,
        )

    def companyfacts_evidence(
        self,
        ticker: str,
        metrics: list[str],
        *,
        periods: Optional[list[str]] = None,
        as_of: Optional[str] = None,
        latest_only: bool = True,
    ) -> list[dict]:
        """Authoritative CompanyFacts as structured fact-evidence rows.

        Returns ``[]`` when CompanyFacts ingestion is disabled (the config gate in
        :meth:`get_companyfacts`) so a disabled source is a strict no-op. When
        ``latest_only`` and no explicit ``periods`` are requested, keeps only the
        most recent filed period per (metric, unit) so a "latest revenue" style
        query is not flooded with every historical quarter; conflicting filed
        values within the kept period remain separate evidence rows.
        """
        rows = self.get_companyfacts(ticker, metrics, periods=periods, as_of=as_of)
        if latest_only and not periods:
            newest: dict[tuple, str] = {}
            for row in rows:
                key = (row.get("metric"), row.get("unit"))
                period = str(row.get("period") or "")
                if key not in newest or period > newest[key]:
                    newest[key] = period
            rows = [
                row for row in rows
                if str(row.get("period") or "") == newest.get(
                    (row.get("metric"), row.get("unit")))
            ]
        return companyfacts_rows_to_evidence(rows)

    # ── Document Storage (ChromaDB) ──────────────────

    # -- Security Universe (2.3.1.1) -----------------------------------------

    def list_securities(
        self,
        index: Optional[str] = None,
        active: Optional[bool] = True,
        sector: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List bounded canonical securities through the SQLite facade."""
        return self.sqlite.list_securities(
            index=index,
            active=active,
            sector=sector,
            limit=limit,
            offset=offset,
        )

    def get_security(self, ticker_or_id: str) -> Optional[dict]:
        """Return one canonical security by ticker or opaque id."""
        return self.sqlite.get_security(ticker_or_id)

    def resolve_security(
        self,
        symbol: str,
        provider: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve canonical, vendor, or historical security symbols."""
        return self.sqlite.resolve_security(symbol, provider=provider, as_of=as_of)

    def list_memberships(
        self,
        security_id: Optional[str] = None,
        index_code: Optional[str] = None,
        active: Optional[bool] = None,
    ) -> list[dict]:
        """List current or historical index memberships."""
        return self.sqlite.list_memberships(
            security_id=security_id,
            index_code=index_code,
            active=active,
        )

    def upsert_universe_snapshot(
        self,
        source: str,
        observed_at: str,
        rows: list[object],
    ) -> dict:
        """Atomically reconcile a validated provider universe snapshot."""
        return self.sqlite.upsert_universe_snapshot(source, observed_at, rows)

    def list_universe_errors(self, run_id: str) -> list[dict]:
        """Return reconciliation errors recorded for one refresh run."""
        return self.sqlite.list_universe_errors(run_id)

    def upsert_narrative(self, record: NarrativeRecord) -> dict:
        """Persist narrative metadata, then index content with retryable status."""
        result = self.sqlite.upsert_narrative_record(record)
        if not result["needs_index"]:
            return result

        item_id = result["corpus_item_id"]
        chroma_metadata = {
            "corpus_item_id": item_id,
            "source_category": record.source_category,
            "item_type": record.item_type,
            "document_family": record.document_family,
            "content_hash": record.content_hash,
            "license_label": record.license_label,
            "normalization_version": record.normalization_version,
            "evidence_authority": record.evidence_authority,
        }
        optional_metadata = {
            "provider_record_id": record.provider_record_id,
            "original_publisher": record.original_publisher,
            "canonical_url": record.canonical_url,
            "event_type": record.event_type,
        }
        chroma_metadata.update({
            key: value for key, value in optional_metadata.items() if value is not None
        })
        chroma_metadata.update(dict(record.metadata))
        if record.tickers:
            chroma_metadata["tickers"] = ",".join(record.tickers)
        if record.index_codes:
            chroma_metadata["index_codes"] = ",".join(record.index_codes)
        if record.sectors:
            chroma_metadata["sectors"] = ",".join(record.sectors)

        try:
            if result["content_changed"]:
                self.chroma.delete_document(item_id)
                self.chroma.delete_filing_section_family(item_id)
            self.chroma.add_document(
                document_id=item_id,
                text=record.body,
                ticker=record.tickers[0] if record.tickers else None,
                source=record.source_name,
                date=record.published_at,
                metadata=chroma_metadata,
            )
        except Exception as exc:  # noqa: BLE001 - metadata must survive Chroma failure
            logger.warning("Narrative indexing failed for %s", item_id, exc_info=True)
            self.sqlite.set_corpus_index_status(item_id, "error", str(exc))
            result["indexing_status"] = "error"
            result["index_error"] = str(exc)[:2_000]
            return result

        self.sqlite.set_corpus_index_status(item_id, "indexed")
        result["indexing_status"] = "indexed"
        result["index_error"] = None
        return result

    def upsert_observation(self, record: ObservationRecord) -> dict:
        """Persist a structured observation without embedding it."""
        return self.sqlite.upsert_observation_record(record)

    def upsert_event(self, record: EventRecord) -> dict:
        """Persist a structured event and its item/security links."""
        return self.sqlite.upsert_event_record(record)

    def save_document(self,
                      document_id: str,
                      text: str,
                      ticker: str = None,
                      source: str = None,
                      date: str = None,
                      metadata: dict = None) -> str:
        """
        Store a document with its embedding in ChromaDB.

        Args:
            document_id: Unique ID (e.g., 'sec/NVDA/10-K-2025')
            text: Document content
            ticker: Associated stock ticker
            source: Source type
            date: Document date (ISO format)
            metadata: Additional metadata

        Returns:
            The document ID (confirmation of storage)
        """
        self._bump_revision("save_document")
        self.chroma.add_document(
            document_id=document_id,
            text=text,
            ticker=ticker,
            source=source,
            date=date,
            metadata=metadata,
        )
        return document_id

    def save_documents_batch(self,
                             ids: list[str],
                             texts: list[str],
                             metadatas: list[dict] = None):
        """Store multiple documents at once."""
        self._bump_revision("save_documents_batch")
        self.chroma.add_documents_batch(ids, texts, metadatas)

    def add_filing_sections(self, sections: list) -> dict[str, int]:
        """Replace and index SEC section families through Chroma's chunker."""
        # Bump before any delete/add: a same-count section replacement changes
        # the model-visible documents even though Chroma's count is unchanged.
        self._bump_revision("add_filing_sections")
        counts = {
            "sections_written": 0,
            "chunks_written": 0,
            "replacements": 0,
            "skipped": 0,
        }
        for section in sections:
            if not section.text.strip():
                counts["skipped"] += 1
                continue
            existing_count = self.chroma.count_filing_section_chunks(section.document_id)
            self.chroma.delete_filing_section_family(section.document_id)
            self.chroma.add_document(
                document_id=section.document_id,
                text=section.text,
                ticker=section.ticker,
                source="sec_filing",
                date=section.filing_date,
                metadata={
                    "accession": section.accession,
                    "form": section.form,
                    "filing_date": section.filing_date,
                    "report_period": section.report_period,
                    "section_key": section.section_key,
                    "section_heading": section.section_heading,
                    "section_index": section.section_index,
                    "parent_id": section.document_id,
                    "source_url": section.source_url,
                    "parsed_path": section.parsed_path,
                },
            )
            stored_count = self.chroma.count_filing_section_chunks(section.document_id)
            if not stored_count:
                raise RuntimeError(f"No chunks stored for {section.document_id}")
            counts["sections_written"] += 1
            counts["chunks_written"] += stored_count
            counts["replacements"] += int(existing_count > 0)
        return counts

    def get_section_chunks(
        self, parent_id: str, *, limit: int, offset: int = 0,
    ) -> list[dict]:
        return self.chroma.get_section_chunks(parent_id, limit=limit, offset=offset)

    def get_adjacent_sections(
        self, accession: str, section_index: int, *, before: int = 1, after: int = 1,
    ) -> list[dict]:
        return self.chroma.get_adjacent_sections(
            accession, section_index, before=before, after=after,
        )

    def count_filing_sections(self, accession: Optional[str] = None) -> int:
        return self.chroma.count_filing_sections(accession)

    # ── Corpus explorer read facade (2.2.7.2) ─────────────────────────────

    @classmethod
    def _validate_corpus_page(cls, limit: int, offset: int = 0) -> None:
        """Reject unbounded corpus pages before either backend is touched."""
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if not 1 <= limit <= cls.MAX_CORPUS_PAGE_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {cls.MAX_CORPUS_PAGE_LIMIT}")
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise ValueError("offset must be an integer")
        if not 0 <= offset <= cls.MAX_CORPUS_OFFSET:
            raise ValueError(
                f"offset must be between 0 and {cls.MAX_CORPUS_OFFSET}")

    def get_source_counts(self, *, limit: int = 100, offset: int = 0) -> list[dict]:
        """Merge bounded SQLite and Chroma source counts without reading bodies."""
        self._validate_corpus_page(limit, offset)
        counts: dict[str, int] = {}
        for row in self.sqlite.get_source_counts(
            limit=self.MAX_CORPUS_PAGE_LIMIT, offset=0,
        ):
            source = str(row.get("source") or "")
            if source:
                counts[source] = counts.get(source, 0) + int(row.get("count") or 0)
        for row in self.chroma.get_source_counts(
            limit=self.MAX_CORPUS_PAGE_LIMIT, offset=0,
        ):
            source = str(row.get("source") or "")
            if source:
                counts[source] = counts.get(source, 0) + int(row.get("count") or 0)
        rows = [
            {"source": source, "count": count}
            for source, count in sorted(counts.items())
        ]
        return rows[offset:offset + limit]

    def get_ticker_counts(self, *, limit: int = 100, offset: int = 0) -> list[dict]:
        """Merge bounded SQLite and Chroma ticker coverage summaries."""
        self._validate_corpus_page(limit, offset)
        grouped: dict[str, dict] = {}
        for backend in (self.sqlite, self.chroma):
            for row in backend.get_ticker_counts(
                limit=self.MAX_CORPUS_PAGE_LIMIT, offset=0,
            ):
                ticker = str(row.get("ticker") or "").upper()
                if not ticker:
                    continue
                item = grouped.setdefault(
                    ticker,
                    {"ticker": ticker, "record_count": 0, "sources": set(),
                     "source_counts": {}, "company_name": None},
                )
                item["record_count"] += int(row.get("record_count") or 0)
                item["sources"].update(str(source) for source in row.get("sources", []))
                item["company_name"] = item["company_name"] or row.get("company_name")
                for source, count in (row.get("source_counts") or {}).items():
                    item["source_counts"][str(source)] = (
                        item["source_counts"].get(str(source), 0) + int(count or 0)
                    )
        rows = []
        for ticker, item in sorted(grouped.items()):
            rows.append({
                "ticker": ticker,
                "record_count": item["record_count"],
                "sources": sorted(item["sources"]),
                "source_counts": item["source_counts"],
                "company_name": item["company_name"],
            })
        return rows[offset:offset + limit]

    def search_corpus_metrics(
        self,
        query: Optional[str] = None,
        *,
        ticker: Optional[str] = None,
        unit: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Search SQLite metric/concept inventory through the facade."""
        self._validate_corpus_page(limit, offset)
        return self.sqlite.search_corpus_metrics(
            query, ticker=ticker, unit=unit, limit=limit, offset=offset,
        )

    def search_corpus_facts(
        self,
        query: Optional[str] = None,
        *,
        ticker: Optional[str] = None,
        unit: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Search bounded fact metadata through the facade."""
        self._validate_corpus_page(limit, offset)
        return self.sqlite.search_corpus_facts(
            query, ticker=ticker, unit=unit, limit=limit, offset=offset,
        )

    def get_corpus_fact(self, record_kind: str, record_id: int) -> Optional[dict]:
        """Read one allowlisted fact record for corpus detail."""
        return self.sqlite.get_corpus_fact(record_kind, record_id)

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
        """Return bounded filing metadata through the facade."""
        self._validate_corpus_page(limit, offset)
        return self.sqlite.list_filings(
            query=query, ticker=ticker, filing_type=filing_type,
            date_from=date_from, date_to=date_to, limit=limit, offset=offset,
        )

    def get_filing(self, accession: str) -> Optional[dict]:
        """Read one safe filing record through the facade."""
        return self.sqlite.get_filing(accession)

    def list_freshness(
        self,
        *,
        ticker: Optional[str] = None,
        source: Optional[str] = None,
        include_scheduler: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Return bounded freshness rows without triggering refresh logic."""
        self._validate_corpus_page(limit, offset)
        return self.sqlite.list_freshness(
            ticker=ticker, source=source, include_scheduler=include_scheduler,
            limit=limit, offset=offset,
        )

    SCHEDULER_SOURCES = {
        "yfinance": {"ttl_key": "fundamentals"},
        "sec_filings": {"ttl_key": "sec_filings"},
        "sec_companyfacts": {"ttl_key": "sec_companyfacts"},
        "fred": {"ttl_key": "macro"},
        "gdelt": {"ttl_key": "gdelt_news"},
        "earnings_transcripts": {"ttl_key": "transcripts"},
        "ir_pages": {"ttl_key": "ir_pages"},
        "estimates": {"ttl_key": "estimates"},
    }

    def list_scheduler_sources(self, *, limit: int = 100, offset: int = 0) -> list[dict]:
        """Project scheduler cadence from persisted cache rows only."""
        self._validate_corpus_page(limit, offset)
        persisted = {
            str(row.get("source") or "").removeprefix("unified:"): row
            for row in self.sqlite.list_scheduler_sources(
                limit=self.MAX_CORPUS_PAGE_LIMIT, offset=0,
            )
        }
        ttls = self._schedule_ttls()
        rows = []
        for source, cfg in self.SCHEDULER_SOURCES.items():
            row = persisted.get(source, {})
            rows.append({
                "source": source,
                "ttl_key": cfg["ttl_key"],
                "ttl_hours": int(ttls.get(cfg["ttl_key"], 24)),
                "status": row.get("status", "never_fetched"),
                "last_run": row.get("last_run"),
                "next_scheduled_update": row.get("next_scheduled_update"),
                "error_message": row.get("error_message"),
            })
        return rows[offset:offset + limit]

    def search_document_families(
        self,
        *,
        query: Optional[str] = None,
        source: Optional[str] = None,
        ticker: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Search non-SEC Chroma document families through the facade."""
        self._validate_corpus_page(limit, offset)
        return self.chroma.search_document_families(
            query=query, source=source, ticker=ticker, date_from=date_from,
            date_to=date_to, limit=limit, offset=offset,
        )

    def get_filing_section_families(
        self, accession: str, *, limit: int = 100, offset: int = 0,
    ) -> list[dict]:
        """Read one bounded filing's section families through the facade."""
        self._validate_corpus_page(limit, offset)
        return self.chroma.get_filing_section_families(
            accession, limit=limit, offset=offset,
        )

    def get_document_family(
        self, parent_id: str, *, limit: int = 1, offset: int = 0,
    ) -> list[dict]:
        """Read a bounded document-family detail page through the facade."""
        self._validate_corpus_page(limit, offset)
        return self.chroma.get_document_family(
            parent_id, limit=limit, offset=offset,
        )

    def get_document(self, document_id: str) -> Optional[dict]:
        """Read one Chroma document for a bounded explorer detail request."""
        return self.chroma.get_document(document_id)

    search_metrics = search_corpus_metrics
    search_facts_for_corpus = search_corpus_facts
    list_filing_inventory = list_filings
    get_freshness_summaries = list_freshness
    list_document_families = search_document_families
    list_filing_section_families = get_filing_section_families

    def delete_filing_section_family(self, parent_id: str) -> None:
        self._bump_revision("delete_filing_section_family")
        self.chroma.delete_filing_section_family(parent_id)

    # ── Hybrid Search ─────────────────────────────────

    def search(self, query: str, n_results: int = 5,
               ticker: str = None) -> dict:
        """
        Search BOTH stores and return combined results.

        Args:
            query: Natural language query
            n_results: Max semantic results
            ticker: Optional ticker filter

        Returns:
            {
                "documents": [...],   # ChromaDB semantic matches
                "facts": [...],       # SQLite structured matches
                "ticker": "NVDA",     # Detected or provided ticker
            }
        """
        # Detect ticker from query if not provided
        detected_ticker = ticker or self._detect_ticker(query)

        # Parallel search
        documents = self.chroma.search(
            query=query,
            n_results=n_results,
            filter_dict={"ticker": detected_ticker} if detected_ticker else None
        )

        # Also search without ticker filter for broader context
        if detected_ticker:
            broad_results = self.chroma.search(
                query=query,
                n_results=n_results // 2,
            )
            # Merge: ticker-filtered first, then broad results (deduped)
            seen_ids = {d["id"] for d in documents}
            for doc in broad_results:
                if doc["id"] not in seen_ids:
                    documents.append(doc)
                    seen_ids.add(doc["id"])

        # Get structured facts if ticker detected
        facts = []
        if detected_ticker:
            facts = self.sqlite.search_facts(
                ticker=detected_ticker,
                limit=n_results
            )

        return {
            "documents": documents,
            "facts": facts,
            "ticker": detected_ticker,
        }

    def search_by_ticker(self, query: str, ticker: str,
                         n_results: int = 5) -> dict:
        """Search specifically within one ticker's data."""
        return self.search(query=query, n_results=n_results, ticker=ticker)

    # ── Filing Pipeline Support ───────────────────────

    def register_filing(self, ticker: str, filing_type: str,
                        filing_date: str, period: str,
                        accession: str, source_url: str) -> bool:
        """Register a filing as received (before parsing)."""
        return self.sqlite.register_filing(
            ticker, filing_type, filing_date, period,
            accession, source_url
        )

    def process_filing(self, filing_record: dict,
                       extracted_text: str,
                       extracted_facts: list[dict],
                       *,
                       index_document: bool = True,
                       mark_parsed: bool = True):
        """
        Process a full filing through the model-as-parser pipeline.

        1. Saves the complete text as a ChromaDB document
        2. Saves each extracted fact to SQLite
        3. Marks the filing as parsed

        Args:
            filing_record: The dict from filings table (must include source_type)
            extracted_text: Full document text (or summary)
            extracted_facts: List of {metric, value, unit, period} dicts
        """
        # 1. Store the document embedding
        doc_id = (
            f"{filing_record['source_type']}/{filing_record['ticker']}/"
            f"{filing_record['filing_type']}-{filing_record['period']}"
        )
        if index_document:
            self.save_document(
                document_id=doc_id,
                text=extracted_text,
                ticker=filing_record["ticker"],
                source=filing_record["filing_type"],
                date=filing_record["filing_date"],
            )

        # 2. Save each extracted fact
        for fact in extracted_facts:
            self.save_fundamental(
                ticker=filing_record["ticker"],
                metric=fact.get("metric"),
                value=fact.get("value"),
                unit=fact.get("unit", "usd"),
                period=fact.get("period", filing_record["period"]),
                source_type=filing_record["source_type"],
                source_url=filing_record["source_url"],
            )

        # 3. Mark as parsed
        if mark_parsed:
            self.sqlite.mark_filing_parsed(
                filing_record["accession"],
                embedding_id=doc_id
            )

    # ── Cache Management ──────────────────────────────

    def get_cache_status(self, ticker: str, source: str) -> Optional[dict]:
        """Get cache freshness metadata for a ticker + source."""
        return self.sqlite.get_cache_status(ticker, source)

    def mark_cache_fresh(self, ticker: str, source: str, ttl_hours: int = 24):
        self.sqlite.mark_cache_fresh(ticker, source, ttl_hours)

    def mark_cache_stale(self, ticker: str, source: str, error: str = None):
        self.sqlite.mark_cache_stale(ticker, source, error)

    def upsert_cache_stale(self, ticker: str, source: str, error: str = None):
        self.sqlite.upsert_cache_stale(ticker, source, error)

    def get_stale_entries(self, limit: int = 20) -> list[dict]:
        return self.sqlite.get_stale_cache_entries(limit)

    # ── Freshness / Staleness-Aware Querying (Phase 1.7.4) ─

    # Maps a logical data source to its cache_meta source identifier and the
    # watchlist.yaml schedule key that defines its TTL (in hours).
    FRESHNESS_SOURCES = {
        "yfinance_fundamentals": {"cache_source": "yfinance_fundamentals", "ttl_key": "fundamentals"},
        "yfinance_news":         {"cache_source": "yfinance_news",         "ttl_key": "news"},
        "sec_filings":           {"cache_source": "sec_filings_discovery", "ttl_key": "sec_filings"},
        "sec_companyfacts":      {"cache_source": "sec_companyfacts",      "ttl_key": "sec_companyfacts"},
        "gdelt_news":            {"cache_source": "gdelt_news",            "ttl_key": "gdelt_news"},
        "earnings_transcripts":  {"cache_source": "earnings_transcripts",  "ttl_key": "transcripts"},
        "ir_pages":              {"cache_source": "ir_pages",              "ttl_key": "ir_pages"},
        "estimates":             {"cache_source": "estimates",             "ttl_key": "estimates"},
    }

    _DEFAULT_TTLS = {
        "fundamentals": 24, "news": 6, "macro": 24, "sec_filings": 12,
        "sec_companyfacts": 24,
        "gdelt_news": 6, "transcripts": 168, "ir_pages": 24, "estimates": 24,
    }

    def _schedule_ttls(self) -> dict:
        """Lazily load the schedule TTL map from configs/watchlist.yaml."""
        cached = getattr(self, "_ttl_cache", None)
        if cached is not None:
            return cached
        ttls = dict(self._DEFAULT_TTLS)
        try:
            import yaml
            wl_path = Path(__file__).parent.parent.parent / "configs/watchlist.yaml"
            if wl_path.exists():
                with open(wl_path) as f:
                    config = yaml.safe_load(f) or {}
                ttls.update(config.get("schedule", {}) or {})
        except Exception:
            pass
        self._ttl_cache = ttls
        return ttls

    @staticmethod
    def _age_hours(last_updated) -> Optional[float]:
        """Compute age in hours from a cache_meta last_updated value."""
        if not last_updated:
            return None
        from datetime import datetime, timezone
        try:
            if isinstance(last_updated, str):
                try:
                    from dateutil import parser
                    dt = parser.parse(last_updated)
                except Exception:
                    dt = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = last_updated
            return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception:
            return None

    def is_ticker_fresh(self, ticker: str, source: str, ttl_hours: int) -> bool:
        """Check if a specific source for a ticker exists and is within TTL.

        Args:
            ticker: Ticker symbol
            source: cache_meta source identifier (e.g., "yfinance_fundamentals")
            ttl_hours: TTL in hours

        Returns:
            True if data exists and is within TTL, False otherwise.
        """
        status = self.sqlite.get_cache_status(ticker, source)
        if not status:
            return False
        if status.get("status") == "stale":
            return False
        age = self._age_hours(status.get("last_updated"))
        if age is None:
            return False
        return age < ttl_hours

    def mark_source_fresh(self, ticker: str, source: str, ttl_hours: int):
        """Mark a source as freshly updated (now + ttl_hours)."""
        self.sqlite.mark_cache_fresh(ticker, source, ttl_hours)

    def mark_source_stale(self, ticker: str, source: str, error: str = ""):
        """Mark a source as stale (e.g., after a failed fetch). Upserts."""
        self.sqlite.upsert_cache_stale(ticker, source, error or None)

    def get_freshness_report(self, ticker: str) -> dict:
        """Get freshness status for all data sources for a given ticker.

        Returns a dict with per-source status ("fresh" | "stale" |
        "never_fetched"), age_hours, ttl_hours, last_updated, plus an
        overall rollup and the list of stale source names.
        """
        ttls = self._schedule_ttls()
        sources: dict[str, dict] = {}
        stale_sources: list[str] = []
        fresh_count = 0
        present_count = 0

        for name, cfg in self.FRESHNESS_SOURCES.items():
            ttl_hours = ttls.get(cfg["ttl_key"], self._DEFAULT_TTLS.get(cfg["ttl_key"], 24))
            cache = self.sqlite.get_cache_status(ticker, cfg["cache_source"])

            if not cache:
                sources[name] = {
                    "status": "never_fetched",
                    "last_updated": None,
                    "age_hours": None,
                    "ttl_hours": ttl_hours,
                }
                continue

            present_count += 1
            age = self._age_hours(cache.get("last_updated"))
            if cache.get("status") == "stale":
                status = "stale"
            elif age is not None and age < ttl_hours:
                status = "fresh"
            else:
                status = "stale"

            if status == "fresh":
                fresh_count += 1
            else:
                stale_sources.append(name)

            sources[name] = {
                "status": status,
                "last_updated": str(cache.get("last_updated")) if cache.get("last_updated") else None,
                "age_hours": round(age, 2) if age is not None else None,
                "ttl_hours": ttl_hours,
            }

        if present_count == 0:
            overall = "never_fetched"
        elif fresh_count == present_count:
            overall = "fresh"
        elif fresh_count == 0:
            overall = "stale"
        else:
            overall = "partial"

        return {
            "ticker": ticker,
            "sources": sources,
            "overall": overall,
            "stale_sources": stale_sources,
        }

    def get_stale_tickers(self, source: str) -> list[str]:
        """Get all tickers whose data for a given logical source is stale.

        ``source`` may be a logical name from FRESHNESS_SOURCES (e.g.
        "yfinance_fundamentals") or a raw cache_meta source identifier.
        Tickers with no cache entry are not included (use the watchlist for
        never-fetched tickers).
        """
        cfg = self.FRESHNESS_SOURCES.get(source)
        cache_source = cfg["cache_source"] if cfg else source
        ttl_key = cfg["ttl_key"] if cfg else None
        ttl_hours = self._schedule_ttls().get(ttl_key, 24) if ttl_key else 24

        sql = "SELECT ticker, last_updated, status FROM cache_meta WHERE source=?"
        with self.sqlite._connect() as conn:
            rows = conn.execute(sql, (cache_source,)).fetchall()

        stale = []
        for row in rows:
            r = dict(row)
            if r.get("status") == "stale":
                stale.append(r["ticker"])
                continue
            age = self._age_hours(r.get("last_updated"))
            if age is None or age >= ttl_hours:
                stale.append(r["ticker"])
        return stale

    # ── Utilities ─────────────────────────────────────

    @staticmethod
    def _detect_ticker(text: str) -> Optional[str]:
        """
        Simple ticker detection from query text.
        Looks for known company names or uppercase 1-4 letter stock symbols.
        """
        import re
        # Known major tickers to prioritize
        known_tickers = {
            "NVDA", "AMD", "AAPL", "MSFT", "GOOGL", "GOOG", "META",
            "AMZN", "TSLA", "INTC", "CRM", "AVGO", "ORCL", "CSCO",
            "IBM", "QCOM", "TXN", "MU", "MRVL", "PLTR", "SNOW",
            "CRWD", "PANW", "UBER", "SQ", "HOOD", "COIN", "MSTR",
        }

        # Resolve common company names to ticker symbols first.
        # This keeps hybrid search useful for natural-language queries.
        company_name_map = {
            "nvidia": "NVDA",
            "advanced micro devices": "AMD",
            "amd": "AMD",
            "apple": "AAPL",
            "microsoft": "MSFT",
            "alphabet": "GOOGL",
            "google": "GOOGL",
            "meta": "META",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "intel": "INTC",
            "salesforce": "CRM",
            "broadcom": "AVGO",
            "oracle": "ORCL",
            "cisco": "CSCO",
            "ibm": "IBM",
            "qualcomm": "QCOM",
            "texas instruments": "TXN",
            "micron": "MU",
            "marvell": "MRVL",
            "palantir": "PLTR",
            "snowflake": "SNOW",
            "crowdstrike": "CRWD",
            "palo alto networks": "PANW",
            "uber": "UBER",
            "block": "SQ",
            "robinhood": "HOOD",
            "coinbase": "COIN",
            "microstrategy": "MSTR",
            "strategy": "MSTR",
        }
        normalized_text = text.lower()
        for company_name, ticker in company_name_map.items():
            if company_name in normalized_text:
                return ticker

        # Find all uppercase words that look like tickers
        candidates = set(re.findall(r'\b[A-Z]{1,4}\b', text))

        # Match against known tickers first
        for c in candidates:
            if c in known_tickers:
                return c

        # Fallback: return the first match that isn't a common word
        common_words = {"I", "A", "AN", "THE", "IT", "IS", "BE", "TO",
                        "OF", "IN", "ON", "AT", "BY", "AS", "OR", "IF",
                        "NO", "GO", "DO", "WE", "HE", "SHE", "ALL"}

        for c in candidates:
            if c not in common_words and len(c) >= 1:
                return c

        return None

    # ── Cleanup / Reset ───────────────────────────────

    def reset(self):
        """Clear all data (for testing)."""
        self._bump_revision("reset")
        self.chroma.reset_collection()
        # For SQLite, just drop and recreate tables
        with self.sqlite._connect() as conn:
            conn.executescript("""
                DROP TABLE IF EXISTS event_corpus_items;
                DROP TABLE IF EXISTS event_securities;
                DROP TABLE IF EXISTS corpus_events;
                DROP TABLE IF EXISTS observation_securities;
                DROP TABLE IF EXISTS corpus_observations;
                DROP TABLE IF EXISTS corpus_item_securities;
                DROP TABLE IF EXISTS corpus_item_sources;
                DROP TABLE IF EXISTS corpus_items;
                DROP TABLE IF EXISTS universe_errors;
                DROP TABLE IF EXISTS security_memberships;
                DROP TABLE IF EXISTS security_aliases;
                DROP TABLE IF EXISTS securities;
                DROP TABLE IF EXISTS fundamentals;
                DROP TABLE IF EXISTS filings;
                DROP TABLE IF EXISTS cache_meta;
                DROP TABLE IF EXISTS ingestion_log;
            """)
            conn.commit()
        self.sqlite._init_schema()

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

from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash
from src.ingestion.records import EventRecord, NarrativeRecord, ObservationRecord

from .chroma_store import ChromaStore
from .retention import RetentionPolicy, load_retention_policy
from .sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

STRUCTURED_ONLY_NARRATIVE_TYPES = frozenset({
    "observation",
    "market_bar",
    "ohlcv",
    "rate",
    "economic_observation",
    "security",
    "membership",
    "alias",
    "refresh_status",
    "cursor",
    "run_health",
})
STRUCTURED_ONLY_SOURCE_CATEGORIES = frozenset({"market_data", "economic_data"})


def _chroma_evidence_filter(filters: dict, ticker: Optional[str]) -> Optional[dict]:
    """Translate exact/range stable facets into Chroma metadata predicates."""
    clauses: list[dict] = []
    if ticker:
        clauses.append({"ticker": ticker})

    exact_fields = {
        "source_category": "source_category",
        "source": "source_name",
        "item_type": "item_type",
        "event_type": "event_type",
        "form": "form",
        "item": "item",
        "exhibit": "exhibit",
        "freshness_status": "freshness_status",
        "indexing_status": "indexing_status",
        "authority_tier": "authority_tier",
    }
    for facet, metadata_field in exact_fields.items():
        value = filters.get(facet)
        if value not in (None, ""):
            clauses.append({metadata_field: value})

    for prefix, metadata_field in (
        ("published", "published_at"),
        ("effective", "effective_at"),
        ("as_of", "as_of_at"),
    ):
        bounds = {}
        if filters.get(f"{prefix}_from"):
            bounds["$gte"] = str(filters[f"{prefix}_from"])
        if filters.get(f"{prefix}_to"):
            bounds["$lte"] = str(filters[f"{prefix}_to"])
        if bounds:
            clauses.append({metadata_field: bounds})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


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

    def _prepare_lexical_chunks(self, document_id: str, text: str, **kwargs) -> list[dict]:
        """Prepare stable chunks without embedding; tolerate simple test doubles."""
        prepare = getattr(self.chroma, "prepare_document", None)
        if callable(prepare):
            try:
                rows = prepare(document_id=document_id, text=text, **kwargs)
                if isinstance(rows, list):
                    return rows
            except Exception:  # noqa: BLE001 - indexing write still owns failure handling
                logger.debug("Chroma chunk preparation fallback used", exc_info=True)
        metadata = dict(kwargs.get("metadata") or {})
        if kwargs.get("ticker"):
            metadata["ticker"] = str(kwargs["ticker"]).upper()
        if kwargs.get("source"):
            metadata["source"] = kwargs["source"]
        if kwargs.get("date"):
            metadata["date"] = kwargs["date"]
        family_id = str(metadata.get("document_family_id") or document_id)
        chunk_id = f"{family_id}#0" if kwargs.get("replace_family") else document_id
        return [{"id": chunk_id, "document": text, "metadata": metadata}]

    def _mark_chroma_revision(self, revision: int) -> None:
        """Expose a generation only after its SQLite and Chroma writes succeeded."""
        marker = getattr(self.chroma, "mark_corpus_revision", None)
        if not callable(marker):
            return
        try:
            marker(int(revision))
        except Exception:  # noqa: BLE001 - mismatch safely disables lexical fusion
            logger.warning("Failed to publish Chroma corpus revision", exc_info=True)

    # ── Health ─────────────────────────────────────────

    def heartbeat(self) -> dict:
        """Check both storage backends are responsive."""
        return {
            "sqlite": self._check_sqlite(),
            "chroma": self.chroma.heartbeat(),
            "chroma_doc_count": self.chroma.count(),
        }

    def migrate_phase2_3(
        self,
        *,
        batch_size: int = 100,
        max_batches: Optional[int] = None,
    ) -> dict[str, object]:
        """Run the resumable Phase 2.3 metadata backfill through the facade."""
        from src.storage.phase2_3_backfill import Phase23Backfill

        return Phase23Backfill(self, batch_size=batch_size).run(
            max_batches=max_batches
        )

    def rebuild_lexical_index(
        self,
        *,
        batch_size: int = 100,
        max_batches: Optional[int] = None,
        restart: bool = False,
    ) -> dict:
        """Backfill persistent lexical rows from bounded Chroma pages."""
        revision = self.retrieval_revision()

        def _page(offset: int, limit: int) -> list[dict]:
            pages = self.chroma.iter_document_batches(
                batch_size=limit, offset=offset
            )
            return next(pages, [])

        result = self.sqlite.rebuild_lexical_index(
            _page,
            batch_size=batch_size,
            target_revision=revision,
            max_batches=max_batches,
            restart=restart,
        )
        if result.get("status") == "completed":
            self._mark_chroma_revision(revision)
        return result

    def reconcile_lexical_index(
        self,
        *,
        repair: bool = False,
        batch_size: int = 100,
    ) -> dict:
        """Stream Chroma chunks and report/repair lexical drift."""
        revision = self.retrieval_revision()

        def _chunks():
            for batch in self.chroma.iter_document_batches(batch_size=batch_size):
                yield from batch

        result = self.sqlite.reconcile_lexical_index(
            _chunks(), repair=repair, revision=revision
        )
        if repair:
            self._mark_chroma_revision(revision)
        return result

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

    def corpus_revision(self) -> int:
        """Return the generation shared by persistent lexical rows and Chroma."""
        return int(self.sqlite.get_lexical_index_state()["indexed_revision"])

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

    def describe_coverage(
        self,
        operation: str = "summary",
        *,
        ticker: Optional[str] = None,
        filters: Optional[dict] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        ticker_only: bool = False,
    ) -> dict:
        """Return the bounded, read-only capability inventory projection."""
        return self.sqlite.describe_coverage(
            operation,
            ticker=ticker,
            filters=filters,
            limit=limit,
            cursor=cursor,
            ticker_only=ticker_only,
        )

    def resolve_security(
        self,
        symbol: str,
        provider: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve canonical, vendor, or historical security symbols."""
        return self.sqlite.resolve_security(symbol, provider=provider, as_of=as_of)

    def resolve_exact_security(self, identifier: str) -> Optional[dict]:
        """Resolve one exact registry identity; ambiguous names never attach."""
        return self.sqlite.resolve_exact_security(identifier)

    def register_security_alias(
        self,
        security_id: str,
        alias: str,
        *,
        alias_type: str = "issuer_alias",
        provider: Optional[str] = None,
        source: str = "registry",
    ) -> bool:
        """Register an exact issuer, manufacturer, or recipient UEI identity."""
        return self.sqlite.register_security_alias(
            security_id,
            alias,
            alias_type=alias_type,
            provider=provider,
            source=source,
        )

    add_security_alias = register_security_alias

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
        """List structured observations through the Store facade, including vintages."""
        return self.sqlite.list_observations(
            source_name=source_name,
            metric_id=metric_id,
            period_end=period_end,
            vintage_at=vintage_at,
            limit=limit,
            offset=offset,
        )

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
        if (
            record.item_type in STRUCTURED_ONLY_NARRATIVE_TYPES
            or record.source_category in STRUCTURED_ONLY_SOURCE_CATEGORIES
        ):
            raise ValueError(
                f"{record.item_type!r} is structured-only and cannot use narrative placement"
            )
        security_ids = set(record.security_ids)
        tickers = set(record.tickers)
        index_memberships = set(record.index_codes)
        sectors = set(record.sectors)
        industries: set[str] = set()
        for security_id in record.security_ids:
            security = self.sqlite.get_security(security_id)
            if security:
                tickers.add(str(security["ticker"]))
                if security.get("sector"):
                    sectors.add(str(security["sector"]))
                if security.get("industry"):
                    industries.add(str(security["industry"]))
            for membership in self.sqlite.list_memberships(
                security_id=security_id, active=True,
            ):
                index_memberships.add(str(membership["index_code"]))

        chroma_metadata = {
            "source_category": record.source_category,
            "source_name": record.source_name,
            "item_type": record.item_type,
            "document_family": record.document_family,
            "content_hash": record.content_hash,
            "license_label": record.license_label,
            "normalization_version": record.normalization_version,
            "evidence_authority": record.evidence_authority,
            "authority_tier": record.evidence_authority,
        }
        optional_metadata = {
            "provider_record_id": record.provider_record_id,
            "original_publisher": record.original_publisher,
            "canonical_url": record.canonical_url,
            "event_type": record.event_type,
            "published_at": record.published_at,
            "effective_at": record.effective_at,
            "as_of_at": record.as_of_at,
        }
        chroma_metadata.update({
            key: value for key, value in optional_metadata.items() if value is not None
        })
        record_metadata = {
            key: value for key, value in dict(record.metadata).items()
            if value is not None and isinstance(value, (str, int, float, bool))
        }
        chroma_metadata.update(record_metadata)
        if "filing_item" in record_metadata and "item" not in record_metadata:
            chroma_metadata["item"] = record_metadata["filing_item"]
        if security_ids:
            chroma_metadata["security_ids"] = ",".join(sorted(security_ids))
        if tickers:
            chroma_metadata["tickers"] = ",".join(sorted(tickers))
        if index_memberships:
            chroma_metadata["index_memberships"] = ",".join(sorted(index_memberships))
        if sectors:
            chroma_metadata["sectors"] = ",".join(sorted(sectors))
        if industries:
            chroma_metadata["industries"] = ",".join(sorted(industries))

        narrative_text = record.body
        if record.item_type == "news":
            narrative_text = "\n\n".join(
                value for value in (record.title, record.summary) if value
            )

        def _lexical_chunks(item_id: str) -> list[dict]:
            metadata = {
                **chroma_metadata,
                "corpus_item_id": item_id,
                "document_family_id": item_id,
                "title": record.title,
            }
            return self._prepare_lexical_chunks(
                item_id,
                narrative_text,
                ticker=record.tickers[0] if record.tickers else None,
                source=record.source_name,
                date=record.published_at,
                metadata=metadata,
                replace_family=True,
            )

        result = self.sqlite.upsert_narrative_record(
            record, lexical_chunks=_lexical_chunks
        )
        item_id = result["corpus_item_id"]
        if not result["needs_index"]:
            self._mark_chroma_revision(result["revision"])
            return result
        chroma_metadata.update({
            "corpus_item_id": item_id,
            "document_family_id": item_id,
        })

        try:
            self.chroma.add_document(
                document_id=item_id,
                text=narrative_text,
                ticker=record.tickers[0] if record.tickers else None,
                source=record.source_name,
                date=record.published_at,
                metadata=chroma_metadata,
                replace_family=True,
            )
        except Exception as exc:  # noqa: BLE001 - metadata must survive Chroma failure
            logger.warning("Narrative indexing failed for %s", item_id, exc_info=True)
            self.sqlite.set_corpus_index_status(item_id, "error", str(exc))
            result["indexing_status"] = "error"
            result["index_error"] = str(exc)[:2_000]
            return result

        self.sqlite.set_corpus_index_status(item_id, "indexed")
        self._mark_chroma_revision(result["revision"])
        result["indexing_status"] = "indexed"
        result["index_error"] = None
        return result

    def upsert_observation(self, record: ObservationRecord) -> dict:
        """Persist a structured observation without embedding it."""
        return self.sqlite.upsert_observation_record(record)

    def upsert_event(self, record: EventRecord) -> dict:
        """Persist a structured event and its item/security links."""
        return self.sqlite.upsert_event_record(record)

    def repair_corpus_item(self, corpus_item_id: str) -> dict:
        """Retry one stored narrative index without contacting its provider."""
        item_id = str(corpus_item_id or "").strip()
        if not item_id:
            raise ValueError("corpus_item_id is required")
        item = self.sqlite.get_corpus_item(item_id)
        if item is None:
            return {
                "corpus_item_id": item_id,
                "status": "skipped",
                "reason": "not_found",
            }
        if item.get("indexing_status") not in {"pending", "error"}:
            return {
                "corpus_item_id": item_id,
                "status": "skipped",
                "reason": "not_retryable",
            }
        if str(item.get("item_type") or "").lower() != "news":
            return {
                "corpus_item_id": item_id,
                "status": "skipped",
                "reason": "full_content_not_retained",
            }
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        text = "\n\n".join(value for value in (title, summary) if value)
        if not text:
            message = "stored narrative has no indexable title or summary"
            self.sqlite.set_corpus_index_status(item_id, "error", message)
            return {
                "corpus_item_id": item_id,
                "status": "failed",
                "reason": "content_unavailable",
                "error": message,
            }

        security_rows = self.sqlite.list_corpus_item_securities(item_id)
        tickers = [str(value) for value in (item.get("tickers") or []) if value]
        if not tickers:
            tickers = [
                str(row.get("ticker")) for row in security_rows if row.get("ticker")
            ]
        metadata = {
            "corpus_item_id": item_id,
            "document_family_id": str(item.get("document_family_id") or item_id),
            "source_category": item.get("source_category"),
            "source_name": item.get("source"),
            "item_type": item.get("item_type"),
            "document_family": item.get("document_family"),
            "content_hash": item.get("content_hash"),
            "license_label": item.get("license_label"),
            "normalization_version": item.get("normalization_version"),
            "evidence_authority": item.get("evidence_authority"),
            "provider_record_id": item.get("provider_record_id"),
            "original_publisher": item.get("original_publisher"),
            "canonical_url": item.get("canonical_url"),
            "event_type": item.get("event_type"),
            "published_at": item.get("published_at"),
            "effective_at": item.get("effective_at"),
            "as_of_at": item.get("as_of_at"),
            "indexing_status": "indexed",
        }
        for key, value in (item.get("metadata") or {}).items():
            if isinstance(value, (str, int, float, bool)):
                metadata[str(key)] = value
        if tickers:
            metadata["tickers"] = ",".join(sorted(set(tickers)))
        if security_rows:
            metadata["security_ids"] = ",".join(
                sorted({str(row["security_id"]) for row in security_rows})
            )

        try:
            lexical_chunks = self._prepare_lexical_chunks(
                item_id, text, ticker=tickers[0] if tickers else None,
                source=str(item.get("source") or ""), date=item.get("published_at"),
                metadata=metadata, replace_family=True,
            )
            revision = self.sqlite.replace_lexical_family(item_id, lexical_chunks)
            self.chroma.add_document(
                document_id=item_id,
                text=text,
                ticker=tickers[0] if tickers else None,
                source=str(item.get("source") or ""),
                date=item.get("published_at"),
                metadata=metadata,
                replace_family=True,
            )
        except Exception as exc:  # noqa: BLE001 - repair stays item-isolated
            message = str(exc)[:2_000]
            self.sqlite.set_corpus_index_status(item_id, "error", message)
            logger.warning("Narrative repair failed for %s: %s", item_id, message)
            return {
                "corpus_item_id": item_id,
                "status": "failed",
                "error": message,
            }

        self.sqlite.set_corpus_index_status(item_id, "indexed")
        self._mark_chroma_revision(revision)
        return {"corpus_item_id": item_id, "status": "completed"}

    def repair_filing_index(self, accession: str) -> dict:
        """Retry one stored SEC artifact without invoking the SEC fetcher."""
        item_id = str(accession or "").strip()
        if not item_id:
            raise ValueError("accession is required")
        filing = self.sqlite.get_retryable_filing(item_id)
        if filing is None:
            return {"accession": item_id, "status": "skipped", "reason": "not_retryable"}
        artifact = filing.get("file_path")
        if not artifact:
            message = "stored SEC artifact path is unavailable"
            self.sqlite.mark_filing_index_pending(item_id, file_path="", error=message)
            return {"accession": item_id, "status": "failed", "error": message}

        try:
            from src.sec.filing_sections import split_filing_sections

            path = Path(str(artifact))
            text = path.read_text(encoding="utf-8", errors="replace")
            sections = split_filing_sections(text, {
                **filing,
                "accession": item_id,
                "filing_type": filing.get("filing_type"),
                "file_path": str(path),
            })[:500]
            if not sections:
                raise ValueError("stored SEC artifact contains no indexable sections")
            counts = self.add_filing_sections(sections)
            if int(counts.get("sections_written") or 0) != len(sections):
                raise RuntimeError("SEC repair did not index every stored section")
            self.sqlite.mark_filing_parsed(
                item_id,
                embedding_id=f"sec:{item_id}",
                file_path=str(path),
                section_count=int(counts.get("sections_written") or 0),
                chunk_count=int(counts.get("chunks_written") or 0),
            )
            return {
                "accession": item_id,
                "status": "completed",
                "sections": int(counts.get("sections_written") or 0),
                "chunks": int(counts.get("chunks_written") or 0),
            }
        except Exception as exc:  # noqa: BLE001 - one artifact stays isolated
            message = str(exc)[:2_000]
            self.sqlite.mark_filing_index_pending(
                item_id, file_path=str(artifact), error=message,
            )
            logger.warning("SEC filing repair failed for %s: %s", item_id, message)
            return {"accession": item_id, "status": "failed", "error": message}

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
        """List stored narrative indexing failures for repair selection."""
        return self.sqlite.list_retryable_corpus_items(
            limit,
            source=source,
            security=security,
            item_id=item_id,
            date_from=date_from,
            date_to=date_to,
            run_id=run_id,
        )

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
        """List stored SEC indexing artifacts through the repair facade."""
        return self.sqlite.list_retryable_filings(
            limit,
            security=security,
            item_id=item_id,
            date_from=date_from,
            date_to=date_to,
            run_id=run_id,
        )

    def run_retention(
        self,
        *,
        as_of: Optional[str] = None,
        apply: bool = False,
        policy: Optional[RetentionPolicy] = None,
        limit: Optional[int] = None,
        eligible_ids: Optional[list[str]] = None,
    ) -> dict:
        """Preview or explicitly apply configured narrative retention."""
        active_policy = policy or load_retention_policy()
        cutoff = active_policy.company_news_cutoff(as_of)
        run_limit = active_policy.max_items_per_run if limit is None else limit
        candidates = self.sqlite.list_news_retention_candidates(
            cutoff,
            limit=run_limit,
            document_family_ids=eligible_ids,
        )
        family_ids = [str(row["document_family_id"]) for row in candidates]
        result = {
            "apply": apply,
            "cutoff": cutoff,
            "eligible": len(candidates),
            "expired": 0,
            "failed": 0,
            "document_family_ids": family_ids,
        }
        if not apply or not candidates:
            return result

        expired_item_ids: list[str] = []
        for candidate in candidates:
            family_id = str(candidate["document_family_id"])
            try:
                self.chroma.delete_document_family(family_id)
            except Exception:  # noqa: BLE001 - one family must not stop maintenance
                logger.warning(
                    "Retention could not delete narrative family %s", family_id,
                    exc_info=True,
                )
                result["failed"] += 1
                continue
            expired_item_ids.append(str(candidate["corpus_item_id"]))

        retired_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        result["expired"] = self.sqlite.expire_news_narratives(
            expired_item_ids,
            retired_at=retired_at,
            reason=f"company_news_older_than_{active_policy.company_news_months}_months",
        )
        if result["expired"]:
            self._mark_chroma_revision(self.retrieval_revision())
        if result["expired"] != len(expired_item_ids):
            result["failed"] += len(expired_item_ids) - result["expired"]
        return result

    def get_corpus_accounting(
        self,
        group_by: str,
        *,
        source_category: Optional[str] = None,
        source: Optional[str] = None,
        item_type: Optional[str] = None,
        event_type: Optional[str] = None,
        security: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        index: Optional[str] = None,
        coverage_tier: Optional[str] = None,
        year: Optional[str] = None,
        month: Optional[str] = None,
        indexing_state: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Expose bounded SQLite corpus aggregates without scanning Chroma."""
        return self.sqlite.get_corpus_accounting(
            group_by,
            source_category=source_category,
            source=source,
            item_type=item_type,
            event_type=event_type,
            security=security,
            sector=sector,
            industry=industry,
            index=index,
            coverage_tier=coverage_tier,
            year=year,
            month=month,
            indexing_state=indexing_state,
            limit=limit,
            offset=offset,
        )

    def count_corpus_accounting(
        self,
        group_by: str,
        *,
        source_category: Optional[str] = None,
        source: Optional[str] = None,
        item_type: Optional[str] = None,
        event_type: Optional[str] = None,
        security: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        index: Optional[str] = None,
        coverage_tier: Optional[str] = None,
        year: Optional[str] = None,
        month: Optional[str] = None,
        indexing_state: Optional[str] = None,
    ) -> dict:
        """Expose total/distinct corpus accounting counts for one dimension."""
        return self.sqlite.count_corpus_accounting(
            group_by,
            source_category=source_category,
            source=source,
            item_type=item_type,
            event_type=event_type,
            security=security,
            sector=sector,
            industry=industry,
            index=index,
            coverage_tier=coverage_tier,
            year=year,
            month=month,
            indexing_state=indexing_state,
        )

    def list_corpus_items(
        self,
        *,
        source_category: Optional[str] = None,
        source: Optional[str] = None,
        item_type: Optional[str] = None,
        event_type: Optional[str] = None,
        security: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        index: Optional[str] = None,
        year: Optional[str] = None,
        month: Optional[str] = None,
        indexing_state: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Expose one bounded page of corpus-item leaf metadata (no bodies)."""
        return self.sqlite.list_corpus_items(
            source_category=source_category,
            source=source,
            item_type=item_type,
            event_type=event_type,
            security=security,
            sector=sector,
            industry=industry,
            index=index,
            year=year,
            month=month,
            indexing_state=indexing_state,
            limit=limit,
            offset=offset,
        )

    def get_corpus_event(self, event_id: str) -> Optional[dict]:
        """Return one structured corpus event with linked securities."""
        return self.sqlite.get_corpus_event(event_id)

    def list_corpus_item_sources(self, corpus_item_id: str) -> list[dict]:
        """Return provenance source rows for one corpus item."""
        return self.sqlite.list_corpus_item_sources(corpus_item_id)

    def list_corpus_item_securities(self, corpus_item_id: str) -> list[dict]:
        """Return the securities a corpus item links to."""
        return self.sqlite.list_corpus_item_securities(corpus_item_id)

    def get_corpus_item(self, corpus_item_id: str) -> Optional[dict]:
        """Return one corpus-item metadata row (no narrative body)."""
        return self.sqlite.get_corpus_item(corpus_item_id)

    # -- Incremental source cursors -----------------------------------------

    def get_source_cursor_state(self, source: str, partition_key: str) -> Optional[dict]:
        """Read source cursor/status state through the persistence facade."""
        return self.sqlite.get_source_cursor_state(source, partition_key)

    def get_source_cursor(self, source: str, partition_key: str) -> Optional[str]:
        """Read the last committed source cursor for one logical partition."""
        return self.sqlite.get_source_cursor(source, partition_key)

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
        """Commit a source cursor and its bounded status metadata."""
        return self.sqlite.set_source_cursor(
            source,
            partition_key,
            cursor_value,
            cursor_type=cursor_type,
            overlap_value=overlap_value,
            last_successful_run_id=last_successful_run_id,
            version=version,
            status=status,
            error_class=error_class,
            error_message=error_message,
            retry_after=retry_after,
        )

    def set_source_cursors(self, updates: list[dict]) -> list[dict]:
        """Atomically commit multiple logical source cursor partitions."""
        return self.sqlite.set_source_cursors(updates)

    def list_source_cursor_states(self, source: str) -> list[dict]:
        """List persisted cursor states for fair partition work selection."""
        return self.sqlite.list_source_cursor_states(source)

    def get_source_budget_usage(
        self,
        source: str,
        *,
        day_start: str,
        minute_start: str,
    ) -> dict:
        """Load durable day/minute quota usage for one provider source."""
        return self.sqlite.get_source_budget_usage(
            source,
            day_start=day_start,
            minute_start=minute_start,
        )

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
        """Persist one run's quota usage without coupling it to freshness."""
        self.sqlite.record_source_budget_usage(
            source,
            day_start=day_start,
            minute_start=minute_start,
            attempted_requests=attempted_requests,
            successful_requests=successful_requests,
            provider_remaining=provider_remaining,
            provider_reset=provider_reset,
        )

    # -- Scheduler run status (2.3.4.3) -------------------------------------

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
        """Persist one scheduler run header through the Store facade."""
        return self.sqlite.start_scheduler_run(
            mode,
            policy_revision=policy_revision,
            config_revision=config_revision,
            requested_sources=requested_sources,
            run_id=run_id,
            started_at=started_at,
            bootstrap_manifest=bootstrap_manifest,
        )

    def record_scheduler_source_summary(self, summary: dict) -> None:
        """Persist one scheduler source summary without exposing raw payloads."""
        self.sqlite.record_scheduler_source_summary(summary)

    def complete_scheduler_run(self, run_id: str, **kwargs: object) -> None:
        """Complete one scheduler run and its bounded terminal summary."""
        self.sqlite.complete_scheduler_run(run_id, **kwargs)

    def record_bootstrap_partition(self, *args: object, **kwargs: object) -> None:
        """Persist one resumable bootstrap partition checkpoint."""
        self.sqlite.record_bootstrap_partition(*args, **kwargs)

    def list_bootstrap_partitions(
        self, run_id: str, source: Optional[str] = None,
    ) -> list[dict]:
        """List bootstrap checkpoints through the Store facade."""
        return self.sqlite.list_bootstrap_partitions(run_id, source)

    def get_resumable_bootstrap_run(self, source: Optional[str] = None) -> Optional[dict]:
        """Return the newest unfinished bootstrap run."""
        return self.sqlite.get_resumable_bootstrap_run(source)

    def list_scheduler_runs(
        self, *, limit: int = 20, source: Optional[str] = None,
    ) -> list[dict]:
        """Return bounded scheduler run history."""
        return self.sqlite.list_scheduler_runs(limit=limit, source=source)

    def prune_scheduler_history(self, max_runs: int = 100) -> int:
        """Prune old scheduler summaries without touching corpus data."""
        return self.sqlite.prune_scheduler_history(max_runs)

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
        """Persist a source status without changing its last committed cursor."""
        return self.sqlite.set_source_status(
            source,
            partition_key,
            status,
            error_class=error_class,
            error_message=error_message,
            retry_after=retry_after,
        )

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
        lexical_metadata = dict(metadata or {})
        chunks = self._prepare_lexical_chunks(
            document_id, text, ticker=ticker, source=source, date=date,
            metadata=lexical_metadata,
        )
        revision = self.sqlite.replace_lexical_family(document_id, chunks)
        self.chroma.add_document(
            document_id=document_id,
            text=text,
            ticker=ticker,
            source=source,
            date=date,
            metadata=metadata,
        )
        self._mark_chroma_revision(revision)
        return document_id

    def save_documents_batch(self,
                             ids: list[str],
                             texts: list[str],
                             metadatas: list[dict] = None):
        """Store multiple documents at once."""
        metadata_rows = metadatas or [{}] * len(ids)
        families = {
            str(document_id): [{
                "id": str(document_id), "document": texts[index],
                "metadata": dict(metadata_rows[index] or {}),
            }]
            for index, document_id in enumerate(ids)
        }
        revision = self.sqlite.replace_lexical_families(families)
        self.chroma.add_documents_batch(ids, texts, metadatas)
        self._mark_chroma_revision(revision)

    def add_filing_sections(self, sections: list) -> dict[str, int]:
        """Replace and index SEC section families through Chroma's chunker."""
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
            security = self.sqlite.resolve_security(section.ticker)
            security_id = str(security["security_id"]) if security else None
            memberships = self.sqlite.list_memberships(
                security_id=security_id, active=True,
            ) if security_id else []
            section_metadata = {
                "accession": section.accession,
                "as_of_at": section.report_period,
                "authority_tier": "direct_sec",
                "content_hash": content_hash(section.text),
                "corpus_item_id": section.document_id,
                "document_family_id": section.document_id,
                "evidence_authority": "direct_sec",
                "filing_date": section.filing_date,
                "form": section.form,
                "item": section.section_key,
                "item_type": "sec_filing",
                "normalization_version": NORMALIZATION_VERSION,
                "parent_id": section.document_id,
                "provider_record_id": section.accession,
                "published_at": section.filing_date,
                "report_period": section.report_period,
                "section_heading": section.section_heading,
                "section_index": section.section_index,
                "section_key": section.section_key,
                "source_category": "regulatory_filing",
                "source_name": "sec",
                "source_url": section.source_url,
                "parsed_path": section.parsed_path,
                "tickers": section.ticker,
            }
            if security_id:
                section_metadata["security_ids"] = security_id
            if memberships:
                section_metadata["index_memberships"] = ",".join(sorted({
                    str(row["index_code"]) for row in memberships
                }))
            if security and security.get("sector"):
                section_metadata["sectors"] = str(security["sector"])
            if security and security.get("industry"):
                section_metadata["industries"] = str(security["industry"])
            lexical_chunks = self._prepare_lexical_chunks(
                section.document_id, section.text, ticker=section.ticker,
                source="sec_filing", date=section.filing_date,
                metadata=section_metadata, replace_family=True,
            )
            revision = self.sqlite.replace_lexical_family(
                section.document_id, lexical_chunks
            )
            self.chroma.add_document(
                document_id=section.document_id,
                text=section.text,
                ticker=section.ticker,
                source="sec_filing",
                date=section.filing_date,
                metadata=section_metadata,
                replace_family=True,
            )
            self._mark_chroma_revision(revision)
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
        configured = dict(self.SCHEDULER_SOURCES)
        try:
            import yaml

            source_path = Path(__file__).parent.parent.parent / "configs/sources.yaml"
            with open(source_path, encoding="utf-8") as config_file:
                source_config = yaml.safe_load(config_file) or {}
            configured = {
                str(name): {"ttl_key": str(spec["ttl_key"])}
                for name, spec in (source_config.get("sources") or {}).items()
                if isinstance(spec, dict) and spec.get("ttl_key")
            }
        except Exception:
            logger.warning("Could not load scheduler source registry", exc_info=True)
        rows = []
        for source, cfg in configured.items():
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
        revision = self.sqlite.delete_lexical_families([parent_id])
        self.chroma.delete_filing_section_family(parent_id)
        self._mark_chroma_revision(revision)

    # ── Hybrid Search ─────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        ticker: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        Search BOTH stores and return combined results.

        Args:
            query: Natural language query
            n_results: Max semantic results
            ticker: Optional ticker filter
            filters: Optional composable stable evidence facets. ``as_of`` is
                used to resolve historical ticker aliases before searching.

        Returns:
            {
                "documents": [...],   # ChromaDB semantic matches
                "facts": [...],       # SQLite structured matches
                "ticker": "NVDA",     # Detected or provided ticker
            }
        """
        # Detect ticker from query if not provided
        stable_filters = dict(filters or {})
        requested_security = (
            stable_filters.get("security") or stable_filters.get("ticker")
            or stable_filters.get("alias") or ticker
        )
        detected_ticker = ticker or self._detect_ticker(query)
        if requested_security:
            try:
                resolved = self.resolve_security(
                    str(requested_security), as_of=stable_filters.get("as_of")
                )
            except Exception:  # noqa: BLE001 - alias resolution is best-effort
                logger.warning("Historical security resolution failed", exc_info=True)
                resolved = None
            if resolved:
                detected_ticker = str(resolved.get("ticker") or requested_security).upper()
                stable_filters["security"] = detected_ticker
            elif ticker or stable_filters.get("ticker"):
                detected_ticker = str(ticker or stable_filters["ticker"]).upper()

        candidate_n = n_results
        if stable_filters:
            candidate_n = min(100, max(n_results, n_results * 4))

        # Parallel search
        chroma_filter = _chroma_evidence_filter(stable_filters, detected_ticker)
        documents = self.chroma.search(
            query=query,
            n_results=candidate_n,
            filter_dict=chroma_filter,
        )

        # Also search without ticker filter for broader context
        if detected_ticker and not requested_security:
            broad_results = self.chroma.search(
                query=query,
                n_results=max(1, candidate_n // 2),
            )
            # Merge: ticker-filtered first, then broad results (deduped)
            seen_ids = {d["id"] for d in documents}
            for doc in broad_results:
                if doc["id"] not in seen_ids:
                    documents.append(doc)
                    seen_ids.add(doc["id"])

        if stable_filters:
            try:
                from src.middleware.evidence_taxonomy import filter_evidence

                documents = filter_evidence(documents, stable_filters)[:n_results]
            except Exception:  # noqa: BLE001 - taxonomy must never fail a query
                logger.warning("Evidence filtering failed; using semantic order", exc_info=True)
                documents = documents[:n_results]
        else:
            documents = documents[:n_results]

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
                        accession: str, source_url: str, *,
                        cik: Optional[str] = None,
                        primary_document: Optional[str] = None,
                        discovery_scope: str = "deep",
                        items: Optional[list[str]] = None,
                        exhibits: Optional[list[dict]] = None) -> bool:
        """Register a filing as received (before parsing)."""
        metadata = {}
        if cik is not None:
            metadata["cik"] = cik
        if primary_document is not None:
            metadata["primary_document"] = primary_document
        if discovery_scope != "deep":
            metadata["discovery_scope"] = discovery_scope
        if items is not None:
            metadata["items"] = items
        if exhibits is not None:
            metadata["exhibits"] = exhibits
        return self.sqlite.register_filing(
            ticker, filing_type, filing_date, period,
            accession, source_url,
            **metadata,
        )

    def register_sec_daily_index(
        self, index_date: str, source_url: str, filings: list[dict],
    ) -> dict[str, object]:
        """Atomically register a filtered daily index and advance its cursor."""
        return self.sqlite.register_sec_daily_index(index_date, source_url, filings)

    def register_sec_filings(self, filings: list[dict]) -> int:
        """Atomically register a normalized SEC filing batch."""
        return self.sqlite.register_sec_filings(filings)

    def get_sec_daily_index_status(self, index_date: str) -> Optional[str]:
        """Return the processed status for one SEC daily index date."""
        return self.sqlite.get_sec_daily_index_status(index_date)

    def get_sec_daily_index_cursor(self) -> Optional[str]:
        """Return the latest fully committed SEC daily-index date."""
        return self.sqlite.get_sec_daily_index_cursor()

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
        "finnhub_news":          {"cache_source": "finnhub_news",          "ttl_key": "finnhub_news"},
        "massive_market":        {"cache_source": "massive_market",        "ttl_key": "massive_market"},
        "massive_actions":       {"cache_source": "massive_actions",       "ttl_key": "massive_actions"},
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
        "finnhub_news": 6, "massive_market": 24, "massive_actions": 24,
        "gdelt_news": 6, "transcripts": 168, "ir_pages": 24, "estimates": 24,
    }

    def _schedule_ttls(self) -> dict:
        """Lazily load scheduler TTLs from the validated source registry file."""
        cached = getattr(self, "_ttl_cache", None)
        if cached is not None:
            return cached
        ttls = dict(self._DEFAULT_TTLS)
        try:
            import yaml
            source_path = Path(__file__).parent.parent.parent / "configs/sources.yaml"
            if source_path.exists():
                with open(source_path, encoding="utf-8") as config_file:
                    config = yaml.safe_load(config_file) or {}
                for source in (config.get("sources") or {}).values():
                    if not isinstance(source, dict):
                        continue
                    ttl_key = source.get("ttl_key")
                    ttl_hours = source.get("ttl_hours")
                    if ttl_key and ttl_hours is not None:
                        ttls[str(ttl_key)] = int(ttl_hours)
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
                DROP TABLE IF EXISTS corpus_fts;
                DROP TABLE IF EXISTS lexical_index_state;
                DROP TABLE IF EXISTS bootstrap_partitions;
                DROP TABLE IF EXISTS scheduler_run_sources;
                DROP TABLE IF EXISTS scheduler_runs;
                DROP TABLE IF EXISTS source_budget_usage;
                DROP TABLE IF EXISTS source_cursors;
                DROP TABLE IF EXISTS sec_daily_indexes;
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
                DROP TABLE IF EXISTS identity_reconciliation_errors;
                DROP TABLE IF EXISTS phase2_3_backfill_progress;
                DROP TABLE IF EXISTS source_circuit_state;
                DROP TABLE IF EXISTS fundamentals;
                DROP TABLE IF EXISTS sec_companyfacts;
                DROP TABLE IF EXISTS filings;
                DROP TABLE IF EXISTS cache_meta;
                DROP TABLE IF EXISTS ingestion_log;
                DROP TABLE IF EXISTS store_revision;
                DROP TABLE IF EXISTS dead_letter;
                DROP TABLE IF EXISTS schema_migrations;
            """)
            conn.commit()
        self.sqlite._init_schema()

"""
src/middleware/evidence.py
Shared evidence contract for retrieved facts and documents — the canonical
document-body field, usable-row filtering, and evidence counts used to
build prompts, grade grounding, and populate response/eval metadata.

The retriever/Store output contract for a document body is the ``document``
field (Chroma naming). ``text``/``content`` are accepted only as
backward-compatible aliases for older or test-injected callers; every
downstream consumer must read the body through ``document_body()`` rather
than open-coding the fallback chain.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Canonical field first; legacy aliases only as a compatibility fallback.
_DOCUMENT_BODY_FIELDS = ("document", "text", "content")


def evidence_id(row: dict, *, prefix: str = "evidence") -> str:
    """Return a stable best-effort identifier for one evidence row."""
    if not isinstance(row, dict):
        return prefix
    metadata = row.get("metadata") or {}
    explicit = row.get("evidence_id") or row.get("id") or metadata.get("id")
    if explicit not in (None, ""):
        return str(explicit)
    parts = (
        row.get("ticker") or metadata.get("ticker"),
        row.get("metric") or metadata.get("metric"),
        row.get("period") or metadata.get("period"),
        row.get("source_type") or metadata.get("source_type") or metadata.get("source"),
    )
    rendered = ":".join(str(part) for part in parts if part not in (None, ""))
    return f"{prefix}:{rendered}" if rendered else prefix


def evidence_field(row: dict, name: str, default: Any = None) -> Any:
    """Read a common evidence field from the row, then its metadata."""
    if not isinstance(row, dict):
        return default
    value = row.get(name)
    if value is not None:
        return value
    metadata = row.get("metadata") or {}
    return metadata.get(name, default)


def document_body(document: dict) -> str:
    """Return the normalized (stripped) body text for a retrieved document.

    Reads the canonical ``document`` field first, then falls back to the
    legacy ``text``/``content`` aliases. Returns "" if none are present, are
    not strings, or are blank/whitespace-only — such a row must never count
    as retrieved evidence.
    """
    if not isinstance(document, dict):
        return ""
    for field_name in _DOCUMENT_BODY_FIELDS:
        value = document.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def usable_documents(retrieval: Optional[dict]) -> list[dict]:
    """Return retrieved document rows whose normalized body is non-blank.

    A document with metadata but an empty/whitespace body contributes zero
    — it is filtered out here rather than downstream.
    """
    documents = (retrieval or {}).get("documents") or []
    return [d for d in documents if isinstance(d, dict) and document_body(d)]


def usable_facts(retrieval: Optional[dict]) -> list[dict]:
    """Return retrieved fact rows with a metric and a non-``None`` value.

    Numeric zero remains valid evidence; a ``None`` value does not.
    """
    facts = (retrieval or {}).get("facts") or []
    return [
        f for f in facts
        if isinstance(f, dict) and f.get("metric") not in (None, "") and f.get("value") is not None
    ]


def evidence_counts(retrieval: Optional[dict]) -> tuple[int, int]:
    """Return ``(usable_fact_count, usable_document_count)`` for a retrieval dict."""
    return len(usable_facts(retrieval)), len(usable_documents(retrieval))


# ── Authoritative structured-fact reconciliation (2.2.5.3) ────────────────
#
# Merge legacy (Yahoo / stored fundamentals + analyst estimates) facts with
# authoritative SEC CompanyFacts at retrieval normalization only — the durable
# ``fundamentals`` table is never rewritten, so a CompanyFacts rollback stays
# possible. Policy (spec 2.2.5.3 Step 2):
#   - an exact SEC CompanyFacts concept/unit/period/as-of match wins for filed
#     GAAP facts (authoritative rows come first and are never dropped);
#   - estimates never overwrite a realized fact (they carry distinct metric names
#     / source types, so they never share a slot);
#   - legacy fundamentals fill unsupported or more-current market slots (a
#     different metric name or a different period is kept as its own item);
#   - a legacy value that DISAGREES with the authoritative value for the exact
#     same (ticker, metric, period) stays as a separate, conflict-flagged
#     evidence item — never averaged, never silently dropped.

def _norm_metric_key(value) -> str:
    return str(value or "").strip().lower()


def _norm_unit_key(value) -> str:
    return str(value or "").strip().upper()


def _numeric_equalish(a, b) -> Optional[bool]:
    """Compare two values numerically. ``None`` when either is non-numeric."""
    try:
        from decimal import Decimal

        da, db = Decimal(str(a)), Decimal(str(b))
    except Exception:  # noqa: BLE001 - non-numeric values are compared elsewhere
        return None
    if da == db:
        return True
    scale = max(abs(da), abs(db), Decimal(1))
    return (abs(da - db) / scale) <= Decimal("0.0001")


def _values_conflict(legacy: dict, authoritative: dict) -> bool:
    """True when a legacy fact contradicts the authoritative filed value."""
    if _norm_unit_key(legacy.get("unit")) and _norm_unit_key(authoritative.get("unit")):
        if _norm_unit_key(legacy.get("unit")) != _norm_unit_key(authoritative.get("unit")):
            return True
    equalish = _numeric_equalish(legacy.get("value"), authoritative.get("value"))
    if equalish is None:
        return str(legacy.get("value")) != str(authoritative.get("value"))
    return not equalish


def reconcile_structured_facts(
    legacy_facts: Optional[list[dict]],
    authoritative_facts: Optional[list[dict]],
) -> list[dict]:
    """Merge legacy + authoritative structured facts under the 2.2.5.3 policy.

    Authoritative (SEC CompanyFacts) rows lead and are always preserved. A legacy
    fact for the exact same (ticker, metric, period) is dropped when it agrees
    with the authoritative value, and kept as a separate ``conflict``-flagged item
    when it disagrees. Any legacy fact whose (metric, period) slot has no
    authoritative match (a different metric name, a more-current period, or a
    market-only field such as ``pe_ratio``/``market_cap``) is kept unchanged.
    """
    authoritative = [f for f in (authoritative_facts or []) if isinstance(f, dict)]
    legacy = [f for f in (legacy_facts or []) if isinstance(f, dict)]
    if not authoritative:
        return legacy

    # Group authoritative rows by (ticker, metric), preserving order — the
    # companyfacts projection lists the newest filed period first.
    by_metric: dict[tuple, list[dict]] = {}
    for fact in authoritative:
        key = (_norm_metric_key(fact.get("ticker")), _norm_metric_key(fact.get("metric")))
        by_metric.setdefault(key, []).append(fact)

    merged: list[dict] = list(authoritative)
    for fact in legacy:
        key = (_norm_metric_key(fact.get("ticker")), _norm_metric_key(fact.get("metric")))
        candidates = by_metric.get(key)
        if not candidates:
            merged.append(fact)  # unsupported / market-only field → keep legacy
            continue
        legacy_period = str(fact.get("period") or "")
        if legacy_period:
            target = next(
                (c for c in candidates if str(c.get("period") or "") == legacy_period),
                None,
            )
            if target is None:
                merged.append(fact)  # different (more-current) period → keep legacy
                continue
        else:
            target = candidates[0]  # unspecified period reconciles to the newest
        if _values_conflict(fact, target):
            disputed = dict(fact)
            disputed["conflict"] = True
            disputed.setdefault("conflict_reason", "legacy_vs_companyfacts")
            merged.append(disputed)
        # else: legacy duplicates the authoritative value — CompanyFacts wins.
    return merged


# ── Stable model-visible evidence ids (2.2.4.3) ───────────────────────────
#
# ``EvidenceItem`` is the request-local contract for one model-visible fact,
# document, or tool/calculation result. ``assign_evidence_ids`` numbers a
# packed list ``E1``, ``E2``, ... so the prompt can render an ``[E#]`` marker
# and answers can cite it. These ids are request-local only — they never
# replace the durable storage identifier (kept as ``store_id``) in databases,
# APIs, or the evidence trace.

@dataclass
class EvidenceItem:
    """One model-visible evidence unit with a request-local ``[E#]`` id.

    ``store_id`` retains the durable storage/tool identifier; ``evidence_id``
    is the request-local ``E#`` marker and is empty until assigned.
    """

    kind: str
    store_id: Optional[str] = None
    evidence_id: str = ""
    ticker: Optional[str] = None
    entities: tuple[str, ...] = ()
    metric: Optional[str] = None
    value: Any = None
    unit: Optional[str] = None
    period: Optional[str] = None
    as_of: Optional[str] = None
    source_type: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    freshness: Optional[str] = None
    item_type: Optional[str] = None
    event_type: Optional[str] = None
    authority_tier: Optional[str] = None
    date_semantics: dict = field(default_factory=dict)
    canonical_security: Optional[str] = None
    coverage_tier: Optional[str] = None
    document: str = ""
    parent_id: Optional[str] = None
    section: Optional[str] = None
    chunk_index: Any = None
    subquery_ids: tuple[str, ...] = ()
    scores: dict = field(default_factory=dict)
    operands: tuple = ()
    formula: Optional[str] = None

    @classmethod
    def from_row(cls, row: dict, *, kind: str) -> "EvidenceItem":
        """Build an item from a retrieval fact/document row."""
        try:
            from .evidence_taxonomy import normalize_evidence

            row = normalize_evidence(row)
        except Exception:  # noqa: BLE001 - additive metadata is fail-soft
            logger.warning("Evidence taxonomy normalization failed", exc_info=True)
        metadata = row.get("metadata") or {}
        raw_entities = evidence_field(row, "entities")
        if isinstance(raw_entities, (list, tuple, set)):
            entities = tuple(str(e).upper() for e in raw_entities if e)
        else:
            ticker = evidence_field(row, "ticker")
            entities = (str(ticker).upper(),) if ticker else ()
        scores = {
            key: row.get(key)
            for key in ("score", "fusion_score", "rerank_score", "similarity", "distance")
            if row.get(key) is not None
        }
        return cls(
            kind=kind,
            store_id=evidence_id(row),
            ticker=evidence_field(row, "ticker"),
            entities=entities,
            metric=row.get("metric") if kind == "fact" else evidence_field(row, "metric"),
            value=row.get("value"),
            unit=evidence_field(row, "unit"),
            period=evidence_field(row, "period"),
            as_of=evidence_field(row, "as_of"),
            source_type=evidence_field(row, "source_type", evidence_field(row, "source")),
            source=evidence_field(row, "source", evidence_field(row, "source_name")),
            source_url=evidence_field(row, "source_url", evidence_field(row, "url")),
            freshness=evidence_field(row, "freshness_status", evidence_field(row, "freshness")),
            item_type=evidence_field(row, "item_type"),
            event_type=evidence_field(row, "event_type"),
            authority_tier=evidence_field(
                row, "authority_tier", evidence_field(row, "evidence_authority")
            ),
            date_semantics=dict(evidence_field(row, "date_semantics", {}) or {}),
            canonical_security=evidence_field(
                row, "canonical_security", evidence_field(row, "ticker")
            ),
            coverage_tier=evidence_field(
                row, "coverage_tier", evidence_field(row, "discovery_scope")
            ),
            document=document_body(row),
            parent_id=metadata.get("parent_id") or row.get("parent_id"),
            section=metadata.get("section") or metadata.get("section_title"),
            chunk_index=metadata.get("chunk_index", metadata.get("chunk")),
            subquery_ids=tuple(row.get("subquery_ids") or ()),
            scores=scores,
            operands=tuple(row.get("operands") or ()),
            formula=row.get("formula"),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable view (used by the evidence trace)."""
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "store_id": self.store_id,
            "ticker": self.ticker,
            "entities": list(self.entities),
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "period": self.period,
            "as_of": self.as_of,
            "source_type": self.source_type,
            "source": self.source,
            "source_url": self.source_url,
            "freshness": self.freshness,
            "item_type": self.item_type,
            "event_type": self.event_type,
            "authority_tier": self.authority_tier,
            "date_semantics": dict(self.date_semantics),
            "canonical_security": self.canonical_security,
            "coverage_tier": self.coverage_tier,
            "parent_id": self.parent_id,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "subquery_ids": list(self.subquery_ids),
            "scores": dict(self.scores),
            "operands": list(self.operands),
            "formula": self.formula,
        }

    def taxonomy_metadata(self) -> dict:
        """Return the shared prompt/citation/graph taxonomy projection."""
        return {
            "item_type": self.item_type,
            "event_type": self.event_type,
            "authority_tier": self.authority_tier,
            "source": self.source or self.source_type,
            "date_semantics": dict(self.date_semantics),
            "canonical_security": self.canonical_security,
            "coverage_tier": self.coverage_tier,
        }


def build_evidence_items(
    facts: Optional[list[dict]],
    documents: Optional[list[dict]],
) -> list[EvidenceItem]:
    """Build the packed evidence list (facts then documents) from retrieval rows."""
    items: list[EvidenceItem] = []
    for row in facts or []:
        if isinstance(row, dict):
            items.append(EvidenceItem.from_row(row, kind="fact"))
    for row in documents or []:
        if isinstance(row, dict):
            items.append(EvidenceItem.from_row(row, kind="document"))
    return items


def assign_evidence_ids(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Assign request-local, deterministic ``E1``, ``E2``, ... ids in packed order.

    Mutates and returns ``items``. The ids never replace ``store_id`` (the
    durable identifier) — they are model-visible request-local labels only.
    """
    for index, item in enumerate(items, 1):
        item.evidence_id = f"E{index}"
    return items

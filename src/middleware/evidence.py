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
    source_url: Optional[str] = None
    freshness: Optional[str] = None
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
            source_url=evidence_field(row, "source_url", evidence_field(row, "url")),
            freshness=evidence_field(row, "freshness_status", evidence_field(row, "freshness")),
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
            "source_url": self.source_url,
            "freshness": self.freshness,
            "parent_id": self.parent_id,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "subquery_ids": list(self.subquery_ids),
            "scores": dict(self.scores),
            "operands": list(self.operands),
            "formula": self.formula,
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

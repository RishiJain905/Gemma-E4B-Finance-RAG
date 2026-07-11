"""
src/middleware/hierarchical_retrieval.py
Bounded hierarchical expansion of precise SEC filing child hits (Phase 2.2.5.3).

Flat chunk retrieval can return a sentence without its heading, table context,
or neighboring explanation. This module reconstructs *just enough* local context
around each precise child hit — at most one preceding and one following
same-section sibling to complete a sentence/table, and at most one adjacent
section when a requested obligation matches its heading or the evidence grader
reports missing local context — while never injecting an entire filing parent
and never exceeding a shared character budget.

Every returned item preserves the root hit's original retrieval score and is
annotated with ``expansion_reason`` and ``root_hit_id`` so downstream packing and
the evidence trace keep full provenance. All store reads fail soft per item: a
missing sibling/section is logged and skipped, never propagated (the query never
errors because an expansion read failed).

Design contract (docs/phase2.2/2.2.5-authoritative-and-long-document-data/
2.2.5.3-hierarchical-retrieval-migration-and-evaluation.md, Step 1):
- keep the exact hit + its section heading;
- add ≤1 preceding and ≤1 following same-section child when needed to complete a
  sentence/table;
- add an adjacent section only on an obligation-heading match or a grader
  missing-local-context signal;
- deduplicate shared parent/sibling chunks across hits;
- stop before the shared context budget is exceeded;
- never return an entire filing parent.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from .evidence import document_body

logger = logging.getLogger(__name__)

# Sentence/table completion heuristics. A body that begins mid-sentence needs its
# preceding sibling; one that does not end on a terminator needs its following
# sibling (a table row typically ends on a digit / delimiter, prose mid-word).
_SENTENCE_TERMINATORS = frozenset(".?!\"')]")
_CONTINUATION_STARTS = frozenset(",);:%")

# Obligation → heading match ignores short/generic tokens.
_MIN_HEADING_TOKEN = 4
_STOP_TOKENS = frozenset({
    "item", "note", "notes", "total", "the", "and", "for", "with", "from",
    "other", "data", "table", "section", "part",
})


def _hit_score(hit: dict) -> Optional[float]:
    """The root hit's retrieval score (rerank preferred, then fusion, then score)."""
    for key in ("rerank_score", "fusion_score", "score", "fused_score"):
        value = hit.get(key)
        if value is not None:
            return value
    return None


def _identities(doc: dict) -> list[tuple]:
    """All independent de-dup identities a chunk carries (id and parent+chunk).

    Mirrors the adaptive orchestrator / weighted-fusion identity contract so a
    chunk surfaced both by id and by (parent_id, chunk_index) is one duplicate.
    """
    meta = doc.get("metadata") or {}
    identities: list[tuple] = []
    doc_id = doc.get("id") or meta.get("id")
    if doc_id is not None:
        identities.append(("id", doc_id))
    parent = meta.get("parent_id") or doc.get("parent_id")
    chunk = meta.get("chunk_index", meta.get("chunk"))
    if parent is not None and chunk is not None:
        identities.append(("chunk", parent, chunk))
    if not identities:
        identities.append(("body", document_body(doc)))
    return identities


def _meta(doc: dict) -> dict:
    return doc.get("metadata") or {}


def _parent_id(doc: dict) -> Optional[str]:
    meta = _meta(doc)
    return meta.get("parent_id") or doc.get("parent_id")


def _chunk_index(doc: dict) -> Optional[int]:
    meta = _meta(doc)
    value = meta.get("chunk_index", meta.get("chunk"))
    return value if isinstance(value, int) else None


def _section_index(doc: dict) -> Optional[int]:
    meta = _meta(doc)
    value = meta.get("section_index")
    return value if isinstance(value, int) else None


def _accession(doc: dict) -> Optional[str]:
    meta = _meta(doc)
    return meta.get("accession") or doc.get("accession")


def _heading(doc: dict) -> str:
    meta = _meta(doc)
    return str(
        meta.get("section_heading")
        or meta.get("section")
        or meta.get("section_key")
        or ""
    )


def _needs_preceding(body: str) -> bool:
    """True when ``body`` begins mid-sentence and its lead-in must be recovered."""
    stripped = (body or "").lstrip()
    if not stripped:
        return False
    first = stripped[0]
    return first.islower() or first in _CONTINUATION_STARTS


def _needs_following(body: str) -> bool:
    """True when ``body`` does not end on a sentence terminator (prose or table)."""
    stripped = (body or "").rstrip()
    if not stripped:
        return False
    return stripped[-1] not in _SENTENCE_TERMINATORS


def _obligation_tokens(obligations: Iterable[Any]) -> set[str]:
    """Heading-match tokens drawn from requested metrics and operations."""
    tokens: set[str] = set()
    for obligation in obligations or ():
        for metric in getattr(obligation, "metrics", ()) or ():
            for token in str(metric).lower().replace("-", "_").split("_"):
                if len(token) >= _MIN_HEADING_TOKEN and token not in _STOP_TOKENS:
                    tokens.add(token)
        for operation in getattr(obligation, "operations", ()) or ():
            for token in str(operation).lower().replace("-", "_").split("_"):
                if len(token) >= _MIN_HEADING_TOKEN and token not in _STOP_TOKENS:
                    tokens.add(token)
    return tokens


def _heading_matches_obligation(heading: str, tokens: set[str]) -> bool:
    if not tokens:
        return False
    lowered = heading.lower()
    return any(token in lowered for token in tokens)


def _annotate(doc: dict, *, reason: str, root_hit_id: Any, score: Any) -> dict:
    """Return a provenance-annotated copy of an expansion chunk.

    The chunk inherits the root hit's original retrieval score (siblings/adjacent
    chunks were never independently retrieved) and records why and from which hit
    it was expanded.
    """
    clone = dict(doc)
    clone["expansion_reason"] = reason
    clone["root_hit_id"] = root_hit_id
    if score is not None and clone.get("score") is None:
        clone["score"] = score
    return clone


def _hit_id(hit: dict) -> Any:
    meta = _meta(hit)
    return hit.get("id") or meta.get("id")


def expand_filing_hits(
    hits: list[dict],
    store: Any,
    obligations: Iterable[Any] = (),
    context_budget: int = 12000,
    *,
    max_siblings: int = 2,
    max_adjacent_sections: int = 1,
    max_expanded_items: Optional[int] = None,
    missing_local_context: bool = False,
) -> list[dict]:
    """Expand precise child ``hits`` with bounded parent/sibling/adjacent context.

    For each hit this keeps the exact chunk and its section heading, then adds at
    most one preceding and one following same-section child when the hit begins or
    ends mid-sentence/table, and at most ``max_adjacent_sections`` adjacent
    sections when a requested obligation matches the adjacent heading or
    ``missing_local_context`` is set. Shared parent/sibling chunks are deduplicated
    across hits, expansion stops before ``context_budget`` characters are exceeded,
    and an entire filing parent is never returned.

    ``obligations`` are the evidence obligations for the request (used only for
    heading matching). Returns the root hits (annotated ``expansion_reason=
    "root_hit"``) followed by their in-budget expansions, each carrying its
    ``root_hit_id`` and the root hit's original retrieval score. Non-filing hits
    (no ``parent_id``/``accession``) pass through unchanged as roots. Per-item
    store failures are logged and skipped — this function never raises.
    """
    if not hits:
        return []

    max_siblings = max(0, int(max_siblings))
    max_adjacent_sections = max(0, int(max_adjacent_sections))
    budget = max(0, int(context_budget))
    tokens = _obligation_tokens(obligations)

    seen: set[tuple] = set()
    out: list[dict] = []
    used = 0
    expansions_added = 0

    def _reserve(doc: dict) -> None:
        nonlocal used
        for identity in _identities(doc):
            seen.add(identity)
        used += len(document_body(doc))

    def _is_seen(doc: dict) -> bool:
        return any(identity in seen for identity in _identities(doc))

    def _can_add_expansion(doc: dict) -> bool:
        if max_expanded_items is not None and expansions_added >= max_expanded_items:
            return False
        return used + len(document_body(doc)) <= budget

    # Root hits are precise retrieved evidence — always preserved, annotated, and
    # reserved so their siblings are never re-emitted as duplicates.
    root_ids: list[Any] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        if _is_seen(hit):
            continue
        root_hit_id = _hit_id(hit)
        root_ids.append(root_hit_id)
        annotated = _annotate(
            hit, reason="root_hit", root_hit_id=root_hit_id, score=_hit_score(hit))
        out.append(annotated)
        _reserve(hit)

    # Expansions, in hit order: preceding sibling, following sibling, then
    # adjacent section(s). Each is bounded by the shared budget and item cap.
    for hit, root_hit_id in zip(
        (h for h in hits if isinstance(h, dict)), root_ids,
    ):
        parent = _parent_id(hit)
        chunk = _chunk_index(hit)
        body = document_body(hit)

        siblings_here = 0
        if parent is not None and chunk is not None and max_siblings > 0:
            wanted: list[tuple[int, str]] = []
            if _needs_preceding(body) and chunk - 1 >= 0:
                wanted.append((chunk - 1, "preceding_sibling"))
            if _needs_following(body):
                wanted.append((chunk + 1, "following_sibling"))
            if wanted:
                family = _safe_section_chunks(store, parent)
                by_index = {_chunk_index(c): c for c in family
                            if _chunk_index(c) is not None}
                for target_index, reason in wanted:
                    if siblings_here >= max_siblings:
                        break
                    sibling = by_index.get(target_index)
                    if sibling is None or _is_seen(sibling):
                        continue
                    if not document_body(sibling):
                        continue
                    if not _can_add_expansion(sibling):
                        continue
                    out.append(_annotate(
                        sibling, reason=reason, root_hit_id=root_hit_id,
                        score=_hit_score(hit)))
                    _reserve(sibling)
                    siblings_here += 1
                    expansions_added += 1

        accession = _accession(hit)
        section_index = _section_index(hit)
        if (
            max_adjacent_sections > 0
            and accession is not None
            and section_index is not None
        ):
            adjacent = _safe_adjacent_sections(
                store, accession, section_index, before=1, after=1)
            added_sections = 0
            claimed_sections: set[int] = {section_index}
            for chunk_doc in adjacent:
                if added_sections >= max_adjacent_sections:
                    break
                adj_section = _section_index(chunk_doc)
                if adj_section is None or adj_section in claimed_sections:
                    continue
                # One representative chunk per adjacent section (its first child),
                # never the whole parent.
                if _chunk_index(chunk_doc) not in (0, None):
                    continue
                if _is_seen(chunk_doc) or not document_body(chunk_doc):
                    continue
                heading = _heading(chunk_doc)
                if _heading_matches_obligation(heading, tokens):
                    reason = "adjacent_section_obligation"
                elif missing_local_context:
                    reason = "adjacent_section_missing_context"
                else:
                    continue
                if not _can_add_expansion(chunk_doc):
                    continue
                out.append(_annotate(
                    chunk_doc, reason=reason, root_hit_id=root_hit_id,
                    score=_hit_score(hit)))
                _reserve(chunk_doc)
                claimed_sections.add(adj_section)
                added_sections += 1
                expansions_added += 1

    return out


def _safe_section_chunks(store: Any, parent_id: str) -> list[dict]:
    """Read one section family's children, failing soft to ``[]``."""
    try:
        from src.storage.chroma_store import ChromaStore
        limit = getattr(ChromaStore, "MAX_READ_LIMIT", 200)
    except Exception:  # noqa: BLE001 - default when the store type is unavailable
        limit = 200
    try:
        return list(store.get_section_chunks(parent_id, limit=limit) or [])
    except Exception:  # noqa: BLE001 - expansion read is best-effort per item
        logger.debug("Section family unavailable: %s", parent_id, exc_info=True)
        return []


def _safe_adjacent_sections(
    store: Any, accession: str, section_index: int, *, before: int, after: int,
) -> list[dict]:
    """Read a bounded adjacent-section window, failing soft to ``[]``."""
    try:
        return list(store.get_adjacent_sections(
            accession, section_index, before=before, after=after) or [])
    except Exception:  # noqa: BLE001 - expansion read is best-effort per item
        logger.debug(
            "Adjacent sections unavailable: %s@%s", accession, section_index,
            exc_info=True)
        return []

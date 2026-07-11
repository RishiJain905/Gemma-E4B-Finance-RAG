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
from typing import Optional

logger = logging.getLogger(__name__)

# Canonical field first; legacy aliases only as a compatibility fallback.
_DOCUMENT_BODY_FIELDS = ("document", "text", "content")


def document_body(document: dict) -> str:
    """Return the normalized (stripped) body text for a retrieved document.

    Reads the canonical ``document`` field first, then falls back to the
    legacy ``text``/``content`` aliases. Returns "" if none are present, are
    not strings, or are blank/whitespace-only — such a row must never count
    as retrieved evidence.
    """
    if not isinstance(document, dict):
        return ""
    for field in _DOCUMENT_BODY_FIELDS:
        value = document.get(field)
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

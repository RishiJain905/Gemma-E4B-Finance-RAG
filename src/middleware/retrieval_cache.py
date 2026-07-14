"""
src/middleware/retrieval_cache.py
Bounded, thread-safe, versioned retrieval cache (Phase 2.2.6.2).

Caches only *normalized pre-prompt evidence and retrieval metadata* — the merged
facts/documents and retrieval telemetry the adaptive orchestrator produced for a
request — keyed on a fingerprint that includes the persisted store data revision
(:meth:`src.storage.store.Store.retrieval_revision`). It NEVER caches a final
generated answer: for volatile finance data (prices, news, sentiment, freshness,
estimates, guidance, or "latest" asks) an answer cache would serve stale figures
whenever nearby embeddings collide, so semantic final-answer caching is
deliberately not implemented here (2.2.6.2 Step 4). Only exact retrieval evidence,
invalidated the instant the store revision changes, is reused.

Correctness before hit rate:
  - the key carries the store revision, so any fact insert/update or document
    add/replace/delete (including a same-count replacement) invalidates entries;
  - a config/model fingerprint change clears the whole cache;
  - a TTL bounds staleness even if a revision bump is somehow missed;
  - explicit-refresh / newer-than-watermark requests never look up the cache;
  - every value is deep-copied on store AND on read, so a caller can never mutate
    a cached record and poison a later hit;
  - all cache operations fail soft: any internal error is a miss, never a query
    failure.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# ── Config / model fingerprint ────────────────────────────

# Config attributes whose change alters retrieval evidence and therefore must
# invalidate the whole cache. Embedding endpoint + model identity + every
# retrieval-shaping knob are included so a redeploy that swaps the model or a
# retrieval toggle can never serve evidence produced under the old settings.
_FINGERPRINT_ATTRS = (
    "model_name",
    "embedding_endpoint",
    "llama_endpoint",
    "top_k_documents",
    "top_k_facts",
    "enable_lexical",
    "rrf_k",
    "enable_reranker",
    "reranker_backend",
    "reranker_model",
    "rerank_candidates",
    "rerank_top_n",
    "adaptive_conditional_rerank",
    "adaptive_max_context_chars",
    "enable_query_decomposition",
    "enable_hierarchical_retrieval",
    "enable_evidence_sufficiency",
    "enable_corrective_retry",
)


def config_fingerprint(config: Any) -> str:
    """Stable digest of the retrieval-shaping config + model identity."""
    payload = {attr: _plain(getattr(config, attr, None)) for attr in _FINGERPRINT_ATTRS}
    return json.dumps(payload, sort_keys=True, default=str)


def _plain(value: Any) -> Any:
    """Coerce a config value into a JSON-stable primitive."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


# ── Cache key ─────────────────────────────────────────────


def build_cache_key(
    *,
    plan: Any,
    config: Any,
    lane: Any,
    revision: int,
    available_metrics: Iterable[str] = (),
    as_of: Optional[str] = None,
) -> str:
    """Build the exact retrieval-cache key for one planned request.

    Includes the compiled retrieval query, the validated plan fields (entities,
    intents, metrics, periods, primary intent), the selected lane, every
    subquery's shape, top-k / lexical / rerank settings, the config+model
    fingerprint, the store revision, and any freshness/as-of requirement. Two
    requests share a key only when reusing one's evidence for the other is
    provably safe.
    """
    lane_value = getattr(lane, "value", lane)
    subqueries = [
        [
            getattr(sq, "id", ""),
            getattr(sq, "text", ""),
            list(getattr(sq, "entity_tickers", ()) or ()),
            list(getattr(sq, "intents", ()) or ()),
            list(getattr(sq, "metrics", ()) or ()),
            list(getattr(sq, "periods", ()) or ()),
            list(getattr(sq, "retrieval_modes", ()) or ()),
            bool(getattr(sq, "derived", False)),
        ]
        for sq in getattr(plan, "subqueries", []) or []
    ]
    key_obj = {
        "retrieval_query": getattr(plan, "retrieval_query", ""),
        "tickers": list(getattr(plan, "tickers", []) or []),
        "intents": list(getattr(plan, "intents", []) or []),
        "metrics": list(getattr(plan, "metrics", []) or []),
        "periods": list(getattr(plan, "periods", []) or []),
        "primary_intent": getattr(plan, "primary_intent", None),
        "lane": lane_value,
        "subqueries": subqueries,
        "top_k_documents": int(getattr(config, "top_k_documents", 5)),
        "top_k_facts": int(getattr(config, "top_k_facts", 10)),
        "enable_lexical": bool(getattr(config, "enable_lexical", True)),
        "enable_reranker": bool(getattr(config, "enable_reranker", False)),
        "reranker_backend": str(getattr(config, "reranker_backend", "")),
        "adaptive_conditional_rerank": bool(
            getattr(config, "adaptive_conditional_rerank", True)),
        "rrf_k": int(getattr(config, "rrf_k", 60)),
        "config_fingerprint": config_fingerprint(config),
        "revision": int(revision),
        "as_of": as_of,
        "available_metrics": sorted(str(m) for m in (available_metrics or ())),
    }
    return json.dumps(key_obj, sort_keys=True, default=str)


# ── Cache ─────────────────────────────────────────────────


@dataclass
class _Entry:
    value: dict
    expires_at: float
    size_chars: int


class RetrievalCache:
    """Thread-safe LRU + TTL cache of pre-prompt retrieval evidence.

    Bounded by ``max_entries`` (LRU eviction) and ``ttl_s`` (a secondary staleness
    bound); a single value larger than ``max_value_chars`` is refused rather than
    stored. Values are deep-copied on both :meth:`set` and :meth:`get`, so callers
    can never mutate a cached record. All operations acquire one re-entrant lock
    and fail soft — an internal error is a miss, never a raise.
    """

    def __init__(
        self,
        *,
        max_entries: int = 256,
        ttl_s: float = 300.0,
        max_value_chars: int = 200_000,
        fingerprint: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._ttl_s = max(0.0, float(ttl_s))
        self._max_value_chars = max(0, int(max_value_chars))
        self._fingerprint = fingerprint
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        # Diagnostic counters (never affect correctness).
        self.stats = {
            "hits": 0, "misses": 0, "stores": 0,
            "evictions": 0, "expired": 0, "too_large": 0, "clears": 0,
        }

    # ── Fingerprint invalidation ───────────────────────

    def check_fingerprint(self, fingerprint: str) -> bool:
        """Clear the cache if the config/model fingerprint changed. Returns True
        when a clear happened."""
        with self._lock:
            if fingerprint != self._fingerprint:
                self._fingerprint = fingerprint
                self._clear_locked()
                return True
            return False

    # ── Core ───────────────────────────────────────────

    def get(self, key: str) -> Optional[dict]:
        """Return a deep copy of the cached evidence for ``key``, or ``None``."""
        try:
            with self._lock:
                entry = self._entries.get(key)
                if entry is None:
                    self.stats["misses"] += 1
                    return None
                if self._ttl_s and self._clock() >= entry.expires_at:
                    del self._entries[key]
                    self.stats["expired"] += 1
                    self.stats["misses"] += 1
                    return None
                self._entries.move_to_end(key)
                self.stats["hits"] += 1
                return copy.deepcopy(entry.value)
        except Exception:  # noqa: BLE001 - a cache failure is a miss, never a raise
            logger.warning("Retrieval cache get failed; treating as miss", exc_info=True)
            return None

    def set(self, key: str, value: dict) -> bool:
        """Store a deep copy of ``value`` under ``key``. Returns True when stored.

        A value whose serialized size exceeds ``max_value_chars`` is refused (the
        cache stays bounded). Insertion evicts the least-recently-used entry once
        the entry count would exceed ``max_entries``.
        """
        try:
            snapshot = copy.deepcopy(value)
            size_chars = len(json.dumps(snapshot, default=str))
            with self._lock:
                if self._max_value_chars and size_chars > self._max_value_chars:
                    self.stats["too_large"] += 1
                    self._entries.pop(key, None)
                    return False
                expires_at = self._clock() + self._ttl_s if self._ttl_s else float("inf")
                self._entries[key] = _Entry(snapshot, expires_at, size_chars)
                self._entries.move_to_end(key)
                self.stats["stores"] += 1
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
                    self.stats["evictions"] += 1
                return True
        except Exception:  # noqa: BLE001 - failing to cache must never fail a query
            logger.warning("Retrieval cache set failed; skipping", exc_info=True)
            return False

    def clear(self) -> None:
        """Drop every cached entry."""
        with self._lock:
            self._clear_locked()

    def _clear_locked(self) -> None:
        self._entries.clear()
        self.stats["clears"] += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

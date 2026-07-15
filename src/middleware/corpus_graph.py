"""src/middleware/corpus_graph.py
Read-only on-demand projection of Store inventory into a bounded corpus graph.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# Phase 2.2 corpus kinds plus the 2.3.5.3 source-independent aggregate and leaf
# kinds. Aggregate kinds are bounded "counts before expansion" nodes; the two
# leaf kinds (corpus_item, event) are the drill targets under a filter.
_AGGREGATE_NODE_KINDS = frozenset({
    "index", "sector", "industry", "source_category", "item_type",
    "event_type", "time_bucket",
})
_LEAF_NODE_KINDS = frozenset({"corpus_item", "event"})
CORPUS_NODE_KINDS = frozenset({
    "source", "ticker", "metric", "fact", "filing", "section",
    "document_family", "freshness", "scheduler_source",
}) | _AGGREGATE_NODE_KINDS | _LEAF_NODE_KINDS
# Kinds that a metadata search returns as concrete nodes (aggregate levels are
# reached through /groups and /facets, never enumerated by search).
SEARCH_KINDS = CORPUS_NODE_KINDS - _AGGREGATE_NODE_KINDS
CORPUS_RELATIONS = frozenset({
    "contains", "describes", "reported_by", "filed_as", "has_section",
    "has_chunk_family", "freshness_for", "scheduled_by",
    # 2.3.5.3 source-independent relations.
    "member_of", "classified_as", "published_by", "occurred_as",
    "bucketed_by", "links_item", "covered_by",
})

# groups()/drill dimensions -> (accounting group_by, projected node kind). Each
# aggregate node carries the drill filter so a bounded next page can expand it.
GROUP_DIMENSIONS = {
    "source_category": ("source_category", "source_category"),
    "source": ("source", "source"),
    "item_type": ("item_type", "item_type"),
    "event_type": ("event_type", "event_type"),
    "sector": ("sector", "sector"),
    "industry": ("industry", "industry"),
    "index": ("index", "index"),
    "year": ("year", "time_bucket"),
    "month": ("month", "time_bucket"),
}
# Curated facet dimensions counted for the current filter set (bounded per call
# and cached by revision+filters so warm reads never recompute).
FACET_DIMENSIONS = (
    "source_category", "item_type", "event_type", "sector", "index",
    "indexing_state", "year",
)
# Filter keys the explorer accepts everywhere (search, groups, facets, drill).
CORPUS_FILTER_KEYS = frozenset({
    "source_category", "source", "item_type", "event_type", "security",
    "sector", "industry", "index", "coverage_tier", "year", "month",
    "indexing_state",
})

# Aggregation-first explorer (2.3.5.2): the bounded dimensions the landing view
# and facet rail count over. Each maps 1:1 to a Store.get_corpus_accounting
# group_by, which returns COUNT/SUM rows straight from SQLite metadata (never
# Chroma bodies), so every count is authoritative and bounded.
AGGREGATE_DIMENSIONS = frozenset({
    "source_category", "source", "item_type", "event_type", "security",
    "sector", "industry", "index", "coverage_tier",
    "year", "month", "indexing_state",
})

# Human labels + authority tier per canonical source category. The tier mirrors
# evidence_taxonomy._authority_rank so the explorer badges agree with the
# retrieval ranker; global (non-issuer) authorities form the parallel branch.
_CATEGORY_LABELS = {
    "sec": "SEC / EDGAR", "issuer": "Issuer / IR", "market_data": "Market data",
    "company_news": "Company news", "central_bank": "Central bank",
    "treasury": "Treasury", "economic_agency": "Economic agency",
    "regulator": "Regulator", "sector_agency": "Sector agency",
    "transcript": "Transcript", "estimates": "Estimates",
    "global_news": "Global news",
}
_CATEGORY_AUTHORITY = {
    "sec": "primary", "issuer": "primary", "central_bank": "primary",
    "treasury": "primary", "economic_agency": "primary", "regulator": "primary",
    "sector_agency": "primary", "market_data": "structured",
    "company_news": "licensed", "transcript": "analysis",
    "estimates": "analysis", "global_news": "discovery",
}
_GLOBAL_CATEGORIES = frozenset(
    {"central_bank", "treasury", "economic_agency", "regulator"})

MAX_PAGE_LIMIT = 200
MAX_ELEMENT_LIMIT = 2_000
DEFAULT_PAGE_LIMIT = 100
DEFAULT_RESULT_PAGE = 50
DEFAULT_VISIBLE_NODE_TARGET = 450
DEFAULT_OVERVIEW_CACHE_TTL_S = 2.0
DEFAULT_OPAQUE_ID_TTL_S = 300.0
DEFAULT_INSPECTOR_EXCERPT_BYTES = 1_000
DEFAULT_INSPECTOR_METADATA_BYTES = 4_000
MAX_CURSOR_OFFSET = 10_000
MAX_PROVENANCE_ENTRIES = 20

_ID_RE = re.compile(r"^cg1_[A-Za-z0-9_-]{16,64}$")
_CURSOR_RE = re.compile(r"^cc1_[A-Za-z0-9_-]{16,512}$")
_DENIED_KEY = re.compile(
    r"key|token|secret|authorization|cookie|password|credential|path",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"\b(?:api[_-]?key|token|secret|authorization|cookie|password)\b\s*[:=]\s*[^,;\s]+",
    re.IGNORECASE,
)
_LOCAL_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|users|var|tmp|private)/)\S*",
    re.IGNORECASE,
)


class CorpusRevisionChanged(ValueError):
    """Raised when an opaque cursor belongs to an older Store revision."""


@dataclass
class _NodeRef:
    """Private lookup state for a short-lived opaque node id."""

    kind: str
    key: str
    payload: dict
    expires_at: float


def _redact_text(value: object, limit: int = 1_000) -> str:
    """Return bounded text with secrets and local paths removed."""
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = _SENSITIVE_ASSIGNMENT.sub("[REDACTED]", text)
    text = _LOCAL_PATH.sub("[REDACTED]", text)
    return text[:max(0, int(limit))]


def _safe_url(value: object) -> Optional[str]:
    """Keep only bounded HTTP(S) source URLs."""
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    safe_query = [
        (key, item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not _DENIED_KEY.search(key)
    ]
    return urlunsplit((
        parsed.scheme, parsed.netloc, parsed.path,
        urlencode(safe_query), parsed.fragment,
    ))[:2_048]


def _safe_value(value: object) -> object:
    """Recursively redact values used in a safe metadata projection."""
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if not _DENIED_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value, 1_000)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _redact_text(value, 1_000)


def _safe_metadata(metadata: Optional[dict]) -> dict:
    """Apply the recursive denylist to a projection metadata dictionary."""
    result = {}
    for key, value in (metadata or {}).items():
        if _DENIED_KEY.search(str(key)):
            continue
        if str(key).lower() in {"source_url", "url"}:
            url = _safe_url(value)
            if url:
                result[str(key)] = url
            continue
        result[str(key)] = _safe_value(value)
    return result


def _opaque_token(prefix: str, value: str) -> str:
    """Hash a canonical Store key into a URL-safe stable token."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()[:18]
    token = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{prefix}{token}"


class CorpusGraph:
    """Project authoritative Store reads into bounded corpus graph responses."""

    def __init__(
        self,
        store: Any,
        *,
        page_limit: int = DEFAULT_PAGE_LIMIT,
        default_page_limit: int = DEFAULT_RESULT_PAGE,
        element_limit: int = MAX_ELEMENT_LIMIT,
        visible_node_target: int = DEFAULT_VISIBLE_NODE_TARGET,
        overview_ttl_s: float = DEFAULT_OVERVIEW_CACHE_TTL_S,
        id_ttl_s: float = DEFAULT_OPAQUE_ID_TTL_S,
        excerpt_bytes: int = DEFAULT_INSPECTOR_EXCERPT_BYTES,
        metadata_bytes: int = DEFAULT_INSPECTOR_METADATA_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.page_limit = max(1, min(MAX_PAGE_LIMIT, int(page_limit)))
        self.default_page_limit = max(
            1, min(self.page_limit, int(default_page_limit)))
        self.element_limit = max(1, min(MAX_ELEMENT_LIMIT, int(element_limit)))
        self.visible_node_target = max(
            1, min(self.element_limit, int(visible_node_target)))
        self.overview_ttl_s = max(0.001, float(overview_ttl_s))
        self.id_ttl_s = max(0.001, float(id_ttl_s))
        self.excerpt_bytes = max(0, int(excerpt_bytes))
        self.metadata_bytes = max(0, int(metadata_bytes))
        self._clock = clock
        self._refs: dict[str, _NodeRef] = {}
        self._overview_cache: Optional[tuple[float, int, dict]] = None
        # Aggregate/facet cache keyed by (revision, canonical cache key). A
        # revision change drops every stale entry (2.3.5.3 Step 4).
        self._agg_cache: dict[tuple[int, str], dict] = {}

    def _cache_get(self, revision: int, key: str) -> Optional[dict]:
        entry = self._agg_cache.get((revision, key))
        return copy.deepcopy(entry) if entry is not None else None

    def _cache_put(self, revision: int, key: str, value: dict) -> dict:
        if self._agg_cache and next(iter(self._agg_cache))[0] != revision:
            self._agg_cache.clear()
        if len(self._agg_cache) >= 256:
            self._agg_cache.pop(next(iter(self._agg_cache)))
        self._agg_cache[(revision, key)] = copy.deepcopy(value)
        return value

    # ── Validation, cursors, and bounded response helpers ─────────────────

    def _validate_limit(self, limit: int) -> int:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if not 1 <= limit <= self.page_limit:
            raise ValueError(f"limit must be between 1 and {self.page_limit}")
        return limit

    def _revision(self) -> int:
        return int(self.store.retrieval_revision())

    @staticmethod
    def _scope(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    def _encode_cursor(self, *, offset: int, revision: int, scope: str) -> str:
        payload = {"v": 1, "offset": int(offset), "revision": int(revision), "scope": scope}
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return f"cc1_{token}"

    def _decode_cursor(self, cursor: Optional[str], *, scope: str, revision: int) -> int:
        if not cursor:
            return 0
        if not isinstance(cursor, str) or not _CURSOR_RE.fullmatch(cursor):
            raise ValueError("invalid cursor")
        try:
            encoded = cursor[4:]
            encoded += "=" * (-len(encoded) % 4)
            payload = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
        except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid cursor") from exc
        if (
            payload.get("v") != 1
            or payload.get("scope") != scope
            or int(payload.get("revision", -1)) != revision
        ):
            if int(payload.get("revision", -1)) != revision:
                raise CorpusRevisionChanged("cursor revision is stale")
            raise ValueError("invalid cursor")
        offset = payload.get("offset")
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise ValueError("invalid cursor")
        if not 0 <= offset <= MAX_CURSOR_OFFSET:
            raise ValueError("cursor expired")
        return offset

    def _register(self, kind: str, key: str, payload: Optional[dict] = None) -> str:
        if kind not in CORPUS_NODE_KINDS:
            raise ValueError("unsupported corpus node kind")
        node_id = _opaque_token("cg1_", f"{kind}\x00{key}")
        self._refs[node_id] = _NodeRef(
            kind=kind, key=key, payload=dict(payload or {}),
            expires_at=self._clock() + self.id_ttl_s,
        )
        return node_id

    def _resolve(self, node_id: str) -> _NodeRef:
        if not isinstance(node_id, str) or not _ID_RE.fullmatch(node_id):
            raise ValueError("invalid opaque id")
        ref = self._refs.get(node_id)
        if ref is None:
            raise ValueError("invalid opaque id")
        if self._clock() >= ref.expires_at:
            self._refs.pop(node_id, None)
            raise ValueError("opaque id expired")
        return ref

    def _node(
        self,
        kind: str,
        key: str,
        label: object,
        *,
        summary: object = "",
        metadata: Optional[dict] = None,
        payload: Optional[dict] = None,
        excerpt: Optional[object] = None,
    ) -> dict:
        node_id = self._register(kind, key, payload or {})
        result = {
            "id": node_id,
            "kind": kind,
            "label": _redact_text(label, 200),
        }
        safe_summary = _redact_text(summary, 200)
        if safe_summary:
            result["summary"] = safe_summary
        safe_meta = _safe_metadata(metadata)
        if safe_meta:
            result["metadata"] = safe_meta
        if excerpt is not None:
            result["excerpt"] = _redact_text(excerpt, 1_000)
        return result

    def _edge(self, source: str, target: str, relation: str, *, metadata: Optional[dict] = None) -> dict:
        if relation not in CORPUS_RELATIONS:
            raise ValueError("unsupported corpus relation")
        edge_id = _opaque_token("ce1_", f"{source}\x00{target}\x00{relation}")
        return {
            "id": edge_id,
            "source": source,
            "target": target,
            "relation": relation,
            "metadata": _safe_metadata(metadata),
        }

    def _response(
        self,
        nodes: list[dict],
        edges: list[dict],
        *,
        revision: int,
        next_cursor: Optional[str] = None,
        truncated: bool = False,
        node_limit: Optional[int] = None,
    ) -> dict:
        """Enforce the hard element cap on every explorer response."""
        max_nodes = min(self.element_limit, node_limit or self.element_limit)
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
            truncated = True
        original_edge_count = len(edges)
        visible_ids = {node["id"] for node in nodes}
        max_edges = max(0, self.element_limit - len(nodes))
        edges = [
            edge for edge in edges
            if edge.get("source") in visible_ids or edge.get("target") in visible_ids
        ][:max_edges]
        if len(edges) < original_edge_count:
            truncated = True
        return {
            "nodes": nodes,
            "edges": edges,
            "next_cursor": next_cursor,
            "truncated": bool(truncated),
            "corpus_revision": int(revision),
        }

    @staticmethod
    def _row_source(row: dict) -> str:
        return str(row.get("source") or "").strip()

    @staticmethod
    def _row_ticker(row: dict) -> Optional[str]:
        ticker = row.get("ticker")
        return str(ticker).upper() if ticker else None

    # ── Node projections ──────────────────────────────────────────────────

    def _source_node(self, row: dict) -> dict:
        source = self._row_source(row)
        return self._node(
            "source", source, source, summary=f"{int(row.get('count') or 0)} stored records",
            metadata={"source": source, "count": int(row.get("count") or 0)},
            payload={"source": source},
        )

    def _ticker_node(self, row: dict) -> dict:
        ticker = self._row_ticker(row) or ""
        return self._node(
            "ticker", ticker, ticker,
            summary=f"{int(row.get('record_count') or 0)} stored records",
            metadata={
                "ticker": ticker,
                "company_name": row.get("company_name"),
                "record_count": int(row.get("record_count") or 0),
                "sources": list(row.get("sources") or []),
            },
            payload={"ticker": ticker, **row},
        )

    def _metric_node(self, row: dict) -> dict:
        metric = str(row.get("metric") or row.get("concept") or "")
        ticker = self._row_ticker(row) or ""
        source = self._row_source(row)
        unit = str(row.get("unit") or "")
        key = "\x1f".join((source, ticker, metric, unit))
        return self._node(
            "metric", key, metric,
            summary=f"{ticker} · {unit}" if ticker else unit,
            metadata={
                "metric": metric, "concept": row.get("concept"),
                "ticker": ticker, "unit": unit,
                "supported_units": row.get("supported_units") or ([unit] if unit else []),
                "source": source,
                "observation_count": int(row.get("observation_count") or 0),
            },
            payload={"metric": metric, "ticker": ticker, "source": source, **row},
        )

    def _fact_node(self, row: dict) -> dict:
        kind = str(row.get("record_kind") or "fundamental")
        record_id = int(row.get("record_id") or 0)
        key = f"{kind}:{record_id}"
        metric = str(row.get("metric") or row.get("concept") or "")
        ticker = self._row_ticker(row) or ""
        source = self._row_source(row)
        return self._node(
            "fact", key, f"{ticker} {metric}",
            summary=f"{row.get('value')} {row.get('unit') or ''}".strip(),
            metadata={
                "ticker": ticker, "metric": metric, "value": row.get("value"),
                "unit": row.get("unit"), "period": row.get("period"),
                "as_of": row.get("filed_at") or row.get("as_of"),
                "source": source, "source_url": _safe_url(row.get("source_url")),
                "accession": row.get("accession"),
                "conflict": row.get("conflict") or row.get("conflict_reason"),
            },
            payload={**row, "record_kind": kind, "record_id": record_id},
        )

    def _filing_node(self, row: dict) -> dict:
        accession = str(row.get("accession") or "")
        return self._node(
            "filing", accession, accession,
            summary=f"{row.get('filing_type') or 'filing'} · {row.get('status') or 'unknown'}",
            metadata={
                "accession": accession, "ticker": self._row_ticker(row),
                "form": row.get("filing_type"), "filing_date": row.get("filing_date"),
                "period": row.get("period"), "status": row.get("status"),
                "parsed_at": row.get("parsed_at"),
                "index_section_count": int(row.get("index_section_count") or 0),
                "index_chunk_count": int(row.get("index_chunk_count") or 0),
                "index_error": _redact_text(row.get("index_error"), 500),
                "source_url": _safe_url(row.get("source_url")),
            },
            payload={**row, "accession": accession},
        )

    def _section_node(self, row: dict, *, excerpt: Optional[object] = None) -> dict:
        metadata = dict(row.get("metadata") or row)
        parent_id = str(row.get("id") or metadata.get("parent_id") or "")
        accession = str(metadata.get("accession") or "")
        index = int(metadata.get("section_index") or 0)
        key = f"{accession}\x1f{parent_id}\x1f{index}"
        return self._node(
            "section", key,
            metadata.get("section_heading") or metadata.get("section_key") or parent_id,
            summary=f"{int(row.get('chunk_count') or metadata.get('chunk_count') or 0)} chunks",
            metadata={
                "accession": accession, "ticker": self._row_ticker(metadata),
                "section_key": metadata.get("section_key"),
                "section_heading": metadata.get("section_heading"),
                "section_index": index,
                "chunk_count": int(row.get("chunk_count") or metadata.get("chunk_count") or 0),
                "source_url": _safe_url(metadata.get("source_url")),
            },
            payload={"parent_id": parent_id, **metadata},
            excerpt=excerpt,
        )

    def _family_node(self, row: dict, *, excerpt: Optional[object] = None) -> dict:
        metadata = dict(row.get("metadata") or row)
        parent_id = str(row.get("id") or metadata.get("parent_id") or "")
        source = self._row_source(metadata)
        ticker = self._row_ticker(metadata) or ""
        date = metadata.get("date") or metadata.get("filing_date")
        return self._node(
            "document_family", parent_id, f"{source} · {ticker or 'general'}",
            summary=str(date or ""),
            metadata={
                "source": source, "ticker": ticker, "date": date,
                "chunk_count": int(row.get("chunk_count") or metadata.get("chunk_count") or 0),
                "source_url": _safe_url(metadata.get("source_url")),
            },
            payload={"parent_id": parent_id, **metadata},
            excerpt=excerpt,
        )

    def _freshness_node(self, row: dict) -> dict:
        ticker = self._row_ticker(row) or ""
        source = self._row_source(row)
        key = f"{ticker}\x1f{source}"
        return self._node(
            "freshness", key, f"{ticker} · {source}",
            summary=str(row.get("status") or "unknown"),
            metadata={
                "ticker": ticker, "source": source, "status": row.get("status"),
                "last_updated": row.get("last_updated"),
                "next_scheduled_update": row.get("next_scheduled_update"),
                "age_hours": row.get("age_hours"), "ttl_hours": row.get("ttl_hours"),
                "error": _redact_text(row.get("error_message"), 500),
            },
            payload={**row, "ticker": ticker, "source": source},
        )

    def _scheduler_node(self, row: dict) -> dict:
        source = str(row.get("source") or "")
        return self._node(
            "scheduler_source", source, source,
            summary=str(row.get("status") or "never_fetched"),
            metadata={
                "source": source, "status": row.get("status"),
                "ttl_key": row.get("ttl_key"), "ttl_hours": row.get("ttl_hours"),
                "last_run": row.get("last_run"),
                "next_scheduled_update": row.get("next_scheduled_update"),
                "error": _redact_text(row.get("error_message"), 500),
            },
            payload={**row, "source": source},
        )

    # ── Overview ──────────────────────────────────────────────────────────

    def overview(self) -> dict:
        """Return a short revision-keyed source/ticker/freshness projection."""
        revision = self._revision()
        now = self._clock()
        cached = self._overview_cache
        if cached and cached[1] == revision and now < cached[0]:
            return copy.deepcopy(cached[2])

        source_rows = self.store.get_source_counts(limit=MAX_PAGE_LIMIT, offset=0)
        ticker_rows = self.store.get_ticker_counts(limit=MAX_PAGE_LIMIT, offset=0)
        freshness_rows = self.store.list_freshness(limit=MAX_PAGE_LIMIT, offset=0)
        scheduler_rows = self.store.list_scheduler_sources(
            limit=MAX_PAGE_LIMIT, offset=0)

        nodes: list[dict] = []
        edges: list[dict] = []
        source_ids: dict[str, str] = {}
        ticker_ids: dict[str, str] = {}
        for row in source_rows:
            node = self._source_node(row)
            nodes.append(node)
            source_ids[self._row_source(row)] = node["id"]
        for row in ticker_rows:
            node = self._ticker_node(row)
            nodes.append(node)
            ticker = self._row_ticker(row)
            if ticker:
                ticker_ids[ticker] = node["id"]
                for source in row.get("sources") or []:
                    source_id = source_ids.get(str(source))
                    if source_id:
                        edges.append(self._edge(source_id, node["id"], "contains"))
        for row in freshness_rows:
            node = self._freshness_node(row)
            nodes.append(node)
            ticker_id = ticker_ids.get(self._row_ticker(row) or "")
            if ticker_id:
                edges.append(self._edge(node["id"], ticker_id, "freshness_for"))
        for row in scheduler_rows:
            node = self._scheduler_node(row)
            nodes.append(node)
            source_id = source_ids.get(str(row.get("source") or ""))
            if source_id:
                edges.append(self._edge(node["id"], source_id, "scheduled_by"))

        result = self._response(
            nodes, edges, revision=revision,
            truncated=len(nodes) > self.visible_node_target,
            node_limit=self.visible_node_target,
        )
        self._overview_cache = (
            now + self.overview_ttl_s, revision, copy.deepcopy(result))
        return result

    # ── Search ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_filters(values: Optional[list[str]]) -> list[str]:
        return sorted({str(value).strip() for value in (values or []) if str(value).strip()})

    def _search_candidates(
        self,
        *,
        query: str,
        kinds: list[str],
        sources: list[str],
        ticker: Optional[str],
        fetch_limit: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        applied = self._agg_filters(filters)
        candidates: list[dict] = []
        for kind in kinds:
            if kind == "corpus_item":
                leaf_filters = {
                    key: value for key, value in applied.items()
                    if key != "coverage_tier"
                }
                if ticker and "security" not in leaf_filters:
                    leaf_filters["security"] = ticker
                if sources and "source" not in leaf_filters:
                    leaf_filters["source"] = sources[0]
                for row in self.store.list_corpus_items(
                    limit=fetch_limit, offset=0, **leaf_filters,
                ):
                    title = str(row.get("title") or "")
                    if query and query.lower() not in title.lower():
                        continue
                    candidates.append(self._leaf_item_node(row))
            elif kind == "event":
                continue  # events are reached via the event_type aggregate drill
            elif kind == "source":
                for row in self.store.get_source_counts(limit=fetch_limit, offset=0):
                    if query and query.lower() not in str(row.get("source") or "").lower():
                        continue
                    if sources and str(row.get("source")) not in sources:
                        continue
                    candidates.append(self._source_node(row))
            elif kind == "ticker":
                for row in self.store.get_ticker_counts(limit=fetch_limit, offset=0):
                    row_ticker = self._row_ticker(row) or ""
                    if ticker and row_ticker != ticker:
                        continue
                    if query and query.lower() not in row_ticker.lower():
                        continue
                    if sources and not set(sources).intersection(row.get("sources") or []):
                        continue
                    candidates.append(self._ticker_node(row))
            elif kind == "metric":
                for row in self.store.search_corpus_metrics(
                    query=query, ticker=ticker, limit=fetch_limit, offset=0,
                ):
                    if sources and self._row_source(row) not in sources:
                        continue
                    candidates.append(self._metric_node(row))
            elif kind == "fact":
                for row in self.store.search_corpus_facts(
                    query=query, ticker=ticker, limit=fetch_limit, offset=0,
                ):
                    if sources and self._row_source(row) not in sources:
                        continue
                    candidates.append(self._fact_node(row))
            elif kind == "filing":
                for row in self.store.list_filings(
                    query=query, ticker=ticker, limit=fetch_limit, offset=0,
                ):
                    if sources and "sec_filings" not in sources and "sec" not in sources:
                        continue
                    candidates.append(self._filing_node(row))
            elif kind == "section":
                for filing in self.store.list_filings(
                    query=None, ticker=ticker, limit=fetch_limit, offset=0,
                ):
                    accession = filing.get("accession")
                    if not accession:
                        continue
                    for row in self.store.get_filing_section_families(
                        accession, limit=fetch_limit, offset=0,
                    ):
                        metadata = row.get("metadata") or row
                        if sources and self._row_source(metadata) not in sources and "sec_filing" not in sources:
                            continue
                        if query and query.lower() not in str(row).lower():
                            continue
                        candidates.append(self._section_node(row))
            elif kind == "document_family":
                for row in self.store.search_document_families(
                    query=query, ticker=ticker, limit=fetch_limit, offset=0,
                ):
                    if sources and self._row_source(row.get("metadata") or row) not in sources:
                        continue
                    candidates.append(self._family_node(row))
            elif kind == "freshness":
                for row in self.store.list_freshness(
                    ticker=ticker, limit=fetch_limit, offset=0,
                ):
                    if sources and self._row_source(row) not in sources:
                        continue
                    if query and query.lower() not in str(row).lower():
                        continue
                    candidates.append(self._freshness_node(row))
            elif kind == "scheduler_source":
                for row in self.store.list_scheduler_sources(
                    limit=fetch_limit, offset=0,
                ):
                    if sources and str(row.get("source")) not in sources:
                        continue
                    if query and query.lower() not in str(row).lower():
                        continue
                    candidates.append(self._scheduler_node(row))
        # Stable ordering makes opaque cursor pages deterministic across calls.
        return sorted(candidates, key=lambda node: (node["kind"], node["label"], node["id"]))

    def _search_edges(self, nodes: list[dict]) -> list[dict]:
        by_kind = {node["kind"]: {} for node in nodes}
        for node in nodes:
            metadata = node.get("metadata") or {}
            if node["kind"] == "source":
                by_kind["source"][str(metadata.get("source"))] = node["id"]
            elif node["kind"] == "ticker":
                by_kind["ticker"][str(metadata.get("ticker"))] = node["id"]
        edges = []
        for node in nodes:
            metadata = node.get("metadata") or {}
            source_id = by_kind.get("source", {}).get(str(metadata.get("source")))
            ticker_id = by_kind.get("ticker", {}).get(str(metadata.get("ticker")))
            if source_id and node["kind"] != "source":
                relation = "reported_by" if node["kind"] == "fact" else "contains"
                edges.append(self._edge(source_id, node["id"], relation))
            if ticker_id and node["kind"] not in {"ticker", "source"}:
                edges.append(self._edge(node["id"], ticker_id, "describes"))
        return edges

    def search(
        self,
        *,
        q: Optional[str] = None,
        query: Optional[str] = None,
        kinds: Optional[list[str]] = None,
        sources: Optional[list[str]] = None,
        ticker: Optional[str] = None,
        filters: Optional[dict] = None,
        limit: int = DEFAULT_RESULT_PAGE,
        cursor: Optional[str] = None,
    ) -> dict:
        """Search safe corpus metadata with revision-bound opaque pages.

        Server-side facet filters (source_category/item_type/event_type/sector/
        industry/index/year/month/indexing_state) narrow the same authoritative
        metadata; a filter change changes the cursor scope and resets paging.
        """
        limit = self._validate_limit(limit)
        normalized_kinds = self._normalize_filters(kinds) or sorted(SEARCH_KINDS)
        invalid_kinds = set(normalized_kinds) - CORPUS_NODE_KINDS
        if invalid_kinds:
            raise ValueError("invalid corpus kind")
        normalized_sources = self._normalize_filters(sources)
        normalized_ticker = str(ticker).upper().strip() if ticker else None
        applied = self._agg_filters(filters)
        query_text = str(q if q is not None else query or "").strip()
        scope = self._scope({
            "operation": "search", "query": query_text, "kinds": normalized_kinds,
            "sources": normalized_sources, "ticker": normalized_ticker,
            "filters": applied,
        })
        revision = self._revision()
        offset = self._decode_cursor(cursor, scope=scope, revision=revision)
        fetch_limit = min(MAX_PAGE_LIMIT, max(limit, offset + limit + 1))
        candidates = self._search_candidates(
            query=query_text, kinds=normalized_kinds, sources=normalized_sources,
            ticker=normalized_ticker, fetch_limit=fetch_limit, filters=applied,
        )
        page = candidates[offset:offset + limit + 1]
        has_more = len(page) > limit
        page = page[:limit]
        next_cursor = self._encode_cursor(
            offset=offset + limit, revision=revision, scope=scope,
        ) if has_more else None
        return self._response(
            page, self._search_edges(page), revision=revision,
            next_cursor=next_cursor, truncated=has_more,
        ) | {"applied_filters": applied}

    # ── Detail and expansion ──────────────────────────────────────────────

    def _detail_node(self, ref: _NodeRef) -> Optional[dict]:
        if ref.kind == "source":
            rows = self.store.get_source_counts(limit=MAX_PAGE_LIMIT, offset=0)
            return next((self._source_node(row) for row in rows if row.get("source") == ref.key), None)
        if ref.kind == "ticker":
            rows = self.store.get_ticker_counts(limit=MAX_PAGE_LIMIT, offset=0)
            return next((self._ticker_node(row) for row in rows if self._row_ticker(row) == ref.key), None)
        if ref.kind == "metric":
            payload = ref.payload
            rows = self.store.search_corpus_metrics(
                query=payload.get("metric"), ticker=payload.get("ticker"),
                unit=payload.get("unit"), limit=MAX_PAGE_LIMIT, offset=0,
            )
            for row in rows:
                if (
                    row.get("metric") == payload.get("metric")
                    and self._row_source(row) == payload.get("source")
                ):
                    return self._metric_node(row)
            return None
        if ref.kind == "fact":
            row = self.store.get_corpus_fact(ref.payload.get("record_kind"), ref.payload.get("record_id"))
            return self._fact_node(row) if row else None
        if ref.kind == "filing":
            row = self.store.get_filing(ref.key)
            return self._filing_node(row) if row else None
        if ref.kind == "section":
            payload = ref.payload
            rows = self.store.get_section_chunks(payload.get("parent_id"), limit=1, offset=0)
            if not rows:
                return None
            row = rows[0]
            return self._section_node(row, excerpt=row.get("document"))
        if ref.kind == "document_family":
            rows = self.store.get_document_family(ref.payload.get("parent_id"), limit=1, offset=0)
            if not rows:
                return None
            return self._family_node(
                {"id": ref.payload.get("parent_id"), "metadata": rows[0].get("metadata") or {}},
                excerpt=rows[0].get("document"),
            )
        if ref.kind == "freshness":
            payload = ref.payload
            rows = self.store.list_freshness(
                ticker=payload.get("ticker"), source=payload.get("source"),
                limit=MAX_PAGE_LIMIT, offset=0,
            )
            return self._freshness_node(rows[0]) if rows else None
        if ref.kind == "scheduler_source":
            rows = self.store.list_scheduler_sources(limit=MAX_PAGE_LIMIT, offset=0)
            return next((self._scheduler_node(row) for row in rows if row.get("source") == ref.key), None)
        return None

    def detail(self, node_id: str) -> Optional[dict]:
        """Return one bounded node detail, including at most one excerpt."""
        ref = self._resolve(node_id)
        revision = self._revision()
        node = self._detail_node(ref)
        if node is None:
            return None
        return self._response([node], [], revision=revision)

    def _neighbor_rows(self, ref: _NodeRef, limit: int, relations: set[str]) -> list[tuple[dict, str]]:
        rows: list[tuple[dict, str]] = []
        if ref.kind == "filing" and "has_section" in relations:
            for row in self.store.get_filing_section_families(ref.key, limit=MAX_PAGE_LIMIT, offset=0):
                rows.append((self._section_node(row), "has_section"))
        elif ref.kind == "section" and "contains" in relations:
            payload = ref.payload
            for row in self.store.get_adjacent_sections(
                payload.get("accession"), int(payload.get("section_index") or 0),
                before=min(limit, 10), after=min(limit, 10),
            ):
                parent_id = (row.get("metadata") or {}).get("parent_id")
                if parent_id == payload.get("parent_id"):
                    continue
                rows.append((self._section_node(row), "contains"))
        elif ref.kind == "source" and "contains" in relations:
            for row in self.store.get_ticker_counts(limit=MAX_PAGE_LIMIT, offset=0):
                if ref.key in (row.get("sources") or []):
                    rows.append((self._ticker_node(row), "contains"))
        elif ref.kind == "ticker":
            if "contains" in relations:
                source_names = set(ref.payload.get("sources") or [])
                for row in self.store.get_source_counts(limit=MAX_PAGE_LIMIT, offset=0):
                    if row.get("source") in source_names:
                        rows.append((self._source_node(row), "contains"))
            if "describes" in relations:
                for row in self.store.search_corpus_metrics(
                    ticker=ref.key, limit=MAX_PAGE_LIMIT, offset=0,
                ):
                    rows.append((self._metric_node(row), "describes"))
            if "reported_by" in relations:
                for row in self.store.search_corpus_facts(
                    ticker=ref.key, limit=MAX_PAGE_LIMIT, offset=0,
                ):
                    rows.append((self._fact_node(row), "reported_by"))
            if "filed_as" in relations:
                for row in self.store.list_filings(
                    ticker=ref.key, limit=MAX_PAGE_LIMIT, offset=0,
                ):
                    rows.append((self._filing_node(row), "filed_as"))
            if "has_chunk_family" in relations:
                for row in self.store.search_document_families(
                    ticker=ref.key, limit=MAX_PAGE_LIMIT, offset=0,
                ):
                    rows.append((self._family_node(row), "has_chunk_family"))
            if "freshness_for" in relations:
                for row in self.store.list_freshness(
                    ticker=ref.key, limit=MAX_PAGE_LIMIT, offset=0,
                ):
                    rows.append((self._freshness_node(row), "freshness_for"))
        elif ref.kind == "metric" and "describes" in relations:
            payload = ref.payload
            for row in self.store.search_corpus_facts(
                query=payload.get("metric"), ticker=payload.get("ticker"),
                limit=MAX_PAGE_LIMIT, offset=0,
            ):
                rows.append((self._fact_node(row), "describes"))
        elif ref.kind == "fact":
            payload = ref.payload
            if "describes" in relations:
                rows.append((self._metric_node({
                    "metric": payload.get("metric"), "ticker": payload.get("ticker"),
                    "unit": payload.get("unit"), "source": payload.get("source"),
                }), "describes"))
            if "reported_by" in relations:
                rows.append((self._source_node({
                    "source": payload.get("source"), "count": 0,
                }), "reported_by"))
        elif ref.kind == "freshness":
            payload = ref.payload
            if "freshness_for" in relations:
                rows.append((self._ticker_node({
                    "ticker": payload.get("ticker"), "record_count": 0,
                    "sources": [payload.get("source")],
                }), "freshness_for"))
        elif ref.kind == "corpus_item":
            item_id = str(ref.payload.get("corpus_item_id") or "")
            row = self.store.get_corpus_item(item_id) if item_id else None
            if row and "published_by" in relations:
                rows.append((self._source_node(
                    {"source": row.get("source"), "count": 0}), "published_by"))
            if "covered_by" in relations:
                for link in self.store.list_corpus_item_securities(item_id) or []:
                    ticker = str(link.get("ticker") or link.get("security_id") or "")
                    rows.append((self._ticker_node({
                        "ticker": ticker, "record_count": 0, "sources": [],
                    }), "covered_by"))
        elif ref.kind == "event":
            row = self.store.get_corpus_event(str(ref.payload.get("event_id") or ""))
            if row and "published_by" in relations:
                rows.append((self._source_node(
                    {"source": row.get("source_name"), "count": 0}), "published_by"))
        elif ref.kind == "scheduler_source" and "scheduled_by" in relations:
            rows.append((self._source_node({"source": ref.key, "count": 0}), "scheduled_by"))
        return rows

    def _aggregate_neighbors(
        self, ref: _NodeRef, *, node_id: str, limit: int, cursor: Optional[str],
        revision: int,
    ) -> dict:
        """Expand one aggregate node into a bounded, store-paged leaf page.

        The child items are queried with an offset pushed to SQLite so an
        oversized branch is never materialized — the aggregate node and its
        count stay authoritative and the next page requires a cursor.
        """
        filters = self._agg_filters(ref.payload.get("filters"))
        leaf_filters = {k: v for k, v in filters.items() if k != "coverage_tier"}
        scope = self._scope({
            "operation": "aggregate_neighbors", "node_id": node_id,
            "filters": filters,
        })
        offset = self._decode_cursor(cursor, scope=scope, revision=revision)
        rows = self.store.list_corpus_items(
            limit=min(self.page_limit, limit) + 1, offset=offset, **leaf_filters)
        has_more = len(rows) > limit
        rows = rows[:limit]
        nodes = [self._leaf_item_node(row) for row in rows]
        edges = [self._edge(node_id, node["id"], "links_item") for node in nodes]
        next_cursor = self._encode_cursor(
            offset=offset + limit, revision=revision, scope=scope,
        ) if has_more else None
        return self._response(
            nodes, edges, revision=revision, next_cursor=next_cursor,
            truncated=has_more,
        ) | {"applied_filters": filters}

    def neighbors(
        self,
        node_id: str,
        *,
        cursor: Optional[str] = None,
        limit: int = DEFAULT_RESULT_PAGE,
        relations: Optional[list[str]] = None,
    ) -> dict:
        """Expand one node by one bounded, revision-aware neighbor page."""
        limit = self._validate_limit(limit)
        ref = self._resolve(node_id)
        revision = self._revision()
        # Aggregate nodes drill into leaf corpus items with store-side paging.
        if ref.payload.get("aggregate"):
            return self._aggregate_neighbors(
                ref, node_id=node_id, limit=limit, cursor=cursor,
                revision=revision)
        normalized_relations = self._normalize_filters(relations) or sorted(CORPUS_RELATIONS)
        if set(normalized_relations) - CORPUS_RELATIONS:
            raise ValueError("invalid corpus relation")
        scope = self._scope({
            "operation": "neighbors", "node_id": node_id,
            "relations": normalized_relations,
        })
        offset = self._decode_cursor(cursor, scope=scope, revision=revision)
        rows = self._neighbor_rows(ref, MAX_PAGE_LIMIT, set(normalized_relations))
        page = rows[offset:offset + limit + 1]
        has_more = len(page) > limit
        page = page[:limit]
        nodes = [item[0] for item in page]
        edges = [self._edge(node_id, node["id"], relation) for node, relation in page]
        next_cursor = self._encode_cursor(
            offset=offset + limit, revision=revision, scope=scope,
        ) if has_more else None
        return self._response(
            nodes, edges, revision=revision, next_cursor=next_cursor,
            truncated=has_more,
        )

    def filing_sections(
        self,
        accession: str,
        *,
        cursor: Optional[str] = None,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> dict:
        """Return one bounded page of section-family nodes for a filing."""
        limit = self._validate_limit(limit)
        if not accession or len(str(accession)) > 256:
            raise ValueError("invalid accession")
        revision = self._revision()
        scope = self._scope({"operation": "filing_sections", "accession": accession})
        offset = self._decode_cursor(cursor, scope=scope, revision=revision)
        if self.store.get_filing(accession) is None:
            raise KeyError("filing not found")
        rows = self.store.get_filing_section_families(
            accession, limit=MAX_PAGE_LIMIT, offset=0,
        )
        page = rows[offset:offset + limit + 1]
        has_more = len(page) > limit
        page = page[:limit]
        filing_id = self._register("filing", accession, {"accession": accession})
        nodes = [self._section_node(row) for row in page]
        edges = [self._edge(filing_id, node["id"], "has_section") for node in nodes]
        next_cursor = self._encode_cursor(
            offset=offset + limit, revision=revision, scope=scope,
        ) if has_more else None
        return self._response(
            nodes, edges, revision=revision, next_cursor=next_cursor,
            truncated=has_more,
        )

    # ── Aggregation-first facet counts (2.3.5.2) ──────────────────────────

    @staticmethod
    def _agg_filters(filters: Optional[dict]) -> dict:
        """Keep only recognized, bounded accounting filters from a query map."""
        result: dict[str, str] = {}
        for key, value in (filters or {}).items():
            if key in CORPUS_FILTER_KEYS and value not in (None, ""):
                result[key] = str(value)[:128]
        return dict(sorted(result.items()))

    def _bucket(self, group_by: str, row: dict) -> dict:
        """Shape one accounting row into a labelled, redacted facet bucket."""
        key = str(row.get("key") or "unknown")
        bucket = {
            "key": _redact_text(key, 128),
            "count": int(row.get("count") or 0),
            "approximate_bytes": int(row.get("approximate_bytes") or 0),
        }
        if group_by == "source_category":
            bucket["label"] = _CATEGORY_LABELS.get(key, key)
            bucket["authority_tier"] = _CATEGORY_AUTHORITY.get(key, "discovery")
            bucket["global_source"] = key in _GLOBAL_CATEGORIES
        else:
            bucket["label"] = _redact_text(key, 128)
        return bucket

    def aggregates(
        self,
        group_by: str,
        *,
        filters: Optional[dict] = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        cursor: Optional[str] = None,
    ) -> dict:
        """Return one bounded, revision-keyed page of facet counts.

        Projects ``Store.get_corpus_accounting`` so the landing view and facet
        rail can show authoritative "counts before expansion" without rendering
        the corpus. Combinable filters narrow the same authoritative aggregate.
        """
        if group_by not in AGGREGATE_DIMENSIONS:
            raise ValueError(f"group_by must be one of {sorted(AGGREGATE_DIMENSIONS)}")
        limit = self._validate_limit(limit)
        applied = self._agg_filters(filters)
        revision = self._revision()
        scope = self._scope(
            {"operation": "aggregates", "group_by": group_by, "filters": applied})
        offset = self._decode_cursor(cursor, scope=scope, revision=revision)
        fetch_limit = min(MAX_PAGE_LIMIT, limit + 1)
        rows = self.store.get_corpus_accounting(
            group_by, limit=fetch_limit, offset=offset, **applied)
        has_more = len(rows) > limit
        rows = rows[:limit]
        buckets = [self._bucket(group_by, row) for row in rows]
        next_cursor = self._encode_cursor(
            offset=offset + limit, revision=revision, scope=scope,
        ) if has_more else None
        return self._response([], [], revision=revision, truncated=has_more) | {
            "next_cursor": next_cursor,
            "aggregates": {
                "group_by": group_by,
                "filters": applied,
                "buckets": buckets,
                "page_total": sum(bucket["count"] for bucket in buckets),
            },
        }

    # ── Aggregate nodes, faceted groups, and leaf items (2.3.5.3) ─────────

    def _leaf_item_node(self, row: dict) -> dict:
        """Project one corpus_items row into a bounded, redacted leaf node."""
        item_id = str(row.get("corpus_item_id") or "")
        source = str(row.get("source") or "")
        item_type = str(row.get("item_type") or "")
        return self._node(
            "corpus_item", item_id, row.get("title") or item_id,
            summary=f"{source} · {item_type}".strip(" ·"),
            metadata={
                "source": source, "source_category": row.get("source_category"),
                "item_type": item_type, "event_type": row.get("event_type"),
                "published_at": row.get("published_at"),
                "indexing_status": row.get("indexing_status"),
                "evidence_authority": row.get("evidence_authority"),
                "source_url": _safe_url(row.get("source_url")),
                "approximate_bytes": (
                    int(row.get("narrative_bytes") or 0)
                    + int(row.get("metadata_bytes") or 0)),
            },
            payload={"corpus_item_id": item_id},
        )

    def _event_node(self, row: dict) -> dict:
        """Project one corpus_events row into a bounded leaf node."""
        event_id = str(row.get("event_id") or "")
        event_type = str(row.get("event_type") or "event")
        return self._node(
            "event", event_id, event_type,
            summary=str(row.get("status") or ""),
            metadata={
                "event_type": event_type, "status": row.get("status"),
                "effective_at": row.get("effective_at"),
                "announced_at": row.get("announced_at"),
                "source": row.get("source_name"),
                "source_category": row.get("source_category"),
                "source_url": _safe_url(row.get("source_url")),
            },
            payload={"event_id": event_id},
        )

    def _aggregate_node(self, dimension: str, node_kind: str, row: dict,
                        base_filters: dict) -> dict:
        """Project one accounting bucket into a drillable aggregate node.

        The node carries the bucket count and the exact filter needed to expand
        it one bounded page at a time — never the child ids themselves.
        """
        key = str(row.get("key") or "unknown")
        count = int(row.get("count") or 0)
        drill = dict(base_filters)
        drill[dimension] = key
        label = (_CATEGORY_LABELS.get(key, key)
                 if dimension == "source_category" else key)
        # A stable, revision-scoped key keeps the opaque id deterministic.
        node_key = self._scope({"agg": dimension, "key": key, "f": base_filters})
        return self._node(
            node_kind, node_key, label,
            summary=f"{count} items",
            metadata={
                "dimension": dimension, "bucket_value": _redact_text(key, 128),
                "count": count,
                "approximate_bytes": int(row.get("approximate_bytes") or 0),
                "filters": base_filters,
            },
            payload={
                "aggregate": True, "dimension": dimension,
                "node_kind": node_kind, "filters": drill,
            },
        )

    def groups(
        self,
        group_by: str,
        *,
        filters: Optional[dict] = None,
        limit: int = DEFAULT_RESULT_PAGE,
        cursor: Optional[str] = None,
    ) -> dict:
        """Return one paged aggregate level as drillable nodes.

        Each node is a bounded count-before-expansion projection; the response
        carries the applied filters, page cursor, truncation flag, and
        visible/total bucket counts. Cached by revision + canonical filters.
        """
        if group_by not in GROUP_DIMENSIONS:
            raise ValueError(f"group_by must be one of {sorted(GROUP_DIMENSIONS)}")
        limit = self._validate_limit(limit)
        accounting_dim, node_kind = GROUP_DIMENSIONS[group_by]
        applied = self._agg_filters(filters)
        revision = self._revision()
        scope = self._scope({
            "operation": "groups", "group_by": group_by, "filters": applied})
        offset = self._decode_cursor(cursor, scope=scope, revision=revision)
        cache_key = self._scope({"g": group_by, "f": applied, "o": offset, "l": limit})
        cached = self._cache_get(revision, cache_key)
        if cached is not None:
            return cached
        rows = self.store.get_corpus_accounting(
            accounting_dim, limit=min(MAX_PAGE_LIMIT, limit + 1), offset=offset,
            **applied)
        has_more = len(rows) > limit
        rows = rows[:limit]
        totals = self.store.count_corpus_accounting(accounting_dim, **applied)
        nodes = [
            self._aggregate_node(group_by, node_kind, row, applied)
            for row in rows
        ]
        next_cursor = self._encode_cursor(
            offset=offset + limit, revision=revision, scope=scope,
        ) if has_more else None
        result = self._response(
            nodes, [], revision=revision, next_cursor=next_cursor,
            truncated=has_more,
        ) | {
            "group_by": group_by,
            "applied_filters": applied,
            "visible_count": len(nodes),
            "total_count": int(totals.get("distinct_keys") or 0),
            "total_items": int(totals.get("total_rows") or 0),
        }
        return self._cache_put(revision, cache_key, result)

    def facets(self, *, filters: Optional[dict] = None,
               dimensions: Optional[list[str]] = None) -> dict:
        """Return bounded facet counts for the current filter set.

        Counts a curated set of dimensions for the same filters so the facet
        rail can render "counts before expansion" immediately. Read-only,
        SQLite-only, and cached by revision + canonical filters.
        """
        applied = self._agg_filters(filters)
        requested = [d for d in (dimensions or FACET_DIMENSIONS)
                     if d in AGGREGATE_DIMENSIONS]
        if not requested:
            requested = list(FACET_DIMENSIONS)
        revision = self._revision()
        cache_key = self._scope({"facets": sorted(requested), "f": applied})
        cached = self._cache_get(revision, cache_key)
        if cached is not None:
            return cached
        facet_map: dict[str, dict] = {}
        for dimension in requested:
            rows = self.store.get_corpus_accounting(
                dimension, limit=self.default_page_limit, offset=0, **applied)
            totals = self.store.count_corpus_accounting(dimension, **applied)
            facet_map[dimension] = {
                "buckets": [self._bucket(dimension, row) for row in rows],
                "total_count": int(totals.get("distinct_keys") or 0),
                "total_items": int(totals.get("total_rows") or 0),
                "truncated": int(totals.get("distinct_keys") or 0) > len(rows),
            }
        result = self._response([], [], revision=revision) | {
            "applied_filters": applied,
            "facets": facet_map,
        }
        return self._cache_put(revision, cache_key, result)

    def _provenance(self, item_id: str) -> list[dict]:
        """Return bounded, redacted provenance rows for a corpus item."""
        entries: list[dict] = []
        for row in (self.store.list_corpus_item_sources(item_id)
                    or [])[:MAX_PROVENANCE_ENTRIES]:
            entries.append({
                "source_name": _redact_text(row.get("source_name"), 128),
                "source_category": _redact_text(row.get("source_category"), 64),
                "original_publisher": _redact_text(
                    row.get("original_publisher"), 128),
                "published_at": row.get("published_at"),
                "source_url": _safe_url(row.get("source_url")),
                "license_label": _redact_text(row.get("license_label"), 64),
                "evidence_authority": _redact_text(
                    row.get("evidence_authority"), 64),
            })
        return entries

    def _item_detail_node(self, item_id: str) -> Optional[dict]:
        row = self.store.get_corpus_item(item_id)
        if not row:
            return None
        node = self._leaf_item_node(row)
        securities = [
            _redact_text(link.get("ticker") or link.get("security_id"), 32)
            for link in (self.store.list_corpus_item_securities(item_id) or [])
        ][:MAX_PROVENANCE_ENTRIES]
        node["metadata"] = dict(node.get("metadata") or {})
        node["metadata"]["securities"] = [s for s in securities if s]
        node["metadata"]["provenance"] = self._provenance(item_id)
        summary = row.get("summary") or row.get("title")
        if summary:
            node["excerpt"] = _redact_text(summary, self.excerpt_bytes)
        return node

    def item_detail(self, node_id: str) -> Optional[dict]:
        """Return one bounded corpus-item or event detail with provenance."""
        ref = self._resolve(node_id)
        revision = self._revision()
        if ref.kind == "corpus_item":
            node = self._item_detail_node(str(ref.payload.get("corpus_item_id")))
        elif ref.kind == "event":
            row = self.store.get_corpus_event(str(ref.payload.get("event_id")))
            node = self._event_node(row) if row else None
        else:
            raise ValueError("node is not an inspectable corpus item")
        if node is None:
            return None
        return self._response([node], [], revision=revision)

    # ── Refresh/freshness status ──────────────────────────────────────────

    def refresh_status(self) -> dict:
        """Return persisted freshness and scheduler state without running either."""
        revision = self._revision()
        rows = self.store.list_freshness(limit=MAX_PAGE_LIMIT, offset=0)
        known = getattr(self.store, "FRESHNESS_SOURCES", {}) or {}
        by_logical: dict[str, list[dict]] = {name: [] for name in known}
        for row in rows:
            source = row.get("source")
            logical = next(
                (
                    name for name, cfg in known.items()
                    if source == cfg.get("cache_source") or source == name
                ),
                str(source),
            )
            by_logical.setdefault(logical, []).append(row)
        sources = {}
        for logical, source_rows in sorted(by_logical.items()):
            if not source_rows:
                sources[logical] = {"status": "never_fetched", "entries": 0, "error": None}
                continue
            statuses = {str(row.get("status") or "unknown") for row in source_rows}
            status = "stale" if "stale" in statuses else (
                "fresh" if statuses == {"fresh"} else "partial")
            first_error = next(
                (row.get("error_message") for row in source_rows if row.get("error_message")),
                None,
            )
            sources[logical] = {
                "status": status, "entries": len(source_rows),
                "error": _redact_text(first_error, 500) if first_error else None,
            }
        scheduler = self.store.list_scheduler_sources(limit=MAX_PAGE_LIMIT, offset=0)
        return self._response(
            [], [], revision=revision,
            truncated=False,
        ) | {
            "refresh": {"sources": sources, "scheduler": [
                _safe_metadata(row) for row in scheduler
            ]},
        }


# Compatibility name for callers that prefer the projection terminology.
CorpusProjector = CorpusGraph

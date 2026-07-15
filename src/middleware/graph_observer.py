"""src/middleware/graph_observer.py
Bounded, non-blocking, redacted query-trace graph observer.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Schema version. 2.3.5.1 only *adds* optional allowlisted metadata keys and
# reuses the existing node kinds/edge relations/statuses, so Phase 2.2 golden
# traces stay readable and the version does not bump. Bump this only when a
# required field or an enum (kind/status/relation) changes incompatibly.
GRAPH_SCHEMA_VERSION = 1
NODE_KINDS = frozenset({
    "query", "plan", "subquery", "stage", "tool", "evidence", "source",
    "answer", "citation",
})
NODE_STATUSES = frozenset({
    "pending", "active", "complete", "dropped", "fallback", "partial", "error",
})
EDGE_RELATIONS = frozenset({
    "compiled_to", "contains", "routed_to", "retrieved", "returned",
    "from_source", "expanded_from", "supports", "cited_by", "corrected_by",
    "validated_as",
})
DELTA_OPERATIONS = frozenset({
    "upsert_node", "upsert_edge", "remove", "trace_complete", "trace_evicted",
    "reset_required",
})

_DENIED_KEY = re.compile(r"key|token|secret|authorization|cookie|path", re.IGNORECASE)
_NODE_METADATA_ALLOWLIST = {
    "query": frozenset({"question_digest", "question_preview_chars"}),
    "plan": frozenset({"tickers", "intents", "metrics", "periods", "lane", "reason"}),
    "subquery": frozenset({
        "subquery_id", "tickers", "intents", "metrics", "periods", "derived",
        "parent_id", "reason",
    }),
    "stage": frozenset({
        "phase", "elapsed_ms", "reason", "status", "count", "lane",
        "retrieval_rounds", "planning_calls", "reranker_calls", "context_chars",
        "evidence_dropped", "evidence_deduped", "fallback_reason",
        "retrieval_strategy", "citation_support_rate", "numeric_claims_supported",
        "numeric_claims_unsupported", "validation_status", "ticker", "metrics",
        "period", "kind", "score",
    }),
    "tool": frozenset({"subquery_id", "count", "elapsed_ms", "status"}),
    # 2.3.5.1: source-aware provenance. Every field is an already-safe scalar,
    # a bounded date, or an opaque id — never keys, payloads, or full text.
    "evidence": frozenset({
        "evidence_id", "kind", "ticker", "metric", "period", "source_type",
        "rank", "score", "freshness", "unit", "as_of", "store_id", "section",
        "parent_id", "item_type", "event_type", "authority_tier", "source",
        "date_semantics", "canonical_security", "coverage_tier",
        "security_id", "index_memberships", "sector", "source_category",
        "source_name", "provider", "publisher", "form", "filing_item",
        "exhibit", "published_at", "effective_at", "accessed_at",
        "normalization_version", "corpus_item_id", "document_family_id",
        "evidence_role",
    }),
    "source": frozenset({
        "source_type", "ticker", "freshness", "count", "source_category",
        "source_name", "authority_tier", "provider", "publisher",
    }),
    "answer": frozenset({"status", "facts", "documents", "elapsed_ms"}),
    "citation": frozenset({
        "evidence_id", "source_type", "ticker", "metric", "period", "support_status",
        "item_type", "event_type", "authority_tier", "source",
        "date_semantics", "canonical_security", "coverage_tier",
        "source_category", "provider", "publisher",
    }),
}
_EDGE_METADATA_ALLOWLIST = frozenset({"rank", "score", "status", "elapsed_ms", "reason"})
_SUMMARY_ALLOWLIST = frozenset({
    "status", "node_count", "edge_count", "element_count", "completed_at",
    "eviction_reason", "question_digest", "question_preview", "id",
})
_SENSITIVE_TEXT = re.compile(
    r"\b\S*(?:api[_-]?key|token|secret|authorization|cookie)\S*\b",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"\b(?:api[_-]?key|token|secret|authorization|cookie)\b\s*[:=]\s*[^,;]+",
    re.IGNORECASE,
)
_LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|var|tmp)/)\S+")


def _bounded_text(value: object, limit: int) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:max(0, int(limit))]


def _question_preview(value: object, limit: int) -> str:
    text = _redact_sensitive_text(str(value or "").replace("\r", " ").replace("\n", " "))
    return text[:max(0, int(limit))]


def _redact_sensitive_text(value: str) -> str:
    text = _SENSITIVE_ASSIGNMENT.sub("[REDACTED]", value)
    text = _SENSITIVE_TEXT.sub("[REDACTED]", text)
    return _LOCAL_PATH.sub("[REDACTED]", text)


def _safe_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if not _DENIED_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        return _redact_sensitive_text(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)


def _allowlisted_metadata(kind: str, metadata: Optional[dict]) -> dict:
    allowed = _NODE_METADATA_ALLOWLIST.get(kind, frozenset())
    return {
        key: _safe_value(value)
        for key, value in (metadata or {}).items()
        if key in allowed and not _DENIED_KEY.search(key)
    }


@dataclass(frozen=True)
class GraphNode:
    """One query-scoped graph node with safe, allowlisted metadata."""

    id: str
    query_id: str
    kind: str
    label: str
    summary: str = ""
    status: str = "pending"
    group_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_sequence: int = 0
    updated_sequence: int = 0

    def to_dict(self, *, excerpt_chars: int = 1000) -> dict:
        """Return the bounded serialization used by deltas and snapshots."""
        kind = self.kind if self.kind in NODE_KINDS else "stage"
        status = self.status if self.status in NODE_STATUSES else "error"
        summary_limit = excerpt_chars if kind == "evidence" else 200
        safe_summary = (
            _question_preview(self.summary, summary_limit)
            if kind == "query"
            else _bounded_text(_redact_sensitive_text(self.summary), summary_limit)
        )
        payload = {
            "id": _bounded_text(_redact_sensitive_text(self.id), 256),
            "query_id": _bounded_text(self.query_id, 128),
            "kind": kind,
            "label": _bounded_text(_redact_sensitive_text(self.label), 200),
            "summary": safe_summary,
            "status": status,
            "metadata": _allowlisted_metadata(kind, self.metadata),
            "created_sequence": int(self.created_sequence),
            "updated_sequence": int(self.updated_sequence),
        }
        if self.group_id:
            payload["group_id"] = _bounded_text(
                _redact_sensitive_text(self.group_id), 256
            )
        return payload

    @classmethod
    def from_dict(cls, value: dict) -> "GraphNode":
        """Build a node from an internal allowlisted operation payload."""
        return cls(
            id=str(value.get("id", "")), query_id=str(value.get("query_id", "")),
            kind=str(value.get("kind", "stage")), label=str(value.get("label", "")),
            summary=str(value.get("summary", "")), status=str(value.get("status", "pending")),
            group_id=value.get("group_id"), metadata=dict(value.get("metadata") or {}),
            created_sequence=int(value.get("created_sequence", 0) or 0),
            updated_sequence=int(value.get("updated_sequence", 0) or 0),
        )


@dataclass(frozen=True)
class GraphEdge:
    """One stable relationship between two query-scoped nodes."""

    id: str
    query_id: str
    source: str
    target: str
    relation: str
    metadata: dict = field(default_factory=dict)
    sequence: int = 0

    def to_dict(self) -> dict:
        """Return a bounded allowlisted serialization."""
        relation = self.relation if self.relation in EDGE_RELATIONS else "returned"
        metadata = {
            key: _safe_value(value)
            for key, value in self.metadata.items()
            if key in _EDGE_METADATA_ALLOWLIST and not _DENIED_KEY.search(key)
        }
        return {
            "id": _bounded_text(_redact_sensitive_text(self.id), 256),
            "query_id": _bounded_text(self.query_id, 128),
            "source": _bounded_text(_redact_sensitive_text(self.source), 256),
            "target": _bounded_text(_redact_sensitive_text(self.target), 256),
            "relation": relation,
            "metadata": metadata,
            "sequence": int(self.sequence),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "GraphEdge":
        """Build an edge from an internal allowlisted operation payload."""
        return cls(
            id=str(value.get("id", "")), query_id=str(value.get("query_id", "")),
            source=str(value.get("source", "")), target=str(value.get("target", "")),
            relation=str(value.get("relation", "returned")),
            metadata=dict(value.get("metadata") or {}),
            sequence=int(value.get("sequence", 0) or 0),
        )


@dataclass(frozen=True)
class GraphDelta:
    """One versioned mutation in a query trace."""

    query_id: str
    sequence: int
    operation: str
    timestamp: float = field(default_factory=time.time)
    node: Optional[GraphNode] = None
    edge: Optional[GraphEdge] = None
    summary: Optional[dict] = None
    schema_version: int = GRAPH_SCHEMA_VERSION

    def to_dict(self, *, excerpt_chars: int = 1000) -> dict:
        """Return the safe wire representation."""
        operation = self.operation if self.operation in DELTA_OPERATIONS else "reset_required"
        payload = {
            "schema_version": int(self.schema_version),
            "query_id": _bounded_text(self.query_id, 128),
            "sequence": int(self.sequence),
            "timestamp": round(float(self.timestamp), 3),
            "operation": operation,
        }
        if self.node is not None:
            payload["node"] = self.node.to_dict(excerpt_chars=excerpt_chars)
        if self.edge is not None:
            payload["edge"] = self.edge.to_dict()
        if self.summary is not None:
            payload["summary"] = {
                key: _safe_value(value)
                for key, value in self.summary.items()
                if key in _SUMMARY_ALLOWLIST and not _DENIED_KEY.search(key)
            }
        return payload


@dataclass
class _Trace:
    query_id: str
    created_at: float
    updated_at: float
    nodes: OrderedDict[str, dict] = field(default_factory=OrderedDict)
    edges: OrderedDict[str, dict] = field(default_factory=OrderedDict)
    deltas: list[dict] = field(default_factory=list)
    next_sequence: int = 0
    complete: bool = False

    @property
    def element_count(self) -> int:
        return len(self.nodes) + len(self.edges)


@dataclass(eq=False)
class TraceSubscriber:
    """A non-blocking subscriber queue owned by the graph SSE endpoint."""

    queue: asyncio.Queue
    reset_required: bool = False


class TraceHub:
    """Bounded in-memory trace store and non-blocking delta broadcaster."""

    def __init__(
        self,
        *,
        trace_limit: int = 100,
        element_limit: int = 5000,
        trace_ttl_s: float = 3600,
        excerpt_chars: int = 1000,
        question_preview_chars: int = 200,
        subscriber_queue_limit: int = 256,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.trace_limit = max(1, int(trace_limit))
        self.element_limit = max(1, int(element_limit))
        self.trace_ttl_s = max(0.001, float(trace_ttl_s))
        self.excerpt_chars = max(0, int(excerpt_chars))
        self.question_preview_chars = max(0, int(question_preview_chars))
        self.subscriber_queue_limit = max(1, int(subscriber_queue_limit))
        self._clock = clock
        self._traces: OrderedDict[str, _Trace] = OrderedDict()
        self._subscribers: set[TraceSubscriber] = set()
        self._event_ids = itertools.count(1)
        self._events: deque[dict] = deque(maxlen=max(1024, self.element_limit * 2))
        self._replay_floor = 0
        self._lock = threading.RLock()
        self.dropped_events = 0
        self.reset_count = 0
        self._publish_failed = False

    def publish(self, delta: GraphDelta) -> None:
        """Synchronously store and fan out a delta without awaiting subscribers."""
        if self._publish_failed:
            return
        try:
            with self._lock:
                now = self._clock()
                self._evict_expired(now)
                trace = self._traces.get(delta.query_id)
                if trace is None:
                    trace = _Trace(delta.query_id, now, now)
                    self._traces[delta.query_id] = trace
                sequence = trace.next_sequence
                trace.next_sequence += 1
                normalized = replace(delta, sequence=sequence, timestamp=now)
                payload = normalized.to_dict(excerpt_chars=self.excerpt_chars)
                self._apply(trace, payload)
                self._store_delta(trace, payload)
                trace.updated_at = now
                self._traces.move_to_end(delta.query_id)
                self._append_event(payload)
                self._enforce_limits(delta.query_id)
        except Exception:  # noqa: BLE001 - observation must never affect a query
            if not self._publish_failed:
                self._publish_failed = True
                logger.exception("Graph observer failed; discarding future deltas")

    def _apply(self, trace: _Trace, payload: dict) -> None:
        operation = payload["operation"]
        if operation == "upsert_node" and payload.get("node"):
            node = payload["node"]
            existing = trace.nodes.get(node["id"])
            if existing:
                node["created_sequence"] = existing["created_sequence"]
            else:
                node["created_sequence"] = payload["sequence"]
            node["updated_sequence"] = payload["sequence"]
            trace.nodes[node["id"]] = node
        elif operation == "upsert_edge" and payload.get("edge"):
            edge = payload["edge"]
            trace.edges[edge["id"]] = edge
        elif operation == "remove":
            summary = payload.get("summary") or {}
            element_id = summary.get("id")
            trace.nodes.pop(element_id, None)
            trace.edges.pop(element_id, None)
        elif operation == "trace_complete":
            trace.complete = True

    def _store_delta(self, trace: _Trace, payload: dict) -> None:
        """Compact reconstruction state so repeated upserts stay element-bounded."""
        operation = payload["operation"]
        element = payload.get("node") or payload.get("edge")
        element_id = element.get("id") if element else None
        if element_id and operation in {"upsert_node", "upsert_edge"}:
            key = "node" if operation == "upsert_node" else "edge"
            trace.deltas = [
                prior for prior in trace.deltas
                if not (
                    prior["operation"] == operation
                    and (prior.get(key) or {}).get("id") == element_id
                )
            ]
        elif operation == "trace_complete":
            trace.deltas = [
                prior for prior in trace.deltas
                if prior["operation"] != "trace_complete"
            ]
        trace.deltas.append(payload)

    def _append_event(self, payload: dict) -> None:
        if len(self._events) == self._events.maxlen:
            self._replay_floor = max(self._replay_floor, self._events[0]["event_id"])
        event = {"event_id": next(self._event_ids), "delta": payload}
        self._events.append(event)
        for subscriber in tuple(self._subscribers):
            if subscriber.reset_required:
                continue
            try:
                subscriber.queue.put_nowait(event)
            except asyncio.QueueFull:
                subscriber.reset_required = True
                self.dropped_events += 1
                self.reset_count += 1

    def _evict_trace(self, query_id: str, reason: str) -> None:
        trace = self._traces.pop(query_id, None)
        if trace is None:
            return
        retained = deque(maxlen=self._events.maxlen)
        for event in self._events:
            if event["delta"].get("query_id") == query_id:
                self._replay_floor = max(self._replay_floor, event["event_id"])
            else:
                retained.append(event)
        self._events = retained
        payload = GraphDelta(
            query_id=query_id,
            sequence=trace.next_sequence,
            timestamp=self._clock(),
            operation="trace_evicted",
            summary={"eviction_reason": reason, "element_count": trace.element_count},
        ).to_dict(excerpt_chars=self.excerpt_chars)
        self._append_event(payload)

    def _evict_expired(self, now: float) -> None:
        expired = [
            query_id for query_id, trace in self._traces.items()
            if now - trace.updated_at >= self.trace_ttl_s
        ]
        for query_id in expired:
            self._evict_trace(query_id, "ttl")

    def _enforce_limits(self, current_query_id: str) -> None:
        while len(self._traces) > self.trace_limit:
            self._evict_trace(next(iter(self._traces)), "trace_limit")
        while self._element_count() > self.element_limit and self._traces:
            oldest = next(iter(self._traces))
            self._evict_trace(oldest, "element_limit")

    def _element_count(self) -> int:
        return sum(trace.element_count for trace in self._traces.values())

    def list_traces(self, limit: int = 25) -> list[dict]:
        """Return newest-first safe trace summaries."""
        with self._lock:
            self._evict_expired(self._clock())
            traces = list(reversed(self._traces.values()))[:max(0, min(int(limit), 100))]
            return [self._trace_summary(trace) for trace in traces]

    def _trace_summary(self, trace: _Trace) -> dict:
        query = next((node for node in trace.nodes.values() if node["kind"] == "query"), {})
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "query_id": trace.query_id,
            "created_at": round(trace.created_at, 3),
            "updated_at": round(trace.updated_at, 3),
            "complete": trace.complete,
            "node_count": len(trace.nodes),
            "edge_count": len(trace.edges),
            "question_preview": query.get("summary", ""),
            "question_digest": (query.get("metadata") or {}).get("question_digest"),
        }

    def snapshot(self, query_id: str) -> Optional[dict]:
        """Reconstruct a current trace snapshot from its stored deltas."""
        with self._lock:
            self._evict_expired(self._clock())
            trace = self._traces.get(query_id)
            if trace is None:
                return None
            nodes: OrderedDict[str, dict] = OrderedDict()
            edges: OrderedDict[str, dict] = OrderedDict()
            for delta in trace.deltas:
                operation = delta["operation"]
                if operation == "upsert_node" and delta.get("node"):
                    nodes[delta["node"]["id"]] = delta["node"]
                elif operation == "upsert_edge" and delta.get("edge"):
                    edges[delta["edge"]["id"]] = delta["edge"]
                elif operation == "remove":
                    element_id = (delta.get("summary") or {}).get("id")
                    nodes.pop(element_id, None)
                    edges.pop(element_id, None)
            return {
                **self._trace_summary(trace),
                "nodes": list(nodes.values()),
                "edges": list(edges.values()),
                "last_sequence": trace.next_sequence - 1,
            }

    def evidence(self, query_id: str, evidence_id: str) -> Optional[dict]:
        """Return one already-stored bounded evidence node."""
        snapshot = self.snapshot(query_id)
        if snapshot is None:
            return None
        return next((
            node for node in snapshot["nodes"]
            if node["kind"] == "evidence"
            and (node.get("metadata") or {}).get("evidence_id") == evidence_id
        ), None)

    def subscribe(self) -> TraceSubscriber:
        """Create a bounded subscriber without changing query execution."""
        subscriber = TraceSubscriber(asyncio.Queue(maxsize=self.subscriber_queue_limit))
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: TraceSubscriber) -> None:
        """Remove a subscriber if it is still connected."""
        with self._lock:
            self._subscribers.discard(subscriber)

    def next_subscriber_event_nowait(self, subscriber: TraceSubscriber) -> dict:
        """Return a queued event or the subscriber's pending reset signal."""
        if subscriber.reset_required:
            subscriber.reset_required = False
            while not subscriber.queue.empty():
                try:
                    subscriber.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            return self._reset_event()
        return subscriber.queue.get_nowait()

    def _reset_event(self) -> dict:
        payload = GraphDelta(
            query_id="*", sequence=0, operation="reset_required",
            summary={"status": "reset_required"},
        ).to_dict(excerpt_chars=self.excerpt_chars)
        with self._lock:
            event_id = self._events[-1]["event_id"] if self._events else 0
        return {"event_id": event_id, "delta": payload}

    def events_since(self, last_event_id: int) -> tuple[bool, list[dict]]:
        """Return replayable events, or signal that the requested history is gone."""
        with self._lock:
            self._evict_expired(self._clock())
            if not self._events:
                return (int(last_event_id) > 0), []
            oldest = self._events[0]["event_id"]
            newest = self._events[-1]["event_id"]
            if (
                int(last_event_id) < max(oldest - 1, self._replay_floor)
                or int(last_event_id) > newest
            ):
                self.reset_count += 1
                return True, []
            return False, [event for event in self._events if event["event_id"] > last_event_id]

    def health(self) -> dict:
        """Return bounded observer counters and limits."""
        with self._lock:
            self._evict_expired(self._clock())
            return {
                "enabled": True,
                "trace_count": len(self._traces),
                "element_count": self._element_count(),
                "subscriber_count": len(self._subscribers),
                "limits": {
                    "traces": self.trace_limit,
                    "elements": self.element_limit,
                    "ttl_s": self.trace_ttl_s,
                    "excerpt_chars": self.excerpt_chars,
                    "question_preview_chars": self.question_preview_chars,
                    "subscriber_queue": self.subscriber_queue_limit,
                },
                "oldest_sequence": self._events[0]["event_id"] if self._events else None,
                "newest_sequence": self._events[-1]["event_id"] if self._events else None,
                "dropped_events": self.dropped_events,
                "reset_count": self.reset_count,
            }


def _node_id(query_id: str, kind: str, reference: str) -> str:
    return f"{query_id}:{kind}:{reference}"


def _edge_id(query_id: str, source: str, relation: str, target: str) -> str:
    digest = hashlib.sha256(f"{source}|{relation}|{target}".encode()).hexdigest()[:16]
    return f"{query_id}:edge:{digest}"


def event_graph_deltas(event, *, excerpt_chars: int, question_preview_chars: int) -> list[GraphDelta]:
    """Project one internal QueryEvent into zero or more safe graph deltas."""
    from .stream_events import (
        EVENT_GRAPH,
        EVENT_ERROR,
        EVENT_QUERY_STARTED,
        EVENT_STAGE,
        EVENT_TOOL_COMPLETED,
        EVENT_TOOL_STARTED,
    )

    query_id = event.query_id
    payload = event.payload
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    operation = None
    summary = None
    if event.type == EVENT_QUERY_STARTED:
        question = str(payload.get("question", ""))
        preview = _question_preview(question, question_preview_chars)
        nodes.append(GraphNode(
            id=_node_id(query_id, "query", "request"), query_id=query_id,
            kind="query", label="Query", summary=preview, status="active",
            metadata={
                "question_digest": hashlib.sha256(question.encode()).hexdigest(),
                "question_preview_chars": question_preview_chars,
            },
        ))
    elif event.type == EVENT_STAGE:
        name = str(payload.get("stage", "stage"))
        phase = str(payload.get("phase", "started"))
        status = {"started": "active", "completed": "complete", "fallback": "fallback"}.get(
            phase, "error"
        )
        nodes.append(GraphNode(
            id=_node_id(query_id, "stage", name), query_id=query_id, kind="stage",
            label=name.replace("_", " ").title(), status=status,
            metadata={key: payload[key] for key in ("elapsed_ms", "reason") if key in payload},
        ))
    elif event.type in (EVENT_TOOL_STARTED, EVENT_TOOL_COMPLETED):
        name = str(payload.get("tool", "tool"))
        reference = f"{payload.get('subquery_id') or 'query'}:{name}"
        status = "active" if event.type == EVENT_TOOL_STARTED else (
            "error" if payload.get("status") == "error" else "complete"
        )
        tool_node = _node_id(query_id, "tool", reference)
        nodes.append(GraphNode(
            id=tool_node, query_id=query_id, kind="tool",
            label=name, status=status,
            metadata={key: payload[key] for key in (
                "subquery_id", "count", "elapsed_ms", "status"
            ) if key in payload},
        ))
        source_node = (
            _node_id(query_id, "subquery", str(payload["subquery_id"]))
            if payload.get("subquery_id")
            else _node_id(query_id, "stage", "route")
        )
        edges.append(GraphEdge(
            id=_edge_id(query_id, source_node, "routed_to", tool_node),
            query_id=query_id, source=source_node, target=tool_node,
            relation="routed_to",
        ))
        if event.type == EVENT_TOOL_COMPLETED:
            retrieve_node = _node_id(query_id, "stage", "retrieve")
            edges.append(GraphEdge(
                id=_edge_id(query_id, tool_node, "returned", retrieve_node),
                query_id=query_id, source=tool_node, target=retrieve_node,
                relation="returned",
                metadata={"status": payload.get("status")},
            ))
    elif event.type == EVENT_GRAPH:
        operation = payload.get("operation")
        summary = payload.get("summary")
        for value in payload.get("nodes") or []:
            node = GraphNode.from_dict({**value, "query_id": query_id})
            if node.kind == "evidence":
                node = replace(node, summary=_bounded_text(node.summary, excerpt_chars))
            nodes.append(node)
        for value in payload.get("edges") or []:
            edges.append(GraphEdge.from_dict({**value, "query_id": query_id}))
    elif event.type == EVENT_ERROR:
        nodes.append(GraphNode(
            id=_node_id(query_id, "stage", "error"), query_id=query_id,
            kind="stage", label="Query Error", summary=str(payload.get("message", "")),
            status="error", metadata={"status": "error"},
        ))
        if payload.get("terminal"):
            operation = "trace_complete"
            summary = {"status": "error", "completed_at": event.timestamp}

    deltas = [
        GraphDelta(query_id=query_id, sequence=event.sequence, timestamp=event.timestamp,
                   operation="upsert_node", node=node)
        for node in nodes
    ]
    deltas.extend(
        GraphDelta(query_id=query_id, sequence=event.sequence, timestamp=event.timestamp,
                   operation="upsert_edge", edge=edge)
        for edge in edges
    )
    if operation in {"trace_complete", "remove"}:
        deltas.append(GraphDelta(
            query_id=query_id, sequence=event.sequence, timestamp=event.timestamp,
            operation=operation, summary=summary,
        ))
    return deltas


def make_event_observer(hub: TraceHub) -> Callable[[object], None]:
    """Return the single-emitter observer callback for a TraceHub."""
    def observe(event) -> None:
        from .stream_events import graph_observer_delta

        for delta in graph_observer_delta(
            event,
            excerpt_chars=hub.excerpt_chars,
            question_preview_chars=hub.question_preview_chars,
        ):
            hub.publish(delta)

    return observe

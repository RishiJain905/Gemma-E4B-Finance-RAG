"""
src/middleware/stream_events.py
Request-scoped progress-event contract for the query pipeline (2.2.6.1).

One internal :class:`QueryEvent` type plus a request-scoped
:class:`QueryEventEmitter` describe *what the pipeline is doing* — pipeline
stages (compile/route/retrieve/grade/correct/pack/generate/validate), safe tool
starts/completions, tokens, terminal metadata, and errors — without ever
exposing the *contents* the pipeline handled. The emitter is the single
instrumentation point: query stages emit once, and separate serializers project
those events onto the redacted chat SSE shape here (the richer, still-allowlisted
graph-observer projection is 2.2.7 — :func:`graph_observer_delta` is a
deliberately empty seam, not a second instrumentation pass).

Redaction is structural, not best-effort: the serializers only ever read a
fixed allowlist of fields (stage name, phase, safe tool name, status, counts,
elapsed time, stable reason codes, a short safe message). Raw tool arguments,
full tool results, prompts, document bodies, secrets, and HTTP headers are never
carried on an event in the first place, so no serializer can leak them.

Backward compatibility: the pre-2.2.6 ``token``, ``metadata``, and ``error``
SSE payloads are unchanged. Progress events (``query_started``, ``stage``,
``tool_started``, ``tool_completed``, ``error``) carry a versioned envelope
(``schema_version``/``query_id``/``sequence``/``timestamp``); ``token`` keeps its
bare ``{"token": ...}`` shape so existing consumers continue to work.
"""

from __future__ import annotations

import itertools
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

# Bumped only on a breaking change to the progress-event envelope/shape. Chat
# and the 2.2.7 graph observer key their parsing off this.
SCHEMA_VERSION = 1

# Event type names. These double as the SSE ``event:`` field for the chat
# serializer, so ``token``/``metadata``/``error`` keep their historical names.
EVENT_QUERY_STARTED = "query_started"
EVENT_STAGE = "stage"
EVENT_TOOL_STARTED = "tool_started"
EVENT_TOOL_COMPLETED = "tool_completed"
EVENT_TOKEN = "token"
EVENT_METADATA = "metadata"
EVENT_ERROR = "error"
EVENT_GRAPH = "graph_observer"

# Allowlisted pipeline stage names + lifecycle phases. Anything outside these
# sets is dropped by :meth:`QueryEventEmitter.stage` so a typo/injected value can
# never reach a client.
STAGE_NAMES = frozenset(
    {"compile", "route", "retrieve", "grade", "correct", "pack", "generate", "validate"}
)
STAGE_PHASES = frozenset({"started", "completed", "fallback"})

# Tool-completion status values.
TOOL_STATUS_OK = "ok"
TOOL_STATUS_ERROR = "error"
_TOOL_STATUSES = frozenset({TOOL_STATUS_OK, TOOL_STATUS_ERROR})
_CURRENT_EMITTER: ContextVar[Optional["QueryEventEmitter"]] = ContextVar(
    "query_event_emitter", default=None
)


@dataclass
class QueryEvent:
    """One internal progress event.

    ``type`` is one of the ``EVENT_*`` constants; ``payload`` holds only
    already-safe scalar fields (never raw args/results/prompts). ``sequence`` is
    monotonic within a single request; ``query_id`` scopes the event to the
    request. Serialization to a wire shape is done by the module-level
    serializers, never on this object, so one event can feed multiple UIs.
    """

    type: str
    query_id: str
    sequence: int
    timestamp: float
    payload: dict = field(default_factory=dict)


def new_query_id() -> str:
    """Return a fresh opaque per-request query id (tracing/scoping only)."""
    return uuid.uuid4().hex


class QueryEventEmitter:
    """Request-scoped factory + buffer for :class:`QueryEvent`.

    Every event is created with a monotonic ``sequence`` and appended to an
    internal buffer; the caller drains the buffer (:meth:`drain`) whenever it is
    ready to serialize and forward. Context-build stages are created before the
    SSE response starts and drained once streaming begins; streaming-time events
    are created and drained immediately. Both routes go through the same counter,
    so ``sequence`` stays globally monotonic and ordered for the whole request.

    Construction never raises; the individual emit helpers validate their inputs
    and drop (log-and-ignore) anything malformed so instrumentation can never
    fail a query.
    """

    def __init__(
        self,
        query_id: Optional[str] = None,
        *,
        include_counts: bool = True,
        observers: Optional[Iterable[Callable[[QueryEvent], None]]] = None,
        buffer_events: bool = True,
    ) -> None:
        self.query_id = query_id or new_query_id()
        self.include_counts = bool(include_counts)
        self._seq = itertools.count()
        self._pending: list[QueryEvent] = []
        self._observers = list(observers or ())
        self._failed_observers: set[int] = set()
        self._buffer_events = bool(buffer_events)

    def _emit(self, type_: str, payload: dict) -> QueryEvent:
        event = QueryEvent(
            type=type_,
            query_id=self.query_id,
            sequence=next(self._seq),
            timestamp=time.time(),
            payload=payload,
        )
        if self._buffer_events:
            self._pending.append(event)
        for observer in self._observers:
            observer_id = id(observer)
            if observer_id in self._failed_observers:
                continue
            try:
                observer(event)
            except Exception:  # noqa: BLE001 - observation must never fail a query
                self._failed_observers.add(observer_id)
                logger.exception("Query event observer failed; discarding it")
        return event

    def drain(self) -> list[QueryEvent]:
        """Return and clear the buffered events (in creation order)."""
        events = self._pending
        self._pending = []
        return events

    # ── Emit helpers (each returns the created event, or None if dropped) ──

    def query_started(self, *, question: Optional[str] = None) -> QueryEvent:
        """Emit the opening ``query_started`` event (should be sequence 0)."""
        payload = {"question": str(question)} if question is not None else {}
        return self._emit(EVENT_QUERY_STARTED, payload)

    def stage(
        self,
        name: str,
        phase: str,
        *,
        elapsed_ms: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> Optional[QueryEvent]:
        """Emit a pipeline ``stage`` event; drop unknown stage/phase names.

        ``reason`` must be a stable, safe code (e.g. ``adaptive_fallback`` or a
        corrective-action name) — never free-form model/user text.
        """
        if name not in STAGE_NAMES or phase not in STAGE_PHASES:
            logger.debug("Dropping stage event with unknown name/phase: %s/%s", name, phase)
            return None
        payload: dict = {"stage": name, "phase": phase}
        if elapsed_ms is not None:
            payload["elapsed_ms"] = round(float(elapsed_ms), 1)
        if reason:
            payload["reason"] = str(reason)
        return self._emit(EVENT_STAGE, payload)

    def tool_started(self, name: str, *, subquery_id: Optional[str] = None) -> Optional[QueryEvent]:
        """Emit a ``tool_started`` event carrying only the safe tool name + sq id."""
        safe = _safe_tool_name(name)
        if not safe:
            return None
        payload: dict = {"tool": safe}
        if subquery_id:
            payload["subquery_id"] = str(subquery_id)
        return self._emit(EVENT_TOOL_STARTED, payload)

    def tool_completed(
        self,
        name: str,
        status: str,
        *,
        count: Optional[int] = None,
        elapsed_ms: Optional[float] = None,
        subquery_id: Optional[str] = None,
    ) -> Optional[QueryEvent]:
        """Emit a ``tool_completed`` event (name, status, optional row/item count).

        The count is emitted only when the emitter was built with
        ``include_counts=True`` (``stream_progress_include_counts``). The tool
        *result* itself is never carried on the event.
        """
        safe = _safe_tool_name(name)
        if not safe:
            return None
        norm_status = status if status in _TOOL_STATUSES else TOOL_STATUS_ERROR
        payload: dict = {"tool": safe, "status": norm_status}
        if self.include_counts and count is not None:
            try:
                payload["count"] = int(count)
            except (TypeError, ValueError):
                pass
        if elapsed_ms is not None:
            payload["elapsed_ms"] = round(float(elapsed_ms), 1)
        if subquery_id:
            payload["subquery_id"] = str(subquery_id)
        return self._emit(EVENT_TOOL_COMPLETED, payload)

    def token(self, token: str) -> QueryEvent:
        """Emit a ``token`` delta (kept for ordering; serialized bare)."""
        return self._emit(EVENT_TOKEN, {"token": token})

    def error(self, message: str, *, terminal: bool) -> QueryEvent:
        """Emit an ``error`` event with a short, safe message + terminal flag."""
        return self._emit(EVENT_ERROR, {"message": _safe_message(message), "terminal": bool(terminal)})

    def graph_update(
        self,
        *,
        nodes: Optional[list[dict]] = None,
        edges: Optional[list[dict]] = None,
        operation: Optional[str] = None,
        summary: Optional[dict] = None,
    ) -> QueryEvent:
        """Emit graph-only references through this shared event stream."""
        payload: dict = {}
        if nodes:
            payload["nodes"] = list(nodes)
        if edges:
            payload["edges"] = list(edges)
        if operation:
            payload["operation"] = str(operation)
        if summary:
            payload["summary"] = dict(summary)
        return self._emit(EVENT_GRAPH, payload)

    def graph_evidence(
        self,
        *,
        evidence_id: str,
        kind: str,
        excerpt: str,
        metadata: Optional[dict] = None,
        source_type: Optional[str] = None,
        rank: Optional[int] = None,
        retrieved_from: Optional[str] = None,
        status: str = "complete",
    ) -> QueryEvent:
        """Emit one actual evidence reference and its source relationship."""
        from .graph_observer import _edge_id, _node_id

        evidence_ref = str(evidence_id)
        evidence_node_id = _node_id(self.query_id, "evidence", evidence_ref)
        evidence_meta = dict(metadata or {})
        evidence_meta.update({"evidence_id": evidence_ref, "kind": str(kind)})
        if source_type:
            evidence_meta["source_type"] = str(source_type)
        if rank is not None:
            evidence_meta["rank"] = int(rank)
        nodes = [{
            "id": evidence_node_id,
            "kind": "evidence",
            "label": f"Evidence {evidence_ref}",
            "summary": str(excerpt),
            "status": status,
            "metadata": evidence_meta,
        }]
        retrieval_node = retrieved_from or _node_id(self.query_id, "stage", "retrieve")
        edges = [{
            "id": _edge_id(self.query_id, retrieval_node, "retrieved", evidence_node_id),
            "source": retrieval_node,
            "target": evidence_node_id,
            "relation": "retrieved",
            "metadata": {"rank": rank} if rank is not None else {},
        }]
        if source_type:
            source_ref = str(source_type)
            source_node_id = _node_id(self.query_id, "source", source_ref)
            nodes.append({
                "id": source_node_id,
                "kind": "source",
                "label": source_ref.replace("_", " ").upper(),
                "status": "complete",
                "metadata": {"source_type": source_ref},
            })
            edges.append({
                "id": _edge_id(self.query_id, evidence_node_id, "from_source", source_node_id),
                "source": evidence_node_id,
                "target": source_node_id,
                "relation": "from_source",
                "metadata": {},
            })
        return self.graph_update(nodes=nodes, edges=edges)


def set_current_emitter(emitter: Optional[QueryEventEmitter]) -> None:
    """Install the emitter visible to executed pipeline modules in this request."""
    _CURRENT_EMITTER.set(emitter)


def current_emitter() -> Optional[QueryEventEmitter]:
    """Return the request's shared emitter, if observation is active."""
    return _CURRENT_EMITTER.get()


def _safe_tool_name(name: Optional[str]) -> Optional[str]:
    """Return a conservatively-sanitized tool name, or None if unusable.

    Tool names in this system are short identifiers (``query_facts``,
    ``get_price_targets``). We allow only ``[A-Za-z0-9_.-]`` and bound the length
    so a malformed/hostile value can never smuggle prose or markup into an event.
    """
    if not name:
        return None
    text = str(name).strip()
    if not text:
        return None
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in "_.-")
    return cleaned[:64] or None


def _safe_message(message: Optional[str]) -> str:
    """Bound and flatten an error message to a short single-line safe string."""
    text = str(message or "").replace("\n", " ").replace("\r", " ").strip()
    return text[:200]


def _envelope(event: QueryEvent) -> dict:
    """The shared versioned envelope for a progress event."""
    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": event.query_id,
        "sequence": event.sequence,
        "timestamp": round(event.timestamp, 3),
    }


def serialize_chat_sse(event: QueryEvent, *, include_counts: bool = True) -> Optional[tuple[str, dict]]:
    """Project one :class:`QueryEvent` onto the redacted chat SSE shape.

    Returns ``(sse_event_name, data_dict)`` or ``None`` when the event has no
    chat projection (``metadata`` is serialized by the app from the response
    model, not here). Only allowlisted fields are copied, so no raw
    arg/result/prompt/secret can ever appear in the output.
    """
    t = event.type
    p = event.payload

    if t == EVENT_TOKEN:
        # Backward-compatible bare shape — older consumers read ``.token``.
        return (EVENT_TOKEN, {"token": p.get("token", "")})

    if t == EVENT_QUERY_STARTED:
        return (EVENT_QUERY_STARTED, _envelope(event))

    if t == EVENT_STAGE:
        data = _envelope(event)
        data["stage"] = p.get("stage")
        data["phase"] = p.get("phase")
        if "elapsed_ms" in p:
            data["elapsed_ms"] = p["elapsed_ms"]
        if "reason" in p:
            data["reason"] = p["reason"]
        return (EVENT_STAGE, data)

    if t == EVENT_TOOL_STARTED:
        data = _envelope(event)
        data["tool"] = p.get("tool")
        if "subquery_id" in p:
            data["subquery_id"] = p["subquery_id"]
        return (EVENT_TOOL_STARTED, data)

    if t == EVENT_TOOL_COMPLETED:
        data = _envelope(event)
        data["tool"] = p.get("tool")
        data["status"] = p.get("status")
        if include_counts and "count" in p:
            data["count"] = p["count"]
        if "elapsed_ms" in p:
            data["elapsed_ms"] = p["elapsed_ms"]
        if "subquery_id" in p:
            data["subquery_id"] = p["subquery_id"]
        return (EVENT_TOOL_COMPLETED, data)

    if t == EVENT_ERROR:
        data = _envelope(event)
        data["message"] = p.get("message", "")
        data["terminal"] = bool(p.get("terminal"))
        return (EVENT_ERROR, data)

    # EVENT_METADATA is emitted by the app from the QueryResponse model.
    return None


def iter_chat_sse(
    events: Iterable[QueryEvent], *, include_counts: bool = True
) -> Iterator[tuple[str, dict]]:
    """Serialize a batch of events to ``(sse_event_name, data)`` pairs, skipping
    events with no chat projection."""
    for event in events:
        serialized = serialize_chat_sse(event, include_counts=include_counts)
        if serialized is not None:
            yield serialized


def graph_observer_delta(
    event: QueryEvent,
    *,
    excerpt_chars: int = 1000,
    question_preview_chars: int = 200,
) -> list:
    """Project one event through the 2.2.7 localhost graph-observer seam.

    The graph observer owns the richer (still allowlisted) delta
    shape. Kept here so the single :class:`QueryEventEmitter` instrumentation is
    the one place both UIs are fed from — the pipeline is never instrumented
    twice.
    """
    from .graph_observer import event_graph_deltas

    return event_graph_deltas(
        event,
        excerpt_chars=excerpt_chars,
        question_preview_chars=question_preview_chars,
    )

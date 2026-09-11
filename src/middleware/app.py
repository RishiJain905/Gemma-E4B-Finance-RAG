"""
src/middleware/app.py
FastAPI application — the main entry point for the middleware layer.

Endpoints:
  GET  /health              — Health check (storage + model)
  POST /query               — Ask a financial question (full pipeline + tools)
  POST /search              — Raw hybrid search (bypasses model)
  POST /financebot/rag      — FinanceBot retrieval with hit/miss contract
  POST /financebot/tools    — FinanceBot dispatch of registered RAG tools
  GET  /tools               — List tools (including classify_trade_bias)

Usage:
    uvicorn src.middleware.app:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import contextvars
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from src.storage.store import Store

from . import prompt_policy
from .config import MiddlewareConfig
from .evidence import (
    assign_evidence_ids,
    build_evidence_items,
    evidence_counts,
    usable_documents,
    usable_facts,
)
from .evidence_trace import EvidenceTraceCollector
from .financebot import (
    invoke_financebot_tool,
    list_financebot_tools,
    run_financebot_retrieval,
)
from .graph_api import create_graph_router
from .models import (
    MAX_QUESTION_CHARS,
    EvidenceCitation,
    FinanceBotRagRequest,
    FinanceBotRagResponse,
    FinanceBotToolRequest,
    FinanceBotToolResponse,
    FreshnessResponse,
    HealthResponse,
    MacroSnapshotResponse,
    QueryRequest,
    QueryResponse,
    RefreshRequest,
    RefreshResponse,
    SearchRequest,
    SearchResponse,
    SentimentResponse,
    SourceCitation,
)

logger = logging.getLogger(__name__)

# Backward-compatible aliases — prompt_policy.py is now the single owner of
# these strings (src/middleware/prompt_policy.py). Kept here so older
# imports/tests referencing middleware_app.SYSTEM_PROMPT still resolve.
SYSTEM_PROMPT = prompt_policy.STRICT_SYSTEM_PROMPT
TOOLS_SYSTEM_PROMPT = prompt_policy.STRICT_TOOLS_SYSTEM_PROMPT

GENERAL_FALLBACK_PREFIX = "Not from your data - general knowledge:"
GENERAL_FALLBACK_CAVEAT = "Please verify against a primary source before relying on it."
NO_GENERAL_FALLBACK_MESSAGE = (
    "I don't have enough data in my knowledge base to answer this. "
    "General-knowledge fallback is disabled for this deployment."
)

# ── Global state (set during lifespan) ─────────────────

config: Optional[MiddlewareConfig] = None
store: Optional[Store] = None
model_client: Optional[httpx.AsyncClient] = None
_tools_supported: bool = True
MACRO_SNAPSHOT_METRICS = ["GDP", "CPIAUCSL", "FEDFUNDS", "UNRATE", "DGS10", "T10Y2Y"]
_MODEL_TASKS_CACHE: Optional[dict] = None
FETCH_ON_MISS_MIN_CONFIDENCE = 0.9
retriever = None  # Shared Retriever (built on startup) — Phase 2.1.2
# Phase 2.2.6.2 — process-global versioned retrieval cache (built lazily when
# enable_retrieval_cache is on) and the llama-server prompt-reuse capability flag
# (flipped off for the process if the backend rejects cache_prompt once).
_retrieval_cache = None
_prompt_cache_supported: bool = True
_MODEL_HEALTH_TTL_S = 10.0
_HEALTH_SUMMARY_TTL_S = 3.0
_model_health = {"ok": False, "ts": 0.0}
_health_cache = {"ts": 0.0, "value": None}
_scheduler = None
graph_hub = None  # Process-local TraceHub; created only when observer is enabled.

# Per-request state (Phase 2.1.8.3). Each incoming request runs in its own
# asyncio Task, which copies the context at creation time, so these never
# leak between concurrent requests as long as they're reset at the top of
# _build_query_context — the single entry point shared by /query and
# /query/stream.
_tools_used_var: contextvars.ContextVar[Optional[list[str]]] = contextvars.ContextVar(
    "tools_used", default=None
)
_answer_policy_override_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "answer_policy_override", default=None
)
_evidence_trace_var: contextvars.ContextVar[Optional[EvidenceTraceCollector]] = contextvars.ContextVar(
    "evidence_trace", default=None
)

# Request-scoped progress-event emitter (2.2.6.1). Non-None only for a
# /query/stream request that has stream progress events enabled; every other
# request path (including /query) leaves it None so the _emit_* helpers are
# no-ops and behavior is byte-identical. Set in the stream endpoint BEFORE
# _build_query_context so context-build stages can record into it, and never
# touched by _reset_request_scoped_state.
_stream_emitter_var: contextvars.ContextVar[Optional["object"]] = contextvars.ContextVar(
    "stream_emitter", default=None
)


def _stream_emitter():
    """Return the current request's progress-event emitter, or None."""
    return _stream_emitter_var.get()


def _graph_observer_enabled() -> bool:
    """Return whether the local query graph is enabled for this process."""
    return bool(getattr(config, "enable_graph_observer", False))


def _install_query_emitter(request: QueryRequest, *, chat_events: bool):
    """Install the one request-scoped emitter used by chat and graph projections."""
    progress = bool(chat_events and _stream_progress_enabled())
    observe = _graph_observer_enabled() and graph_hub is not None
    if not progress and not observe:
        _stream_emitter_var.set(None)
        from .stream_events import set_current_emitter

        set_current_emitter(None)
        return None
    from .stream_events import QueryEventEmitter, set_current_emitter

    observers = []
    if observe:
        from .graph_observer import make_event_observer

        observers.append(make_event_observer(graph_hub))
    emitter = QueryEventEmitter(
        include_counts=bool(getattr(config, "stream_progress_include_counts", True)),
        observers=observers,
        buffer_events=progress,
    )
    _stream_emitter_var.set(emitter)
    set_current_emitter(emitter)
    emitter.query_started(question=request.question)
    return emitter


def _emit_stage(name: str, phase: str, *, elapsed_ms=None, reason=None) -> None:
    """Record one pipeline stage progress event, if an emitter is installed."""
    emitter = _stream_emitter_var.get()
    if emitter is not None:
        emitter.stage(name, phase, elapsed_ms=elapsed_ms, reason=reason)


def _emit_tool_started(name: str, *, subquery_id=None) -> None:
    """Record a tool-started progress event, if an emitter is installed."""
    emitter = _stream_emitter_var.get()
    if emitter is not None:
        emitter.tool_started(name, subquery_id=subquery_id)


def _emit_tool_completed(name: str, status: str, *, count=None, elapsed_ms=None,
                         subquery_id=None) -> None:
    """Record a tool-completed progress event, if an emitter is installed."""
    emitter = _stream_emitter_var.get()
    if emitter is not None:
        emitter.tool_completed(
            name, status, count=count, elapsed_ms=elapsed_ms, subquery_id=subquery_id)


def _emit_graph_legacy_intent(intent: dict, *, fallback_reason: Optional[str] = None) -> None:
    """Emit the compact legacy query -> route(intent) -> retrieval step.

    The legacy intent-parse *is* the routing stage, so its node uses the shared
    ``stage:route`` id (label stays "Intent" for the reader). This keeps it in
    the ROUTE column and, critically, lets the observer's tool ``routed_to``
    edges — which source from ``stage:route`` whenever a tool has no subquery,
    as legacy tools never do — resolve to a real node instead of dangling.

    ``fallback_reason`` is set only when the adaptive layer demoted to legacy;
    it marks the routing node as ``fallback`` so that signal survives the merge
    with the ``stage:route`` node the fallback branch already emitted.
    """
    emitter = _stream_emitter_var.get()
    if emitter is None or not _graph_observer_enabled():
        return
    from .graph_observer import _edge_id, _node_id

    query_id = emitter.query_id
    query_node = _node_id(query_id, "query", "request")
    route_node = _node_id(query_id, "stage", "route")
    retrieve_node = _node_id(query_id, "stage", "retrieve")
    metadata = {
        "ticker": intent.get("ticker"),
        "metrics": list(intent.get("metrics") or ()),
        "period": intent.get("timeframe"),
        "kind": intent.get("question_type"),
    }
    if fallback_reason:
        metadata["reason"] = fallback_reason
    emitter.graph_update(
        nodes=[{
            "id": route_node,
            "kind": "stage",
            "label": "Intent",
            "status": "fallback" if fallback_reason else "complete",
            "metadata": metadata,
        }],
        edges=[
            {
                "id": _edge_id(query_id, query_node, "compiled_to", route_node),
                "source": query_node,
                "target": route_node,
                "relation": "compiled_to",
            },
            {
                "id": _edge_id(query_id, route_node, "routed_to", retrieve_node),
                "source": route_node,
                "target": retrieve_node,
                "relation": "routed_to",
            },
        ],
    )


def _emit_graph_plan(plan, *, lane: Optional[str] = None) -> None:
    """Emit the validated plan and its actual request-local subqueries."""
    emitter = _stream_emitter_var.get()
    if emitter is None or not _graph_observer_enabled():
        return
    from .graph_observer import _edge_id, _node_id

    query_id = emitter.query_id
    query_node = _node_id(query_id, "query", "request")
    plan_node = _node_id(query_id, "plan", "validated")
    nodes = [{
        "id": plan_node,
        "kind": "plan",
        "label": "Validated Query Plan",
        "status": "complete",
        "metadata": {
            "tickers": list(plan.tickers),
            "intents": list(plan.intents),
            "metrics": list(plan.metrics),
            "periods": list(plan.periods),
            "lane": lane,
        },
    }]
    edges = [{
        "id": _edge_id(query_id, query_node, "compiled_to", plan_node),
        "source": query_node,
        "target": plan_node,
        "relation": "compiled_to",
    }]
    for subquery in plan.subqueries:
        subquery_node = _node_id(query_id, "subquery", subquery.id)
        nodes.append({
            "id": subquery_node,
            "kind": "subquery",
            "label": subquery.id,
            "status": "complete",
            "group_id": plan_node,
            "metadata": {
                "subquery_id": subquery.id,
                "tickers": list(subquery.entity_tickers),
                "intents": list(subquery.intents),
                "metrics": list(subquery.metrics),
                "periods": list(subquery.periods),
                "derived": bool(subquery.derived),
                "parent_id": subquery.parent_id,
                "reason": subquery.reason_code,
            },
        })
        edges.append({
            "id": _edge_id(query_id, plan_node, "contains", subquery_node),
            "source": plan_node,
            "target": subquery_node,
            "relation": "contains",
        })
        if subquery.derived and subquery.parent_id:
            parent_node = _node_id(query_id, "subquery", subquery.parent_id)
            edges.append({
                "id": _edge_id(query_id, parent_node, "expanded_from", subquery_node),
                "source": parent_node,
                "target": subquery_node,
                "relation": "expanded_from",
            })
    emitter.graph_update(nodes=nodes, edges=edges)


def _emit_graph_adaptive_result(result) -> None:
    """Emit actual adaptive lane, grader, and correction outcome counters."""
    emitter = _stream_emitter_var.get()
    if emitter is None or not _graph_observer_enabled():
        return
    from .graph_observer import _edge_id, _node_id

    query_id = emitter.query_id
    plan_node = _node_id(query_id, "plan", "validated")
    route_node = _node_id(query_id, "stage", "route")
    nodes = [{
        "id": route_node,
        "kind": "stage",
        "label": "Route",
        "status": "fallback" if result.fallback_reason else "complete",
        "metadata": result.graph_trace_metadata(),
    }]
    edges = [{
        "id": _edge_id(query_id, plan_node, "routed_to", route_node),
        "source": plan_node,
        "target": route_node,
        "relation": "routed_to",
    }]
    if result.sufficiency is not None:
        grade_node = _node_id(query_id, "stage", "grade")
        nodes.append({
            "id": grade_node,
            "kind": "stage",
            "label": "Evidence Grade",
            "status": "complete",
            "metadata": result.sufficiency.graph_trace_metadata(),
        })
        edges.append({
            "id": _edge_id(query_id, route_node, "returned", grade_node),
            "source": route_node,
            "target": grade_node,
            "relation": "returned",
        })
    if result.retry_performed:
        correct_node = _node_id(query_id, "stage", "correct")
        nodes.append({
            "id": correct_node,
            "kind": "stage",
            "label": "Corrective Retrieval",
            "status": "complete",
            "metadata": {"reason": result.corrective_action.value},
        })
        edges.append({
            "id": _edge_id(query_id, route_node, "corrected_by", correct_node),
            "source": route_node,
            "target": correct_node,
            "relation": "corrected_by",
        })
    emitter.graph_update(nodes=nodes, edges=edges)


def _emit_graph_evidence(
    retrieval: dict, ledger: list, *, status: str = "complete"
) -> list[str]:
    """Emit only the actual normalized evidence selected for prompt packing."""
    emitter = _stream_emitter_var.get()
    if emitter is None or not _graph_observer_enabled():
        return []
    items = list(ledger) if ledger else build_evidence_items(
        usable_facts(retrieval), usable_documents(retrieval))
    reference_by_store_id = {
        str(item.store_id): str(item.evidence_id or item.store_id)
        for item in items
        if item.store_id and (item.evidence_id or item.store_id)
    }
    references: list[str] = []
    excerpt_limit = int(getattr(config, "graph_excerpt_chars", 1000))
    for rank, item in enumerate(items, 1):
        reference = item.evidence_id or item.store_id
        if not reference:
            continue
        reference = str(reference)
        references.append(reference)
        if item.kind == "document":
            excerpt = item.document[:excerpt_limit]
        else:
            excerpt = " ".join(str(value) for value in (
                item.metric, item.value, item.unit, item.period
            ) if value not in (None, ""))[:excerpt_limit]
        score = next(iter(item.scores.values()), None)
        metadata = {
            "ticker": item.ticker,
            "metric": item.metric,
            "period": item.period,
            "freshness": item.freshness,
            "score": score,
            "store_id": item.store_id,
            "section": item.section,
            "parent_id": item.parent_id,
        }
        metadata.update(item.graph_evidence_metadata())
        retrieved_from = None
        if item.subquery_ids:
            from .graph_observer import _node_id

            retrieved_from = _node_id(emitter.query_id, "subquery", item.subquery_ids[0])
        emitter.graph_evidence(
            evidence_id=reference,
            kind=item.kind,
            excerpt=excerpt,
            metadata=metadata,
            source_type=item.source_type or "unknown",
            rank=rank,
            retrieved_from=retrieved_from,
            status=status,
            source_metadata=item.source_node_metadata(),
        )
        parent_reference = reference_by_store_id.get(str(item.parent_id))
        if parent_reference:
            from .graph_observer import _edge_id, _node_id

            parent_node = _node_id(emitter.query_id, "evidence", parent_reference)
            evidence_node = _node_id(emitter.query_id, "evidence", reference)
            emitter.graph_update(edges=[{
                "id": _edge_id(
                    emitter.query_id, parent_node, "expanded_from", evidence_node
                ),
                "source": parent_node,
                "target": evidence_node,
                "relation": "expanded_from",
            }])
    return references


def _emit_graph_dropped_evidence(result, selected_retrieval: dict) -> None:
    """Emit actual retrieved rows excluded by the adaptive context budget."""
    if result.context is None:
        return
    selected = build_evidence_items(
        usable_facts(selected_retrieval), usable_documents(selected_retrieval))
    selected_ids = {(item.kind, item.store_id) for item in selected}
    candidates = build_evidence_items(result.merged_facts, result.merged_documents)
    dropped_facts = []
    dropped_documents = []
    for item, row in zip(candidates, [*result.merged_facts, *result.merged_documents]):
        if (item.kind, item.store_id) in selected_ids:
            continue
        if item.kind == "fact":
            dropped_facts.append(row)
        else:
            dropped_documents.append(row)
    if dropped_facts or dropped_documents:
        _emit_graph_evidence(
            {"facts": dropped_facts, "documents": dropped_documents},
            [],
            status="dropped",
        )


def _emit_graph_terminal(
    context: dict,
    *,
    model_available: bool,
    evidence_citations: Optional[list],
    validation: Optional[dict],
    citations: Optional[list] = None,
) -> None:
    """Emit answer/citation/validation nodes from the produced terminal result."""
    emitter = _stream_emitter_var.get()
    if emitter is None or not _graph_observer_enabled():
        return
    from .graph_observer import _edge_id, _node_id

    query_id = emitter.query_id
    answer_node = _node_id(query_id, "answer", "final")
    nodes = [{
        "id": answer_node,
        "kind": "answer",
        "label": "Final Answer",
        "status": "complete" if model_available else "fallback",
        "metadata": {
            "status": context.get("grounding_level"),
            "facts": len(usable_facts(context.get("retrieval"))),
            "documents": len(usable_documents(context.get("retrieval"))),
            "answer_origin": context.get("answer_origin"),
            "generation_skipped": context.get("generation_skipped"),
            "generation_skip_reason": context.get("generation_skip_reason"),
        },
    }]
    query_node = _node_id(query_id, "query", "request")
    edges = [{
        "id": _edge_id(query_id, query_node, "returned", answer_node),
        "source": query_node,
        "target": answer_node,
        "relation": "returned",
    }]
    for reference in context.get("graph_evidence_ids") or []:
        evidence_node = _node_id(query_id, "evidence", str(reference))
        edges.append({
            "id": _edge_id(query_id, evidence_node, "supports", answer_node),
            "source": evidence_node,
            "target": answer_node,
            "relation": "supports",
        })
    for index, citation in enumerate(evidence_citations or (), 1):
        data = citation.model_dump() if hasattr(citation, "model_dump") else citation.dict()
        reference = data.get("evidence_id") or f"legacy-{index}"
        citation_node = _node_id(query_id, "citation", str(reference))
        nodes.append({
            "id": citation_node,
            "kind": "citation",
            "label": f"Citation {reference}",
            "status": "complete" if data.get("support_status") == "supported" else "error",
            "metadata": {
                "evidence_id": data.get("evidence_id"),
                "source_type": data.get("source_type"),
                "ticker": data.get("ticker"),
                "metric": data.get("metric"),
                "period": data.get("period"),
                "support_status": data.get("support_status"),
                "item_type": data.get("item_type"),
                "event_type": data.get("event_type"),
                "authority_tier": data.get("authority_tier"),
                "source": data.get("source"),
                "date_semantics": data.get("date_semantics"),
                "canonical_security": data.get("canonical_security"),
                "coverage_tier": data.get("coverage_tier"),
                "source_category": data.get("source_category"),
                "provider": data.get("provider"),
                "publisher": data.get("publisher"),
            },
        })
        if data.get("evidence_id"):
            evidence_node = _node_id(query_id, "evidence", str(data["evidence_id"]))
            edges.append({
                "id": _edge_id(query_id, evidence_node, "cited_by", citation_node),
                "source": evidence_node,
                "target": citation_node,
                "relation": "cited_by",
            })
        edges.append({
            "id": _edge_id(query_id, citation_node, "supports", answer_node),
            "source": citation_node,
            "target": answer_node,
            "relation": "supports",
        })
    if not evidence_citations:
        for index, citation in enumerate(citations or (), 1):
            data = citation.model_dump() if hasattr(citation, "model_dump") else citation.dict()
            reference = f"legacy-{index}"
            citation_node = _node_id(query_id, "citation", reference)
            nodes.append({
                "id": citation_node,
                "kind": "citation",
                "label": f"Citation {index}",
                "status": "complete",
                "metadata": {
                    "source_type": data.get("source_type"),
                    "ticker": data.get("ticker"),
                    "metric": data.get("metric"),
                    "period": data.get("period"),
                    "support_status": "supported",
                },
            })
            edges.append({
                "id": _edge_id(query_id, citation_node, "supports", answer_node),
                "source": citation_node,
                "target": answer_node,
                "relation": "supports",
            })
    validate_node = _node_id(query_id, "stage", "validate")
    validation_status = (validation or {}).get("validation_status") or "off"
    nodes.append({
        "id": validate_node,
        "kind": "stage",
        "label": "Validate",
        "status": (
            "complete"
            if validation_status in {"supported", "no_claims", "off"}
            else "partial"
        ),
        "metadata": {
            "validation_status": validation_status,
            "citation_support_rate": (validation or {}).get("citation_support_rate"),
            "numeric_claims_supported": (validation or {}).get("numeric_claims_supported"),
            "numeric_claims_unsupported": (validation or {}).get("numeric_claims_unsupported"),
        },
    })
    edges.append({
        "id": _edge_id(query_id, answer_node, "validated_as", validate_node),
        "source": answer_node,
        "target": validate_node,
        "relation": "validated_as",
        "metadata": {"status": validation_status},
    })
    emitter.graph_update(nodes=nodes, edges=edges)
    emitter.graph_update(
        operation="trace_complete",
        summary={
            "status": context.get("grounding_level"),
            "completed_at": time.time(),
        },
    )


def _grounding_level(retrieval: dict) -> str:
    """Return the graded grounding level from usable facts/documents.

    Blank document bodies and None-valued facts are not usable evidence
    (see src/middleware/evidence.py) and must not inflate grounding.
    """
    n_facts, n_docs = evidence_counts(retrieval)
    n = n_facts + n_docs
    return "grounded" if n >= 3 else "partial" if n >= 1 else "none"


def _answer_mode_from_sufficiency(
    sufficiency,
    *,
    requires_specific_figures: bool,
) -> str:
    """Convert deterministic obligation coverage into one answer policy mode."""
    from .evidence_grader import SufficiencyStatus

    if sufficiency.status is SufficiencyStatus.SUFFICIENT:
        return "grounded"
    if any(row.covered_fields for row in sufficiency.coverage):
        return "partial"
    if not requires_specific_figures and _allow_general_fallback():
        return "general"
    return "refused"


def _answer_policy() -> str:
    """Return the effective answer policy: per-request override, else configured default."""
    override = _answer_policy_override_var.get()
    if override in ("strict", "graded"):
        return override
    policy = str(getattr(config, "answer_policy", "graded") or "graded").lower()
    return "strict" if policy == "strict" else "graded"


def _answer_validation_mode() -> str:
    """Return the effective citation/numeric validation policy (2.2.4.3).

    off (or an unknown value) preserves legacy behavior byte-for-byte; report
    attaches validation metadata without changing the answer; enforce may
    downgrade grounding or refuse a wholly-unsupported answer.
    """
    mode = str(getattr(config, "answer_validation", "off") or "off").strip().lower()
    return mode if mode in ("off", "report", "enforce") else "off"


def _evidence_ids_enabled() -> bool:
    """Whether request-local ``[E#]`` evidence ids should be built/rendered."""
    return _answer_validation_mode() != "off"


def _build_evidence_ledger(retrieval: dict) -> list:
    """Build the packed, ``E#``-numbered ledger from usable retrieval evidence."""
    return assign_evidence_ids(
        build_evidence_items(usable_facts(retrieval), usable_documents(retrieval)))


def _reset_request_scoped_state(answer_policy: Optional[str]) -> None:
    """Reset per-request contextvars: tool-call log, answer-policy override,
    and evidence-trace collector."""
    _tools_used_var.set([])
    _evidence_trace_var.set(None)
    normalized = str(answer_policy or "").strip().lower()
    _answer_policy_override_var.set(normalized if normalized in ("strict", "graded") else None)


def _record_tool_used(name: str) -> None:
    """Append a dispatched tool name to the current request's tool-call log."""
    used = _tools_used_var.get()
    if used is not None and name and name not in used:
        used.append(name)


def _get_tools_used() -> Optional[list[str]]:
    """Return the current request's dispatched tool names, or None if empty."""
    used = _tools_used_var.get()
    return list(used) if used else None


def _record_trace_prompt(system_prompt: str, user_prompt: str) -> None:
    """Record the exact system/user messages for the request's evidence trace
    (2.2.1.2). No-op when no trace was requested for this request."""
    collector = _evidence_trace_var.get()
    if collector is not None:
        collector.record_prompt(system_prompt=system_prompt, user_prompt=user_prompt)


def _record_trace_tool_result(name: str, arguments: dict, result: dict) -> None:
    """Append one dispatched tool call to the request's evidence trace, if any."""
    collector = _evidence_trace_var.get()
    if collector is not None:
        collector.record_tool_result(name, arguments, result)


def _discard_trace_tool_results() -> None:
    """Drop any tool results recorded so far on the request's evidence trace.

    Called before a plain-call fallback (tools unsupported / empty tool-mode
    response) records its messages, so the abandoned tool attempt's results
    never leak into the successful answer path's trace.
    """
    collector = _evidence_trace_var.get()
    if collector is not None:
        collector.discard_tool_results()


def _resolved_ticker_field(intent: dict) -> Optional[dict]:
    """Return {'name','source'} when the resolver mapped a non-exact ticker.

    Omitted for exact matches (known_ticker/override) so the field only
    fires for name lookups and typo-corrected fuzzy matches.
    """
    source = intent.get("ticker_source")
    name = intent.get("resolved_name")
    if not source or not name or source in ("known_ticker", "override"):
        return None
    return {"name": name, "source": source}


def _allow_general_fallback() -> bool:
    """Return whether no-context general fallback answers are allowed."""
    return bool(getattr(config, "allow_general_fallback", True))


def _system_prompt_for_request(
    intent: Optional[dict],
    grounding_level: str,
    tools_enabled: bool = False,
) -> str:
    """Return this request's system prompt via the shared prompt_policy builder.

    Shared by the plain, streaming, and tool-loop call sites so they cannot
    silently enforce different rules (2.2.1.1).
    """
    return prompt_policy.build_system_prompt(
        answer_policy=_answer_policy(),
        allow_general_fallback=_allow_general_fallback(),
        intent=intent,
        grounding_level=grounding_level,
        tools_enabled=tools_enabled,
    )


def _is_declined_answer(answer: str) -> bool:
    """Heuristically detect model refusals/unsafe declines."""
    text = (answer or "").strip().lower()
    if not text:
        return True
    decline_markers = (
        "i don't have enough data",
        "i do not have enough data",
        "i can't answer",
        "i cannot answer",
        "i'm unable to answer",
        "i am unable to answer",
        "cannot provide",
        "can't provide",
        "unsafe",
        "not enough information",
        "data is unavailable",
    )
    return any(marker in text for marker in decline_markers)


def _apply_answer_policy(answer: str, grounding_level: str) -> str:
    """Enforce deterministic labels/refusals that should not depend on sampling."""
    if _answer_policy() == "strict":
        return answer
    if not answer or answer.startswith(("Error calling model:", "Model unavailable.")):
        return answer
    if grounding_level == "refused":
        return NO_GENERAL_FALLBACK_MESSAGE
    if grounding_level not in {"none", "general"}:
        return answer
    if not _allow_general_fallback():
        return NO_GENERAL_FALLBACK_MESSAGE
    if _is_declined_answer(answer):
        return answer

    labeled = answer.strip()
    if not labeled.lower().startswith(GENERAL_FALLBACK_PREFIX.lower()):
        labeled = f"{GENERAL_FALLBACK_PREFIX} {labeled}"
    if "verify against a primary source" not in labeled.lower():
        labeled = f"{labeled}\n\n{GENERAL_FALLBACK_CAVEAT}"
    return labeled


def _response_grounding(answer: str, grounding_level: str) -> str:
    """Map the actual answer path to response metadata."""
    if _is_declined_answer(answer):
        return "refused"
    if grounding_level == "grounded":
        return "grounded"
    if grounding_level == "partial":
        return "partial"
    if grounding_level in {"none", "general"} and _allow_general_fallback():
        return "general"
    return "refused"


async def _invoke_model(
    prompt: str,
    temperature: float,
    max_tokens: int,
    intent: dict,
    grounding_level: str,
) -> tuple[str, list[SourceCitation]]:
    """Call _call_model while preserving old-signature test monkeypatches."""
    import inspect

    params = inspect.signature(_call_model).parameters
    if "intent" not in params:
        return await _call_model(prompt, temperature, max_tokens)
    return await _call_model(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        intent=intent,
        grounding_level=grounding_level,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global config, store, model_client, retriever, graph_hub

    logger.info("Starting middleware...")
    from src.utils.env import load_env
    load_env()  # load .env credentials before initializing components
    config = MiddlewareConfig()
    if config.enable_graph_observer:
        from .graph_observer import TraceHub

        graph_hub = TraceHub(
            trace_limit=config.graph_trace_limit,
            element_limit=config.graph_element_limit,
            trace_ttl_s=config.graph_trace_ttl_s,
            excerpt_chars=config.graph_excerpt_chars,
            question_preview_chars=config.graph_question_preview_chars,
        )
    else:
        graph_hub = None
    store = Store(
        embedding_endpoint=config.embedding_endpoint,
        embedding_cache_size=config.embedding_cache_size,
        embedding_model=config.model_name,
    )
    model_client = httpx.AsyncClient(timeout=60)

    # Shared retriever so the BM25 lexical index is built once and reused
    # across requests (Phase 2.1.2). Warm it eagerly on startup.
    from .retriever import Retriever
    retriever = Retriever(store=store, config=config)
    if config.enable_lexical:
        retriever.warm_lexical_index()

    yield  # App runs here

    # Shutdown
    if model_client:
        await model_client.aclose()
    graph_hub = None
    logger.info("Middleware shut down.")


app = FastAPI(
    title="FinanceBot hybrid RAG (Gemma-E4B-Finance-RAG)",
    description=(
        "Hybrid finance RAG for FinanceBot — retrieval, registered tools "
        "(including long/short classify_trade_bias), and optional local generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(create_graph_router(
    lambda: graph_hub,
    _graph_observer_enabled,
    lambda: store,
    lambda: config,
))

# ── Static single-page graph UI (Phase 2.2.7.3) ──────────────────────────────
# Served from disk only when the observer is enabled, mirroring the 404-when-off
# convention of the graph API router. No StaticFiles mount / directory listing:
# a fixed extension allowlist and a resolved-path containment check keep the
# surface to the pinned bundle only.
_GRAPH_STATIC_DIR = (Path(__file__).parent / "static" / "graph").resolve()
_GRAPH_MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".map": "application/json",
}


def _graph_static_file(relative: str) -> Path:
    """Resolve a request path inside the pinned bundle or raise 404.

    Rejects traversal (``..``), absolute paths, disallowed extensions, and any
    resolved path that escapes ``_GRAPH_STATIC_DIR``.
    """
    if not _graph_observer_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    candidate = (_GRAPH_STATIC_DIR / relative).resolve()
    try:
        candidate.relative_to(_GRAPH_STATIC_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc
    if candidate.suffix.lower() not in _GRAPH_MEDIA_TYPES:
        raise HTTPException(status_code=404, detail="Not found")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return candidate


@app.get("/graph", include_in_schema=False)
def graph_index() -> FileResponse:
    """Return the single-page graph interface (404 when the observer is off)."""
    index = _graph_static_file("index.html")
    return FileResponse(
        index,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/graph/static/{asset:path}", include_in_schema=False)
def graph_static(asset: str) -> FileResponse:
    """Serve one pinned bundle asset with a safe MIME type and no listing."""
    target = _graph_static_file(asset)
    return FileResponse(
        target,
        media_type=_GRAPH_MEDIA_TYPES[target.suffix.lower()],
        headers={"Cache-Control": "no-cache"},
    )


# ── Local-safe graph security posture (Phase 2.2.7.4) ────────────────────────
# The observer UI and API are a strictly local, read-only side channel. This one
# middleware is the single chokepoint that (1) hides every /graph* surface from a
# non-loopback client (no bypass flag — remote exposure is a separate future
# design covering auth/TLS/proxy trust) and (2) stamps a strict same-origin CSP
# and hardening headers on the responses. The graph router keeps its own loopback
# dependency so it is still safe when mounted standalone in tests.
_GRAPH_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# default-src 'none' denies everything not explicitly granted. Scripts/fonts/
# connect (fetch + SSE) are same-origin only; img allows data: for the inline
# favicon and canvas tiles. style-src additionally allows 'unsafe-inline' because
# Cytoscape injects one fixed ``position: relative`` <style> element at runtime —
# inline STYLE cannot execute code, and script-src stays strict 'self', which is
# the XSS-critical directive. No third-party origin is ever allowed.
_GRAPH_CSP = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)


def _is_graph_path(path: str) -> bool:
    return path == "/graph" or path.startswith("/graph/")


def _apply_graph_security_headers(response, path: str) -> None:
    """Stamp the strict same-origin CSP + hardening headers on a graph response."""
    response.headers["Content-Security-Policy"] = _GRAPH_CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    # Read-only observability data is never cacheable. Handlers that need a
    # different directive (SSE keep-alive uses no-cache; the static bundle uses
    # no-cache) set Cache-Control themselves; only add the default when absent.
    response.headers.setdefault("Cache-Control", "no-store")


@app.middleware("http")
async def _graph_security_middleware(request: Request, call_next):
    """Enforce loopback-only access and same-origin CSP for every /graph* route."""
    path = request.url.path
    if not _is_graph_path(path):
        return await call_next(request)
    host = request.client.host if request.client else ""
    if host not in _GRAPH_LOOPBACK_HOSTS:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    response = await call_next(request)
    _apply_graph_security_headers(response, path)
    return response


def _find_tasks_block(node) -> dict:
    if not isinstance(node, dict):
        return {}
    tasks = node.get("tasks")
    if isinstance(tasks, dict):
        return tasks
    for value in node.values():
        found = _find_tasks_block(value)
        if found:
            return found
    return {}


def _task_params(task_name) -> dict:
    """Return model task params from configs/model.yaml; fail soft."""
    global _MODEL_TASKS_CACHE
    try:
        if _MODEL_TASKS_CACHE is None:
            import yaml

            path = Path(__file__).resolve().parents[2] / "configs" / "model.yaml"
            with open(path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            _MODEL_TASKS_CACHE = _find_tasks_block(loaded)
        task = _MODEL_TASKS_CACHE.get(str(task_name), {})
        return task if isinstance(task, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load model task params: %s", exc)
        _MODEL_TASKS_CACHE = {}
        return {}


def _macro_snapshot_data() -> dict:
    return store.get_fundamentals_batch("MACRO", metrics=MACRO_SNAPSHOT_METRICS)


def _sentiment_data(ticker: str, days: int = 7) -> dict:
    from src.macros.gdelt_ingestor import GDELTIngestor

    return GDELTIngestor(store=store).get_sentiment_summary(ticker.upper(), days=days)


def _guidance_data(ticker: str) -> dict:
    from src.macros.earnings_transcripts import EarningsTranscriptIngestor

    return EarningsTranscriptIngestor(store=store).get_latest_guidance(ticker.upper())


# ── Health ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Enhanced health check with storage, model, scheduler, and freshness."""
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    storage_health = store.heartbeat()
    model_ok = await _check_model_health()
    summary = _cached_health_summary()

    capabilities = None
    if config:
        # Effective streaming capability (2.2.6.1): whether this deployment can
        # serve the /query/stream endpoint at all. With tools enabled, streaming
        # is available only when tool-final streaming is on; otherwise the
        # endpoint 404s. streaming_tool_final is advertised (true) only when that
        # path is actually active, so a client learns tools-enabled requests can
        # still stream their final synthesis.
        streaming_enabled = bool(getattr(config, "enable_streaming", True))
        tool_final_on = bool(getattr(config, "enable_tool_final_streaming", False))
        tools_on = bool(config.enable_tools)
        streaming_capable = streaming_enabled and (not tools_on or tool_final_on)
        capabilities = {
            "tools": tools_on,
            "streaming": streaming_capable,
            "streaming_tool_final": bool(streaming_enabled and tools_on and tool_final_on),
            "answer_policy": str(getattr(config, "answer_policy", "graded") or "graded").lower(),
            # Bounded adaptive-RAG (2.2.3.4) effective booleans. Clients/eval
            # read these to know whether the adaptive route + deterministic tool
            # routing are actually live on this deployment.
            "adaptive_rag": bool(getattr(config, "enable_adaptive_rag", False)),
            "deterministic_tool_routing": bool(
                getattr(config, "enable_deterministic_tool_routing", False)),
            # Conversation-memory (2.2.2) capabilities + effective limits. The
            # client reads these to size its multiline composer and history
            # controls; older servers omit them and the client uses local
            # defaults. history/multiline are always-true on this build since
            # the request contract accepts bounded history and multi-line
            # questions up to max_question_chars.
            "history": True,
            "multiline": True,
            "max_question_chars": int(MAX_QUESTION_CHARS),
            "conversation_max_turns": int(getattr(config, "conversation_max_turns", 8)),
            "conversation_max_history_chars": int(
                getattr(config, "conversation_max_history_chars", 8000)),
            "lexical_backend": str(getattr(config, "lexical_backend", "fts5")),
            "lexical_mode": (
                retriever.lexical.mode
                if bool(getattr(config, "enable_lexical", True)) and retriever is not None
                else "disabled"
            ),
        }
        # Local query-graph observer (2.2.7.4). Advertised only when enabled so
        # older/observer-off servers stay byte-compatible and the chat client
        # knows the effective, same-origin graph URL + bounded observer limits.
        if _graph_observer_enabled():
            capabilities["graph_observer"] = True
            capabilities["graph_url"] = str(request.base_url).rstrip("/") + "/graph"
            if graph_hub is not None:
                capabilities["graph_observer_limits"] = graph_hub.health().get("limits")

    return HealthResponse(
        status="ok" if storage_health.get("sqlite") else "degraded",
        storage=storage_health,
        model_available=model_ok,
        scheduler=summary.get("scheduler"),
        freshness=summary.get("freshness"),
        capabilities=capabilities,
    )


@app.get("/tools")
async def tools():
    """List model-callable middleware tools and tool-gating state."""
    from .tools import REGISTRY

    return {
        "enabled": bool(config.enable_tools) if config else False,
        "allow_write_tools": bool(config.allow_write_tools) if config else False,
        "financebot_dispatch": "POST /financebot/tools",
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "write": tool.write,
            }
            for tool in REGISTRY.values()
        ],
    }


def _mark_model_health(ok: bool) -> None:
    """Refresh the in-process model-health cache."""
    _model_health["ok"] = bool(ok)
    _model_health["ts"] = time.monotonic()


def _get_scheduler():
    """Return the lazily-built scheduler used by /health."""
    global _scheduler
    if _scheduler is None:
        from src.scheduler import UnifiedScheduler
        _scheduler = UnifiedScheduler(store=store)
    return _scheduler


def _cached_health_summary(ttl: float = _HEALTH_SUMMARY_TTL_S) -> dict:
    """Return cached scheduler status and ticker freshness for /health."""
    now = time.monotonic()
    cached = _health_cache.get("value")
    if cached is not None and now - float(_health_cache.get("ts", 0.0)) < ttl:
        return cached

    scheduler_status = None
    try:
        scheduler_status = _get_scheduler().status_report()
    except Exception as e:  # noqa: BLE001
        logger.warning("Scheduler status unavailable: %s", e)

    freshness_summary: dict[str, str] = {}
    for ticker in ["NVDA", "AMD", "AAPL", "MSFT", "META", "CRWD"]:
        try:
            report = store.get_freshness_report(ticker)
            freshness_summary[ticker] = report["overall"]
        except Exception:  # noqa: BLE001
            freshness_summary[ticker] = "error"

    value = {"scheduler": scheduler_status, "freshness": freshness_summary}
    _health_cache["value"] = value
    _health_cache["ts"] = now
    return value


async def _check_model_health(ttl: float = _MODEL_HEALTH_TTL_S) -> bool:
    """Ping the llama-server to check if the model is available."""
    if not config or not model_client:
        return False
    now = time.monotonic()
    if now - float(_model_health.get("ts", 0.0)) < ttl:
        return bool(_model_health.get("ok", False))
    try:
        resp = await model_client.get(
            config.llama_endpoint.replace("/v1/chat/completions", "/health"),
            timeout=5,
        )
        ok = resp.status_code == 200
        _mark_model_health(ok)
        return ok
    except Exception:
        _mark_model_health(False)
        return False


# ── Query (Full Pipeline) ──────────────────────────────

async def _maybe_fetch_on_miss(ticker: str) -> dict:
    """Run fetch-on-miss ingestion off the event loop with a hard timeout."""
    from .on_demand import fetch_ticker_on_miss

    # Validate config BEFORE starting any work: once the to_thread task is
    # created the fetch runs (with network I/O) even if wait_for errors out.
    # Strict type check on purpose — mock/partial configs must not trigger
    # a live network fetch.
    timeout = getattr(config, "fetch_on_miss_timeout_s", None)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        logger.debug("Fetch-on-miss skipped for %s: invalid timeout config", ticker)
        return {"fetched": False, "ticker": ticker, "sources": [], "error": "invalid_config"}

    try:
        task = asyncio.create_task(asyncio.to_thread(fetch_ticker_on_miss, store, ticker))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Fetch-on-miss timed out for %s", ticker)
        return {"fetched": False, "ticker": ticker, "sources": [], "error": "timeout"}
    except Exception as e:  # noqa: BLE001 - never let fetch-on-miss crash a query
        logger.warning("Fetch-on-miss failed unexpectedly for %s: %s", ticker, e)
        return {"fetched": False, "ticker": ticker, "sources": [], "error": str(e)}


def _stage_timing(timings: dict[str, object], name: str, stage_start: float) -> float:
    """Record and return a stage duration in milliseconds."""
    elapsed = round((time.perf_counter() - stage_start) * 1000, 1)
    timings[name] = elapsed
    return elapsed


QUALITY_STAGE_TIMING_FIELDS = (
    "intent_plan_ms",
    "catalog_tools_ms",
    "dense_retrieval_ms",
    "lexical_retrieval_ms",
    "fusion_rerank_ms",
    "prompt_construction_ms",
    "model_ttft_ms",
    "model_total_ms",
    "end_to_end_ms",
)


def _available_timing(value) -> Optional[float]:
    """Normalize an observed duration; missing/unobserved stages stay unavailable."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    return round(float(value), 1)


def _quality_stage_timings(
    timings: dict[str, object], *, end_to_end_ms: Optional[float] = None
) -> dict[str, Optional[float]]:
    """Project legacy timing details onto the stable Phase 2.3.7.7 stage schema."""
    retrieval = timings.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, dict) else {}
    existing = timings.get("stages")
    existing = existing if isinstance(existing, dict) else {}

    intent_parts = [
        value for value in (timings.get("intent_parse"), timings.get("query_plan"))
        if _available_timing(value) is not None
    ]
    dense_parts = [
        value for value in (retrieval.get("embedding"), retrieval.get("chroma"))
        if _available_timing(value) is not None
    ]
    stages = {
        "intent_plan_ms": _available_timing(sum(intent_parts)) if intent_parts else None,
        "catalog_tools_ms": _available_timing(timings.get("catalog_tools")),
        "dense_retrieval_ms": _available_timing(sum(dense_parts)) if dense_parts else None,
        "lexical_retrieval_ms": _available_timing(retrieval.get("lexical")),
        "fusion_rerank_ms": _available_timing(retrieval.get("fusion_rerank")),
        "prompt_construction_ms": _available_timing(timings.get("prompt_build")),
        "model_ttft_ms": _available_timing(timings.get("model_ttft")),
        "model_total_ms": _available_timing(
            timings.get("model_total", timings.get("model_call"))
        ),
        "end_to_end_ms": _available_timing(end_to_end_ms),
    }
    for timing_field in QUALITY_STAGE_TIMING_FIELDS:
        if stages[timing_field] is None:
            stages[timing_field] = _available_timing(existing.get(timing_field))
    return stages


def _refresh_quality_stage_timings(
    timings: dict[str, object], *, end_to_end_ms: Optional[float] = None
) -> None:
    timings["stages"] = _quality_stage_timings(
        timings, end_to_end_ms=end_to_end_ms,
    )


def _return_timings_enabled() -> bool:
    """Return whether response timing metadata should be included."""
    return_timings = getattr(config, "return_timings", True)
    return return_timings if isinstance(return_timings, bool) else True


async def _build_query_context(request: QueryRequest) -> dict:
    """Run the shared query pipeline up to the augmented prompt.

    ONE request-path switch (2.2.3.4): the shared prefix (intent parse, bounded
    history selection, follow-up compilation) is identical for both endpoints,
    then a single decision selects the complete path —

    - ``enable_adaptive_rag=false`` -> the legacy ``IntentParser -> Retriever ->
      PromptAugmenter`` path (:func:`_build_legacy_query_context`);
    - ``enable_adaptive_rag=true`` -> ``QueryPlan -> deterministic route ->
      AdaptiveOrchestrator -> selected evidence -> PromptAugmenter``
      (:func:`_build_adaptive_query_context`).

    Both return the same context keys, so ``/query`` and ``/query/stream`` share
    one compiled context. If the adaptive path raises (e.g. an invalid plan), we
    record ``adaptive_fallback`` and run the legacy path with the untouched raw
    question — the query never fails because the new layer failed.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    start = time.time()
    timings: dict[str, object] = {
        "stages": {field: None for field in QUALITY_STAGE_TIMING_FIELDS},
    }
    _reset_request_scoped_state(request.answer_policy)

    compile_start = time.perf_counter()
    _emit_stage("compile", "started")
    stage_start = compile_start
    from .intent_parser import IntentParser

    parser = IntentParser()
    intent = parser.parse(request.question, override_ticker=request.ticker)
    _stage_timing(timings, "intent_parse", stage_start)

    # Bounded client-owned history selection (2.2.2.1). The middleware stays
    # stateless: it validates + selects the turns the client sent and reports
    # how many it used. No history -> single-turn behavior is unchanged and the
    # conversation metadata block is omitted.
    conversation_meta: Optional[dict] = None
    history_turns: list = []
    if request.history:
        from .conversation import select_history

        selection = select_history(
            request.history,
            max_turns=getattr(config, "conversation_max_turns", 8),
            max_chars=getattr(config, "conversation_max_history_chars", 8000),
        )
        conversation_meta = selection.as_metadata()
        history_turns = selection.turns

    # Follow-up rewriting & entity carryover (2.2.2.2). Behind a config flag and
    # only when history is present. Compiles a SEPARATE retrieval query and an
    # effective retrieval intent (carried entity/metric/timeframe) while leaving
    # request.question untouched. Off (or no history) -> retrieval_intent is the
    # raw intent and retrieval_query is the raw question, so nothing changes.
    compiled = None
    retrieval_query = request.question
    retrieval_intent = intent
    if history_turns and getattr(config, "enable_conversation_rewrite", False):
        from .conversation import compile_question

        compiled = compile_question(
            request.question, history_turns, override_ticker=request.ticker,
        )
        from .query_rewriter import should_use_llm_fallback

        if should_use_llm_fallback(compiled, config):
            from .query_rewriter import rewrite_query

            compiled = await asyncio.to_thread(
                rewrite_query, request.question, history_turns, compiled, config=config,
            )
        retrieval_query = compiled.retrieval_query or request.question
        retrieval_intent = _effective_intent(intent, compiled)
        if conversation_meta is not None:
            conversation_meta["topic_reset"] = bool(compiled.topic_reset)

    _emit_stage("compile", "completed",
                elapsed_ms=(time.perf_counter() - compile_start) * 1000)

    shared = {
        "request": request,
        "start": start,
        "timings": timings,
        "parser": parser,
        "intent": intent,
        "conversation_meta": conversation_meta,
        "history_turns": history_turns,
        "compiled": compiled,
        "retrieval_query": retrieval_query,
        "retrieval_intent": retrieval_intent,
    }

    if bool(getattr(config, "enable_adaptive_rag", False)):
        try:
            return await _build_adaptive_query_context(shared)
        except Exception:  # noqa: BLE001 - adaptive layer must never fail a query
            logger.exception("Adaptive context build failed; using legacy path")
            _emit_stage("route", "fallback", reason="adaptive_fallback")
            return await _build_legacy_query_context(
                shared, orchestration={"lane": None, "fallback_reason": "adaptive_fallback"}
            )
    return await _build_legacy_query_context(shared)


async def _freshness_stage(
    intent_like: dict, do_refresh: bool, timings: dict[str, object]
) -> dict:
    """Evaluate + (optionally) refresh freshness for one entity, with fetch-on-miss.

    Shared by the legacy and adaptive context builders so both apply the exact
    same freshness/auto-refresh/fetch-on-miss policy to the request's primary
    entity. ``intent_like`` only needs ``ticker`` and ``ticker_confidence`` keys.
    """
    stage_start = time.perf_counter()
    freshness_meta = _evaluate_and_refresh(intent_like.get("ticker"), do_refresh)
    ticker = intent_like.get("ticker")
    if (
        getattr(config, "enable_fetch_on_miss", True)
        and ticker
        and freshness_meta.get("overall") == "never_fetched"
        and intent_like.get("ticker_confidence", 0.0) >= FETCH_ON_MISS_MIN_CONFIDENCE
    ):
        res = await _maybe_fetch_on_miss(ticker)
        if res.get("fetched"):
            freshness_meta.setdefault("fetched_on_miss", []).append(ticker)
            freshness_meta["overall"] = "fresh"
        else:
            warning = f"Couldn't fetch live data for {ticker} right now."
            if res.get("error"):
                warning = f"{warning} Reason: {res['error']}"
            freshness_meta["warning"] = warning
    _stage_timing(timings, "freshness_check", stage_start)
    return freshness_meta


async def _build_legacy_query_context(
    shared: dict, *, orchestration: Optional[dict] = None
) -> dict:
    """The legacy ``IntentParser -> Retriever -> PromptAugmenter`` path.

    Behavior is byte-for-byte identical to the pre-2.2.3.4 pipeline. ``orchestration``
    is ``None`` on the pure legacy path and carries an ``adaptive_fallback``
    marker only when the adaptive path failed and demoted here.
    """
    request: QueryRequest = shared["request"]
    timings = shared["timings"]
    retrieval_query = shared["retrieval_query"]
    retrieval_intent = shared["retrieval_intent"]
    _emit_graph_legacy_intent(
        retrieval_intent, fallback_reason=(orchestration or {}).get("fallback_reason")
    )

    freshness_meta = await _freshness_stage(retrieval_intent, request.refresh, timings)
    write_requested = bool(freshness_meta.pop("_write_requested", False))

    stage_start = time.perf_counter()
    _emit_stage("retrieve", "started")
    from .retriever import Retriever

    r = retriever or Retriever(store=store, config=config)
    retrieval = r.retrieve(
        query=retrieval_query,
        intent=retrieval_intent,
        top_k_documents=config.top_k_documents,
        top_k_facts=config.top_k_facts,
    )
    retrieval_ms = _stage_timing(timings, "retrieval", stage_start)
    _emit_stage("retrieve", "completed", elapsed_ms=retrieval_ms)
    retrieval_timings = retrieval.get("timings", {}) if isinstance(retrieval, dict) else {}
    timings["retrieval"] = {
        "total": retrieval_ms,
        "embedding": _available_timing(retrieval_timings.get("embedding")),
        "chroma": _available_timing(retrieval_timings.get("chroma")),
        "sqlite": _available_timing(retrieval_timings.get("sqlite")),
        "lexical": _available_timing(retrieval_timings.get("lexical")),
        "fusion_rerank": _available_timing(retrieval_timings.get("fusion_rerank")),
    }
    grounding_level = _grounding_level(retrieval)

    # Request-local [E#] evidence ledger (2.2.4.3). Empty when validation is
    # off, so the prompt and response stay byte-for-byte legacy.
    evidence_ledger = _build_evidence_ledger(retrieval) if _evidence_ids_enabled() else []
    graph_evidence_ids = _emit_graph_evidence(retrieval, evidence_ledger)

    # Request-scoped evidence-trace collector (2.2.1.2), created right after
    # evidence normalization so facts/documents are the exact usable rows
    # (see src/middleware/evidence.py) — untruncated, full provenance. Left
    # None (and thus omitted from the response) unless explicitly requested.
    trace_collector: Optional[EvidenceTraceCollector] = None
    if request.include_evidence_trace:
        trace_collector = EvidenceTraceCollector(
            answer_policy=_answer_policy(),
            grounding_level=grounding_level,
            raw_question=request.question,
            retrieval_query=retrieval_query,
            facts=usable_facts(retrieval),
            documents=usable_documents(retrieval),
        )
        trace_collector.record_evidence_ledger(evidence_ledger)
    _evidence_trace_var.set(trace_collector)

    stage_start = time.perf_counter()
    _emit_stage("pack", "started")
    from .prompt_augmenter import PromptAugmenter

    # Prompt/answer use the RAW question and the effective (carried) intent;
    # only retrieval used the compiled query.
    augmenter = PromptAugmenter(config=config)
    augmented_prompt = augmenter.build_prompt(
        question=request.question,
        intent=retrieval_intent,
        retrieval=retrieval,
        grounding_level=grounding_level,
        evidence_ledger=evidence_ledger or None,
    )
    timings["prompt"] = _prompt_efficiency_metrics(
        augmented_prompt=augmented_prompt, intent=retrieval_intent,
        grounding_level=grounding_level,
        facts=usable_facts(retrieval), documents=usable_documents(retrieval))
    _emit_stage("pack", "completed",
                elapsed_ms=_stage_timing(timings, "prompt_build", stage_start))

    return {
        "start": shared["start"],
        "timings": timings,
        "intent": retrieval_intent,
        "freshness": freshness_meta,
        "retrieval": retrieval,
        "grounding_level": grounding_level,
        "augmented_prompt": augmented_prompt,
        "include_evidence_trace": request.include_evidence_trace,
        "conversation": shared["conversation_meta"],
        "history_turns": shared["history_turns"],
        "compiled": shared["compiled"],
        "retrieval_query": retrieval_query,
        "orchestration": orchestration,
        "evidence_ledger": evidence_ledger,
        "graph_evidence_ids": graph_evidence_ids,
        "calculations": [],
        "_write_requested": write_requested,
    }


def _adaptive_available_metrics() -> tuple:
    """The metric names present in the store, for the deterministic router.

    Fails soft to an empty tuple (router then abstains on any named metric) so a
    store without ``list_metrics`` never breaks the adaptive path.
    """
    try:
        return tuple(store.sqlite.list_metrics())
    except Exception:  # noqa: BLE001 - metric catalog is best-effort
        return ()


def _get_retrieval_cache():
    """Return the process-global versioned retrieval cache, or None when off.

    Built lazily from config the first time it is needed. Behind
    ``enable_retrieval_cache`` (off by default) so the default path never
    allocates it. See src/middleware/retrieval_cache.py for the invalidation
    contract (store revision + config/model fingerprint + TTL).
    """
    global _retrieval_cache
    if not bool(getattr(config, "enable_retrieval_cache", False)):
        return None
    if _retrieval_cache is None:
        from .retrieval_cache import RetrievalCache, config_fingerprint

        _retrieval_cache = RetrievalCache(
            max_entries=int(getattr(config, "retrieval_cache_max_entries", 256)),
            ttl_s=float(getattr(config, "retrieval_cache_ttl_s", 300.0)),
            max_value_chars=int(getattr(config, "retrieval_cache_max_value_chars", 200_000)),
            fingerprint=config_fingerprint(config),
        )
    return _retrieval_cache


def _prompt_efficiency_metrics(
    *, augmented_prompt: str, intent: Optional[dict], grounding_level: str,
    facts: list, documents: list,
) -> dict:
    """Record prompt-prefix efficiency telemetry (2.2.6.2 Step 3).

    Captures the augmented-prompt characters + estimated tokens, the rendered
    evidence characters, and a short digest of the FIXED system-policy prefix (the
    reusable prompt prefix). Pure measurement — it never changes the prompt.
    """
    from .prompt_augmenter import PromptAugmenter

    system_prompt = _system_prompt_for_request(
        intent=intent, grounding_level=grounding_level,
        tools_enabled=bool(getattr(config, "enable_tools", False)),
    )
    prompt_chars = len(augmented_prompt)
    return {
        "prompt_chars": prompt_chars,
        "estimated_tokens": prompt_chars // 4,
        "evidence_chars": PromptAugmenter.evidence_char_count(facts, documents),
        "fixed_prefix_digest": prompt_policy.fixed_prefix_digest(system_prompt),
    }


def _tool_result_count(result) -> Optional[int]:
    """Best-effort row/item count from a tool result for a progress event.

    Reads only the shape/length of the standard read-tool result envelopes
    (``results`` list, ``fundamentals``/``estimates``/``price_targets`` maps) —
    never the values themselves. Returns None when no count is meaningful, so a
    count is emitted only when it genuinely reflects rows/items retrieved.
    """
    from .tools.base import tool_result_count

    return tool_result_count(result)


def _emit_adaptive_progress(result) -> None:
    """Emit grade/correct stages and deterministic-tool events from a result.

    Called after orchestration completes so the progress stream reflects the
    ACTUAL bounded work done (grader verdict, one corrective round if it ran, and
    each deterministic read tool the router dispatched). Redaction is inherent:
    only stage names/phases, safe tool names, statuses, and row/item counts are
    emitted — never plan text, arguments, or tool result bodies. No-op when no
    emitter is installed.
    """
    if _stream_emitter_var.get() is None:
        return
    # Deterministic tool invocations (skip model tool-planning on the stream path).
    if result.tool_execution is not None:
        for inv in result.tool_execution.invocations:
            if not inv.name:
                continue
            _emit_tool_started(inv.name, subquery_id=inv.subquery_id)
            status = "error" if inv.error else "ok"
            _emit_tool_completed(
                inv.name, status,
                count=_tool_result_count(inv.result),
                subquery_id=inv.subquery_id,
            )
    # Grader verdict (once), then the one corrective round if it actually ran.
    if result.sufficiency is not None:
        status = getattr(result.sufficiency.status, "value", str(result.sufficiency.status))
        _emit_stage("grade", "completed", reason=str(status))
    if getattr(result, "retry_performed", False):
        action = getattr(result.corrective_action, "value", str(result.corrective_action))
        _emit_stage("correct", "completed", reason=str(action))


def _orchestration_metadata(result) -> dict:
    """Map an OrchestrationResult to the response ``orchestration`` block.

    Every counter is the ACTUAL executed value (not a configured maximum), so a
    client/eval can see exactly how bounded the request stayed (2.2.3.4 Step 2).
    """
    ctx = result.context
    tools = []
    if result.tool_execution is not None:
        tools = [inv.name for inv in result.tool_execution.invocations if inv.name]
    dropped = 0
    if ctx is not None:
        dropped = int(ctx.dropped_facts) + int(ctx.dropped_documents)
    return {
        "lane": result.lane.value,
        "reason_codes": list(result.reason_codes),
        "subqueries_executed": len(result.subqueries_executed),
        "retrieval_rounds": int(result.retrieval_rounds_used),
        "planning_calls": 1 if result.planning_ran else 0,
        "reranker_calls": 1 if result.rerank_ran else 0,
        "deterministic_tools": tools,
        "set_complete": result.set_complete,
        "result_set_size": result.result_set_size,
        "model_calls": 0,
        "context_chars": int(result.context_size),
        "evidence_dropped": dropped,
        "fallback_reason": result.fallback_reason,
        "query_plan": _plan_trace(result.plan),
    }


def _sufficiency_metadata(result, answer_mode: str) -> Optional[dict]:
    """Build optional response metadata from the same graded result."""
    if result.sufficiency is None:
        return None
    metadata = result.sufficiency.to_metadata()
    metadata["status"] = answer_mode
    metadata.pop("sufficiency", None)
    metadata["corrective_action"] = result.corrective_action.value
    metadata["retry_performed"] = bool(result.retry_performed)
    return metadata


def _fact_trace_id(fact: dict) -> dict:
    return {
        "ticker": fact.get("ticker"),
        "metric": fact.get("metric"),
        "period": fact.get("period"),
    }


def _doc_trace_id(doc: dict):
    meta = doc.get("metadata") or {}
    return doc.get("id") or meta.get("id") or meta.get("parent_id")


def _plan_trace(plan) -> dict:
    obligations = plan.obligations
    return {
        "retrieval_query": plan.retrieval_query,
        "entities": list(plan.tickers),
        "intents": list(plan.intents),
        "metrics": list(plan.metrics),
        "periods": list(plan.periods),
        "primary_intent": plan.primary_intent,
        "subqueries": [sq.id for sq in plan.subqueries],
        "reason_codes": list(plan.reason_codes),
        "obligations": {
            "entity_set": list(obligations.entity_set),
            "universe_scope": obligations.universe_scope,
            "operation": obligations.operation,
            "metrics": list(obligations.metrics),
            "item_types": list(obligations.item_types),
            "sources": list(obligations.sources),
            "completeness": obligations.completeness,
            "limit": obligations.limit,
            "as_of": obligations.as_of,
            "qualitative": obligations.qualitative,
            "evidence_modes": list(obligations.evidence_modes),
        },
    }


def _evidence_trace_orchestration(plan, result) -> dict:
    """The exact adaptive route trace (2.2.3.4 Step 3) for the evidence trace.

    Records only actions that actually occurred on the successful answer path:
    the validated plan, the selected lane + reason codes, executed
    subqueries/rounds, deterministic tool + calculation results, the re-rank
    decision + fallback reason, and the final selected/dropped evidence ids
    under the context budget. A rejected planner attempt is a reason code only —
    its output is never mixed into the answer evidence.
    """
    ctx = result.context
    tool_results: list[dict] = []
    calculations: list[dict] = []
    if result.tool_execution is not None:
        for inv in result.tool_execution.invocations:
            tool_results.append({
                "name": inv.name,
                "arguments": dict(inv.arguments or {}),
                "reason_code": inv.reason_code,
                "subquery_id": inv.subquery_id,
                "result": inv.result,
                "error": inv.error,
            })
        calculations = [dict(c) for c in result.tool_execution.calculations]

    meta = _orchestration_metadata(result)
    meta.update({
        "query_plan": _plan_trace(plan),
        "deterministic_tool_results": tool_results,
        "calculations": calculations,
        "subquery_ids": list(result.subqueries_executed),
        "rerank_decision": {
            "ran": bool(result.rerank_ran),
            "reason_codes": [c for c in result.reason_codes if c.startswith("rerank")],
        },
        "selected_evidence": {
            "fact_ids": [_fact_trace_id(f) for f in (ctx.facts if ctx else [])],
            "document_ids": [_doc_trace_id(d) for d in (ctx.documents if ctx else [])],
        },
        "dropped_evidence": {
            "facts": int(ctx.dropped_facts) if ctx else 0,
            "documents": int(ctx.dropped_documents) if ctx else 0,
        },
        "context_budget": {
            "context_chars": int(ctx.context_chars) if ctx else 0,
            "estimated_tokens": int(ctx.estimated_tokens) if ctx else 0,
            "truncated": bool(ctx.truncated) if ctx else False,
        },
    })
    return meta


async def _build_adaptive_query_context(shared: dict) -> dict:
    """The adaptive ``QueryPlan -> route -> orchestrate -> PromptAugmenter`` path.

    Builds one validated multi-entity plan from the compiled retrieval query,
    runs the bounded orchestrator (deterministic route + lane selection + one
    execution budget + one context budget + conditional re-rank), and grounds
    the prompt with the pre-budgeted evidence. Raises only if the plan cannot be
    built or validated; :func:`_build_query_context` then demotes to the legacy
    path with the untouched raw question.
    """
    request: QueryRequest = shared["request"]
    timings = shared["timings"]
    parser = shared["parser"]
    retrieval_query = shared["retrieval_query"]

    # Build + validate the plan from the compiled standalone query (2.2.2). A
    # QueryPlanError propagates to the caller's legacy fallback.
    stage_start = time.perf_counter()
    _emit_stage("route", "started")
    plan = parser.parse_plan(
        request.question,
        retrieval_query=(retrieval_query if retrieval_query != request.question else None),
        override_ticker=request.ticker,
    )
    _emit_graph_plan(plan)
    _emit_stage("route", "completed",
                elapsed_ms=_stage_timing(timings, "query_plan", stage_start))

    intent = plan.to_legacy_intent()
    freshness_meta = await _freshness_stage(intent, request.refresh, timings)
    write_requested = bool(freshness_meta.pop("_write_requested", False))

    stage_start = time.perf_counter()
    _emit_stage("retrieve", "started")
    from .adaptive_orchestrator import orchestrate
    from .retriever import Retriever

    r = retriever or Retriever(store=store, config=config)
    from . import deterministic_router

    def timed_execute_route(*args, **kwargs):
        tool_started = time.perf_counter()
        try:
            return deterministic_router.execute_route(*args, **kwargs)
        finally:
            timings["catalog_tools"] = float(timings.get("catalog_tools") or 0.0) + (
                time.perf_counter() - tool_started
            ) * 1000

    result = orchestrate(
        plan,
        store,
        config,
        retriever=r,
        available_metrics=_adaptive_available_metrics(),
        execute_fn=timed_execute_route,
        retrieval_cache=_get_retrieval_cache(),
        refresh=bool(getattr(request, "refresh", False)),
    )
    _emit_stage("retrieve", "completed",
                elapsed_ms=_stage_timing(timings, "orchestration", stage_start))
    _emit_graph_plan(result.plan, lane=result.lane.value)
    _emit_graph_adaptive_result(result)

    # Build a retrieve()-compatible view from the single budgeted context so
    # grounding, evidence counts, degraded answers, and the response builder all
    # operate on exactly the evidence the model will see.
    ctx_sel = result.context
    sel_facts = list(ctx_sel.facts) if ctx_sel is not None else list(result.merged_facts)
    sel_docs = list(ctx_sel.documents) if ctx_sel is not None else list(result.merged_documents)
    retrieval = {
        "facts": sel_facts,
        "documents": sel_docs,
        "ticker": intent.get("ticker"),
        "strategy": result.lane.value,
        "retrieval_strategy": result.retrieval_strategy or "vector",
        "timings": {},
    }
    retrieval_timings = (
        getattr(r, "_timings", {})
        if int(result.retrieval_rounds_used or 0) > 0
        else {}
    )
    retrieval_timings = retrieval_timings if isinstance(retrieval_timings, dict) else {}
    timings["retrieval"] = {
        "total": _available_timing(timings.get("orchestration")),
        "embedding": _available_timing(retrieval_timings.get("embedding")),
        "chroma": _available_timing(retrieval_timings.get("chroma")),
        "sqlite": _available_timing(retrieval_timings.get("sqlite")),
        "lexical": _available_timing(retrieval_timings.get("lexical")),
        "fusion_rerank": _available_timing(retrieval_timings.get("fusion_rerank")),
    }
    if result.sufficiency is not None:
        specific = bool(plan.entities) or bool(plan.metrics) or bool(
            set(plan.intents) & {"fact_lookup", "comparison", "trend", "projection"}
        )
        grounding_level = _answer_mode_from_sufficiency(
            result.sufficiency, requires_specific_figures=specific)
    else:
        grounding_level = _grounding_level(retrieval)
    sufficiency_meta = _sufficiency_metadata(result, grounding_level)

    # Recorded deterministic calculations feed numeric validation (2.2.4.3):
    # a cited figure produced by a bounded calculation is supported evidence.
    calculations: list[dict] = []
    if result.tool_execution is not None:
        calculations = [dict(c) for c in result.tool_execution.calculations]

    # Request-local [E#] evidence ledger over the pre-budgeted evidence.
    deterministic_evidence = bool(
        getattr(config, "enable_deterministic_answers", False)
        and result.tool_execution is not None
    )
    evidence_ledger = (
        _build_evidence_ledger(retrieval)
        if _evidence_ids_enabled() or deterministic_evidence
        else []
    )
    graph_evidence_ids = _emit_graph_evidence(retrieval, evidence_ledger)
    _emit_graph_dropped_evidence(result, retrieval)

    # Evidence-trace collector (opt-in). The adaptive route trace records only
    # the successful answer path (2.2.3.4 Step 3).
    trace_collector: Optional[EvidenceTraceCollector] = None
    if request.include_evidence_trace:
        trace_collector = EvidenceTraceCollector(
            answer_policy=_answer_policy(),
            grounding_level=grounding_level,
            raw_question=request.question,
            retrieval_query=retrieval_query,
            facts=usable_facts(retrieval),
            documents=usable_documents(retrieval),
        )
        trace_collector.record_orchestration(_evidence_trace_orchestration(plan, result))
        trace_collector.record_evidence_ledger(evidence_ledger)
    _evidence_trace_var.set(trace_collector)

    stage_start = time.perf_counter()
    _emit_stage("pack", "started")
    from .prompt_augmenter import PromptAugmenter

    # Prompt uses the RAW question, the legacy-projected plan intent, and the
    # pre-budgeted evidence as the single budget owner (no re-filtering).
    augmenter = PromptAugmenter(config=config)
    augmented_prompt = augmenter.build_prompt(
        question=request.question,
        intent=intent,
        retrieval={},
        grounding_level=grounding_level,
        preselected={"facts": sel_facts, "documents": sel_docs},
        evidence_sufficiency=result.sufficiency,
        evidence_ledger=evidence_ledger or None,
    )
    timings["prompt"] = _prompt_efficiency_metrics(
        augmented_prompt=augmented_prompt, intent=intent,
        grounding_level=grounding_level, facts=sel_facts, documents=sel_docs)
    _emit_stage("pack", "completed",
                elapsed_ms=_stage_timing(timings, "prompt_build", stage_start))

    return {
        "start": shared["start"],
        "timings": timings,
        "intent": intent,
        "freshness": freshness_meta,
        "retrieval": retrieval,
        "grounding_level": grounding_level,
        "augmented_prompt": augmented_prompt,
        "include_evidence_trace": request.include_evidence_trace,
        "conversation": shared["conversation_meta"],
        "history_turns": shared["history_turns"],
        "compiled": shared["compiled"],
        "retrieval_query": retrieval_query,
        "orchestration": _orchestration_metadata(result),
        "deterministic_answer": result.deterministic_answer,
        "answer_origin": None,
        "coverage_metadata": result.answer_metadata,
        "evidence_sufficiency": sufficiency_meta,
        "evidence_ledger": evidence_ledger,
        "graph_evidence_ids": graph_evidence_ids,
        "calculations": calculations,
        "_orchestration_result": result,
        "_write_requested": write_requested,
    }


def _effective_intent(intent: dict, compiled) -> dict:
    """Overlay carried/resolved slots onto the parsed intent for retrieval.

    The raw question's intent is preserved except where follow-up compilation
    (2.2.2.2) resolved an entity, metric, or timeframe the raw turn lacked. The
    carried ticker is marked ``ticker_source="carryover"`` so the response's
    resolved-ticker field (which only fires for non-exact name lookups) is not
    spuriously populated for it.
    """
    eff = dict(intent)
    entity = getattr(compiled, "entity", None)
    if entity and entity != eff.get("ticker"):
        eff["ticker"] = entity
        if entity in getattr(compiled, "carried_entities", []):
            eff["ticker_source"] = "carryover"
            eff["resolved_name"] = None
            eff["ticker_confidence"] = 1.0
    metrics = getattr(compiled, "metrics", None)
    if metrics:
        eff["metrics"] = list(metrics)
    timeframe = getattr(compiled, "timeframe", None)
    if timeframe and not eff.get("timeframe"):
        eff["timeframe"] = timeframe
    return eff


def _task_settings(request: QueryRequest, intent: dict) -> tuple[float, int]:
    """Return temperature and max_tokens for this request."""
    task = _task_params("analysis") if intent.get("question_type") == "projection" else {}
    temperature = request.temperature or task.get("temperature") or config.default_temperature
    max_tokens = request.max_tokens or task.get("max_tokens") or config.max_tokens
    return temperature, max_tokens


def _evidence_citation_model(record) -> EvidenceCitation:
    """Map an answer_validator CitationRecord onto the response model."""
    data = record.graph_reference()
    data["source_url"] = record.source_url
    return EvidenceCitation(**data)


def _enforce_validation(answer_text: str, grounding_level: str, report) -> tuple[str, str, str]:
    """Apply the enforce policy (2.2.4.3). Returns (answer, grounding, action).

    A wholly-unsupported answer is replaced with the existing honest refusal;
    otherwise a violation (unsupported number or unresolved citation) downgrades
    grounded->partial and appends a short support warning. No model call is made.
    """
    require_ids = bool(getattr(config, "require_evidence_ids", False))
    if report.wholly_unsupported():
        return NO_GENERAL_FALLBACK_MESSAGE, "refused", "refuse"
    if not report.has_violations(require_evidence_ids=require_ids):
        return answer_text, grounding_level, "none"
    if grounding_level == "grounded":
        grounding_level = "partial"
    warning = report.support_warning()
    if warning and warning not in answer_text:
        answer_text = f"{answer_text}\n\n{warning}"
    return answer_text, grounding_level, "downgrade"


def _apply_answer_validation(
    context: dict, answer_text: str, grounding_level: str,
) -> tuple[str, str, Optional[dict], Optional[list]]:
    """Run deterministic citation/numeric validation per policy (2.2.4.3).

    Returns ``(answer_text, grounding_level, validation_meta, evidence_citations)``.
    off -> all-None passthrough. report -> metadata + log, answer unchanged.
    enforce -> may downgrade/refuse with a support warning. The validator never
    makes a model call and always fails soft to ``report_unavailable``.
    """
    mode = _answer_validation_mode()
    if mode == "off":
        return answer_text, grounding_level, None, None

    ledger = context.get("evidence_ledger") or []
    calculations = context.get("calculations") or []
    try:
        from .answer_validator import AnswerValidation, validate_answer

        report = validate_answer(
            answer_text, ledger, calculations=calculations,
            require_evidence_ids=bool(getattr(config, "require_evidence_ids", False)),
        )
    except Exception:  # noqa: BLE001 - validation must never crash a query
        logger.exception("Answer validation failed; reporting report_unavailable")
        from .answer_validator import AnswerValidation

        meta = AnswerValidation.unavailable().to_metadata()
        meta["enforcement"] = "none"
        return answer_text, grounding_level, meta, None

    meta = report.to_metadata()
    evidence_citations = [_evidence_citation_model(c) for c in report.citations]
    action = "none"
    if mode == "enforce":
        answer_text, grounding_level, action = _enforce_validation(
            answer_text, grounding_level, report)
    elif report.numeric_claims_unsupported or report.citations_missing or report.citations_malformed:
        logger.info(
            "answer_validation report: status=%s unsupported=%d missing=%d malformed=%d",
            report.status, report.numeric_claims_unsupported,
            report.citations_missing, report.citations_malformed)
    meta["enforcement"] = action
    return answer_text, grounding_level, meta, evidence_citations


def _build_query_response(
    *,
    context: dict,
    answer_text: str,
    citations: list[SourceCitation],
    model_available: bool,
    validation: Optional[dict] = None,
    evidence_citations: Optional[list] = None,
) -> QueryResponse:
    """Build a QueryResponse from shared query context and model output."""
    intent = context["intent"]
    retrieval = context["retrieval"]
    elapsed_ms = round((time.time() - context["start"]) * 1000, 1)
    _refresh_quality_stage_timings(context["timings"], end_to_end_ms=elapsed_ms)
    # Usable-evidence counts (2.2.1.1) — shared by /query and the
    # /query/stream terminal metadata event since both call this function.
    n_facts, n_docs = evidence_counts(retrieval)

    # Evidence trace (2.2.1.2) — opt-in, and only ever non-None when a model
    # call actually recorded a prompt (never for the degraded path).
    evidence_trace = None
    if context.get("include_evidence_trace"):
        collector = _evidence_trace_var.get()
        trace = collector.finalize() if collector is not None else None
        if trace is not None:
            evidence_trace = trace.to_dict()

    # Follow-up rewriting metadata (2.2.2.2) — only present when compilation ran
    # (flag on + history). Exposes the compiled query and the carried slots so a
    # client can show what carried and the eval harness can record explicit
    # resolved values.
    compiled = context.get("compiled")
    retrieval_query = None
    carried_context = None
    resolved_tickers = None
    resolved_metrics = None
    resolved_timeframe = None
    if compiled is not None:
        retrieval_query = context.get("retrieval_query")
        carried_context = compiled.as_metadata()
        resolved_tickers = [compiled.entity] if compiled.entity else list(compiled.carried_entities)
        resolved_metrics = list(compiled.metrics)
        resolved_timeframe = compiled.timeframe

    # Optional local query-graph trace id (2.2.7.4). Non-None only when the
    # observer is enabled and this request installed an emitter; excluded from
    # the serialized response otherwise so behavior stays byte-compatible.
    graph_trace_id = None
    if _graph_observer_enabled():
        emitter = _stream_emitter_var.get()
        if emitter is not None:
            graph_trace_id = emitter.query_id

    return QueryResponse(
        answer=answer_text,
        answer_origin=context.get("answer_origin"),
        generation_skipped=context.get("generation_skipped"),
        generation_skip_reason=context.get("generation_skip_reason"),
        coverage_metadata=context.get("coverage_metadata"),
        citations=citations,
        detected_ticker=intent.get("ticker"),
        detected_intent=intent.get("question_type"),
        facts_used=n_facts,
        documents_used=n_docs,
        grounding=_response_grounding(answer_text, context["grounding_level"]),
        latency_ms=elapsed_ms,
        timings=context["timings"] if _return_timings_enabled() else None,
        model_available=model_available,
        evidence_trace=evidence_trace,
        conversation=context.get("conversation"),
        freshness=context["freshness"],
        retrieval_strategy=retrieval.get("retrieval_strategy"),
        tools_used=_get_tools_used(),
        resolved_ticker=_resolved_ticker_field(intent),
        retrieval_query=retrieval_query,
        carried_context=carried_context,
        resolved_tickers=resolved_tickers,
        resolved_metrics=resolved_metrics,
        resolved_timeframe=resolved_timeframe,
        orchestration=context.get("orchestration"),
        evidence_sufficiency=context.get("evidence_sufficiency"),
        evidence_citations=evidence_citations,
        answer_validation=validation,
        graph_trace_id=graph_trace_id,
    )


def _source_citations_from_validation(report) -> list[SourceCitation]:
    """Project supported evidence citations onto the legacy source list."""
    citations: list[SourceCitation] = []
    seen: set[tuple] = set()
    for record in report.citations:
        if record.support_status != "supported" or not record.ticker:
            continue
        key = (
            record.source_type or "tool", record.ticker, record.metric,
            record.period, record.source_url,
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(SourceCitation(
            source_type=record.source_type or "tool",
            ticker=record.ticker,
            metric=record.metric,
            period=record.period,
            source_url=record.source_url,
        ))
    return citations


def _discard_deterministic_trace() -> None:
    collector = _evidence_trace_var.get()
    if collector is not None and hasattr(collector, "discard_deterministic_answer"):
        collector.discard_deterministic_answer()


def _try_deterministic_response(context: dict) -> Optional[QueryResponse]:
    """Validate and finalize the shared deterministic fast path, or fall back.

    The entire render/validate/response/graph sequence is one fail-soft
    boundary. Returning ``None`` means the caller must run normal generation
    exactly once.
    """
    if not bool(getattr(config, "enable_deterministic_answers", False)):
        return None
    result = context.get("_orchestration_result")
    if result is None:
        return None

    from . import deterministic_answers

    contract = deterministic_answers.check_final_answer_contract(
        result,
        evidence_ledger=context.get("evidence_ledger") or [],
        freshness=context.get("freshness"),
        write_requested=bool(context.get("_write_requested")),
    )
    if not contract.eligible:
        return None

    started_at = time.perf_counter()
    try:
        rendered = deterministic_answers.render_deterministic_answer(
            result, evidence_ledger=context.get("evidence_ledger") or [],
        )
        if rendered.template is not contract.template:
            raise deterministic_answers.DeterministicAnswerError(
                "contract and renderer selected different templates"
            )

        from .answer_validator import validate_deterministic_answer

        validation_values = [
            *(context.get("calculations") or []),
            *rendered.validation_values,
        ]
        report = validate_deterministic_answer(
            rendered.text,
            context.get("evidence_ledger") or [],
            calculations=validation_values,
        )
        validation = report.to_metadata()
        validation["enforcement"] = "deterministic"
        evidence_citations = [_evidence_citation_model(c) for c in report.citations]
        citations = _source_citations_from_validation(report)

        context["answer_origin"] = "deterministic"
        context["generation_skipped"] = True
        context["generation_skip_reason"] = deterministic_answers.GENERATION_SKIP_REASON
        context["grounding_level"] = "grounded"
        if isinstance(context.get("orchestration"), dict):
            context["orchestration"]["model_calls"] = 0

        collector = _evidence_trace_var.get()
        if collector is not None:
            collector.record_deterministic_answer(rendered.template.value)

        context["timings"]["model_call"] = None
        context["timings"]["deterministic_answer"] = round(
            (time.perf_counter() - started_at) * 1000, 1
        )
        _emit_stage(
            "generate", "skipped",
            reason=deterministic_answers.GENERATION_SKIP_REASON,
        )
        response = _build_query_response(
            context=context,
            answer_text=rendered.text,
            citations=citations,
            model_available=True,
            validation=validation,
            evidence_citations=evidence_citations,
        )
        _emit_graph_terminal(
            context,
            model_available=True,
            evidence_citations=evidence_citations,
            validation=validation,
            citations=citations,
        )
        return response
    except Exception:  # noqa: BLE001 - the normal model path is the safety net
        logger.exception("Deterministic final-answer path failed; using model generation")
        _discard_deterministic_trace()
        context["answer_origin"] = None
        context["generation_skipped"] = None
        context["generation_skip_reason"] = None
        _emit_stage("generate", "fallback", reason="deterministic_fast_path_error")
        return None


def _deterministic_contract_eligible(context: dict) -> bool:
    """Cheap streaming preflight used to avoid probing the model unnecessarily."""
    if not bool(getattr(config, "enable_deterministic_answers", False)):
        return False
    result = context.get("_orchestration_result")
    if result is None:
        return False
    from .deterministic_answers import check_final_answer_contract

    return check_final_answer_contract(
        result,
        evidence_ledger=context.get("evidence_ledger") or [],
        freshness=context.get("freshness"),
        write_requested=bool(context.get("_write_requested")),
    ).eligible


async def _answer_query_context(request: QueryRequest, context: dict) -> QueryResponse:
    """Complete a prepared query context through the non-streaming model path."""
    intent = context["intent"]
    retrieval = context["retrieval"]
    grounding_level = context["grounding_level"]
    deterministic = _try_deterministic_response(context)
    if deterministic is not None:
        return deterministic

    stage_start = time.perf_counter()
    _emit_stage("generate", "started")
    model_available = await _check_model_health()
    if model_available:
        if (
            isinstance(context.get("orchestration"), dict)
            and "model_calls" in context["orchestration"]
        ):
            context["orchestration"]["model_calls"] = 1
        temperature, max_tokens = _task_settings(request, intent)
        model_started = time.perf_counter()
        try:
            answer_text, citations = await _invoke_model(
                prompt=context["augmented_prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
                intent=intent,
                grounding_level=grounding_level,
            )
        finally:
            context["timings"]["model_total"] = (
                time.perf_counter() - model_started
            ) * 1000
        if not answer_text.startswith("Error calling model:"):
            _mark_model_health(True)
        if intent.get("question_type") == "projection":
            from .guardrails import apply_projection_guardrail

            answer_text, _flagged = apply_projection_guardrail(
                answer_text,
                context["augmented_prompt"],
            )
        if bool(getattr(config, "enable_deterministic_answers", False)):
            context["answer_origin"] = "model"
            context["generation_skipped"] = False
    else:
        logger.warning("Model unavailable - returning degraded answer")
        _emit_stage("generate", "fallback", reason="model_unavailable")
        answer_text = _format_degraded_answer(
            retrieval, intent, context.get("evidence_sufficiency"))
        citations = []
        if bool(getattr(config, "enable_deterministic_answers", False)):
            context["answer_origin"] = "degraded"
            context["generation_skipped"] = False

    # Deterministic citation/numeric validation (2.2.4.3). off -> passthrough;
    # report -> metadata only; enforce -> may downgrade/refuse. Updates the
    # context grounding so _build_query_response reflects an enforced downgrade.
    answer_text, grounding_level, validation, evidence_citations = _apply_answer_validation(
        context, answer_text, grounding_level)

    _stage_timing(context["timings"], "model_call", stage_start)
    if model_available:
        _emit_stage("generate", "completed")

    context["grounding_level"] = grounding_level
    _emit_graph_terminal(
        context,
        model_available=model_available,
        evidence_citations=evidence_citations,
        validation=validation,
        citations=citations,
    )

    return _build_query_response(
        context=context,
        answer_text=answer_text,
        citations=citations,
        model_available=model_available,
        validation=validation,
        evidence_citations=evidence_citations,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a financial question.

    Full pipeline:
      1. Parse intent (ticker, metrics, question type)
      2. Dual retrieval (SQLite facts + ChromaDB documents)
      3. Build augmented prompt
      4. Call TraceAlchemy model
      5. Return grounded answer with citations

    Shares `_build_query_context` / `_answer_query_context` with
    `/query/stream` so the two paths cannot drift.
    """
    emitter = _install_query_emitter(request, chat_events=False)
    try:
        context = await _build_query_context(request)
        return await _answer_query_context(request, context)
    except Exception:
        if emitter is not None:
            emitter.error("query failed", terminal=True)
        raise


def _response_to_dict(response: QueryResponse) -> dict:
    """Return a pydantic model as a JSON-serializable dict."""
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    else:
        data = response.dict()
    # Keep optional rollout/observer fields byte-compatible on runtimes whose
    # pydantic serializer does not yet honor Field(exclude_if=...).
    for key in (
        "generation_skipped", "generation_skip_reason", "evidence_sufficiency",
        "evidence_citations", "answer_validation", "graph_trace_id",
    ):
        if data.get(key) is None:
            data.pop(key, None)
    return data


def _sse(event: str, data: dict) -> str:
    """Format one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _model_messages(
    prompt: str, intent: Optional[dict], grounding_level: str, *, tools_enabled: bool
) -> list[dict]:
    """Build the [system, user] chat messages for one model request."""
    return [
        {
            "role": "system",
            "content": _system_prompt_for_request(
                intent=intent, grounding_level=grounding_level, tools_enabled=tools_enabled,
            ),
        },
        {"role": "user", "content": prompt},
    ]


async def _stream_chat_tokens(*, messages: list[dict], temperature: float, max_tokens: int):
    """Yield token deltas from llama-server for a prebuilt chat messages list.

    The single low-level streaming primitive: the plain (no-tools) path and the
    tool-final path (which streams the accumulated tool messages) both go through
    here so the SSE parsing is shared. No ``tools`` key is ever sent — the final
    answer request must not re-enter tool planning.
    """
    if not model_client or not config:
        raise RuntimeError("model client unavailable")

    payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    async with model_client.stream("POST", config.llama_endpoint, json=payload) as resp:
        if hasattr(resp, "raise_for_status"):
            resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = (line or "").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Ignoring malformed model stream line: %s", line[:120])
                continue
            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            token = delta.get("content")
            if token is None:
                token = (choice.get("message") or {}).get("content")
            if token:
                yield token
    _mark_model_health(True)


async def _stream_model_tokens(
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
    intent: dict,
    grounding_level: str,
):
    """Yield token deltas for the plain (tools-disabled) final answer path."""
    messages = _model_messages(prompt, intent, grounding_level, tools_enabled=False)
    async for token in _stream_chat_tokens(
        messages=messages, temperature=temperature, max_tokens=max_tokens
    ):
        yield token


def _stream_progress_enabled() -> bool:
    """Whether versioned/redacted pipeline progress events are emitted (2.2.6.1)."""
    return bool(getattr(config, "enable_stream_progress_events", False))


def _tool_final_streaming_enabled() -> bool:
    """Whether a tools-enabled request may stream its final answer (2.2.6.1)."""
    return bool(getattr(config, "enable_tool_final_streaming", False))


def _context_used_deterministic_tools(context: dict) -> bool:
    """True when the adaptive route already dispatched deterministic tools.

    In that case the evidence is already packed into the prompt (2.2.3.2), so the
    stream path skips the model tool-planning rounds and streams the final answer
    directly.
    """
    orch = context.get("orchestration") or {}
    return bool(orch.get("deterministic_tools"))


def _drain_progress(emitter):
    """Yield SSE strings for any buffered progress events (sync generator)."""
    if emitter is None:
        return
    from .stream_events import iter_chat_sse

    for name, data in iter_chat_sse(emitter.drain(), include_counts=emitter.include_counts):
        yield _sse(name, data)


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream the final answer as SSE token deltas, then terminal metadata.

    Tool planning (2.2.6.1) — when ``enable_tools`` and
    ``enable_tool_final_streaming`` are both on — runs as bounded NON-streaming
    rounds first; only the final synthesis streams. With ``enable_tools`` on but
    the flag off, streaming stays disabled (404) exactly as before. When
    ``enable_stream_progress_events`` is on, versioned/redacted stage + tool
    progress events precede the tokens. Both flags off -> byte-identical to the
    pre-2.2.6 behavior.
    """
    if not bool(getattr(config, "enable_streaming", True)):
        raise HTTPException(status_code=404, detail="Streaming disabled")

    tools_enabled = bool(getattr(config, "enable_tools", False))
    tool_final = tools_enabled and _tool_final_streaming_enabled()
    if tools_enabled and not tool_final:
        # Legacy capability: streaming is genuinely unavailable while the tool
        # loop is active and tool-final streaming is off (404 -> client caches).
        raise HTTPException(status_code=404, detail="Streaming disabled while tools are enabled")

    # Install a request-scoped progress emitter BEFORE building the context so
    # the compile/route/retrieve/grade/correct/pack stages are captured. None
    # (progress off) leaves the _emit_* helpers as no-ops -> zero overhead.
    request_emitter = _install_query_emitter(request, chat_events=True)
    emitter = request_emitter if _stream_progress_enabled() else None

    try:
        context = await _build_query_context(request)
    except Exception:
        if request_emitter is not None:
            request_emitter.error("query failed", terminal=True)
        raise
    run_tool_final = (
        tool_final and _tools_supported and not _context_used_deterministic_tools(context)
    )
    model_available: Optional[bool] = None
    if not _deterministic_contract_eligible(context):
        model_available = await _check_model_health()
        # Feature-off/ineligible requests retain the legacy capability response.
        if not model_available and emitter is None and not run_tool_final:
            raise HTTPException(
                status_code=404,
                detail="Streaming unavailable when model is unavailable",
            )

    return StreamingResponse(
        _query_stream_events(
            request, context, emitter, run_tool_final, model_available,
        ),
        media_type="text/event-stream",
    )


async def _stream_degraded_answer(request: QueryRequest, context: dict):
    """Stream a single degraded (model-unavailable) answer + terminal metadata."""
    fallback = await _answer_query_context(request, context)
    if fallback.answer:
        yield _sse("token", {"token": fallback.answer})
    metadata = _response_to_dict(fallback)
    metadata.pop("answer", None)
    yield _sse("metadata", metadata)


async def _plan_tools_for_stream(context: dict, temperature: float, max_tokens: int):
    """Run the bounded non-streaming tool/planning rounds for the stream path.

    Returns ``(final_messages, kind)``. ``kind`` is the planning outcome:
    ``answered``/``final``/``plain_fallback`` all yield a messages list to stream
    (no ``tools`` key); ``error`` yields ``(None, "error")`` so the caller fails
    soft to the full non-streaming answer.
    """
    intent = context["intent"]
    grounding_level = context["grounding_level"]
    prompt = context["augmented_prompt"]
    plain_messages = _model_messages(prompt, intent, grounding_level, tools_enabled=False)
    base_payload = {
        "model": config.model_name,
        "messages": plain_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    tool_messages = _model_messages(prompt, intent, grounding_level, tools_enabled=True)
    outcome = await _run_tool_planning_rounds(
        base_payload=base_payload, messages=tool_messages, prompt=prompt)
    if outcome.kind == "error":
        return None, "error"
    if outcome.kind == "plain_fallback":
        _discard_trace_tool_results()
        return plain_messages, outcome.kind
    return outcome.messages, outcome.kind  # answered | final


async def _query_stream_events(
    request: QueryRequest,
    context: dict,
    emitter,
    run_tool_final: bool,
    model_available: Optional[bool],
):
    """SSE generator: flush progress, stream the final answer, emit metadata."""
    # 1. Flush the buffered context-build progress (query_started + stages).
    for chunk in _drain_progress(emitter):
        yield chunk

    # 2.3.7.3: render/validate/finalize before any model health probe or call.
    deterministic = _try_deterministic_response(context)
    if deterministic is not None:
        for chunk in _drain_progress(emitter):
            yield chunk
        yield _sse("token", {"token": deterministic.answer})
        metadata = _response_to_dict(deterministic)
        metadata.pop("answer", None)
        yield _sse("metadata", metadata)
        return

    if model_available is None:
        model_available = await _check_model_health()
    # Model-unavailable: the pre-2.2.6 legacy path historically returned a
    # capability 404. Once the SSE generator has started, preserve the newer
    # fail-soft behavior and return one degraded answer instead.

    stage_start = time.perf_counter()
    temperature, max_tokens = _task_settings(request, context["intent"])

    # 2. Model unavailable (new paths only — legacy 404'd already): degrade.
    if not model_available:
        _emit_stage("generate", "fallback", reason="model_unavailable")
        for chunk in _drain_progress(emitter):
            yield chunk
        async for chunk in _stream_degraded_answer(request, context):
            yield chunk
        return

    # 3. Select the final-answer token source.
    trace_system: Optional[str] = None
    if run_tool_final:
        final_messages, kind = await _plan_tools_for_stream(context, temperature, max_tokens)
        for chunk in _drain_progress(emitter):  # flush tool_started/tool_completed
            yield chunk
        if kind == "error":
            _emit_stage("generate", "fallback", reason="tool_planning_error")
            for chunk in _drain_progress(emitter):
                yield chunk
            async for chunk in _stream_degraded_answer(request, context):
                yield chunk
            return
        token_source = _stream_chat_tokens(
            messages=final_messages, temperature=temperature, max_tokens=max_tokens)
        trace_system = final_messages[0]["content"]
    else:
        token_source = _stream_model_tokens(
            prompt=context["augmented_prompt"],
            temperature=temperature,
            max_tokens=max_tokens,
            intent=context["intent"],
            grounding_level=context["grounding_level"],
        )
        trace_system = _system_prompt_for_request(
            intent=context["intent"],
            grounding_level=context["grounding_level"],
            tools_enabled=False,
        )

    # 4. Stream the final synthesis.
    _emit_stage("generate", "started")
    model_started = time.perf_counter()
    for chunk in _drain_progress(emitter):
        yield chunk
    answer_parts: list[str] = []
    try:
        async for token in token_source:
            if "model_ttft" not in context["timings"]:
                context["timings"]["model_ttft"] = (
                    time.perf_counter() - model_started
                ) * 1000
            answer_parts.append(token)
            yield _sse("token", {"token": token})
    except Exception as exc:  # noqa: BLE001 - streaming must never fail a query
        if answer_parts:
            # Tokens already emitted: send a terminal error, never a second
            # conflicting answer (2.2.6.1 Step 2).
            logger.warning("Final answer stream interrupted after tokens: %s", exc)
            if emitter is not None:
                emitter.error("final answer stream interrupted", terminal=True)
                for chunk in _drain_progress(emitter):
                    yield chunk
            else:
                yield _sse("error", {"message": "final answer stream interrupted", "terminal": True})
            return
        # Pre-token failure: fall back once to the non-streaming final answer.
        logger.warning("Streaming model call failed; falling back server-side: %s", exc)
        _emit_stage("generate", "fallback", reason="stream_failed")
        for chunk in _drain_progress(emitter):
            yield chunk
        async for chunk in _stream_degraded_answer(request, context):
            yield chunk
        return
    context["timings"]["model_total"] = (
        time.perf_counter() - model_started
    ) * 1000
    _emit_stage("generate", "completed")
    for chunk in _drain_progress(emitter):
        yield chunk

    # 5. Finalize: answer policy, trace, projection guardrail, validation.
    answer_text = _apply_answer_policy("".join(answer_parts), context["grounding_level"])
    citations = _extract_citations(answer_text)
    _record_trace_prompt(trace_system, context["augmented_prompt"])
    if context["intent"].get("question_type") == "projection":
        from .guardrails import apply_projection_guardrail

        answer_text, _flagged = apply_projection_guardrail(answer_text, context["augmented_prompt"])
    # Deterministic citation/numeric validation (2.2.4.3). Enforced downgrades/
    # refusals reach the terminal metadata event; already-streamed tokens are
    # not rewritten (a known streaming limitation).
    answer_text, context["grounding_level"], validation, evidence_citations = \
        _apply_answer_validation(context, answer_text, context["grounding_level"])
    for chunk in _drain_progress(emitter):
        yield chunk

    _stage_timing(context["timings"], "model_call", stage_start)
    response = _build_query_response(
        context=context,
        answer_text=answer_text,
        citations=citations,
        model_available=True,
        validation=validation,
        evidence_citations=evidence_citations,
    )
    _emit_graph_terminal(
        context,
        model_available=True,
        evidence_citations=evidence_citations,
        validation=validation,
        citations=citations,
    )
    metadata = _response_to_dict(response)
    metadata.pop("answer", None)
    yield _sse("metadata", metadata)


def _format_degraded_answer(
    retrieval: dict,
    intent: dict,
    evidence_sufficiency: Optional[dict] = None,
) -> str:
    """Format retrieved data as a readable answer when the model is unavailable."""
    parts = ["⚠️ Model unavailable — showing raw retrieved data:\n"]
    facts = retrieval.get("facts", [])
    docs = retrieval.get("documents", [])

    if facts:
        parts.append("**Structured Facts:**")
        for f in facts[:5]:
            parts.append(
                f"- {f.get('metric')}: {f.get('value')} "
                f"({f.get('period', 'N/A')})"
            )
        parts.append("")

    if docs:
        parts.append("**Relevant Documents:**")
        for d in docs[:3]:
            meta = d.get("metadata", {}) or {}
            parts.append(
                f"- {d.get('id', 'unknown')} "
                f"({meta.get('source', d.get('source', 'unknown'))})"
            )
        parts.append("")

    if not facts and not docs:
        parts.append("No stored data found for this question.")
        parts.append("")

    if evidence_sufficiency:
        missing = evidence_sufficiency.get("missing_subqueries") or []
        reasons = evidence_sufficiency.get("reason_codes") or []
        if missing or reasons:
            parts.append(
                "Evidence status: " + str(evidence_sufficiency.get("status", "refused"))
                + "; missing=" + (", ".join(missing) or "none")
                + "; reasons=" + (", ".join(reasons) or "none")
            )
            parts.append("")

    parts.append("Start llama-server to get AI-grounded answers.")
    return "\n".join(parts)


# ── Freshness / Refresh (Phase 1.7.4) ──────────────────

# Maps user-facing short source aliases to the logical source names used by
# Store.FRESHNESS_SOURCES.
_SOURCE_ALIASES = {
    "fundamentals": "yfinance_fundamentals",
    "yfinance_fundamentals": "yfinance_fundamentals",
    "news": "yfinance_news",
    "yfinance_news": "yfinance_news",
    "finnhub": "finnhub_news",
    "finnhub_news": "finnhub_news",
    "massive": "massive_market",
    "massive_market": "massive_market",
    "sec": "sec_filings",
    "sec_filings": "sec_filings",
    "gdelt": "gdelt_news",
    "gdelt_news": "gdelt_news",
    "earnings": "earnings_transcripts",
    "earnings_transcripts": "earnings_transcripts",
    "ir": "ir_pages",
    "ir_pages": "ir_pages",
    "estimates": "estimates",
}


def _normalize_sources(sources: Optional[list[str]]) -> list[str]:
    """Map short aliases to logical source names, dropping unknown ones."""
    if not sources:
        return []
    out = []
    for s in sources:
        logical = _SOURCE_ALIASES.get(str(s).lower().strip())
        if logical and logical not in out:
            out.append(logical)
    return out


def _stale_source_names(report: dict) -> list[str]:
    """Return logical source names that are stale or have never been fetched."""
    return [
        name for name, info in report.get("sources", {}).items()
        if info.get("status") in ("stale", "never_fetched")
    ]


# Logical sources that have a scheduler-managed ingestion pipeline. Refreshing
# these routes through the UnifiedScheduler so TTL tracking, staggered
# execution, and dead-letter handling stay consistent with cron-driven runs.
SCHEDULER_SOURCE_MAP = {
    "sec_filings": "sec_filings",
    "earnings_transcripts": "earnings_transcripts",
    "ir_pages": "ir_pages",
}


def _refresh_via_scheduler(ticker: str, logical: str) -> None:
    """Route a per-ticker refresh through the UnifiedScheduler's source runner.

    For scheduler-managed sources (SEC filings, earnings transcripts, IR pages)
    this ensures the same TTL tracking and error handling as the cron path.
    The underlying ingestors mark per-ticker cache freshness; we additionally
    mark the requested ticker fresh so the freshness report reflects the run.
    Falls through to direct ingestion for non-scheduler sources.
    """
    source_name = SCHEDULER_SOURCE_MAP.get(logical)
    if not source_name:
        _refresh_one_source_direct(ticker, logical)
        return

    from src.scheduler import UnifiedScheduler
    from src.storage.store import Store

    sched = UnifiedScheduler(store=store)
    sched._run_source(source_name, force=True)

    # Reflect the run in this ticker's freshness even if the underlying
    # ingestor only marks the synthetic scheduler ticker.
    cfg = Store.FRESHNESS_SOURCES.get(logical)
    if cfg:
        ttl = store._schedule_ttls().get(cfg["ttl_key"], 24)
        store.mark_source_fresh(ticker, cfg["cache_source"], ttl)


def _refresh_one_source(ticker: str, logical: str) -> None:
    """Run the ingestion for a single logical source for one ticker.

    Scheduler-managed sources are routed through the UnifiedScheduler bridge;
    all others are ingested directly. Marks the source fresh in cache_meta on
    success. Raises on failure so the caller can record the error.
    """
    if logical in SCHEDULER_SOURCE_MAP:
        _refresh_via_scheduler(ticker, logical)
        return
    from src.scheduler import UnifiedScheduler

    UnifiedScheduler(store=store).run_bounded_security_refresh(ticker, logical)


def _refresh_one_source_direct(ticker: str, logical: str) -> None:
    """Direct per-ticker ingestion for a single logical source.

    Marks the source fresh in cache_meta on success. Raises on failure so the
    caller can record the error.
    """
    from src.storage.store import Store
    cfg = Store.FRESHNESS_SOURCES.get(logical)
    cache_source = cfg["cache_source"] if cfg else logical
    ttl = store._schedule_ttls().get(cfg["ttl_key"], 24) if cfg else 24

    if logical == "yfinance_fundamentals":
        from src.ingestion.yfinance_ingestor import YFinanceIngestor
        ing = YFinanceIngestor(store=store)
        t = ing._fetch_ticker(ticker)
        if t is not None:
            ing._ingest_ticker_fundamentals(ticker, t)  # marks cache fresh
    elif logical == "yfinance_news":
        from src.ingestion.yfinance_ingestor import YFinanceIngestor
        ing = YFinanceIngestor(store=store)
        t = ing._fetch_ticker(ticker)
        if t is not None:
            ing._ingest_ticker_news(ticker, t)  # marks cache fresh
    elif logical == "finnhub_news":
        from src.ingestion.finnhub_ingestor import FinnhubIngestor
        result = FinnhubIngestor(store=store).ingest_ticker_news(ticker)
        if result.get("status") not in {"ok", "success", "no_data"}:
            raise RuntimeError(
                f"finnhub refresh failed for {ticker}: "
                f"{result.get('status') or result.get('error_class') or 'error'}"
            )
    elif logical == "massive_market":
        from src.ingestion.massive_ingestor import MassiveIngestor
        result = MassiveIngestor(store=store).ingest_market_data(tickers=[ticker])
        if result.get("status") not in {"ok", "success", "no_data"}:
            raise RuntimeError(
                f"massive refresh failed for {ticker}: "
                f"{result.get('status') or result.get('error_class') or 'error'}"
            )
    elif logical == "sec_filings":
        from src.sec import FilingScheduler
        sched = FilingScheduler(store=store)
        sched.processor.discover_new_filings(ticker)
        store.mark_source_fresh(ticker, cache_source, ttl)
    elif logical == "gdelt_news":
        from src.macros.gdelt_ingestor import GDELTIngestor
        GDELTIngestor(store=store).fetch_and_store_for_ticker(ticker)
        store.mark_source_fresh(ticker, cache_source, ttl)
    elif logical == "earnings_transcripts":
        from src.macros.earnings_transcripts import EarningsTranscriptIngestor
        EarningsTranscriptIngestor(store=store).fetch_and_process(ticker)
        store.mark_source_fresh(ticker, cache_source, ttl)
    elif logical == "ir_pages":
        from src.macros.ir_ingestor import IRIngestor
        IRIngestor(store=store).fetch_for_ticker(ticker)
        store.mark_source_fresh(ticker, cache_source, ttl)
    elif logical == "estimates":
        from src.macros.estimates_ingestor import EstimatesIngestor
        result = EstimatesIngestor(store=store).fetch_for_ticker(ticker)
        status = result.get("status")
        if status not in ("success", "no_data"):
            raise RuntimeError(
                f"estimates refresh failed for {ticker}: {status} "
                f"{result.get('errors') or []}"
            )
        # no_data still marks fresh: lack of analyst coverage (e.g. ETFs)
        # shouldn't trigger a refetch on every stale check.
        store.mark_source_fresh(ticker, cache_source, ttl)
    else:
        raise ValueError(f"Unknown source: {logical}")


def _refresh_ticker_sources(ticker: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """Refresh the given logical sources for a ticker.

    Returns (refreshed, errors).
    """
    refreshed: list[str] = []
    errors: list[str] = []
    for logical in sources:
        try:
            _refresh_one_source(ticker, logical)
            refreshed.append(logical)
        except Exception as e:  # noqa: BLE001 - never let a refresh crash the query
            logger.warning("Refresh failed for %s/%s: %s", ticker, logical, e)
            store.mark_source_stale(ticker, _logical_cache_source(logical), str(e))
            errors.append(f"{logical}: {e}")
    return refreshed, errors


def _logical_cache_source(logical: str) -> str:
    from src.storage.store import Store
    cfg = Store.FRESHNESS_SOURCES.get(logical)
    return cfg["cache_source"] if cfg else logical


def _evaluate_and_refresh(ticker: Optional[str], do_refresh: bool) -> dict:
    """Check freshness for a ticker and optionally refresh stale sources.

    Returns the freshness metadata block for the query response.
    """
    meta = {
        "overall": "unknown",
        "refreshed_during_query": [],
        "stale_sources_used": [],
        "fetched_on_miss": [],
        "warning": None,
    }
    if not ticker or not store:
        return meta

    try:
        report = store.get_freshness_report(ticker)
    except Exception as e:  # noqa: BLE001
        logger.warning("Freshness check failed for %s: %s", ticker, e)
        return meta

    meta["overall"] = report.get("overall", "unknown")
    # Only present-but-expired sources are auto-refreshed during a query;
    # never_fetched sources are left to the scheduler / explicit refresh.
    stale = [
        name for name, info in report.get("sources", {}).items()
        if info.get("status") == "stale"
    ]
    if not stale:
        return meta

    if do_refresh:
        meta["_write_requested"] = True
        refreshed, _errors = _refresh_ticker_sources(ticker, stale)
        meta["refreshed_during_query"] = refreshed
        meta["stale_sources_used"] = [s for s in stale if s not in refreshed]
    else:
        meta["stale_sources_used"] = stale
        meta["warning"] = (
            f"{ticker} has stale data for: {', '.join(stale)}. "
            "Answer may not reflect the latest information."
        )
    return meta


@app.get("/freshness/{ticker}", response_model=FreshnessResponse)
async def get_freshness(ticker: str):
    """Get a freshness report for a ticker across all data sources."""
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    report = store.get_freshness_report(ticker.upper())
    return FreshnessResponse(
        ticker=report["ticker"],
        overall=report["overall"],
        sources=report["sources"],
        stale_sources=report["stale_sources"],
    )


@app.post("/refresh/{ticker}", response_model=RefreshResponse)
async def refresh_ticker(ticker: str, body: Optional[RefreshRequest] = None):
    """Trigger on-demand refresh for a ticker.

    If ``sources`` is omitted, all currently-stale sources are refreshed.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    ticker = ticker.upper()
    start = time.time()

    # Distinguish "no sources field" (refresh all stale) from "sources field
    # provided but all unknown" (refresh nothing — silently skip unknowns).
    sources_provided = bool(body and body.sources)
    requested = _normalize_sources(body.sources if body else None)
    report = store.get_freshness_report(ticker)
    stale = _stale_source_names(report)

    if sources_provided:
        to_refresh = requested
        skipped = [s for s in report.get("sources", {}) if s not in requested]
    else:
        to_refresh = stale
        skipped = [s for s in report.get("sources", {}) if s not in stale]

    refreshed, errors = _refresh_ticker_sources(ticker, to_refresh)

    # Keep the BM25 lexical index fresh after ingestion (Phase 2.1.2.3).
    if retriever is not None:
        try:
            retriever.refresh_lexical_index()
        except Exception as e:  # noqa: BLE001 - never fail the refresh response
            logger.warning("Lexical index refresh failed: %s", e)

    return RefreshResponse(
        ticker=ticker,
        refreshed=refreshed,
        skipped=skipped,
        errors=errors,
        duration_s=round(time.time() - start, 2),
    )


@dataclass
class _ToolPlanningOutcome:
    """How the bounded tool/planning rounds ended (2.2.6.1 Step 2).

    ``kind`` selects the caller's final-answer branch:
      - ``answered``       — the model returned content with no tool call; the
                             non-streaming caller uses ``content`` directly, the
                             stream caller re-requests ``messages`` streamed.
      - ``final``          — the iteration budget was consumed with tool results;
                             the caller issues one final answer request over
                             ``messages``.
      - ``plain_fallback`` — tools were unsupported / returned empty; the caller
                             answers from the plain (no-tools) base payload.
      - ``error``          — a transport/malformed error; the non-streaming caller
                             returns ``error_message``.
    ``messages`` is the (mutated) tool-loop messages list including any assistant
    tool-call turns and role=tool results.
    """

    kind: str
    messages: list = field(default_factory=list)
    content: str = ""
    error_message: str = ""


async def _run_tool_planning_rounds(
    *, base_payload: dict, messages: list, prompt: str
) -> _ToolPlanningOutcome:
    """Run the bounded, NON-streaming tool/planning rounds shared by both paths.

    Executes tools until the model stops requesting them or the iteration budget
    is spent; it never produces the final answer itself. ``messages`` is mutated
    in place with assistant tool-call turns and role=tool results. Mirrors the
    pre-2.2.6 tool loop exactly (tools-unsupported / empty-first-response
    fallbacks, malformed/transport errors, evidence-trace + tools_used recording)
    so the non-streaming ``/query`` path is byte-for-byte unchanged; the stream
    path reuses the same rounds and then streams a final request.
    """
    global _tools_supported
    from .tools import ToolContext, dispatch_tool_traced, openai_schema

    schema = openai_schema()
    ctx = ToolContext(
        allow_write=config.allow_write_tools,
        max_refreshes=config.max_refreshes_per_query,
    )
    for iteration in range(config.max_tool_iterations):
        payload = {**base_payload, "messages": messages, "tools": schema}
        try:
            resp = await model_client.post(config.llama_endpoint, json=payload)
            resp.raise_for_status()
            _mark_model_health(True)
            data = resp.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text if e.response is not None else ""
            status = e.response.status_code if e.response is not None else None
            if status in (400, 404, 500) and "tool" in body.lower():
                logger.warning("Model tools unsupported; falling back to plain calls")
                _tools_supported = False
                return _ToolPlanningOutcome(kind="plain_fallback", messages=messages)
            logger.error("Model call failed: %s", e)
            return _ToolPlanningOutcome(
                kind="error", messages=messages, error_message=f"Error calling model: {e}")
        except Exception as e:  # noqa: BLE001
            logger.error("Model call failed: %s", e)
            return _ToolPlanningOutcome(
                kind="error", messages=messages, error_message=f"Error calling model: {e}")

        try:
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
        except (AttributeError, IndexError, TypeError) as e:
            logger.error("Malformed model response in tool loop: %s", e)
            return _ToolPlanningOutcome(
                kind="error", messages=messages,
                error_message=f"Error calling model: malformed response ({e})")
        if iteration == 0 and not tool_calls and not content.strip():
            logger.warning("Model returned empty content with tools; disabling tools")
            _tools_supported = False
            return _ToolPlanningOutcome(kind="plain_fallback", messages=messages)
        if not tool_calls:
            return _ToolPlanningOutcome(kind="answered", messages=messages, content=content)

        messages.append(msg)
        for call in tool_calls:
            tool_name = (call.get("function") or {}).get("name")
            if tool_name:
                _record_tool_used(tool_name)
            result, dispatched_name, dispatched_args = await asyncio.to_thread(
                dispatch_tool_traced, call, store, ctx
            )
            # Recorded immediately after dispatch returns and before the
            # matching role=tool message is appended (2.2.1.2 Step 2).
            _record_trace_tool_result(dispatched_name or tool_name or "", dispatched_args, result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result),
                }
            )

    return _ToolPlanningOutcome(kind="final", messages=messages)


async def _call_model(
    prompt: str,
    temperature: float,
    max_tokens: int,
    intent: Optional[dict] = None,
    grounding_level: str = "grounded",
) -> tuple[str, list[SourceCitation]]:
    """Send the augmented prompt to TraceAlchemy and parse the response.

    Non-streaming path. When tools are enabled it runs the bounded tool/planning
    rounds (:func:`_run_tool_planning_rounds`) and then produces the final
    non-streaming answer; the stream path shares those same rounds.
    """
    if not model_client or not config:
        return "Model unavailable. Please ensure llama-server is running.", []

    tools_on = bool(getattr(config, "enable_tools", False)) and _tools_supported
    plain_messages = _model_messages(prompt, intent, grounding_level, tools_enabled=False)
    base_payload = {
        "model": config.model_name,
        "messages": plain_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if not tools_on:
        answer, citations = await _post_and_parse(base_payload)
        _record_trace_prompt(plain_messages[0]["content"], prompt)
        return _apply_answer_policy(answer, grounding_level), citations

    # The tool loop keeps its own messages list so every fallback to
    # _post_and_parse(base_payload) still sends the exact pre-tools prompt.
    tool_messages = _model_messages(prompt, intent, grounding_level, tools_enabled=True)
    outcome = await _run_tool_planning_rounds(
        base_payload=base_payload, messages=tool_messages, prompt=prompt)

    if outcome.kind == "error":
        return outcome.error_message, []
    if outcome.kind == "plain_fallback":
        answer, citations = await _post_and_parse(base_payload)
        _discard_trace_tool_results()
        _record_trace_prompt(base_payload["messages"][0]["content"], prompt)
        return _apply_answer_policy(answer, grounding_level), citations
    if outcome.kind == "answered":
        answer = _apply_answer_policy(outcome.content, grounding_level)
        _record_trace_prompt(outcome.messages[0]["content"], prompt)
        return answer, _extract_citations(answer)

    # kind == "final": the budget was consumed with tool results — one final call.
    answer, citations = await _post_and_parse({**base_payload, "messages": outcome.messages})
    _record_trace_prompt(outcome.messages[0]["content"], prompt)
    return _apply_answer_policy(answer, grounding_level), citations


def _prompt_cache_enabled() -> bool:
    """Whether llama-server prompt reuse (cache_prompt) is active for this process."""
    return bool(getattr(config, "llama_cache_prompt", False)) and _prompt_cache_supported


def _capture_prompt_reuse_timings(data: dict) -> None:
    """Log llama-server's reused-token timings when present (2.2.6.2 Step 3).

    llama-server returns a ``timings`` object (``cached_n``/``prompt_n``) when
    prompt reuse is active. Recorded only; live TTFT measurement is separate.
    """
    try:
        timings = data.get("timings") if isinstance(data, dict) else None
        if isinstance(timings, dict) and ("cached_n" in timings or "prompt_n" in timings):
            logger.info(
                "llama prompt reuse: cached_n=%s prompt_n=%s",
                timings.get("cached_n"), timings.get("prompt_n"))
    except Exception:  # noqa: BLE001 - telemetry only, never fail a call
        pass


async def _post_and_parse(payload: dict) -> tuple[str, list[SourceCitation]]:
    """POST a chat payload and parse content plus inline citations.

    When ``llama_cache_prompt`` is on and still supported this sends the official
    llama-server prompt-reuse field (``cache_prompt: true``) on the request. If
    the backend rejects it, prompt reuse is disabled for the process and the
    plain payload is retried once (fail-soft capability behavior); with the flag
    off the payload is byte-identical to the pre-2.2.6.2 request.
    """
    global _prompt_cache_supported
    use_cache_prompt = _prompt_cache_enabled()
    send_payload = {**payload, "cache_prompt": True} if use_cache_prompt else payload
    try:
        resp = await model_client.post(config.llama_endpoint, json=send_payload)
        resp.raise_for_status()
        _mark_model_health(True)
        data = resp.json()
        if use_cache_prompt:
            _capture_prompt_reuse_timings(data)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse citations from the response (simple heuristic)
        citations = _extract_citations(content)

        return content, citations
    except Exception as e:
        # Capability fallback: if the backend rejected cache_prompt, disable it
        # for the process and retry the plain payload exactly once. The retry runs
        # with _prompt_cache_supported=False, so _prompt_cache_enabled() is False
        # and no cache_prompt key is sent — there is no second retry.
        if use_cache_prompt:
            logger.warning(
                "llama-server rejected cache_prompt (%s); disabling prompt reuse "
                "for this process and retrying plain once", e)
            _prompt_cache_supported = False
            return await _post_and_parse(payload)
        logger.error("Model call failed: %s", e)
        return f"Error calling model: {e}", []


def _extract_citations(text: str) -> list[SourceCitation]:
    """Extract [Source: ...] citations from model output."""
    import re
    if config is not None and not bool(getattr(config, "enable_citations", True)):
        return []
    citations = []
    pattern = r'\[Source:\s*([^\]]+)\]'
    for match in re.finditer(pattern, text):
        parts = match.group(1).split("/")
        citation = SourceCitation(
            source_type=parts[0] if len(parts) > 0 else "unknown",
            ticker=parts[1] if len(parts) > 1 else "",
        )
        citations.append(citation)
    return citations


# ── Raw Search (Bypass Model) ──────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Raw hybrid search — returns retrieved data without model inference.

    Documents come through the shared retriever's hybrid path (vector + BM25 +
    re-rank, per config) so ``fusion_score`` / ``rerank_score`` are exposed for
    inspecting ranking quality (Phase 2.1.2.3).
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    from .retriever import Retriever
    r = retriever or Retriever(store=store, config=config)

    documents: list[dict] = []
    facts: list[dict] = []
    ticker_out = request.ticker
    try:
        facts_results = store.search(
            query=request.query,
            n_results=request.n_results,
            ticker=request.ticker,
        )
        facts = facts_results.get("facts", [])
        ticker_out = facts_results.get("ticker") or request.ticker
    except Exception as e:  # noqa: BLE001
        logger.warning("Search facts failed (embedding server may be down): %s", e)

    try:
        documents = r.retrieve_documents(
            query=request.query,
            ticker=request.ticker,
            n_results=request.n_results,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Search documents failed: %s", e)

    return SearchResponse(
        documents=documents,
        facts=facts,
        ticker=ticker_out,
    )


# ── FinanceBot adapter (RAG source of truth; all tools retained) ──

@app.post("/financebot/rag", response_model=FinanceBotRagResponse)
async def financebot_rag(request: FinanceBotRagRequest):
    """Hybrid retrieval for FinanceBot with an explicit hit/miss contract.

    Does not call a local chat model. On ``status=hit`` FinanceBot must answer
    from the returned facts/documents only. On ``status=miss``,
    ``web_search_allowed`` is true and open-web fallback is permitted.
    All registered RAG tools remain available at ``POST /financebot/tools`` and
    ``GET /tools`` (including ``classify_trade_bias`` for long vs short).
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    from .retriever import Retriever
    r = retriever or Retriever(store=store, config=config)
    min_facts = request.min_facts
    min_documents = request.min_documents
    min_score = request.min_document_score
    if config is not None:
        if min_facts == 1:
            min_facts = int(getattr(config, "financebot_min_facts", 1) or 1)
        if min_documents == 1:
            min_documents = int(getattr(config, "financebot_min_documents", 1) or 1)
        if min_score == 0.0:
            min_score = float(getattr(config, "financebot_min_document_score", 0.0) or 0.0)

    payload = run_financebot_retrieval(
        store=store,
        retriever=r,
        query=request.query,
        ticker=request.ticker,
        n_results=request.n_results,
        min_facts=min_facts,
        min_documents=min_documents,
        min_document_score=min_score,
    )
    return FinanceBotRagResponse(**payload)


@app.post("/financebot/tools", response_model=FinanceBotToolResponse)
async def financebot_tools(request: FinanceBotToolRequest):
    """Invoke any registered RAG tool without a local chat model.

    This is the FinanceBot path for ``classify_trade_bias`` (long vs short),
    ``get_price_targets``, ``query_facts``, ``describe_coverage``, and every
    other tool attached to this RAG. Write tools still require
    ``allow_write_tools``.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")
    allow_write = bool(getattr(config, "allow_write_tools", False)) if config else False
    max_refreshes = int(getattr(config, "max_refreshes_per_query", 2) or 2) if config else 2
    payload = invoke_financebot_tool(
        store=store,
        name=request.name,
        arguments=request.arguments,
        allow_write=allow_write,
        max_refreshes=max_refreshes,
    )
    return FinanceBotToolResponse(**payload)


@app.get("/financebot/tools")
async def financebot_list_tools():
    """List every registered RAG tool FinanceBot can call."""
    enabled = bool(getattr(config, "enable_tools", False)) if config else False
    allow_write = bool(getattr(config, "allow_write_tools", False)) if config else False
    return {
        "enabled": enabled,
        "allow_write_tools": allow_write,
        "financebot_direct_dispatch": True,
        "tools": list_financebot_tools(),
    }


# ── Macro / Sentiment / Guidance ───────────────────────

@app.get("/macro/snapshot", response_model=MacroSnapshotResponse)
async def macro_snapshot():
    """
    Get a quick snapshot of key macro-economic indicators.
    Returns cached data from SQLite (no live FRED API call).
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    macro = _macro_snapshot_data()

    return MacroSnapshotResponse(
        gdp=macro.get("GDP"),
        inflation_cpi=macro.get("CPIAUCSL"),
        fed_rate=macro.get("FEDFUNDS"),
        unemployment=macro.get("UNRATE"),
        ten_year_treasury=macro.get("DGS10"),
        ten_two_spread=macro.get("T10Y2Y"),
    )


@app.get("/sentiment/{ticker}", response_model=SentimentResponse)
async def sentiment(ticker: str, days: int = 7):
    """
    Get GDELT sentiment summary for a ticker.

    Args:
        ticker: Stock ticker symbol
        days: Lookback period in days (default: 7)

    Returns average tone score, article count, and sentiment ratios.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    summary = _sentiment_data(ticker, days=days)

    return SentimentResponse(
        ticker=summary.get("ticker", ticker.upper()),
        average_tone=summary.get("average_tone"),
        article_count=summary.get("article_count", 0),
        positive_ratio=summary.get("positive_ratio", 0.0),
        negative_ratio=summary.get("negative_ratio", 0.0),
    )


@app.get("/guidance/{ticker}", response_model=dict)
async def guidance(ticker: str):
    """
    Get the latest earnings guidance for a ticker.
    Returns revenue guidance range, EPS, and margin from the most
    recent earnings transcript.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    guidance_data = _guidance_data(ticker)

    if not guidance_data:
        return {"ticker": ticker.upper(), "guidance": {}, "status": "not_found"}

    return {
        "ticker": ticker.upper(),
        "guidance": guidance_data,
        "status": "found",
    }


# ── Root ───────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "FinanceBot hybrid RAG (Gemma-E4B-Finance-RAG)",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
        "search": "POST /search",
        "financebot_rag": "POST /financebot/rag",
        "financebot_tools": "POST /financebot/tools",
        "tools": "GET /tools",
    }

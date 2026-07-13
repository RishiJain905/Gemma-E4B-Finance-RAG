"""tests/test_graph_observer.py
Offline tests for bounded, redacted query graph observation.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from types import SimpleNamespace

import pytest

from src.middleware.stream_events import QueryEventEmitter, serialize_chat_sse


def _observer_module():
    from src.middleware.graph_observer import (
        GraphDelta,
        GraphEdge,
        GraphNode,
        TraceHub,
        make_event_observer,
    )

    return GraphDelta, GraphEdge, GraphNode, TraceHub, make_event_observer


def test_records_have_stable_serializable_shapes_and_redact_recursively():
    GraphDelta, GraphEdge, GraphNode, _TraceHub, _make = _observer_module()
    node = GraphNode(
        id="q1:evidence:E1",
        query_id="q1",
        kind="evidence",
        label="Evidence E1",
        summary="safe excerpt",
        status="complete",
        metadata={
            "evidence_id": "E1",
            "ticker": "AAPL",
            "rank": 1,
            "nested": {"api_token": "CANARY", "period": "2025"},
            "model_path": "C:/CANARY/model.gguf",
            "ignored": "CANARY",
        },
        created_sequence=1,
        updated_sequence=2,
    )
    edge = GraphEdge(
        id="q1:edge:E1-source",
        query_id="q1",
        source="q1:evidence:E1",
        target="q1:source:sec",
        relation="from_source",
        metadata={"rank": 1, "authorization": "CANARY"},
    )
    delta = GraphDelta(query_id="q1", sequence=3, operation="upsert_node", node=node)

    payload = delta.to_dict()
    encoded = json.dumps(payload)
    assert payload["schema_version"] == 1
    assert payload["node"]["metadata"] == {
        "evidence_id": "E1",
        "ticker": "AAPL",
        "rank": 1,
    }
    assert "CANARY" not in encoded
    assert edge.to_dict()["metadata"] == {"rank": 1}


def test_hub_reconstructs_snapshot_and_exact_evidence_source_edges():
    GraphDelta, GraphEdge, GraphNode, TraceHub, _make = _observer_module()
    hub = TraceHub(trace_limit=10, element_limit=20, trace_ttl_s=60)
    evidence = GraphNode(
        id="q1:evidence:E1", query_id="q1", kind="evidence",
        label="Evidence E1", summary="Revenue was 10", status="complete",
        metadata={"evidence_id": "E1", "source_type": "sec_10k"},
    )
    source = GraphNode(
        id="q1:source:sec_10k", query_id="q1", kind="source",
        label="SEC 10-K", status="complete", metadata={"source_type": "sec_10k"},
    )
    edge = GraphEdge(
        id="q1:edge:E1:sec_10k", query_id="q1",
        source=evidence.id, target=source.id, relation="from_source",
    )
    for operation, item in (
        ("upsert_node", evidence), ("upsert_node", source), ("upsert_edge", edge)
    ):
        hub.publish(GraphDelta(
            query_id="q1", sequence=0, operation=operation,
            node=item if operation == "upsert_node" else None,
            edge=item if operation == "upsert_edge" else None,
        ))
    hub.publish(GraphDelta(query_id="q1", sequence=0, operation="trace_complete"))

    snapshot = hub.snapshot("q1")
    assert snapshot is not None
    assert [node["id"] for node in snapshot["nodes"]] == [evidence.id, source.id]
    assert snapshot["edges"] == [edge.to_dict()]
    assert hub.evidence("q1", "E1")["summary"] == "Revenue was 10"


def test_hub_evicts_deterministically_by_trace_count_elements_and_ttl():
    GraphDelta, _GraphEdge, GraphNode, TraceHub, _make = _observer_module()
    now = [100.0]
    hub = TraceHub(
        trace_limit=2, element_limit=2, trace_ttl_s=10, clock=lambda: now[0]
    )
    subscriber = hub.subscribe()

    def publish(query_id: str) -> None:
        hub.publish(GraphDelta(
            query_id=query_id, sequence=0, operation="upsert_node",
            node=GraphNode(
                id=f"{query_id}:query", query_id=query_id, kind="query",
                label="Query", status="active",
            ),
        ))

    publish("q1")
    publish("q2")
    publish("q3")
    assert [item["query_id"] for item in hub.list_traces()] == ["q3", "q2"]
    assert hub.snapshot("q1") is None
    queued = []
    while not subscriber.queue.empty():
        queued.append(subscriber.queue.get_nowait())
    assert any(event["delta"]["operation"] == "trace_evicted" for event in queued)

    now[0] = 111.0
    assert hub.list_traces() == []


def test_replay_before_an_evicted_delta_requires_reset_without_stale_events():
    GraphDelta, _GraphEdge, GraphNode, TraceHub, _make = _observer_module()
    hub = TraceHub(trace_limit=1)
    for query_id in ("q1", "q2"):
        hub.publish(GraphDelta(
            query_id=query_id, sequence=0, operation="upsert_node",
            node=GraphNode(
                id=f"{query_id}:query", query_id=query_id, kind="query",
                label="Query", status="active",
            ),
        ))
    reset, events = hub.events_since(0)
    assert reset is True
    assert events == []


def test_future_replay_cursor_requires_reset_after_restart():
    GraphDelta, _GraphEdge, GraphNode, TraceHub, _make = _observer_module()
    hub = TraceHub()
    hub.publish(GraphDelta(
        query_id="q1", sequence=0, operation="upsert_node",
        node=GraphNode(
            id="q1:query", query_id="q1", kind="query",
            label="Query", status="active",
        ),
    ))
    assert hub.events_since(999)[0] is True
    reset_event = hub._reset_event()
    assert hub.events_since(reset_event["event_id"])[0] is False


def test_repeated_upserts_compact_reconstruction_memory():
    GraphDelta, _GraphEdge, GraphNode, TraceHub, _make = _observer_module()
    hub = TraceHub(element_limit=2)
    for index in range(100):
        hub.publish(GraphDelta(
            query_id="q1", sequence=index, operation="upsert_node",
            node=GraphNode(
                id="q1:stage:retrieve", query_id="q1", kind="stage",
                label="Retrieve", status="complete", metadata={"count": index},
            ),
        ))
    assert len(hub._traces["q1"].deltas) == 1
    assert hub.snapshot("q1")["nodes"][0]["metadata"]["count"] == 99


def test_full_subscriber_is_marked_reset_without_blocking_publish():
    GraphDelta, _GraphEdge, GraphNode, TraceHub, _make = _observer_module()
    hub = TraceHub(subscriber_queue_limit=1)
    subscriber = hub.subscribe()
    for index in range(2):
        hub.publish(GraphDelta(
            query_id="q1", sequence=index, operation="upsert_node",
            node=GraphNode(
                id=f"q1:stage:{index}", query_id="q1", kind="stage",
                label="Stage", status="active",
            ),
        ))
    assert subscriber.reset_required is True
    reset = hub.next_subscriber_event_nowait(subscriber)
    assert reset["delta"]["operation"] == "reset_required"
    hub.unsubscribe(subscriber)


def test_emitter_observer_is_fail_soft_and_chat_sse_is_backward_compatible(caplog):
    _GraphDelta, _GraphEdge, _GraphNode, TraceHub, make_event_observer = _observer_module()
    hub = TraceHub()
    emitter = QueryEventEmitter(
        query_id="q1", observers=[make_event_observer(hub)], include_counts=True
    )
    event = emitter.stage("retrieve", "completed", elapsed_ms=1.25)
    assert serialize_chat_sse(event) == (
        "stage",
        {
            "schema_version": 1,
            "query_id": "q1",
            "sequence": 0,
            "timestamp": round(event.timestamp, 3),
            "stage": "retrieve",
            "phase": "completed",
            "elapsed_ms": 1.2,
        },
    )
    assert hub.snapshot("q1")["nodes"][0]["id"] == "q1:stage:retrieve"

    calls = 0

    def broken(_event):
        nonlocal calls
        calls += 1
        raise RuntimeError("observer broke")

    bad = QueryEventEmitter(query_id="q2", observers=[broken])
    bad.query_started(question="hello")
    bad.stage("route", "started")
    assert calls == 1
    assert "observer broke" in caplog.text


def test_question_preview_digest_excerpt_cap_and_canary_redaction():
    _GraphDelta, _GraphEdge, _GraphNode, TraceHub, make_event_observer = _observer_module()
    hub = TraceHub(excerpt_chars=12, question_preview_chars=8)
    emitter = QueryEventEmitter(query_id="q1", observers=[make_event_observer(hub)])
    emitter.query_started(question="What is CANARY_SECRET revenue?")
    emitter.graph_evidence(
        evidence_id="E1",
        kind="document",
        excerpt="abcdefghijklmnop CANARY_SECRET",
        metadata={"ticker": "AAPL", "api_key": "CANARY_SECRET"},
        source_type="sec_10k",
        rank=1,
    )
    snapshot = hub.snapshot("q1")
    query = next(node for node in snapshot["nodes"] if node["kind"] == "query")
    evidence = next(node for node in snapshot["nodes"] if node["kind"] == "evidence")
    assert query["summary"] == "What is "
    assert len(query["metadata"]["question_digest"]) == 64
    assert evidence["summary"] == "abcdefghijkl"
    assert "CANARY_SECRET" not in json.dumps(snapshot)


def test_terminal_error_completes_trace_and_identifiers_hide_local_paths():
    _GraphDelta, _GraphEdge, _GraphNode, TraceHub, make_event_observer = _observer_module()
    hub = TraceHub()
    emitter = QueryEventEmitter(query_id="q1", observers=[make_event_observer(hub)])
    emitter.query_started(question="hello")
    emitter.graph_evidence(
        evidence_id=r"C:\Users\alice\secret.txt", kind="document", excerpt="safe"
    )
    emitter.error("stream failed", terminal=True)
    snapshot = hub.snapshot("q1")
    assert snapshot["complete"] is True
    assert "alice" not in json.dumps(snapshot)
    assert any(node["status"] == "error" for node in snapshot["nodes"])


def test_normal_publish_p95_below_two_ms():
    GraphDelta, _GraphEdge, GraphNode, TraceHub, _make = _observer_module()
    hub = TraceHub(trace_limit=100, element_limit=5000)
    durations = []
    for index in range(600):
        delta = GraphDelta(
            query_id=f"q{index % 50}", sequence=index, operation="upsert_node",
            node=GraphNode(
                id=f"q{index % 50}:stage:retrieve", query_id=f"q{index % 50}",
                kind="stage", label="Retrieve", status="complete",
                metadata={"elapsed_ms": 1.0, "count": index},
            ),
        )
        started = time.perf_counter_ns()
        hub.publish(delta)
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    p95 = statistics.quantiles(durations, n=100)[94]
    assert p95 < 2.0
    assert not asyncio.iscoroutinefunction(hub.publish)


def test_hierarchical_edge_maps_store_parent_to_request_evidence_id(monkeypatch):
    from src.middleware import app as middleware_app
    from src.middleware.config import MiddlewareConfig
    from src.middleware.evidence import assign_evidence_ids, build_evidence_items
    from src.middleware.graph_observer import TraceHub
    from src.middleware.models import QueryRequest

    config = MiddlewareConfig()
    config.enable_graph_observer = True
    hub = TraceHub()
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "graph_hub", hub)
    request = QueryRequest(question="filing context")
    emitter = middleware_app._install_query_emitter(request, chat_events=False)
    retrieval = {"facts": [], "documents": [
        {"id": "root", "document": "root", "metadata": {"source_type": "sec_10k"}},
        {"id": "child", "document": "child", "metadata": {
            "source_type": "sec_10k", "parent_id": "root",
        }},
    ]}
    ledger = assign_evidence_ids(build_evidence_items([], retrieval["documents"]))
    middleware_app._emit_graph_evidence(retrieval, ledger)
    snapshot = hub.snapshot(emitter.query_id)
    expanded = [edge for edge in snapshot["edges"] if edge["relation"] == "expanded_from"]
    assert len(expanded) == 1
    assert expanded[0]["source"].endswith(":E1")
    assert expanded[0]["target"].endswith(":E2")


def test_grade_and_tool_events_are_emitted_at_actual_execution_boundaries():
    from src.middleware.evidence_grader import grade_evidence
    from src.middleware.graph_observer import TraceHub
    from src.middleware.query_plan import QueryPlan, QuerySubquery
    from src.middleware.stream_events import set_current_emitter
    from src.middleware.tools.base import (
        REGISTRY,
        Tool,
        ToolContext,
        dispatch_named_tool,
    )

    hub = TraceHub()
    emitter = QueryEventEmitter(query_id="q1", observers=[_observer_module()[4](hub)])
    set_current_emitter(emitter)
    plan = QueryPlan(
        original_question="revenue", retrieval_query="revenue",
        metrics=["revenue"],
        subqueries=[QuerySubquery(id="sq0", text="revenue", metrics=("revenue",))],
    ).validate()
    tool = Tool(
        name="graph_test_tool", description="test", parameters={"type": "object"},
        handler=lambda _store: {"results": [1]},
    )
    REGISTRY[tool.name] = tool
    try:
        grade_evidence(plan, {"facts": [], "documents": []})
        dispatch_named_tool(
            tool.name, {}, object(), ToolContext(allow_write=False, max_refreshes=0)
        )
    finally:
        REGISTRY.pop(tool.name, None)
        set_current_emitter(None)

    events = [event["delta"] for event in hub.events_since(0)[1]]
    grade = [
        event["node"]["status"] for event in events
        if event.get("node", {}).get("id") == "q1:stage:grade"
    ]
    tools = [
        event["node"]["status"] for event in events
        if event.get("node", {}).get("kind") == "tool"
    ]
    assert grade == ["active", "complete"]
    assert tools == ["active", "complete"]


@pytest.mark.asyncio
async def test_legacy_path_uses_executed_retrieval_once_and_builds_small_trace(monkeypatch):
    from src.middleware import app as middleware_app
    from src.middleware.config import MiddlewareConfig
    from src.middleware.graph_observer import TraceHub
    from src.middleware.models import QueryRequest

    class FakeRetriever:
        calls = 0

        def retrieve(self, **_kwargs):
            self.calls += 1
            return {
                "facts": [{
                    "id": "fact-1", "ticker": "AAPL", "metric": "revenue",
                    "value": 10, "period": "2025", "source_type": "sec_10k",
                }],
                "documents": [],
                "retrieval_strategy": "hybrid",
                "timings": {},
            }

    async def freshness(*_args, **_kwargs):
        return {"overall": "fresh"}

    config = MiddlewareConfig()
    config.enable_graph_observer = True
    config.enable_adaptive_rag = False
    fake = FakeRetriever()
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "retriever", fake)
    monkeypatch.setattr(middleware_app, "graph_hub", TraceHub())
    monkeypatch.setattr(middleware_app, "_freshness_stage", freshness)

    request = QueryRequest(question="What was AAPL revenue?", refresh=False)
    emitter = middleware_app._install_query_emitter(request, chat_events=False)
    context = await middleware_app._build_query_context(request)
    middleware_app._emit_graph_terminal(
        context, model_available=False, evidence_citations=[], validation=None
    )

    assert fake.calls == 1
    snapshot = middleware_app.graph_hub.snapshot(emitter.query_id)
    kinds = {node["kind"] for node in snapshot["nodes"]}
    assert {"query", "stage", "evidence", "source", "answer"} <= kinds
    assert snapshot["complete"] is True
    node_ids = {node["id"] for node in snapshot["nodes"]}
    assert all(edge["source"] in node_ids and edge["target"] in node_ids
               for edge in snapshot["edges"])


@pytest.mark.asyncio
async def test_adaptive_path_projects_actual_plan_lane_and_selected_evidence(monkeypatch):
    from src.middleware import adaptive_orchestrator, app as middleware_app
    from src.middleware.adaptive_orchestrator import ContextSelection, Lane, OrchestrationResult
    from src.middleware.config import MiddlewareConfig
    from src.middleware.graph_observer import TraceHub
    from src.middleware.models import QueryRequest
    from src.middleware.query_plan import QueryEntity, QueryPlan, QuerySubquery

    plan = QueryPlan(
        original_question="Compare AAPL revenue",
        retrieval_query="Compare AAPL revenue",
        entities=[QueryEntity("AAPL", "Apple", 1.0, "known_ticker", "AAPL", 8)],
        intents=["comparison"], metrics=["revenue"],
        subqueries=[QuerySubquery(
            id="sq0", text="Compare AAPL revenue", entity_tickers=("AAPL",),
            intents=("comparison",), metrics=("revenue",), retrieval_modes=("facts",),
        )],
        primary_intent="comparison",
    ).validate()
    fact = {
        "id": "fact-1", "ticker": "AAPL", "metric": "revenue", "value": 10,
        "period": "2025", "source_type": "sec_10k", "subquery_ids": ["sq0"],
    }
    result = OrchestrationResult(
        lane=Lane.STANDARD, plan=plan, merged_facts=[fact],
        subqueries_executed=["sq0"], retrieval_rounds_used=1,
        context=ContextSelection(facts=[fact], context_chars=100), context_size=100,
    )
    calls = 0

    def orchestrate_once(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return result

    async def freshness(*_args, **_kwargs):
        return {"overall": "fresh"}

    config = MiddlewareConfig()
    config.enable_graph_observer = True
    config.enable_adaptive_rag = True
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "store", SimpleNamespace(sqlite=SimpleNamespace()))
    monkeypatch.setattr(middleware_app, "retriever", object())
    monkeypatch.setattr(middleware_app, "graph_hub", TraceHub())
    monkeypatch.setattr(middleware_app, "_freshness_stage", freshness)
    monkeypatch.setattr(middleware_app, "_adaptive_available_metrics", lambda: ("revenue",))
    monkeypatch.setattr(adaptive_orchestrator, "orchestrate", orchestrate_once)

    request = QueryRequest(question=plan.original_question, refresh=False)
    emitter = middleware_app._install_query_emitter(request, chat_events=False)
    shared = {
        "request": request,
        "start": time.time(),
        "timings": {},
        "parser": SimpleNamespace(parse_plan=lambda *_args, **_kwargs: plan),
        "intent": plan.to_legacy_intent(),
        "conversation_meta": None,
        "history_turns": [],
        "compiled": None,
        "retrieval_query": plan.retrieval_query,
        "retrieval_intent": plan.to_legacy_intent(),
    }
    context = await middleware_app._build_adaptive_query_context(shared)

    assert calls == 1
    assert context["orchestration"]["lane"] == "standard"
    snapshot = middleware_app.graph_hub.snapshot(emitter.query_id)
    assert any(node["kind"] == "plan" for node in snapshot["nodes"])
    assert any(node["kind"] == "subquery" for node in snapshot["nodes"])
    assert any(node["kind"] == "evidence" for node in snapshot["nodes"])

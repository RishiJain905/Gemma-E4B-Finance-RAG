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
async def test_legacy_tool_edges_resolve_to_the_shared_route_node(monkeypatch):
    """Legacy tools carry no subquery, so the observer sources their ``routed_to``
    edges from ``stage:route``. The legacy intent node must therefore BE
    ``stage:route`` or every such edge dangles and the ROUTE column shows no
    fan-out (Defect 2). Drive the real tool-emit path (which the small-trace test
    above never exercises) and assert the routing node exists and the edges land."""
    from src.middleware import app as middleware_app
    from src.middleware.config import MiddlewareConfig
    from src.middleware.graph_observer import TraceHub, _node_id
    from src.middleware.models import QueryRequest

    class ToolEmittingRetriever:
        def retrieve(self, **_kwargs):
            # A legacy dispatch: tool events with no subquery_id.
            middleware_app._emit_tool_started("query_facts")
            middleware_app._emit_tool_completed("query_facts", "ok", count=1)
            return {
                "facts": [{
                    "id": "fact-1", "ticker": "AAPL", "metric": "revenue",
                    "value": 10, "period": "2025", "source_type": "sec_10k",
                }],
                "documents": [], "retrieval_strategy": "hybrid", "timings": {},
            }

    async def freshness(*_args, **_kwargs):
        return {"overall": "fresh"}

    config = MiddlewareConfig()
    config.enable_graph_observer = True
    config.enable_adaptive_rag = False
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "retriever", ToolEmittingRetriever())
    monkeypatch.setattr(middleware_app, "graph_hub", TraceHub())
    monkeypatch.setattr(middleware_app, "_freshness_stage", freshness)

    request = QueryRequest(question="What was AAPL revenue?", refresh=False)
    emitter = middleware_app._install_query_emitter(request, chat_events=False)
    context = await middleware_app._build_query_context(request)
    middleware_app._emit_graph_terminal(
        context, model_available=False, evidence_citations=[], validation=None
    )

    snapshot = middleware_app.graph_hub.snapshot(emitter.query_id)
    _assert_graph_integrity(snapshot)  # no dangling edges anywhere in the trace
    node_ids = {node["id"] for node in snapshot["nodes"]}
    route_id = _node_id(emitter.query_id, "stage", "route")
    tool_ids = {n["id"] for n in snapshot["nodes"] if n["kind"] == "tool"}
    assert route_id in node_ids, "legacy routing node must use the shared stage:route id"
    assert tool_ids, "the dispatched tool must appear as a node"
    routed = [
        edge for edge in snapshot["edges"]
        if edge["relation"] == "routed_to" and edge["target"] in tool_ids
    ]
    assert routed, "the ROUTE column must fan out to the tool"
    assert all(edge["source"] == route_id for edge in routed)


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


# ── 2.2.7.4: completeness, noninterference, overhead ────────────────────────

def _assert_graph_integrity(snapshot: dict) -> None:
    """Every node id is unique and every edge endpoint resolves to a node."""
    ids = [node["id"] for node in snapshot["nodes"]]
    assert len(ids) == len(set(ids)), "duplicate node ids in trace"
    node_ids = set(ids)
    edge_ids = [edge["id"] for edge in snapshot["edges"]]
    assert len(edge_ids) == len(set(edge_ids)), "duplicate edge ids in trace"
    for edge in snapshot["edges"]:
        assert edge["source"] in node_ids, f"dangling source {edge['source']}"
        assert edge["target"] in node_ids, f"dangling target {edge['target']}"


def _seeded_trace(hub, query_id="q1"):
    """One representative completed trace across the executed pipeline stages."""
    emitter = QueryEventEmitter(
        query_id=query_id, observers=[_observer_module()[4](hub)])
    emitter.query_started(question="What was NVDA revenue in 2025?")
    emitter.stage("compile", "completed", elapsed_ms=1.0)
    emitter.graph_update(
        nodes=[
            {"id": f"{query_id}:plan:validated", "kind": "plan",
             "label": "Validated Query Plan", "status": "complete",
             "metadata": {"tickers": ["NVDA"], "metrics": ["revenue"], "lane": "standard"}},
            {"id": f"{query_id}:subquery:sq0", "kind": "subquery", "label": "sq0",
             "status": "complete", "group_id": f"{query_id}:plan:validated",
             "metadata": {"subquery_id": "sq0"}},
        ],
        edges=[
            {"id": f"{query_id}:plan-edge", "source": f"{query_id}:query:request",
             "target": f"{query_id}:plan:validated", "relation": "compiled_to"},
            {"id": f"{query_id}:sq-edge", "source": f"{query_id}:plan:validated",
             "target": f"{query_id}:subquery:sq0", "relation": "contains"},
        ],
    )
    emitter.stage("retrieve", "started")
    emitter.tool_started("query_facts", subquery_id="sq0")
    emitter.tool_completed("query_facts", "ok", count=1, subquery_id="sq0")
    emitter.stage("retrieve", "completed", elapsed_ms=2.0)
    emitter.graph_evidence(
        evidence_id="E1", kind="fact", excerpt="NVDA revenue 60.9B",
        metadata={"ticker": "NVDA", "metric": "revenue", "period": "2025", "rank": 1},
        source_type="sec_10k", rank=1,
    )
    emitter.graph_update(
        nodes=[
            {"id": f"{query_id}:answer:final", "kind": "answer", "label": "Final Answer",
             "status": "complete", "metadata": {"status": "grounded"}},
            {"id": f"{query_id}:citation:E1", "kind": "citation", "label": "Citation E1",
             "status": "complete",
             "metadata": {"evidence_id": "E1", "support_status": "supported"}},
        ],
        edges=[
            {"id": f"{query_id}:ans-edge", "source": f"{query_id}:query:request",
             "target": f"{query_id}:answer:final", "relation": "returned"},
            {"id": f"{query_id}:sup-edge", "source": f"{query_id}:evidence:E1",
             "target": f"{query_id}:answer:final", "relation": "supports"},
            {"id": f"{query_id}:cit-edge", "source": f"{query_id}:evidence:E1",
             "target": f"{query_id}:citation:E1", "relation": "cited_by"},
        ],
    )
    emitter.graph_update(operation="trace_complete", summary={"status": "grounded"})
    return emitter


def test_seeded_trace_is_complete_truthful_and_integral():
    _GD, _GE, _GN, TraceHub, _make = _observer_module()
    hub = TraceHub()
    emitter = _seeded_trace(hub)
    snapshot = hub.snapshot(emitter.query_id)

    assert snapshot["complete"] is True
    _assert_graph_integrity(snapshot)
    kinds = [node["kind"] for node in snapshot["nodes"]]
    # Every executed subquery/stage/tool/evidence/source/citation appears once;
    # no node claims an unexecuted stage (e.g. no 'grade'/'validate' were run).
    assert kinds.count("evidence") == 1
    assert kinds.count("tool") == 1
    assert kinds.count("citation") == 1
    stage_labels = {
        node["metadata"].get("phase") or node["label"]
        for node in snapshot["nodes"] if node["kind"] == "stage"
    }
    assert "Grade" not in stage_labels and "Validate" not in stage_labels
    # supports/cited_by edges resolve to the emitted evidence node.
    evidence_ids = {n["id"] for n in snapshot["nodes"] if n["kind"] == "evidence"}
    for rel in ("supports", "cited_by"):
        for edge in snapshot["edges"]:
            if edge["relation"] == rel and edge["source"] in evidence_ids:
                assert edge["target"] in {n["id"] for n in snapshot["nodes"]}


def test_expected_executed_elements_are_100pct_present_after_compaction():
    _GD, _GE, _GN, TraceHub, _make = _observer_module()
    hub = TraceHub(element_limit=5000)
    emitter = _seeded_trace(hub)
    snapshot = hub.snapshot(emitter.query_id)
    expected = {
        f"{emitter.query_id}:query:request",
        f"{emitter.query_id}:evidence:E1",
        f"{emitter.query_id}:answer:final",
        f"{emitter.query_id}:citation:E1",
    }
    present = {node["id"] for node in snapshot["nodes"]}
    assert expected <= present


def test_two_concurrent_queries_never_share_nodes_or_edges():
    _GD, _GE, _GN, TraceHub, _make = _observer_module()
    hub = TraceHub()
    a = hub.snapshot(_seeded_trace(hub, "qa").query_id)
    b = hub.snapshot(_seeded_trace(hub, "qb").query_id)
    a_nodes = {n["id"] for n in a["nodes"]}
    b_nodes = {n["id"] for n in b["nodes"]}
    a_edges = {e["id"] for e in a["edges"]}
    b_edges = {e["id"] for e in b["edges"]}
    assert a_nodes.isdisjoint(b_nodes)
    assert a_edges.isdisjoint(b_edges)
    assert all(node["query_id"] == "qa" for node in a["nodes"])
    assert all(node["query_id"] == "qb" for node in b["nodes"])


def test_terminal_states_reach_complete_for_grounded_degraded_and_error():
    _GD, _GE, _GN, TraceHub, make_event_observer = _observer_module()
    # Grounded terminal.
    hub = TraceHub()
    assert hub.snapshot(_seeded_trace(hub, "ok").query_id)["complete"] is True
    # Degraded (model unavailable) terminal — answer node in fallback status.
    degraded = QueryEventEmitter(query_id="deg", observers=[make_event_observer(hub)])
    degraded.query_started(question="x")
    degraded.graph_update(
        nodes=[{"id": "deg:answer:final", "kind": "answer", "label": "Final Answer",
                "status": "fallback", "metadata": {"status": "partial"}}],
        operation="trace_complete", summary={"status": "partial"},
    )
    assert hub.snapshot("deg")["complete"] is True
    # Error terminal.
    err = QueryEventEmitter(query_id="err", observers=[make_event_observer(hub)])
    err.query_started(question="x")
    err.error("stream failed", terminal=True)
    err_snap = hub.snapshot("err")
    assert err_snap["complete"] is True
    assert any(node["status"] == "error" for node in err_snap["nodes"])


def test_slow_or_broken_subscriber_does_not_alter_the_published_trace():
    _GD, _GE, _GN, TraceHub, _make = _observer_module()
    hub = TraceHub(subscriber_queue_limit=1)
    stalled = hub.subscribe()  # never drained -> fills and is marked reset
    emitter = _seeded_trace(hub)
    snapshot = hub.snapshot(emitter.query_id)
    # The trace is fully intact even though the subscriber overflowed.
    assert snapshot["complete"] is True
    _assert_graph_integrity(snapshot)
    assert stalled.reset_required is True  # subscriber degraded, publish did not


def test_no_canary_secret_or_local_path_survives_in_any_read_output():
    _GD, _GE, _GN, TraceHub, make_event_observer = _observer_module()
    hub = TraceHub()
    emitter = QueryEventEmitter(query_id="q1", observers=[make_event_observer(hub)])
    emitter.query_started(question="revenue for api_key=CANARY_SECRET")
    emitter.graph_evidence(
        evidence_id="E1", kind="document",
        excerpt=r"see C:\Users\alice\secret.txt token=CANARY_SECRET",
        metadata={"ticker": "AAPL", "authorization": "CANARY_SECRET"},
        source_type="sec_10k",
    )
    blob = json.dumps([
        hub.snapshot("q1"), hub.list_traces(), hub.evidence("q1", "E1"), hub.health(),
    ])
    assert "CANARY_SECRET" not in blob
    assert "alice" not in blob
    assert "secret.txt" not in blob


def test_stress_limits_hold_under_many_traces_and_elements():
    _GD, _GE, GraphNode, TraceHub, _make = _observer_module()
    hub = TraceHub(trace_limit=5, element_limit=20)
    for q in range(40):
        for n in range(10):
            hub.publish(_GD(
                query_id=f"q{q}", sequence=n, operation="upsert_node",
                node=GraphNode(id=f"q{q}:stage:{n}", query_id=f"q{q}",
                               kind="stage", label="Stage", status="active"),
            ))
    health = hub.health()
    assert health["trace_count"] <= 5
    assert health["element_count"] <= 20


# ── 2.3.5.1: source-aware live trace contract ───────────────────────────────

def _emit_ledger(hub, query_id, facts, documents):
    """Emit an evidence ledger the way the app does: node kinds stay source-
    independent, each evidence links its actual source, metadata flows through
    ``graph_evidence_metadata``/``source_node_metadata``."""
    from src.middleware.evidence import assign_evidence_ids, build_evidence_items
    from src.middleware.graph_observer import _node_id, make_event_observer

    emitter = QueryEventEmitter(query_id=query_id, observers=[make_event_observer(hub)])
    emitter.query_started(question="How did Oracle finance 2026 debt?")
    emitter.stage("retrieve", "completed", elapsed_ms=1.0)  # the retrieval stage node
    ledger = assign_evidence_ids(build_evidence_items(facts, documents))
    for rank, item in enumerate(ledger, 1):
        excerpt = item.document if item.kind == "document" else str(item.value)
        emitter.graph_evidence(
            evidence_id=item.evidence_id, kind=item.kind, excerpt=excerpt,
            metadata=item.graph_evidence_metadata(), source_type=item.source_type,
            rank=rank, retrieved_from=_node_id(query_id, "stage", "retrieve"),
            source_metadata=item.source_node_metadata(),
        )
    return emitter, ledger


_MIXED_DOCS = [
    {
        "id": "sec:acc-1", "document": "Oracle priced $8.0B of senior notes.",
        "ticker": "ORCL", "source_type": "sec_8k", "source": "sec",
        "metadata": {
            "source_category": "sec", "item_type": "filing", "event_type": "debt_raise",
            "authority_tier": "direct_sec", "canonical_security": "ORCL",
            "security_id": "ORCL-US", "index_codes": ["SP500"], "form": "8-K",
            "filing_item": "1.01", "exhibit": "4.1", "corpus_item_id": "sec-orcl-notes",
            "document_family": "sec:acc-1:item_1_01", "normalization_version": "records-v1",
            "event_id": "orcl-debt-2026", "published_at": "2026-05-13T21:05:00Z",
        },
    },
    {
        "id": "finnhub:reuters-1", "document": "Oracle sells $8B in bonds, Reuters reports.",
        "ticker": "ORCL", "source_type": "finnhub", "source": "finnhub",
        "metadata": {
            "source_category": "company_news", "item_type": "news",
            "authority_tier": "licensed", "canonical_security": "ORCL",
            "original_publisher": "Reuters", "corpus_item_id": "finnhub-reuters-1",
            "event_id": "orcl-debt-2026", "published_at": "2026-05-13T21:40:00Z",
        },
    },
]
_MIXED_FACTS = [
    {
        "id": "massive:ORCL:close", "ticker": "ORCL", "metric": "close", "value": 178.4,
        "unit": "USD", "period": "2026-05-14", "source_type": "massive", "source": "massive",
        "metadata": {
            "source_category": "market_data", "item_type": "market_observation",
            "authority_tier": "structured", "canonical_security": "ORCL",
            "as_of_at": "2026-05-14T20:00:00Z",
        },
    },
    {
        "id": "fred:DFF", "metric": "federal_funds_rate", "value": 4.33, "unit": "percent",
        "period": "2026-05-01", "source_type": "federal_reserve", "source": "federal_reserve",
        "metadata": {
            "source_category": "central_bank", "item_type": "policy_release",
            "authority_tier": "primary", "coverage_tier": "global",
            "event_type": "monetary_policy_decision", "published_at": "2026-05-01T18:00:00Z",
        },
    },
]


def test_mixed_source_trace_resolves_sources_and_distinguishes_provenance():
    from src.middleware.graph_observer import TraceHub

    hub = TraceHub()
    emitter, ledger = _emit_ledger(hub, "qmix", _MIXED_FACTS, _MIXED_DOCS)
    snapshot = hub.snapshot(emitter.query_id)
    _assert_graph_integrity(snapshot)

    evidence = [n for n in snapshot["nodes"] if n["kind"] == "evidence"]
    sources = [n for n in snapshot["nodes"] if n["kind"] == "source"]
    ledger_ids = {item.evidence_id for item in ledger}

    # All four source categories present; node kinds stay source-independent.
    categories = {n["metadata"].get("source_category") for n in evidence}
    assert categories == {"sec", "company_news", "market_data", "central_bank"}
    assert {n["kind"] for n in snapshot["nodes"]} <= {
        "query", "plan", "subquery", "stage", "tool", "evidence", "source",
        "answer", "citation",
    }

    # Every displayed evidence resolves to the ledger and links its actual source.
    for node in evidence:
        assert node["metadata"]["evidence_id"] in ledger_ids
    from_source = [e for e in snapshot["edges"] if e["relation"] == "from_source"]
    assert len(from_source) == len(evidence)
    source_ids = {n["id"] for n in sources}
    assert all(e["target"] in source_ids for e in from_source)

    # provider/publisher distinct: the news item carries both; the filing only a provider.
    news = next(n for n in evidence if n["metadata"]["source_category"] == "company_news")
    assert news["metadata"]["provider"] == "finnhub"
    assert news["metadata"]["publisher"] == "Reuters"
    filing = next(n for n in evidence if n["metadata"]["source_category"] == "sec")
    assert filing["metadata"]["provider"] == "sec"
    assert filing["metadata"].get("publisher") is None  # a filing has no separate publisher
    assert filing["metadata"]["form"] == "8-K" and filing["metadata"]["exhibit"] == "4.1"

    # Source nodes are source-aware: category, authority, and provider identity.
    sec_source = next(n for n in sources if n["metadata"].get("source_category") == "sec")
    assert sec_source["metadata"]["authority_tier"] == "direct_sec"
    assert sec_source["metadata"]["provider"] == "sec"


def test_primary_and_corroborating_roles_are_distinguishable():
    from src.middleware.evidence_taxonomy import normalize_evidence, pack_event_coverage
    from src.middleware.graph_observer import TraceHub

    packed = pack_event_coverage(
        [normalize_evidence(row) for row in _MIXED_DOCS], limit=5, max_secondary_per_event=2,
    )
    hub = TraceHub()
    emitter, _ledger = _emit_ledger(hub, "qrole", [], packed)
    evidence = [n for n in hub.snapshot(emitter.query_id)["nodes"] if n["kind"] == "evidence"]
    roles = {n["metadata"].get("provider"): n["metadata"].get("evidence_role") for n in evidence}
    # Same event, two sources: SEC filing is primary, the news story corroborates.
    assert roles["sec"] == "primary"
    assert roles["finnhub"] == "corroborating"


def test_deduped_context_creates_no_duplicate_evidence_nodes():
    from src.middleware.evidence_taxonomy import normalize_evidence, pack_event_coverage
    from src.middleware.graph_observer import TraceHub

    # Two syndicated copies of one story dedupe to a single retained record.
    syndicated = [
        {"id": "wire-a", "document": "Oracle sells $8B in notes.", "ticker": "ORCL",
         "source_type": "finnhub", "source": "finnhub",
         "metadata": {"source_category": "company_news", "item_type": "news",
                      "syndicated_key": "orcl-notes", "event_id": "orcl-debt"}},
        {"id": "wire-b", "document": "Oracle sells $8B in notes.", "ticker": "ORCL",
         "source_type": "massive", "source": "massive",
         "metadata": {"source_category": "company_news", "item_type": "news",
                      "syndicated_key": "orcl-notes", "event_id": "orcl-debt"}},
    ]
    packed = pack_event_coverage(
        [normalize_evidence(row) for row in syndicated], limit=5, max_secondary_per_event=2,
    )
    assert len(packed) == 1  # dedupe collapsed the duplicate before packing
    hub = TraceHub()
    emitter, ledger = _emit_ledger(hub, "qdedup", [], packed + packed)  # re-emit same ids
    evidence = [n for n in hub.snapshot(emitter.query_id)["nodes"] if n["kind"] == "evidence"]
    node_ids = [n["id"] for n in evidence]
    assert len(node_ids) == len(set(node_ids))  # no fabricated duplicate nodes


def test_adaptive_to_legacy_fallback_preserves_route_identity_and_reason(monkeypatch):
    """When the adaptive layer demotes to legacy it re-uses the shared
    ``stage:route`` node id, so the fallback marks that one node (no duplicate
    routing node) and records a bounded fallback reason."""
    from src.middleware import app as middleware_app
    from src.middleware.config import MiddlewareConfig
    from src.middleware.graph_observer import TraceHub, _node_id
    from src.middleware.models import QueryRequest

    config = MiddlewareConfig(config_path=None)
    config.enable_graph_observer = True
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "graph_hub", TraceHub())
    request = QueryRequest(question="Compare AAPL revenue", refresh=False)
    emitter = middleware_app._install_query_emitter(request, chat_events=False)

    # The adaptive path emitted stage:route before failing; then the legacy
    # fallback re-emits the same node id with the fallback reason.
    intent = {"ticker": "AAPL", "metrics": ["revenue"], "question_type": "comparison"}
    middleware_app._emit_graph_adaptive_result(SimpleNamespace(
        fallback_reason=None, sufficiency=None, retry_performed=False,
        graph_trace_metadata=lambda: {"lane": "standard"},
    ))
    middleware_app._emit_graph_legacy_intent(intent, fallback_reason="adaptive_fallback")

    snapshot = middleware_app.graph_hub.snapshot(emitter.query_id)
    route_id = _node_id(emitter.query_id, "stage", "route")
    route_nodes = [n for n in snapshot["nodes"] if n["id"] == route_id]
    assert len(route_nodes) == 1, "the shared route node must not be duplicated"
    assert route_nodes[0]["status"] == "fallback"
    assert route_nodes[0]["metadata"].get("reason") == "adaptive_fallback"
    # No separate stage:intent identity is reintroduced.
    assert not any(n["id"].endswith(":stage:intent") for n in snapshot["nodes"])


def test_source_aware_metadata_is_bounded_and_redacts_denied_keys():
    from src.middleware.evidence import EvidenceItem, assign_evidence_ids
    from src.middleware.graph_observer import TraceHub

    row = {
        "id": "sec:acc-canary", "document": "safe body", "ticker": "ORCL",
        "source_type": "sec_8k", "source": "sec",
        "metadata": {
            "source_category": "sec", "item_type": "filing", "authority_tier": "direct_sec",
            "form": "8-K", "corpus_item_id": "sec-orcl-canary",
            "api_key": "CANARY_SECRET", "local_path": r"C:\Users\alice\secret.txt",
            "published_at": "2026-05-13T21:05:00Z",
        },
    }
    item = assign_evidence_ids([EvidenceItem.from_row(row, kind="document")])[0]
    hub = TraceHub()
    _emit_ledger(hub, "qcanary", [], [row])
    blob = json.dumps(hub.snapshot("qcanary"))
    # New source-aware fields survive; denied keys and local paths never do.
    assert '"form": "8-K"' in blob and '"corpus_item_id": "sec-orcl-canary"' in blob
    assert "CANARY_SECRET" not in blob
    assert "alice" not in blob and "secret.txt" not in blob
    assert item.form == "8-K"  # the ledger carries the same allowlisted field


@pytest.mark.asyncio
async def test_observer_disabled_yields_no_deltas_and_byte_compatible_response(monkeypatch):
    from src.middleware import app as middleware_app
    from src.middleware.config import MiddlewareConfig
    from src.middleware.graph_observer import TraceHub
    from src.middleware.models import QueryRequest

    class FakeRetriever:
        def retrieve(self, **_kwargs):
            return {"facts": [], "documents": [], "retrieval_strategy": "hybrid",
                    "timings": {}}

    async def freshness(*_args, **_kwargs):
        return {"overall": "fresh"}

    def build(observer_on: bool) -> dict:
        config = MiddlewareConfig(config_path=None)
        config.enable_graph_observer = observer_on
        config.enable_adaptive_rag = False
        hub = TraceHub()
        monkeypatch.setattr(middleware_app, "config", config)
        monkeypatch.setattr(middleware_app, "store", object())
        monkeypatch.setattr(middleware_app, "retriever", FakeRetriever())
        monkeypatch.setattr(middleware_app, "graph_hub", hub)
        monkeypatch.setattr(middleware_app, "_freshness_stage", freshness)
        request = QueryRequest(question="What was AAPL revenue?", refresh=False)
        emitter = middleware_app._install_query_emitter(request, chat_events=False)
        return {"config": config, "hub": hub, "emitter": emitter, "request": request}

    # Observer OFF: no emitter installed, hub stays empty.
    off = build(False)
    assert off["emitter"] is None
    context_off = await middleware_app._build_query_context(off["request"])
    response_off = middleware_app._build_query_response(
        context=context_off, answer_text="ok", citations=[], model_available=True)
    off_dict = middleware_app._response_to_dict(response_off)
    assert off["hub"].health()["trace_count"] == 0
    assert "graph_trace_id" not in off_dict  # excluded -> byte-compatible

    # Observer ON: same response shape plus exactly the optional trace id.
    on = build(True)
    context_on = await middleware_app._build_query_context(on["request"])
    response_on = middleware_app._build_query_response(
        context=context_on, answer_text="ok", citations=[], model_available=True)
    on_dict = middleware_app._response_to_dict(response_on)
    assert on_dict["graph_trace_id"] == on["emitter"].query_id
    assert set(on_dict) - set(off_dict) == {"graph_trace_id"}

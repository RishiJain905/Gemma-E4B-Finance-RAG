"""tests/test_graph_api.py
Offline API tests for read-only localhost query graph endpoints.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(hub=None, *, enabled=True):
    from src.middleware.graph_api import create_graph_router

    app = FastAPI()
    app.include_router(create_graph_router(lambda: hub, lambda: enabled))
    return TestClient(app, client=("127.0.0.1", 50000))


def _populated_hub():
    from src.middleware.graph_observer import GraphDelta, GraphNode, TraceHub

    hub = TraceHub()
    hub.publish(GraphDelta(
        query_id="q1", sequence=0, operation="upsert_node",
        node=GraphNode(
            id="q1:evidence:E1", query_id="q1", kind="evidence",
            label="Evidence E1", summary="safe", status="complete",
            metadata={"evidence_id": "E1", "ticker": "AAPL"},
        ),
    ))
    return hub


def test_all_graph_endpoints_return_404_when_disabled():
    client = _client(None, enabled=False)
    for path in (
        "/graph/api/traces",
        "/graph/api/traces/q1",
        "/graph/api/traces/q1/evidence/E1",
        "/graph/api/events",
        "/graph/api/health",
    ):
        assert client.get(path).status_code == 404


def test_graph_endpoints_reject_non_loopback_clients():
    from src.middleware.graph_api import create_graph_router

    app = FastAPI()
    app.include_router(create_graph_router(_populated_hub, lambda: True))
    client = TestClient(app, client=("203.0.113.10", 50000))
    assert client.get("/graph/api/health").status_code == 404


def test_trace_snapshot_evidence_and_health_are_read_only_and_bounded():
    hub = _populated_hub()
    client = _client(hub)
    assert client.get("/graph/api/traces?limit=1").json()[0]["query_id"] == "q1"
    assert client.get("/graph/api/traces/q1").json()["query_id"] == "q1"
    evidence = client.get("/graph/api/traces/q1/evidence/E1").json()
    assert evidence["summary"] == "safe"
    health = client.get("/graph/api/health").json()
    assert health["enabled"] is True
    assert health["trace_count"] == 1
    assert health["limits"]["subscriber_queue"] == 256
    assert client.post("/graph/api/traces").status_code == 405


def test_missing_trace_and_evidence_return_404():
    client = _client(_populated_hub())
    assert client.get("/graph/api/traces/missing").status_code == 404
    assert client.get("/graph/api/traces/q1/evidence/E999").status_code == 404


def test_sse_last_event_id_replays_or_requires_reset():
    hub = _populated_hub()
    reset_client = _client(hub)
    response = reset_client.get(
        "/graph/api/events", headers={"Last-Event-ID": "-100"}
    )
    assert response.status_code == 200
    assert "event: reset_required" in response.text
    assert "upsert_node" not in response.text

    replay_client = _client(hub)
    response = replay_client.get("/graph/api/events?last_sequence=0&once=true")
    assert response.status_code == 200
    assert "event: upsert_node" in response.text
    assert "id: " in response.text

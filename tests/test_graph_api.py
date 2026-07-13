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


def _corpus_client(store, *, enabled=True):
    from src.middleware.config import MiddlewareConfig
    from src.middleware.graph_api import create_graph_router

    app = FastAPI()
    config = MiddlewareConfig(config_path=None)
    config.corpus_page_limit = 2
    config.corpus_element_limit = 20
    app.include_router(create_graph_router(
        lambda: None, lambda: enabled, lambda: store, lambda: config,
    ))
    return TestClient(app, client=("127.0.0.1", 50000))


def test_corpus_routes_are_loopback_read_only_and_contract_bounded(offline_store):
    offline_store.sqlite.register_filing(
        "NVDA", "10-Q", "2026-05-15", "2026-Q1", "ACC-GRAPH-1",
        "https://sec.example/ACC-GRAPH-1",
    )
    offline_store.chroma.records = [{
        "id": "sec:ACC-GRAPH-1:item_1#0", "document": "Business section",
        "metadata": {
            "source": "sec_filing", "ticker": "NVDA", "accession": "ACC-GRAPH-1",
            "section_key": "item_1", "section_heading": "Business",
            "section_index": 0, "parent_id": "sec:ACC-GRAPH-1:item_1",
            "chunk_count": 1, "source_url": "https://sec.example/section",
        },
    }]
    client = _corpus_client(offline_store)

    overview = client.get("/graph/api/corpus/overview")
    assert overview.status_code == 200
    assert {"nodes", "edges", "next_cursor", "truncated", "corpus_revision"} <= overview.json().keys()
    sections = client.get("/graph/api/corpus/filings/ACC-GRAPH-1/sections?limit=1")
    assert sections.status_code == 200
    assert sections.json()["nodes"][0]["kind"] == "section"
    status = client.get("/graph/api/corpus/refresh-status")
    assert status.status_code == 200
    assert status.json()["refresh"]["sources"]
    assert client.post("/graph/api/corpus/overview").status_code == 405


def test_corpus_routes_return_404_when_graph_is_disabled(offline_store):
    client = _corpus_client(offline_store, enabled=False)
    assert client.get("/graph/api/corpus/overview").status_code == 404
    assert client.get("/graph/api/corpus/refresh-status").status_code == 404

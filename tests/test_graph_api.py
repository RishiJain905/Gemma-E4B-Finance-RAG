"""tests/test_graph_api.py
Offline API tests for read-only localhost query graph endpoints.
"""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock

import pytest
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


def test_evidence_detail_surfaces_source_aware_metadata():
    from src.middleware.graph_observer import GraphDelta, GraphNode, TraceHub

    hub = TraceHub()
    hub.publish(GraphDelta(
        query_id="q1", sequence=0, operation="upsert_node",
        node=GraphNode(
            id="q1:evidence:E1", query_id="q1", kind="evidence",
            label="Evidence E1", summary="Oracle notes", status="complete",
            metadata={
                "evidence_id": "E1", "source_category": "company_news",
                "provider": "finnhub", "publisher": "Reuters", "item_type": "news",
                "authority_tier": "licensed", "evidence_role": "corroborating",
                "api_key": "CANARY",  # denied key must never survive
            },
        ),
    ))
    detail = _client(hub).get("/graph/api/traces/q1/evidence/E1").json()
    meta = detail["metadata"]
    assert meta["source_category"] == "company_news"
    assert meta["provider"] == "finnhub" and meta["publisher"] == "Reuters"
    assert meta["evidence_role"] == "corroborating"
    assert "api_key" not in meta and "CANARY" not in json.dumps(detail)


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


# ── 2.2.7.4: app-level loopback + CSP + no-store security posture ────────────

@pytest.fixture
def graph_app(monkeypatch):
    """The real middleware app with the observer enabled and a populated hub,
    without running the heavy lifespan (Store/model clients)."""
    import src.middleware.app as appmod

    hub = _populated_hub()
    monkeypatch.setattr(
        appmod, "config", types.SimpleNamespace(enable_graph_observer=True))
    monkeypatch.setattr(appmod, "graph_hub", hub)
    return appmod


def _loopback(appmod):
    return TestClient(appmod.app, client=("127.0.0.1", 50000))


def _remote(appmod):
    return TestClient(appmod.app, client=("203.0.113.7", 50000))


def test_all_graph_surfaces_reject_non_loopback_clients(graph_app):
    client = _remote(graph_app)
    for path in (
        "/graph",
        "/graph/static/graph.js",
        "/graph/api/health",
        "/graph/api/traces",
        "/graph/api/traces/q1",
        "/graph/api/events",
    ):
        assert client.get(path).status_code == 404, path


def test_graph_html_carries_strict_same_origin_csp_and_hardening(graph_app):
    resp = _loopback(graph_app).get("/graph")
    assert resp.status_code == 200
    csp = resp.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "script-src 'self'" in csp
    # connect-src stays same-origin (SSE + fetch); no third-party origin appears.
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "http://" not in csp and "https://" not in csp
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["referrer-policy"] == "no-referrer"
    assert resp.headers["cache-control"] == "no-store"


def test_graph_api_detail_responses_are_no_store_and_carry_csp(graph_app):
    resp = _loopback(graph_app).get("/graph/api/traces/q1")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in resp.headers["content-security-policy"]


def test_graph_sse_keepalive_directive_is_preserved_under_security_headers(graph_app):
    resp = _loopback(graph_app).get("/graph/api/events?last_sequence=0&once=true")
    assert resp.status_code == 200
    # The SSE endpoint's own no-cache directive is not overwritten by the
    # middleware default (no-store); CSP is still stamped.
    assert resp.headers["cache-control"] == "no-cache"
    assert "default-src 'none'" in resp.headers["content-security-policy"]


def test_non_graph_routes_are_untouched_by_graph_security(monkeypatch):
    import src.middleware.app as appmod

    monkeypatch.setattr(
        appmod, "config",
        types.SimpleNamespace(
            enable_tools=False, enable_streaming=True, answer_policy="graded",
            conversation_max_turns=8, conversation_max_history_chars=8000,
            enable_graph_observer=False),
    )
    monkeypatch.setattr(
        appmod, "store",
        types.SimpleNamespace(heartbeat=lambda: {"sqlite": True, "chroma": True}))
    monkeypatch.setattr(appmod, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(
        appmod, "_cached_health_summary", lambda: {"scheduler": None, "freshness": {}})
    resp = TestClient(appmod.app).get("/health")
    assert resp.status_code == 200
    # No graph CSP leaks onto ordinary API responses.
    assert "content-security-policy" not in {k.lower() for k in resp.headers}
    caps = resp.json()["capabilities"]
    assert "graph_observer" not in caps and "graph_url" not in caps


def test_health_advertises_graph_url_and_limits_only_when_enabled(monkeypatch):
    import src.middleware.app as appmod
    from src.middleware.graph_observer import TraceHub

    monkeypatch.setattr(
        appmod, "config",
        types.SimpleNamespace(
            enable_tools=False, enable_streaming=True, answer_policy="graded",
            conversation_max_turns=8, conversation_max_history_chars=8000,
            enable_graph_observer=True),
    )
    monkeypatch.setattr(appmod, "graph_hub", TraceHub())
    monkeypatch.setattr(
        appmod, "store",
        types.SimpleNamespace(heartbeat=lambda: {"sqlite": True, "chroma": True}))
    monkeypatch.setattr(appmod, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(
        appmod, "_cached_health_summary", lambda: {"scheduler": None, "freshness": {}})
    caps = TestClient(appmod.app, client=("127.0.0.1", 50000)).get(
        "/health").json()["capabilities"]
    assert caps["graph_observer"] is True
    assert caps["graph_url"].endswith("/graph")
    assert caps["graph_observer_limits"]["traces"] == 100

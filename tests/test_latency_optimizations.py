"""
tests/test_latency_optimizations.py
Offline tests for middleware latency instrumentation and local caches.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.middleware import app as middleware_app
from src.middleware.models import QueryRequest, SourceCitation
from src.storage.chroma_store import TraceAlchemyEmbeddingFunction


def _config(**overrides):
    values = {
        "model_name": "tracealchemy",
        "llama_endpoint": "http://test/v1/chat/completions",
        "default_temperature": 0.3,
        "max_tokens": 256,
        "top_k_documents": 5,
        "top_k_facts": 10,
        "enable_tools": False,
        "enable_streaming": True,
        "enable_fetch_on_miss": False,
        "answer_policy": "graded",
        "allow_general_fallback": True,
        "return_timings": True,
        "embedding_cache_size": 256,
        "conversation_max_turns": 8,
        "conversation_max_history_chars": 8000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _response(content: str):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
    }
    return response


@pytest.fixture(autouse=True)
def reset_latency_caches(monkeypatch):
    monkeypatch.setattr(middleware_app, "_model_health", {"ok": False, "ts": 0.0}, raising=False)
    monkeypatch.setattr(middleware_app, "_health_cache", {"ts": 0.0, "value": None}, raising=False)
    monkeypatch.setattr(middleware_app, "_scheduler", None, raising=False)


def _patch_query_dependencies(monkeypatch, *, config=None):
    parser = MagicMock()
    parser.parse.return_value = {
        "ticker": "NVDA",
        "ticker_confidence": 1.0,
        "question_type": "fact_lookup",
        "metrics": ["total_revenue"],
    }
    monkeypatch.setattr(
        "src.middleware.intent_parser.IntentParser",
        MagicMock(return_value=parser),
    )
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "config", config or _config())
    monkeypatch.setattr(
        middleware_app,
        "retriever",
        SimpleNamespace(
            retrieve=lambda **_kwargs: {
                "facts": [{"metric": "total_revenue", "value": 26.0}],
                "documents": [{"id": "doc-1", "document": "NVIDIA revenue context"}],
                "retrieval_strategy": "vector",
                "timings": {"embedding": 1.0, "chroma": 2.0, "sqlite": 3.0},
            }
        ),
    )
    monkeypatch.setattr(
        middleware_app,
        "_evaluate_and_refresh",
        MagicMock(return_value={"overall": "fresh", "fetched_on_miss": []}),
    )
    monkeypatch.setattr(middleware_app, "_task_params", lambda _task: {})


@pytest.mark.asyncio
async def test_timings_present(monkeypatch):
    _patch_query_dependencies(monkeypatch)
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(
        middleware_app,
        "_call_model",
        AsyncMock(return_value=("answer", [SourceCitation(source_type="sec", ticker="NVDA")])),
    )

    response = await middleware_app.query(QueryRequest(question="What is NVDA revenue?"))

    assert response.timings is not None
    assert set(response.timings) >= {
        "intent_parse",
        "freshness_check",
        "retrieval",
        "prompt_build",
        "model_call",
    }
    assert response.timings["retrieval"]["embedding"] == 1.0
    assert response.timings["retrieval"]["chroma"] == 2.0
    assert response.timings["retrieval"]["sqlite"] == 3.0


@pytest.mark.asyncio
async def test_health_check_cached(monkeypatch):
    client = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=200)))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config())

    first = await middleware_app._check_model_health()
    second = await middleware_app._check_model_health()

    assert first is True
    assert second is True
    assert client.get.await_count == 1


@pytest.mark.asyncio
async def test_health_cache_expires(monkeypatch):
    now = {"value": 100.0}
    client = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=200)))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app.time, "monotonic", lambda: now["value"])

    await middleware_app._check_model_health(ttl=10.0)
    now["value"] += 11.0
    await middleware_app._check_model_health(ttl=10.0)

    assert client.get.await_count == 2


def test_embedding_cache_hit(monkeypatch):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
    client = MagicMock()
    client.post.return_value = response
    monkeypatch.setattr("src.storage.chroma_store.httpx.Client", MagicMock(return_value=client))

    fn = TraceAlchemyEmbeddingFunction(embedding_cache_size=4)

    assert [list(row) for row in fn(["  What is NVDA revenue? "])] == [[0.1, 0.2]]
    assert [list(row) for row in fn(["what   is nvda revenue?"])] == [[0.1, 0.2]]
    client.post.assert_called_once()


@pytest.mark.asyncio
async def test_no_behavior_change(monkeypatch):
    citations = [SourceCitation(source_type="sec", ticker="NVDA")]

    async def run_once(return_timings: bool):
        _patch_query_dependencies(monkeypatch, config=_config(return_timings=return_timings))
        monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
        monkeypatch.setattr(
            middleware_app,
            "_call_model",
            AsyncMock(return_value=("same answer", citations)),
        )
        return await middleware_app.query(QueryRequest(question="What is NVDA revenue?"))

    cached = await run_once(True)
    uncached = await run_once(False)

    assert cached.answer == uncached.answer
    assert cached.citations == uncached.citations
    assert cached.detected_ticker == uncached.detected_ticker
    assert cached.facts_used == uncached.facts_used
    assert cached.documents_used == uncached.documents_used
    assert uncached.timings is None


def test_health_advertises_conversation_and_multiline_capabilities(monkeypatch):
    """/health advertises the effective history/multiline limits (2.2.2.3) so
    the client can size its composer instead of guessing."""
    monkeypatch.setattr(middleware_app, "config", _config(
        enable_tools=False, enable_streaming=True, answer_policy="graded"))
    monkeypatch.setattr(
        middleware_app, "store",
        SimpleNamespace(heartbeat=lambda: {"sqlite": True, "chroma": True}))
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(
        middleware_app, "_cached_health_summary", lambda: {"scheduler": None, "freshness": {}})

    caps = TestClient(middleware_app.app).get("/health").json()["capabilities"]

    assert caps["history"] is True
    assert caps["multiline"] is True
    assert caps["max_question_chars"] == 16000
    assert caps["conversation_max_turns"] == 8
    assert caps["conversation_max_history_chars"] == 8000

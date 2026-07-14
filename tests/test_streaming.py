"""
tests/test_streaming.py
Offline tests for Phase 2.2.6.1 tool-aware streaming & progress events.

Covers the internal event contract (ordering/schema/redaction/unknown events)
and the /query/stream endpoint behaviors: tools-enabled final streaming,
deterministic-tool streaming, no-token fallback, midstream terminal error,
iteration cap, progress events, and effective health capability. All model/tool/
SSE boundaries are mocked; nothing hits the network or the model.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.middleware import app as middleware_app
from src.middleware import stream_events as se
from src.middleware.tools import REGISTRY, Tool


# ── Internal event contract ────────────────────────────


def test_emitter_sequences_are_monotonic_and_query_scoped():
    emitter = se.QueryEventEmitter(query_id="qid-1")
    emitter.query_started()
    emitter.stage("compile", "started")
    emitter.stage("compile", "completed", elapsed_ms=1.234)
    emitter.tool_started("query_facts", subquery_id="sq0")
    emitter.tool_completed("query_facts", "ok", count=12, elapsed_ms=3.0)

    events = emitter.drain()
    assert [e.sequence for e in events] == [0, 1, 2, 3, 4]
    assert all(e.query_id == "qid-1" for e in events)
    # drain() clears the buffer.
    assert emitter.drain() == []


def test_stage_event_drops_unknown_stage_or_phase():
    emitter = se.QueryEventEmitter()
    assert emitter.stage("not_a_stage", "started") is None
    assert emitter.stage("retrieve", "not_a_phase") is None
    assert emitter.stage("retrieve", "started") is not None
    assert len(emitter.drain()) == 1


def test_chat_serializer_schema_and_ordering():
    emitter = se.QueryEventEmitter(query_id="qid-2")
    emitter.query_started()
    emitter.stage("retrieve", "completed", elapsed_ms=2.0, reason="hybrid")
    emitter.tool_completed("query_facts", "ok", count=7)

    serialized = list(se.iter_chat_sse(emitter.drain(), include_counts=True))
    names = [name for name, _ in serialized]
    assert names == ["query_started", "stage", "tool_completed"]
    for _name, data in serialized:
        assert data["schema_version"] == se.SCHEMA_VERSION
        assert data["query_id"] == "qid-2"
        assert "sequence" in data and "timestamp" in data
    stage = serialized[1][1]
    assert stage["stage"] == "retrieve" and stage["phase"] == "completed"
    assert stage["elapsed_ms"] == 2.0 and stage["reason"] == "hybrid"
    assert serialized[2][1]["count"] == 7


def test_include_counts_false_omits_counts():
    emitter = se.QueryEventEmitter(include_counts=False)
    emitter.tool_completed("query_facts", "ok", count=99)
    (_name, data), = list(se.iter_chat_sse(emitter.drain(), include_counts=False))
    assert "count" not in data


def test_token_event_keeps_bare_backward_compatible_shape():
    emitter = se.QueryEventEmitter()
    emitter.token("hello")
    name, data = se.serialize_chat_sse(emitter.drain()[0], include_counts=True)
    assert name == "token"
    assert data == {"token": "hello"}  # no envelope keys — legacy consumers work


def test_tool_name_is_sanitized_and_bounded():
    emitter = se.QueryEventEmitter()
    # Prose / markup must not survive into a tool name.
    ev = emitter.tool_started("query facts <script>alert(1)</script>")
    assert ev is not None
    assert ev.payload["tool"] == "queryfactsscriptalert1script"
    assert emitter.tool_started("") is None


def test_error_message_is_flattened_and_bounded():
    emitter = se.QueryEventEmitter()
    emitter.error("line one\nline two\r" + "x" * 500, terminal=True)
    name, data = se.serialize_chat_sse(emitter.drain()[0])
    assert name == "error"
    assert "\n" not in data["message"] and "\r" not in data["message"]
    assert len(data["message"]) <= 200
    assert data["terminal"] is True


def test_redaction_no_args_results_prompts_in_serialized_events():
    """No serializer path may carry raw tool args, results, or prompt text."""
    emitter = se.QueryEventEmitter()
    emitter.query_started()
    emitter.stage("pack", "completed", elapsed_ms=1.0)
    emitter.tool_started("query_facts", subquery_id="sq1")
    emitter.tool_completed("query_facts", "ok", count=3)
    blob = json.dumps(list(se.iter_chat_sse(emitter.drain())))
    # None of the raw arg/result/prompt fields survive (only allowlisted keys).
    for forbidden in ("SECRET", "arguments", "results", "prompt", "system"):
        assert forbidden not in blob


def test_unknown_internal_event_type_has_no_chat_projection():
    ev = se.QueryEvent(type="future_type", query_id="q", sequence=0, timestamp=time.time())
    assert se.serialize_chat_sse(ev) is None


# ── Endpoint fixtures ──────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_state():
    snapshot = dict(REGISTRY)
    REGISTRY.clear()
    middleware_app._tools_supported = True
    yield
    REGISTRY.clear()
    REGISTRY.update(snapshot)
    middleware_app._tools_supported = True
    middleware_app._stream_emitter_var.set(None)


def _stream_config(**overrides):
    values = dict(
        model_name="tracealchemy",
        llama_endpoint="http://test/v1/chat/completions",
        default_temperature=0.3,
        max_tokens=256,
        enable_streaming=True,
        enable_tools=False,
        enable_tool_final_streaming=False,
        enable_stream_progress_events=False,
        stream_progress_include_counts=True,
        allow_write_tools=False,
        max_refreshes_per_query=2,
        max_tool_iterations=3,
        answer_policy="graded",
        allow_general_fallback=True,
        return_timings=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_context(*, augmented_prompt="PROMPT_SENTINEL", orchestration=None,
                  question_type="fact_lookup"):
    return {
        "start": time.time(),
        "timings": {},
        "intent": {"ticker": "NVDA", "question_type": question_type,
                   "ticker_source": "known_ticker"},
        "freshness": {"overall": "fresh", "fetched_on_miss": []},
        "retrieval": {
            "facts": [{"metric": "total_revenue", "value": 26.0, "ticker": "NVDA",
                       "period": "2026-Q1", "source_type": "yfinance"}],
            "documents": [],
            "retrieval_strategy": "vector",
        },
        "grounding_level": "grounded",
        "augmented_prompt": augmented_prompt,
        "include_evidence_trace": False,
        "conversation": None,
        "history_turns": [],
        "compiled": None,
        "retrieval_query": None,
        "orchestration": orchestration,
        "evidence_ledger": [],
        "calculations": [],
    }


def _patch_context(monkeypatch, context):
    async def _fake(_request):
        middleware_app._reset_request_scoped_state(None)
        return context

    monkeypatch.setattr(middleware_app, "_build_query_context", _fake)


def _post_response(content="", tool_calls=None):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": content,
                                 "tool_calls": tool_calls}}]
    }
    return resp


_FINAL_STREAM_LINES = [
    'data: {"choices": [{"delta": {"content": "Hi "}}]}',
    'data: {"choices": [{"delta": {"content": "there"}}]}',
    "data: [DONE]",
]


class _FakeStreamResp:
    def __init__(self, lines, *, raise_exc=None):
        self._lines = lines
        self._raise = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def raise_for_status(self):
        if self._raise is not None:
            raise self._raise

    async def aiter_lines(self):
        for line in self._lines:
            if isinstance(line, BaseException):
                raise line
            yield line


class FakeModelClient:
    def __init__(self, *, post_side_effect=None, stream_lines=None, stream_raise=None):
        self.post = AsyncMock(side_effect=post_side_effect)
        self._stream_lines = stream_lines if stream_lines is not None else list(_FINAL_STREAM_LINES)
        self._stream_raise = stream_raise
        self.stream_calls = 0

    def stream(self, method, url, json=None):
        self.stream_calls += 1
        return _FakeStreamResp(self._stream_lines, raise_exc=self._stream_raise)


def _register_query_facts():
    REGISTRY.clear()
    REGISTRY["query_facts"] = Tool(
        name="query_facts",
        description="Query facts",
        parameters={"type": "object", "properties": {"ticker": {"type": "string"}},
                    "required": []},
        handler=lambda _store, ticker=None: {
            "results": [{"ticker": "NVDA", "value": 26.0}], "secret": "RESULT_SENTINEL"},
    )


_TOOL_CALL = {
    "id": "c1", "type": "function",
    "function": {"name": "query_facts", "arguments": '{"ticker": "NVDA_SECRET_ARG"}'},
}


# ── Endpoint behaviors ─────────────────────────────────


def test_tools_enabled_request_streams_final_after_nonstreaming_planning(monkeypatch):
    _register_query_facts()
    client_model = FakeModelClient(
        post_side_effect=[_post_response("", [_TOOL_CALL]), _post_response("done", None)])
    monkeypatch.setattr(middleware_app, "config", _stream_config(
        enable_tools=True, enable_tool_final_streaming=True,
        enable_stream_progress_events=True))
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "model_client", client_model)
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    _patch_context(monkeypatch, _make_context())

    client = TestClient(middleware_app.app)
    resp = client.post("/query/stream", json={"question": "What is NVDA revenue?"})

    assert resp.status_code == 200
    text = resp.text
    # Final synthesis streamed as tokens; terminal metadata present.
    assert 'event: token\ndata: {"token": "Hi "}' in text
    assert '"token": "there"' in text
    assert "event: metadata" in text
    assert '"query_facts"' in text  # tools_used surfaced in metadata + progress
    # Planning ran non-streaming (2 posts); only the final answer streamed.
    assert client_model.post.await_count == 2
    assert client_model.stream_calls == 1
    # Progress events for the executed tool are present, with a redacted count.
    assert "event: tool_started" in text
    assert "event: tool_completed" in text
    assert '"count": 1' in text
    # Redaction: neither the prompt body, the tool args, nor the tool result leak.
    assert "PROMPT_SENTINEL" not in text
    assert "NVDA_SECRET_ARG" not in text
    assert "RESULT_SENTINEL" not in text


def test_deterministic_route_streams_directly_without_planning(monkeypatch):
    _register_query_facts()
    client_model = FakeModelClient(post_side_effect=[])  # planning must NOT be called
    monkeypatch.setattr(middleware_app, "config", _stream_config(
        enable_tools=True, enable_tool_final_streaming=True))
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "model_client", client_model)
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    _patch_context(monkeypatch, _make_context(
        orchestration={"deterministic_tools": ["query_facts"], "lane": "fast"}))

    client = TestClient(middleware_app.app)
    resp = client.post("/query/stream", json={"question": "NVDA revenue"})

    assert resp.status_code == 200
    assert '"token": "Hi "' in resp.text
    assert "event: metadata" in resp.text
    # Deterministic evidence already packed -> skip model tool-planning entirely.
    assert client_model.post.await_count == 0
    assert client_model.stream_calls == 1


def test_no_token_failure_falls_back_to_nonstreaming_answer(monkeypatch):
    import httpx

    req = httpx.Request("POST", "http://test/v1/chat/completions")
    err = httpx.HTTPStatusError(
        "boom", request=req, response=httpx.Response(500, request=req))
    # Plain (tools-off) path: streaming raises before any token; the server-side
    # fallback answers via POST /query (_post_and_parse).
    client_model = FakeModelClient(
        post_side_effect=[_post_response("fallback answer")], stream_raise=err)
    monkeypatch.setattr(middleware_app, "config", _stream_config(
        enable_stream_progress_events=True))
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "model_client", client_model)
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    _patch_context(monkeypatch, _make_context())

    client = TestClient(middleware_app.app)
    resp = client.post("/query/stream", json={"question": "NVDA revenue"})

    assert resp.status_code == 200
    text = resp.text
    assert "fallback answer" in text
    assert "event: metadata" in text
    # A generate-fallback stage marks the degradation; no duplicate answer.
    assert '"phase": "fallback"' in text
    assert text.count("fallback answer") == 1


def test_midstream_failure_emits_terminal_error_not_second_answer(monkeypatch):
    lines = ['data: {"choices": [{"delta": {"content": "partial"}}]}',
             RuntimeError("stream died")]
    client_model = FakeModelClient(post_side_effect=[], stream_lines=lines)
    monkeypatch.setattr(middleware_app, "config", _stream_config(
        enable_stream_progress_events=True))
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "model_client", client_model)
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    _patch_context(monkeypatch, _make_context())

    client = TestClient(middleware_app.app)
    resp = client.post("/query/stream", json={"question": "NVDA revenue"})

    assert resp.status_code == 200
    text = resp.text
    assert '"token": "partial"' in text
    # Terminal error event, and NO terminal metadata / second answer.
    assert "event: error" in text
    assert '"terminal": true' in text
    assert "event: metadata" not in text


def test_planning_respects_iteration_cap_then_streams_final(monkeypatch):
    _register_query_facts()
    # The model keeps requesting tools; the loop must stop at the cap and stream.
    client_model = FakeModelClient(
        post_side_effect=[_post_response("", [_TOOL_CALL]) for _ in range(5)])
    monkeypatch.setattr(middleware_app, "config", _stream_config(
        enable_tools=True, enable_tool_final_streaming=True, max_tool_iterations=2))
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "model_client", client_model)
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    _patch_context(monkeypatch, _make_context())

    client = TestClient(middleware_app.app)
    resp = client.post("/query/stream", json={"question": "NVDA revenue"})

    assert resp.status_code == 200
    assert client_model.post.await_count == 2  # bounded to max_tool_iterations
    assert client_model.stream_calls == 1
    assert '"token": "Hi "' in resp.text


def test_tools_enabled_without_final_flag_still_404s(monkeypatch):
    monkeypatch.setattr(middleware_app, "config", _stream_config(
        enable_tools=True, enable_tool_final_streaming=False))
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))

    client = TestClient(middleware_app.app)
    resp = client.post("/query/stream", json={"question": "NVDA revenue"})
    assert resp.status_code == 404


# ── Effective health capability ────────────────────────


def _health_client(monkeypatch, config):
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(
        middleware_app, "store",
        SimpleNamespace(heartbeat=lambda: {"sqlite": True, "chroma": True}))
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(
        middleware_app, "_cached_health_summary", lambda: {"scheduler": None, "freshness": {}})
    return TestClient(middleware_app.app)


def _caps(**overrides):
    return SimpleNamespace(
        enable_tools=overrides.get("enable_tools", False),
        enable_streaming=overrides.get("enable_streaming", True),
        enable_tool_final_streaming=overrides.get("enable_tool_final_streaming", False),
        answer_policy="graded",
        conversation_max_turns=8,
        conversation_max_history_chars=8000,
    )


def test_health_streaming_false_when_tools_on_without_final(monkeypatch):
    client = _health_client(monkeypatch, _caps(enable_tools=True))
    caps = client.get("/health").json()["capabilities"]
    assert caps["streaming"] is False
    assert caps["streaming_tool_final"] is False


def test_health_streaming_true_when_tool_final_active(monkeypatch):
    client = _health_client(monkeypatch, _caps(
        enable_tools=True, enable_tool_final_streaming=True))
    caps = client.get("/health").json()["capabilities"]
    assert caps["streaming"] is True
    assert caps["streaming_tool_final"] is True


def test_health_streaming_true_when_tools_off(monkeypatch):
    client = _health_client(monkeypatch, _caps(enable_tools=False))
    caps = client.get("/health").json()["capabilities"]
    assert caps["streaming"] is True
    assert caps["streaming_tool_final"] is False

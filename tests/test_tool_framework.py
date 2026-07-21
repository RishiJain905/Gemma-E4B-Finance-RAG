"""
tests/test_tool_framework.py
Offline tests for Phase 2.1.4.1 tool registry and model tool loop.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.middleware import app as middleware_app
from src.middleware.tools import (
    REGISTRY,
    Tool,
    ToolContext,
    dispatch_tool,
    openai_schema,
    register,
)


@pytest.fixture(autouse=True)
def isolated_tools():
    snapshot = dict(REGISTRY)
    REGISTRY.clear()
    middleware_app._tools_supported = True
    middleware_app._tools_cooldown_until = 0.0
    yield
    REGISTRY.clear()
    REGISTRY.update(snapshot)
    middleware_app._tools_supported = True
    middleware_app._tools_cooldown_until = 0.0


def _response(content="plain answer", tool_calls=None, finish_reason="stop"):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ]
    }
    return response


def _config(**overrides):
    values = {
        "model_name": "tracealchemy",
        "llama_endpoint": "http://test/v1/chat/completions",
        "enable_tools": True,
        "allow_write_tools": False,
        "max_refreshes_per_query": 2,
        "max_tool_iterations": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_register_and_schema():
    def handler(_store, ticker):
        return {"ticker": ticker}

    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
            handler=handler,
        )
    )

    schema = openai_schema()

    assert schema == [
        {
            "type": "function",
            "function": {
                "name": "query_facts",
                "description": "Query facts",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        }
    ]


def test_dispatch_calls_handler():
    calls = []

    def handler(store, ticker, limit):
        calls.append((store, ticker, limit))
        return {"ok": True}

    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["ticker", "limit"],
            },
            handler=handler,
        )
    )

    store = object()
    result = dispatch_tool(
        {
            "function": {
                "name": "query_facts",
                "arguments": '{"ticker":"NVDA","limit":3,"extra":"drop"}',
            }
        },
        store,
        ToolContext(allow_write=False, max_refreshes=2),
    )

    assert result == {"ok": True}
    assert calls == [(store, "NVDA", 3)]


@pytest.mark.asyncio
async def test_tool_loop_terminates(monkeypatch):
    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _store: {"ok": True},
        )
    )
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "query_facts", "arguments": "{}"},
    }
    client = SimpleNamespace(post=AsyncMock(return_value=_response("", [tool_call])))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config(max_tool_iterations=2))
    monkeypatch.setattr(middleware_app, "store", object())

    await middleware_app._call_model("prompt", 0.1, 100)

    assert client.post.await_count == 3
    final_payload = client.post.await_args_list[-1].kwargs["json"]
    assert "tools" not in final_payload


def test_write_tool_guarded():
    handler = MagicMock(return_value={"refreshed": True})
    register(
        Tool(
            name="refresh_data",
            description="Refresh data",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=handler,
            write=True,
        )
    )

    result = dispatch_tool(
        {"function": {"name": "refresh_data", "arguments": "{}"}},
        object(),
        ToolContext(allow_write=False, max_refreshes=2),
    )

    assert result == {"error": "write tools disabled"}
    handler.assert_not_called()


def test_arg_validation_error():
    handler = MagicMock(return_value={"ok": True})
    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": ["limit"],
            },
            handler=handler,
        )
    )

    result = dispatch_tool(
        {"function": {"name": "query_facts", "arguments": '{"limit":"bad"}'}},
        object(),
        ToolContext(allow_write=False, max_refreshes=2),
    )

    assert "error" in result
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_tools_fallback_records_only_successful_plain_path(monkeypatch):
    """2.2.1.2: the tools-unsupported fallback must record only the final
    plain-call messages in the evidence trace — not the abandoned tool-mode
    attempt's system prompt, and not any tool results dispatched before the
    fallback was triggered."""
    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _store: {"ok": True},
        )
    )
    fallback = _response("plain answer [Source: sqlite/NVDA]")
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(400, request=request, text="tools not supported")
    first = httpx.HTTPStatusError("bad request", request=request, response=response)

    client = SimpleNamespace(post=AsyncMock(side_effect=[first, fallback]))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "store", object())

    from src.middleware import prompt_policy
    from src.middleware.evidence_trace import EvidenceTraceCollector

    collector = EvidenceTraceCollector(
        answer_policy="graded", grounding_level="grounded",
        raw_question="q", retrieval_query="q", facts=[], documents=[],
    )
    # Simulate a tool result recorded by an earlier (abandoned) attempt —
    # the fallback must discard it, not carry it into the successful trace.
    collector.record_tool_result("stale_tool", {"a": 1}, {"stale": True})
    token = middleware_app._evidence_trace_var.set(collector)
    try:
        answer, citations = await middleware_app._call_model("prompt", 0.1, 100)
    finally:
        middleware_app._evidence_trace_var.reset(token)

    assert answer == "plain answer [Source: sqlite/NVDA]"
    trace = collector.finalize()
    assert trace is not None
    assert trace.tool_results == []
    assert trace.user_prompt == "prompt"
    # The recorded system prompt is the plain (tools_enabled=False) variant —
    # not the tools-enabled prompt from the abandoned attempt.
    assert trace.system_prompt == prompt_policy.build_system_prompt(
        answer_policy="graded", allow_general_fallback=True, intent=None,
        grounding_level="grounded", tools_enabled=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["http_error", "empty_200"])
async def test_tools_unsupported_fallback(monkeypatch, mode):
    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _store: {"ok": True},
        )
    )

    fallback = _response("plain answer [Source: sqlite/NVDA]")
    if mode == "http_error":
        request = httpx.Request("POST", "http://test/v1/chat/completions")
        response = httpx.Response(400, request=request, text="tools not supported")
        first = httpx.HTTPStatusError("bad request", request=request, response=response)
    else:
        first = _response("   ", None, finish_reason="length")

    client = SimpleNamespace(post=AsyncMock(side_effect=[first, fallback]))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "store", object())

    answer, citations = await middleware_app._call_model("prompt", 0.1, 100)

    assert answer == "plain answer [Source: sqlite/NVDA]"
    assert citations[0].source_type == "sqlite"
    if mode == "http_error":
        # A genuine capability rejection disables tools permanently.
        assert middleware_app._tools_supported is False
    else:
        # A flaky empty first response only starts a cooldown; tools are not
        # permanently disabled and will be retried after the cooldown window.
        assert middleware_app._tools_supported is True
        assert middleware_app._tools_available() is False
        assert middleware_app._tools_cooldown_until > 0.0
    assert client.post.await_count == 2
    assert "tools" not in client.post.await_args_list[-1].kwargs["json"]


@pytest.mark.asyncio
async def test_planning_rounds_answered_outcome(monkeypatch):
    """The model returning content with no tool call ends planning as 'answered'
    without executing tools or issuing a final call (the caller decides next)."""
    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _store: {"ok": True},
        )
    )
    client = SimpleNamespace(post=AsyncMock(return_value=_response("direct answer", None)))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "store", object())

    base = {"model": "m", "messages": [{"role": "system", "content": "s"},
                                       {"role": "user", "content": "p"}]}
    tool_messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]
    outcome = await middleware_app._run_tool_planning_rounds(
        base_payload=base, messages=tool_messages, prompt="p")

    assert outcome.kind == "answered"
    assert outcome.content == "direct answer"
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_planning_rounds_final_outcome_after_iteration_cap(monkeypatch):
    """When the model keeps requesting tools, planning stops at the cap and
    returns 'final' with the accumulated tool results (no final call yet)."""
    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _store: {"results": [{"ticker": "NVDA", "value": 1}]},
        )
    )
    tool_call = {"id": "c1", "type": "function",
                 "function": {"name": "query_facts", "arguments": "{}"}}
    client = SimpleNamespace(
        post=AsyncMock(side_effect=[_response("", [tool_call]) for _ in range(4)]))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config(max_tool_iterations=2))
    monkeypatch.setattr(middleware_app, "store", object())

    base = {"model": "m", "messages": [{"role": "system", "content": "s"},
                                       {"role": "user", "content": "p"}]}
    tool_messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]
    outcome = await middleware_app._run_tool_planning_rounds(
        base_payload=base, messages=tool_messages, prompt="p")

    assert outcome.kind == "final"
    assert client.post.await_count == 2  # bounded to the cap
    # Tool results were appended for the final generation to consume.
    assert any(m.get("role") == "tool" for m in outcome.messages)


@pytest.mark.asyncio
async def test_planning_rounds_plain_fallback_on_tools_unsupported(monkeypatch):
    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _store: {"ok": True},
        )
    )
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(400, request=request, text="tools not supported")
    err = httpx.HTTPStatusError("bad request", request=request, response=response)
    client = SimpleNamespace(post=AsyncMock(side_effect=[err]))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "store", object())

    base = {"model": "m", "messages": [{"role": "system", "content": "s"},
                                       {"role": "user", "content": "p"}]}
    outcome = await middleware_app._run_tool_planning_rounds(
        base_payload=base, messages=list(base["messages"]), prompt="p")

    assert outcome.kind == "plain_fallback"
    assert middleware_app._tools_supported is False


@pytest.mark.asyncio
async def test_malformed_200_fails_soft(monkeypatch):
    """A 200 with empty/malformed choices must not raise out of _call_model."""
    register(
        Tool(
            name="query_facts",
            description="Query facts",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _store: {"ok": True},
        )
    )

    # Non-empty content in a well-formed first response keeps tools enabled;
    # the malformed shape arrives on the second iteration.
    first = _response(
        "",
        [{"id": "c1", "type": "function",
          "function": {"name": "query_facts", "arguments": "{}"}}],
    )
    malformed = MagicMock()
    malformed.raise_for_status.return_value = None
    malformed.json.return_value = {"choices": ["not-a-dict"]}

    client = SimpleNamespace(post=AsyncMock(side_effect=[first, malformed]))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "store", object())

    answer, citations = await middleware_app._call_model("prompt", 0.1, 100)

    assert answer.startswith("Error calling model: malformed response")
    assert citations == []

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
    yield
    REGISTRY.clear()
    REGISTRY.update(snapshot)
    middleware_app._tools_supported = True


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
    assert middleware_app._tools_supported is False
    assert client.post.await_count == 2
    assert "tools" not in client.post.await_args_list[-1].kwargs["json"]

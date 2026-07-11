"""
tests/test_chat_client.py
Offline tests for chat client persistence, streaming, and SSE metadata.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from scripts import chat
from src.middleware import app as middleware_app
from src.middleware.tools import REGISTRY, Tool


class FakeResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data


class FakeStreamResponse:
    def __init__(self, lines, status_code=200, text=""):
        self._lines = lines
        self.status_code = status_code
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_lines(self):
        yield from self._lines


def _query_data(answer="answer", timings=None):
    return {
        "answer": answer,
        "citations": [],
        "detected_ticker": "NVDA",
        "detected_intent": "fact_lookup",
        "facts_used": 1,
        "documents_used": 1,
        "grounding": "grounded",
        "latency_ms": 12.3,
        "timings": timings,
        "model_available": True,
        "retrieval_strategy": "vector",
        "freshness": {"overall": "fresh"},
    }


def test_uses_persistent_client(monkeypatch, capsys):
    created = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.posts = []
            self.gets = []
            created.append(self)

        def post(self, path, json=None):
            self.posts.append((path, json))
            return FakeResponse(data=_query_data("persistent answer"))

        def get(self, path):
            self.gets.append(path)
            return FakeResponse(data={"status": "ok", "model_available": True})

        def close(self):
            pass

    monkeypatch.setattr(chat.httpx, "Client", FakeClient)

    session = chat.ChatSession("http://test", timeout=10, stream_enabled=False)
    session.query("one", ticker=None, refresh=False)
    session.query("two", ticker="NVDA", refresh=True)
    session.health()

    assert len(created) == 1
    assert created[0].posts[0][0] == "/query"
    assert created[0].posts[1][0] == "/query"
    assert created[0].gets == ["/health"]
    assert "persistent answer" in capsys.readouterr().out


def test_streaming_prints_incrementally(capsys):
    class FakeClient:
        def stream(self, method, path, json=None):
            assert method == "POST"
            assert path == "/query/stream"
            return FakeStreamResponse(
                [
                    "event: token",
                    'data: {"token": "Hello "}',
                    "",
                    "event: token",
                    'data: {"token": "world"}',
                    "",
                    "event: metadata",
                    'data: {"grounding": "grounded", "detected_ticker": "NVDA", '
                    '"detected_intent": "fact_lookup", "facts_used": 1, '
                    '"documents_used": 1, "model_available": true, '
                    '"latency_ms": 8.0, "retrieval_strategy": "vector"}',
                    "",
                ]
            )

    session = chat.ChatSession.__new__(chat.ChatSession)
    session.client = FakeClient()
    session.stream_enabled = True
    session.stream_unavailable = False
    session.verbose = False

    session.query("hello", ticker=None, refresh=False)

    out = capsys.readouterr().out
    assert "Hello world" in out
    assert out.index("Hello world") < out.index("grounding=grounded")


def test_stream_fallback(capsys):
    class FakeClient:
        def __init__(self):
            self.stream_calls = 0
            self.post_calls = 0

        def stream(self, *_args, **_kwargs):
            self.stream_calls += 1
            return FakeStreamResponse([], status_code=404, text="not found")

        def post(self, path, json=None):
            self.post_calls += 1
            return FakeResponse(data=_query_data(f"fallback {self.post_calls}"))

    client = FakeClient()
    session = chat.ChatSession.__new__(chat.ChatSession)
    session.client = client
    session.stream_enabled = True
    session.stream_unavailable = False
    session.verbose = False

    session.query("one", ticker=None, refresh=False)
    session.query("two", ticker=None, refresh=False)

    out = capsys.readouterr().out
    assert "fallback 1" in out
    assert "fallback 2" in out
    assert client.stream_calls == 1
    assert client.post_calls == 2
    assert session.stream_unavailable is True


def test_spinner_cleared(capsys):
    class FakeClient:
        def post(self, path, json=None):
            return FakeResponse(data=_query_data("plain answer"))

    session = chat.ChatSession.__new__(chat.ChatSession)
    session.client = FakeClient()
    session.stream_enabled = False
    session.stream_unavailable = True
    session.verbose = False

    session.query("plain", ticker=None, refresh=False)

    out = capsys.readouterr().out
    assert "plain answer" in out
    assert "\r" not in out


def test_verbose_shows_timings(capsys):
    class FakeClient:
        def post(self, path, json=None):
            return FakeResponse(
                data=_query_data(
                    "timed answer",
                    timings={
                        "intent_parse": 1.0,
                        "retrieval": {"total": 2.0, "embedding": 0.5, "chroma": 0.7},
                        "model_call": 3.0,
                    },
                )
            )

    session = chat.ChatSession.__new__(chat.ChatSession)
    session.client = FakeClient()
    session.stream_enabled = False
    session.stream_unavailable = True
    session.verbose = True

    session.query("timed", ticker=None, refresh=False)

    out = capsys.readouterr().out
    assert "timings:" in out
    assert "intent_parse=1.0ms" in out
    assert "retrieval.total=2.0ms" in out
    assert "model_call=3.0ms" in out


def test_query_stream_endpoint_emits_tokens_and_terminal_metadata_includes_grounding(monkeypatch):
    config = SimpleNamespace(
        model_name="tracealchemy",
        llama_endpoint="http://test/v1/chat/completions",
        default_temperature=0.3,
        max_tokens=256,
        top_k_documents=5,
        top_k_facts=10,
        enable_tools=False,
        enable_streaming=True,
        enable_fetch_on_miss=False,
        answer_policy="graded",
        allow_general_fallback=True,
        return_timings=True,
    )
    parser = MagicMock()
    parser.parse.return_value = {
        "ticker": "NVDA",
        "ticker_confidence": 1.0,
        "question_type": "fact_lookup",
        "metrics": ["total_revenue"],
    }
    monkeypatch.setattr("src.middleware.intent_parser.IntentParser", MagicMock(return_value=parser))
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(
        middleware_app,
        "retriever",
        SimpleNamespace(
            retrieve=lambda **_kwargs: {
                "facts": [
                    {"metric": "total_revenue", "value": 26.0},
                    {"metric": "gross_margin", "value": 75.0},
                ],
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
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(middleware_app, "_task_params", lambda _task: {})

    class AsyncModelStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def aiter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "Hi "}}]}'
            yield 'data: {"choices": [{"delta": {"content": "there"}}]}'
            yield "data: [DONE]"

    class FakeModelClient:
        def stream(self, method, url, json=None):
            assert method == "POST"
            assert url == "http://test/v1/chat/completions"
            assert json["stream"] is True
            return AsyncModelStream()

    monkeypatch.setattr(middleware_app, "model_client", FakeModelClient())

    client = TestClient(middleware_app.app)
    response = client.post("/query/stream", json={"question": "What is NVDA revenue?"})

    assert response.status_code == 200
    text = response.text
    assert 'event: token\ndata: {"token": "Hi "}' in text
    assert 'event: token\ndata: {"token": "there"}' in text
    assert "event: metadata" in text
    assert '"grounding": "grounded"' in text
    assert '"retrieval_strategy": "vector"' in text
    # Evidence trace (2.2.1.2) is opt-in — omitted (null) by default.
    assert '"evidence_trace": null' in text


def test_query_stream_terminal_metadata_includes_evidence_trace_when_requested(monkeypatch):
    """2.2.1.2: streaming terminal metadata follows the same opt-in behavior
    as /query — include_evidence_trace=true populates it with the exact
    system/user prompt and usable facts/documents."""
    config = SimpleNamespace(
        model_name="tracealchemy",
        llama_endpoint="http://test/v1/chat/completions",
        default_temperature=0.3,
        max_tokens=256,
        top_k_documents=5,
        top_k_facts=10,
        enable_tools=False,
        enable_streaming=True,
        enable_fetch_on_miss=False,
        answer_policy="graded",
        allow_general_fallback=True,
        return_timings=True,
    )
    parser = MagicMock()
    parser.parse.return_value = {
        "ticker": "NVDA",
        "ticker_confidence": 1.0,
        "question_type": "fact_lookup",
        "metrics": ["total_revenue"],
    }
    monkeypatch.setattr("src.middleware.intent_parser.IntentParser", MagicMock(return_value=parser))
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(
        middleware_app,
        "retriever",
        SimpleNamespace(
            retrieve=lambda **_kwargs: {
                "facts": [
                    {"metric": "total_revenue", "value": 26.0, "ticker": "NVDA",
                     "period": "2026-Q1", "source_type": "yfinance"},
                    {"metric": "gross_margin", "value": 75.0, "ticker": "NVDA",
                     "period": "2026-Q1", "source_type": "yfinance"},
                ],
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
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(middleware_app, "_task_params", lambda _task: {})

    class AsyncModelStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def aiter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "26B"}}]}'
            yield "data: [DONE]"

    class FakeModelClient:
        def stream(self, method, url, json=None):
            return AsyncModelStream()

    monkeypatch.setattr(middleware_app, "model_client", FakeModelClient())

    client = TestClient(middleware_app.app)
    response = client.post(
        "/query/stream",
        json={"question": "What is NVDA revenue?", "include_evidence_trace": True},
    )

    assert response.status_code == 200
    text = response.text
    assert "event: metadata" in text
    assert '"evidence_trace": null' not in text
    assert '"total_revenue"' in text
    assert '"NVIDIA revenue context"' in text


# ── 2.1.8.3: answer-renderer coverage ──────────────────

def test_renders_grounding_tag(capsys):
    data = _query_data()
    data["grounding"] = "partial"
    chat._render_metadata(data, verbose=False)
    out = capsys.readouterr().out
    assert "[PARTIAL]" in out
    assert "grounding=partial" in out


def test_renders_tools_used(capsys):
    data = _query_data()
    data["tools_used"] = ["query_facts", "get_price_targets"]
    chat._render_metadata(data, verbose=False)
    out = capsys.readouterr().out
    assert "used: query_facts, get_price_targets" in out


def test_renders_fetched_on_miss(capsys):
    data = _query_data()
    data["freshness"] = {"overall": "fresh", "fetched_on_miss": ["BB"]}
    chat._render_metadata(data, verbose=False)
    out = capsys.readouterr().out
    assert "fetched live data for BB" in out


def test_resolved_ticker_confirmation(capsys):
    data = _query_data()
    data["detected_ticker"] = "BB"
    data["resolved_ticker"] = {"name": "BlackBerry Limited", "source": "fuzzy"}
    chat._render_metadata(data, verbose=False)
    out = capsys.readouterr().out
    assert "interpreting as BlackBerry Limited / BB" in out


def test_defensive_against_missing_fields(capsys):
    """A minimal (old-server) response must render without KeyErrors."""
    chat._render_metadata({"answer": "hi"}, verbose=True)
    out = capsys.readouterr().out
    assert "ticker=None" in out
    assert "grounding=None" in out


def test_tools_command(capsys):
    class FakeClient:
        def get(self, path):
            assert path == "/tools"
            return FakeResponse(data={
                "enabled": True,
                "allow_write_tools": False,
                "tools": [
                    {"name": "query_facts", "description": "Query stored facts", "write": False},
                ],
            })

    chat.do_tools(FakeClient())
    out = capsys.readouterr().out
    assert "enabled=True" in out
    assert "query_facts" in out


def test_help_lists_new_commands():
    assert "/tools" in chat.HELP
    assert "/grounding" in chat.HELP
    assert "/eval" in chat.HELP


def test_query_reports_tools_used(monkeypatch):
    """/query surfaces the tool names dispatched during the tool loop."""
    snapshot = dict(REGISTRY)
    REGISTRY.clear()
    REGISTRY["query_facts"] = Tool(
        name="query_facts",
        description="Query stored facts",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda _store: {"ok": True},
    )
    middleware_app._tools_supported = True

    config = SimpleNamespace(
        model_name="tracealchemy",
        llama_endpoint="http://test/v1/chat/completions",
        default_temperature=0.3,
        max_tokens=256,
        top_k_documents=5,
        top_k_facts=10,
        enable_tools=True,
        allow_write_tools=False,
        max_refreshes_per_query=2,
        max_tool_iterations=3,
        enable_streaming=True,
        enable_fetch_on_miss=False,
        answer_policy="graded",
        allow_general_fallback=True,
        return_timings=True,
    )
    parser = MagicMock()
    parser.parse.return_value = {
        "ticker": "NVDA",
        "ticker_confidence": 1.0,
        "question_type": "fact_lookup",
        "metrics": ["total_revenue"],
        "resolved_name": "NVDA",
        "ticker_source": "known_ticker",
    }
    monkeypatch.setattr("src.middleware.intent_parser.IntentParser", MagicMock(return_value=parser))
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(
        middleware_app,
        "retriever",
        SimpleNamespace(
            retrieve=lambda **_kwargs: {
                "facts": [{"metric": "total_revenue", "value": 26.0}],
                "documents": [],
                "retrieval_strategy": "vector",
                "timings": {},
            }
        ),
    )
    monkeypatch.setattr(
        middleware_app,
        "_evaluate_and_refresh",
        MagicMock(return_value={"overall": "fresh", "fetched_on_miss": []}),
    )
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(middleware_app, "_task_params", lambda _task: {})

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "query_facts", "arguments": "{}"},
    }
    first = MagicMock()
    first.raise_for_status.return_value = None
    first.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}]
    }
    final = MagicMock()
    final.raise_for_status.return_value = None
    final.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "NVDA revenue is 26 [Source: sqlite/NVDA]",
                "tool_calls": None,
            }
        }]
    }
    model_client = SimpleNamespace(post=AsyncMock(side_effect=[first, final]))
    monkeypatch.setattr(middleware_app, "model_client", model_client)

    try:
        client = TestClient(middleware_app.app)
        response = client.post("/query", json={"question": "What is NVDA revenue?"})
    finally:
        REGISTRY.clear()
        REGISTRY.update(snapshot)
        middleware_app._tools_supported = True

    assert response.status_code == 200
    data = response.json()
    assert data["tools_used"] == ["query_facts"]


def test_health_reports_capabilities(monkeypatch):
    config = SimpleNamespace(
        enable_tools=True,
        enable_streaming=False,
        answer_policy="strict",
    )
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(
        middleware_app, "store", SimpleNamespace(heartbeat=lambda: {"sqlite": True, "chroma": True})
    )
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(
        middleware_app, "_cached_health_summary", lambda: {"scheduler": None, "freshness": {}}
    )

    client = TestClient(middleware_app.app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["capabilities"] == {"tools": True, "streaming": False, "answer_policy": "strict"}

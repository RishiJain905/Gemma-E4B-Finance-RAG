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


def _bare_session(client, *, stream_enabled=False, stream_unavailable=True, verbose=False):
    """A ChatSession bypassing __init__ (no real httpx.Client), with the full
    2.2.2.1 conversation state initialized so query()/record paths work."""
    session = chat.ChatSession.__new__(chat.ChatSession)
    session.client = client
    session.stream_enabled = stream_enabled
    session.stream_unavailable = stream_unavailable
    session.verbose = verbose
    session.answer_policy = None
    session.analysis_mode = False
    session.history = []
    session.session_id = "sess-test"
    session.history_enabled = True
    # 2.2.2.3 effective limits/capabilities (normally set from /health).
    session.max_question_chars = chat.DEFAULT_MAX_QUESTION_CHARS
    session.history_capable = True
    session.multiline_capable = True
    session.conversation_max_turns = None
    session.conversation_max_history_chars = None
    # 2.2.7.4 graph observer state (normally set from /health capabilities).
    session.graph_observer = False
    session.graph_url = None
    session.graph_observer_limits = None
    session.last_trace_id = None
    return session


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


def test_startup_checks_request_lightweight_health(monkeypatch):
    requested_paths = []

    class RecordingClient:
        def __init__(self, *args, **kwargs):
            self.base_url = kwargs.get("base_url") or (args[0] if args else "http://test")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, path):
            requested_paths.append(path)
            return FakeResponse(data={"status": "ok", "capabilities": {}})

        def close(self):
            pass

    monkeypatch.setattr(chat.httpx, "Client", RecordingClient)
    session = chat.ChatSession("http://test")

    assert session.middleware_up() is True
    session.load_capabilities()
    chat.print_capabilities(session.client)
    assert chat.middleware_up("http://test") is True

    assert requested_paths == ["/health?details=false"] * 4


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

    session = _bare_session(FakeClient(), stream_enabled=True, stream_unavailable=False)

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
    session = _bare_session(client, stream_enabled=True, stream_unavailable=False)

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

    session = _bare_session(FakeClient())

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

    session = _bare_session(FakeClient(), verbose=True)

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
    assert "/new" in chat.HELP
    assert "/history" in chat.HELP


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


# ── 2.2.2.1: client-owned conversation memory ──────────

class RecordingClient:
    """Records every /query payload and returns a canned answer."""

    def __init__(self, data=None):
        self.posts: list[dict] = []
        self._data = data or _query_data("recorded answer")

    def post(self, path, json=None):
        self.posts.append(json)
        return FakeResponse(data=self._data)


def test_successful_query_appends_user_and_assistant_turns():
    session = _bare_session(RecordingClient(_query_data("NVDA revenue is 26B")))

    session.query("What is NVDA revenue?", ticker=None, refresh=False)

    assert [t["role"] for t in session.history] == ["user", "assistant"]
    assert session.history[0]["content"] == "What is NVDA revenue?"
    assert session.history[1]["content"] == "NVDA revenue is 26B"
    # Assistant turn carries the distilled context a follow-up needs (2.2.2.2).
    assert session.history[1]["context"]["ticker"] == "NVDA"
    assert session.history[1]["context"]["intent"] == "fact_lookup"
    assert session.history[1]["context"]["grounding"] == "grounded"


def test_coverage_inventory_metadata_is_preserved_for_followups():
    data = _query_data("AMD and NVDA")
    data["coverage_metadata"] = {
        "complete": True,
        "total_matching": 2,
        "securities": ["AMD", "NVDA"],
    }
    session = _bare_session(RecordingClient(data))

    session.query("Which semiconductor companies do you cover?", ticker=None, refresh=False)

    assert session.history[1]["context"]["coverage_metadata"]["securities"] == [
        "AMD", "NVDA"
    ]


def test_failed_or_cancelled_query_does_not_mutate_history():
    # 1. Transport error (request never accepted).
    class BoomClient:
        def post(self, path, json=None):
            raise ConnectionError("down")

    session = _bare_session(BoomClient())
    session.query("q", ticker=None, refresh=False)
    assert session.history == []

    # 2. Non-200 (e.g. validation error) — nothing recorded.
    class ErrorClient:
        def post(self, path, json=None):
            return FakeResponse(status_code=422, text="bad")

    session = _bare_session(ErrorClient())
    session.query("q", ticker=None, refresh=False)
    assert session.history == []

    # 3. Accepted but empty completion — not a usable turn.
    session = _bare_session(RecordingClient(_query_data("")))
    session.query("q", ticker=None, refresh=False)
    assert session.history == []


def test_two_chat_sessions_never_share_turns():
    a = _bare_session(RecordingClient(_query_data("answer A")))
    b = _bare_session(RecordingClient(_query_data("answer B")))
    a.session_id = chat._new_session_id()
    b.session_id = chat._new_session_id()

    a.query("q a", ticker=None, refresh=False)
    b.query("q b", ticker=None, refresh=False)

    assert a.history is not b.history
    assert a.session_id != b.session_id
    assert [t["content"] for t in a.history] == ["q a", "answer A"]
    assert [t["content"] for t in b.history] == ["q b", "answer B"]


def test_new_clears_turns_and_rotates_session_id():
    session = _bare_session(RecordingClient())
    session.answer_policy = "strict"
    session.query("q1", ticker=None, refresh=False)
    assert session.history
    old_id = session.session_id

    session.new_session()

    assert session.history == []
    assert session.session_id != old_id
    # Explicit CLI settings survive a conversation reset.
    assert session.answer_policy == "strict"


def test_history_off_sends_no_turns():
    client = RecordingClient()
    session = _bare_session(client)
    # Seed a prior turn, then disable history.
    session.query("q1", ticker=None, refresh=False)
    assert session.history
    session.history_enabled = False

    session.query("q2", ticker=None, refresh=False)

    last_payload = client.posts[-1]
    assert "history" not in last_payload
    assert "session_id" not in last_payload
    # Disabled history is neither sent nor grown by the new turn.
    assert [t["content"] for t in session.history] == ["q1", "recorded answer"]


def test_stream_and_non_stream_paths_record_equivalent_history():
    class StreamClient:
        def stream(self, method, path, json=None):
            return FakeStreamResponse([
                "event: token",
                'data: {"token": "answer"}',
                "",
                "event: metadata",
                'data: {"detected_ticker": "NVDA", "detected_intent": "fact_lookup", '
                '"grounding": "grounded", "model_available": true}',
                "",
            ])

    streamed = _bare_session(StreamClient(), stream_enabled=True, stream_unavailable=False)
    streamed.query("What is NVDA revenue?", ticker=None, refresh=False)

    non_stream = _bare_session(RecordingClient(_query_data("answer")))
    non_stream.query("What is NVDA revenue?", ticker=None, refresh=False)

    assert streamed.history == non_stream.history


def test_health_reports_capabilities(monkeypatch):
    config = SimpleNamespace(
        enable_tools=True,
        enable_streaming=False,
        answer_policy="strict",
        conversation_max_turns=8,
        conversation_max_history_chars=8000,
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
    caps = response.json()["capabilities"]
    # 2.2.2.1 flags preserved …
    assert caps["tools"] is True
    assert caps["streaming"] is False
    assert caps["answer_policy"] == "strict"
    # … plus 2.2.2.3 conversation/multiline capability flags + effective limits.
    assert caps["history"] is True
    assert caps["multiline"] is True
    assert caps["max_question_chars"] == 16000
    assert caps["conversation_max_turns"] == 8
    assert caps["conversation_max_history_chars"] == 8000


# ── 2.2.2.3: multiline composer, limits/errors, session visibility, eval ──

def _scripted_input(lines, *, exhaust=EOFError):
    """An ``input()`` stand-in that returns queued lines then raises ``exhaust``
    (EOFError/KeyboardInterrupt) — mimics the terminal EOF/Ctrl+C boundary."""
    it = iter(lines)

    def _fn(_prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise exhaust

    return _fn


def test_multiline_buffer_preserves_newlines_and_punctuation():
    buf = chat.MultilineBuffer()
    buf.add("Compare NVDA and AMD revenue growth from FY2023 to FY2025.")
    buf.add("Include margin changes; summarize the strongest cited risk for each!")
    text = buf.text()
    # Newlines join the lines exactly; punctuation is untouched.
    assert text == (
        "Compare NVDA and AMD revenue growth from FY2023 to FY2025.\n"
        "Include margin changes; summarize the strongest cited risk for each!"
    )
    assert text.count("\n") == 1
    assert buf.char_count() == len(text)
    assert not buf.is_empty()


def test_send_submits_exactly_one_question():
    client = RecordingClient(_query_data("combined answer"))
    session = _bare_session(client)
    input_fn = _scripted_input([
        "Compare NVDA and AMD revenue growth from FY2023 to FY2025.",
        "Include margin changes and summarize the strongest cited risk for each.",
        "/send",
    ])

    submitted = chat.compose_multiline(session, None, False, input_fn=input_fn)

    assert submitted is True
    # Exactly one request, carrying the exact newline-joined question.
    assert len(client.posts) == 1
    assert client.posts[0]["question"] == (
        "Compare NVDA and AMD revenue growth from FY2023 to FY2025.\n"
        "Include margin changes and summarize the strongest cited risk for each."
    )


def test_cancel_and_interrupt_do_not_submit_or_change_history():
    # /cancel discards the draft.
    client = RecordingClient()
    session = _bare_session(client)
    submitted = chat.compose_multiline(
        session, None, False,
        input_fn=_scripted_input(["a buffered line", "/cancel"]))
    assert submitted is False
    assert client.posts == []
    assert session.history == []

    # EOF (Ctrl+D) cancels the buffer, not the session.
    submitted = chat.compose_multiline(
        session, None, False,
        input_fn=_scripted_input(["half a thought"], exhaust=EOFError))
    assert submitted is False
    assert client.posts == []
    assert session.history == []

    # Ctrl+C cancels the buffer, not the session.
    submitted = chat.compose_multiline(
        session, None, False,
        input_fn=_scripted_input(["half a thought"], exhaust=KeyboardInterrupt))
    assert submitted is False
    assert client.posts == []
    assert session.history == []


def test_oversized_multiline_question_is_not_sent():
    client = RecordingClient()
    session = _bare_session(client)
    session.max_question_chars = 50
    long_line = "x" * 200
    # Oversized /send is rejected locally; a following /cancel exits.
    submitted = chat.compose_multiline(
        session, None, False,
        input_fn=_scripted_input([long_line, "/send", "/cancel"]))

    assert submitted is False
    assert client.posts == []  # never left the client
    assert session.history == []


def test_422_prints_structured_validation_detail(capsys):
    long_msg = ("String should have at most 16000 characters — the composed "
                "question exceeded the server limit and was rejected without "
                "truncation so no partial question was ever answered (detail-tail-marker)")
    detail = [{
        "type": "string_too_long",
        "loc": ["body", "question"],
        "msg": long_msg,
        "ctx": {"max_length": 16000},
    }]

    class Client422:
        def post(self, path, json=None):
            return FakeResponse(status_code=422, data={"detail": detail}, text="clip")

    session = _bare_session(Client422())
    session.query("q", ticker=None, refresh=False)

    out = capsys.readouterr().out
    assert "422" in out
    assert "body.question" in out
    assert "string_too_long" in out
    # Full detail, not clipped to 200 chars — the tail marker survives.
    assert "detail-tail-marker" in out


def test_422_does_not_disable_streaming():
    class Stream422Client:
        def __init__(self):
            self.stream_calls = 0
            self.post_calls = 0

        def stream(self, *_args, **_kwargs):
            self.stream_calls += 1
            return FakeStreamResponse([], status_code=422, text="unprocessable")

        def post(self, path, json=None):
            self.post_calls += 1
            return FakeResponse(data=_query_data("fallback answer"))

    client = Stream422Client()
    session = _bare_session(client, stream_enabled=True, stream_unavailable=False)

    session.query("first", ticker=None, refresh=False)
    # A 422 must not poison streaming: still enabled for the next question.
    assert session.stream_unavailable is False
    session.query("second", ticker=None, refresh=False)

    assert client.stream_calls == 2   # streaming attempted both times
    assert client.post_calls == 2     # each falls back once to POST /query
    assert session.stream_unavailable is False


def test_old_server_without_conversation_capability_still_works(capsys):
    class OldClient:
        def __init__(self):
            self.posts = []

        def get(self, path):
            # Old middleware: /health has no conversation capability keys.
            return FakeResponse(data={
                "status": "ok",
                "capabilities": {"tools": False, "streaming": True, "answer_policy": "graded"},
            })

        def post(self, path, json=None):
            self.posts.append(json)
            return FakeResponse(data=_query_data("still answered"))

    client = OldClient()
    session = _bare_session(client)

    caps = session.load_capabilities()
    # Missing keys leave local defaults in place.
    assert session.max_question_chars == chat.DEFAULT_MAX_QUESTION_CHARS
    assert "max_question_chars" not in (caps or {})

    session.query("What is NVDA revenue?", ticker=None, refresh=False)
    out = capsys.readouterr().out
    assert "still answered" in out
    assert len(client.posts) == 1


def test_verbose_metadata_shows_carried_context(capsys):
    data = _query_data()
    data["carried_context"] = {
        "entities": ["AMD"], "metrics": ["revenue"], "timeframe": "FY2025"}

    chat._render_metadata(data, verbose=True)
    verbose_out = capsys.readouterr().out
    assert "context: AMD · revenue · FY2025" in verbose_out

    chat._render_metadata(data, verbose=False)
    plain_out = capsys.readouterr().out
    assert "context:" not in plain_out


def test_history_preview_does_not_expose_evidence_trace(capsys):
    session = _bare_session(RecordingClient())
    session.history = [
        {"role": "user", "content": "What is NVDA revenue?"},
        {
            "role": "assistant",
            "content": "NVDA revenue is 26B",
            # Neither the follow-up context nor any leaked evidence trace / system
            # prompt / tool schema must ever reach the /history preview.
            "context": {"ticker": "NVDA", "intent": "fact_lookup"},
            "evidence_trace": {"system_prompt": "SECRET-SYSTEM-PROMPT",
                               "documents": ["SECRET-EVIDENCE-DOC"]},
        },
    ]

    session.print_history()
    out = capsys.readouterr().out

    assert "NVDA revenue is 26B" in out          # the user-visible turn preview
    assert "SECRET-SYSTEM-PROMPT" not in out     # no system prompt
    assert "SECRET-EVIDENCE-DOC" not in out      # no retrieved evidence
    assert "evidence_trace" not in out
    assert "system_prompt" not in out


def test_conversation_eval_command_uses_expected_runner_arguments():
    calls: list[list[str]] = []

    def fake_runner(argv):
        calls.append(list(argv))
        if argv and str(argv[0]).endswith("score.py"):
            return 0, ("=== Conversational (Phase 2.2) ===\n"
                       "entity_carryover_accuracy              0.90   (n=10)\n"
                       "cross_session_leakage_rate             0.00   (n=10)\n")
        return 0, "Wrote 12 results -> eval/runs/1.jsonl"

    # Conversation mode: run the fixtures, then score deterministically.
    chat.do_eval(3, conversations=True, runner=fake_runner)
    assert calls[0] == [str(chat.RUN_EVAL), "--limit", "3"]
    assert calls[1] == [str(chat.SCORE_EVAL), "--no-judge"]
    assert len(calls) == 2

    # Single-turn mode: single-turn cases only, no conversations, no scoring.
    calls.clear()
    chat.do_eval(10, runner=fake_runner)
    assert calls == [[str(chat.RUN_EVAL), "--limit", "10", "--no-conversations"]]


# ── 2.2.6.1: streaming progress rendering ──────────────


def test_stream_progress_renders_single_inplace_line(capsys):
    progress = chat._StreamProgress(tty=True, verbose=False)
    progress.handle("query_started", {})
    progress.handle("stage", {"stage": "compile", "phase": "started"})
    progress.handle("stage", {"stage": "retrieve", "phase": "started"})
    progress.handle("tool_completed", {"tool": "query_facts", "status": "ok", "count": 12})
    progress.handle("stage", {"stage": "generate", "phase": "started"})

    out = capsys.readouterr().out
    # A single in-place line built from the pipeline stages + tool (with count).
    assert "resolve -> retrieval -> query_facts (12 rows) -> answer" in out
    # Rendered in place with carriage returns, not as a growing log.
    assert "\r" in out
    assert "\n" not in out


def test_stream_progress_is_silent_on_non_tty(capsys):
    progress = chat._StreamProgress(tty=False, verbose=False)
    progress.handle("stage", {"stage": "compile", "phase": "started"})
    progress.handle("tool_completed", {"tool": "query_facts", "status": "ok", "count": 3})
    progress.finish()
    assert capsys.readouterr().out == ""


def test_stream_progress_verbose_logs_timings_and_reasons(capsys):
    progress = chat._StreamProgress(tty=True, verbose=True)
    progress.handle("stage", {"stage": "retrieve", "phase": "completed",
                              "elapsed_ms": 12.5})
    progress.handle("stage", {"stage": "correct", "phase": "completed",
                              "reason": "run_derived_subqueries"})
    progress.handle("tool_completed", {"tool": "query_facts", "status": "ok", "count": 7})

    out = capsys.readouterr().out
    assert "retrieval" in out and "12.5ms" in out
    assert "correct" in out and "run_derived_subqueries" in out
    assert "query_facts (7 rows)" in out and "[ok]" in out


def test_stream_ignores_unknown_and_progress_events(capsys):
    """Unknown event types and progress events never break token/metadata
    handling, and no progress line is drawn on a non-TTY test stdout."""
    class FakeClient:
        def stream(self, method, path, json=None):
            return FakeStreamResponse([
                "event: query_started",
                'data: {"schema_version": 1, "query_id": "q", "sequence": 0}',
                "",
                "event: stage",
                'data: {"stage": "retrieve", "phase": "started", "sequence": 1}',
                "",
                "event: some_future_event",
                'data: {"anything": true}',
                "",
                "event: token",
                'data: {"token": "answer text"}',
                "",
                "event: metadata",
                'data: {"grounding": "grounded", "detected_ticker": "NVDA", '
                '"detected_intent": "fact_lookup", "facts_used": 1, '
                '"documents_used": 1, "model_available": true, '
                '"latency_ms": 8.0, "retrieval_strategy": "vector"}',
                "",
            ])

    session = _bare_session(FakeClient(), stream_enabled=True, stream_unavailable=False)
    session.query("hello", ticker=None, refresh=False)

    out = capsys.readouterr().out
    assert "answer text" in out
    assert "grounding=grounded" in out
    # The turn was recorded (complete stream), proving metadata parsed cleanly.
    assert session.history[-1]["content"] == "answer text"


def test_health_reports_streaming_tool_final_capability(monkeypatch):
    config = SimpleNamespace(
        enable_tools=True,
        enable_streaming=True,
        enable_tool_final_streaming=True,
        answer_policy="graded",
        conversation_max_turns=8,
        conversation_max_history_chars=8000,
    )
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(
        middleware_app, "store", SimpleNamespace(heartbeat=lambda: {"sqlite": True, "chroma": True}))
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(
        middleware_app, "_cached_health_summary", lambda: {"scheduler": None, "freshness": {}})

    client = TestClient(middleware_app.app)
    caps = client.get("/health").json()["capabilities"]
    # Tools + tool-final streaming -> streaming is genuinely available here.
    assert caps["streaming"] is True
    assert caps["streaming_tool_final"] is True


# ── 2.2.7.4: /graph command, capabilities, and terminal trace pointer ────────

class _GraphClient:
    """Minimal client exposing /health capabilities + a base_url for /graph."""

    def __init__(self, caps, base_url="http://127.0.0.1:8000"):
        self._caps = caps
        self.base_url = base_url

    def get(self, path):
        return FakeResponse(data={"status": "ok", "capabilities": self._caps})


class _Opener:
    """Injected browser opener that records URLs instead of launching one."""

    def __init__(self, result=True):
        self.result = result
        self.opened = []

    def __call__(self, url):
        self.opened.append(url)
        return self.result


def _graph_caps(**over):
    caps = {
        "tools": False, "streaming": True, "answer_policy": "graded",
        "graph_observer": True, "graph_url": "http://127.0.0.1:8000/graph",
        "graph_observer_limits": {"traces": 100, "elements": 5000},
    }
    caps.update(over)
    return caps


def test_load_capabilities_reads_graph_observer_fields():
    session = _bare_session(_GraphClient(_graph_caps()))
    session.load_capabilities()
    assert session.graph_observer is True
    assert session.graph_url == "http://127.0.0.1:8000/graph"
    assert session.graph_observer_limits["traces"] == 100


def test_load_capabilities_omits_graph_on_older_server():
    session = _bare_session(_GraphClient({"tools": False, "streaming": True}))
    session.load_capabilities()
    assert session.graph_observer is False
    assert session.graph_url is None


def test_graph_command_opens_effective_url(capsys):
    session = _bare_session(_GraphClient(_graph_caps()))
    session.load_capabilities()
    opener = _Opener()
    session.graph("", opener=opener)
    assert opener.opened == ["http://127.0.0.1:8000/graph"]
    assert "opened" in capsys.readouterr().out


def test_graph_url_prints_without_opening(capsys):
    session = _bare_session(_GraphClient(_graph_caps()))
    session.load_capabilities()
    opener = _Opener()
    session.graph("url", opener=opener)
    assert opener.opened == []
    assert "http://127.0.0.1:8000/graph" in capsys.readouterr().out


def test_graph_trace_deep_links_last_trace():
    session = _bare_session(_GraphClient(_graph_caps()))
    session.load_capabilities()
    session.last_trace_id = "abcdef123456"
    opener = _Opener()
    session.graph("trace", opener=opener)
    assert opener.opened == ["http://127.0.0.1:8000/graph#trace=abcdef123456"]


def test_graph_disabled_server_prints_note_and_never_opens(capsys):
    session = _bare_session(_GraphClient({"tools": False, "streaming": True}))
    session.load_capabilities()
    opener = _Opener()
    session.graph("", opener=opener)
    assert opener.opened == []
    assert "disabled" in capsys.readouterr().out.lower()


def test_graph_open_failure_prints_manual_url(capsys):
    session = _bare_session(_GraphClient(_graph_caps()))
    session.load_capabilities()
    session.graph("", opener=_Opener(result=False))
    assert "open:" in capsys.readouterr().out


def test_effective_graph_url_falls_back_to_base_url():
    # Older server does not advertise graph_url, but the observer is on.
    session = _bare_session(_GraphClient(_graph_caps(graph_url=None)))
    session.load_capabilities()
    assert session.graph_observer is True
    assert session.graph_url is None
    assert session.effective_graph_url() == "http://127.0.0.1:8000/graph"


def test_terminal_metadata_shows_trace_pointer_when_present(capsys):
    data = _query_data()
    data["graph_trace_id"] = "0123456789abcdef"
    chat._render_metadata(data, verbose=False)
    out = capsys.readouterr().out
    assert "trace=01234567" in out


def test_query_captures_last_trace_id_from_response():
    data = _query_data()
    data["graph_trace_id"] = "trace-xyz"

    class FakeClient:
        def post(self, path, json=None):
            return FakeResponse(data=data)

    session = _bare_session(FakeClient())
    session.query("q", ticker=None, refresh=False)
    assert session.last_trace_id == "trace-xyz"


# ── Analyst mode (/analysis) ───────────────────────────────────────────────


def _capturing_client():
    class FakeClient:
        def __init__(self):
            self.posts = []

        def post(self, path, json=None):
            self.posts.append((path, json))
            return FakeResponse(data=_query_data("analyst answer"))

    return FakeClient()


def test_analysis_oneshot_sends_mode():
    client = _capturing_client()
    session = _bare_session(client)
    session.query("is NVDA a good buy?", ticker=None, refresh=False, mode="analysis")
    assert client.posts[0][1]["mode"] == "analysis"


def test_plain_query_omits_mode_key():
    client = _capturing_client()
    session = _bare_session(client)
    session.query("what is NVDA revenue?", ticker=None, refresh=False)
    assert "mode" not in client.posts[0][1]


def test_sticky_analysis_toggles_mode_on_plain_queries():
    client = _capturing_client()
    session = _bare_session(client)
    session.analysis_mode = True
    session.query("outlook?", ticker="NVDA", refresh=False)
    assert client.posts[0][1]["mode"] == "analysis"
    session.analysis_mode = False
    session.query("revenue?", ticker="NVDA", refresh=False)
    assert "mode" not in client.posts[1][1]


def test_payload_only_adds_mode_for_analysis():
    assert "mode" not in chat._payload("q", None, True, mode=None)
    assert "mode" not in chat._payload("q", None, True, mode="qa")
    assert chat._payload("q", None, True, mode="analysis")["mode"] == "analysis"


def test_help_lists_analysis_commands():
    assert "/analysis" in chat.HELP


def test_prompt_suffix_shows_analyst_when_sticky_on():
    session = _bare_session(_capturing_client())
    assert "analyst" not in session.prompt_suffix()
    session.analysis_mode = True
    assert "analyst" in session.prompt_suffix()


def test_metadata_tag_shows_analyst_header(capsys):
    data = _query_data()
    data["mode"] = "analysis"
    chat._render_metadata(data, verbose=False)
    assert "[analyst]" in capsys.readouterr().out

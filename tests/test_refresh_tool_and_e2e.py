import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.middleware import app as middleware_app  # noqa: E402
from src.middleware.app import app  # noqa: E402
from src.middleware.tools import REGISTRY, ToolContext, dispatch_tool  # noqa: E402
from src.middleware.tools import data_tools, sanity  # noqa: E402,F401
from src.storage.store import Store as StoreClass  # noqa: E402
from src.storage.store import Store  # noqa: E402


@pytest.fixture(autouse=True)
def reset_middleware_globals(monkeypatch, tmp_path):
    config_path = tmp_path / "analytics.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "excluded_symbols": [],
                "equity_only_metrics": ["forward_pe"],
                "metric_ranges": {"forward_pe": {"min": 0, "max": 500}},
            }
        )
    )
    monkeypatch.setattr(sanity, "_CONFIG_PATH", config_path)
    sanity._reset_cache()
    monkeypatch.setattr(middleware_app, "_tools_supported", True)
    yield
    sanity._reset_cache()
    middleware_app._tools_supported = True


@pytest.fixture
def tmp_store(tmp_path: Path):
    with patch("src.storage.store.ChromaStore") as mock_cls:
        instance = MagicMock()
        instance.heartbeat.return_value = True
        instance.count.return_value = 0
        mock_cls.return_value = instance
        yield Store(db_path=tmp_path / "tools.db", chroma_path=tmp_path / "chroma")


def _tool_call(name, arguments):
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _model_response(content="", tool_calls=None):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            }
        ]
    }
    return response


def _config(**overrides):
    values = {
        "model_name": "tracealchemy",
        "llama_endpoint": "http://test/v1/chat/completions",
        "default_temperature": 0.3,
        "max_tokens": 2048,
        "top_k_documents": 5,
        "top_k_facts": 10,
        "enable_tools": True,
        "allow_write_tools": False,
        "max_refreshes_per_query": 2,
        "max_tool_iterations": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _seed_forward_pe(store, ticker, value):
    store.sqlite.upsert_fundamental(
        ticker=ticker,
        metric="forward_pe",
        value=value,
        unit="ratio",
        period="2026-Q1",
        source_type="test",
    )


def _mark_only_gdelt_stale(store, ticker="NVDA"):
    for cfg in StoreClass.FRESHNESS_SOURCES.values():
        store.mark_source_fresh(ticker, cfg["cache_source"], 24)
    store.mark_source_stale(ticker, "gdelt_news")


def test_refresh_tool_guarded(tmp_store, monkeypatch):
    called = MagicMock(return_value=(["gdelt_news"], []))
    monkeypatch.setattr(middleware_app, "store", tmp_store)
    monkeypatch.setattr(middleware_app, "_refresh_ticker_sources", called)

    denied = dispatch_tool(
        {"function": {"name": "refresh_data", "arguments": '{"ticker":"nvda"}'}},
        tmp_store,
        ToolContext(allow_write=False, max_refreshes=2),
    )

    assert denied == {"error": "write tools disabled"}
    called.assert_not_called()

    _mark_only_gdelt_stale(tmp_store)
    allowed = dispatch_tool(
        {"function": {"name": "refresh_data", "arguments": '{"ticker":"nvda"}'}},
        tmp_store,
        ToolContext(allow_write=True, max_refreshes=2),
    )

    assert allowed["ticker"] == "NVDA"
    assert allowed["refreshed"] == ["gdelt_news"]
    called.assert_called_once_with("NVDA", ["gdelt_news"])


def test_refresh_rate_limit(tmp_store, monkeypatch):
    called = MagicMock(return_value=(["gdelt_news"], []))
    monkeypatch.setattr(middleware_app, "store", tmp_store)
    monkeypatch.setattr(middleware_app, "_refresh_ticker_sources", called)
    _mark_only_gdelt_stale(tmp_store)
    ctx = ToolContext(allow_write=True, max_refreshes=2)
    call = {"function": {"name": "refresh_data", "arguments": '{"ticker":"NVDA"}'}}

    assert "error" not in dispatch_tool(call, tmp_store, ctx)
    assert "error" not in dispatch_tool(call, tmp_store, ctx)
    blocked = dispatch_tool(call, tmp_store, ctx)

    assert blocked == {"error": "max refreshes exceeded", "tool": "refresh_data"}
    assert called.call_count == 2


def test_refresh_cannot_trigger_full_scheduler(tmp_store, monkeypatch):
    assert middleware_app._normalize_sources(["all"]) == []

    called = MagicMock(return_value=(["gdelt_news"], []))
    scheduler_refresh = MagicMock()
    scheduler_cls = MagicMock()
    monkeypatch.setattr(middleware_app, "store", tmp_store)
    monkeypatch.setattr(middleware_app, "_refresh_ticker_sources", called)
    monkeypatch.setattr(middleware_app, "_refresh_via_scheduler", scheduler_refresh)
    monkeypatch.setattr("src.scheduler.UnifiedScheduler", scheduler_cls)
    _mark_only_gdelt_stale(tmp_store)

    # Explicitly invalid sources (e.g. "all") are a structured error, not a
    # fallback into refreshing everything stale.
    result = dispatch_tool(
        {
            "function": {
                "name": "refresh_data",
                "arguments": '{"ticker":"NVDA","sources":["all"]}',
            }
        },
        tmp_store,
        ToolContext(allow_write=True, max_refreshes=2),
    )
    assert "error" in result
    assert "valid_sources" in result
    called.assert_not_called()

    # Scheduler-managed sources are skipped (never routed to the
    # watchlist-wide UnifiedScheduler runners); per-ticker ones refresh.
    result = dispatch_tool(
        {
            "function": {
                "name": "refresh_data",
                "arguments": '{"ticker":"NVDA","sources":["sec","gdelt"]}',
            }
        },
        tmp_store,
        ToolContext(allow_write=True, max_refreshes=2),
    )
    assert result["refreshed"] == ["gdelt_news"]
    assert result["skipped_scheduler_managed"] == ["sec_filings"]
    called.assert_called_once_with("NVDA", ["gdelt_news"])

    # Only scheduler-managed sources requested -> nothing is refreshed.
    result = dispatch_tool(
        {
            "function": {
                "name": "refresh_data",
                "arguments": '{"ticker":"NVDA","sources":["sec","earnings","ir"]}',
            }
        },
        tmp_store,
        ToolContext(allow_write=True, max_refreshes=5),
    )
    assert result["refreshed"] == []
    assert result["skipped_scheduler_managed"] == [
        "sec_filings", "earnings_transcripts", "ir_pages",
    ]
    called.assert_called_once()  # unchanged — no second refresh happened
    scheduler_refresh.assert_not_called()
    scheduler_cls.assert_not_called()


def test_tools_endpoint(monkeypatch):
    with TestClient(app) as client:
        monkeypatch.setattr(
            middleware_app,
            "config",
            _config(enable_tools=True, allow_write_tools=True),
        )
        response = client.get("/tools")

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["allow_write_tools"] is True
    tools = {tool["name"]: tool for tool in data["tools"]}
    assert set(tools) == set(REGISTRY)
    assert len(tools) == 9
    assert tools["refresh_data"]["write"] is True
    assert all(not tool["write"] for name, tool in tools.items() if name != "refresh_data")


@pytest.mark.integration
def test_end_to_end_tool_query(tmp_store, monkeypatch):
    _seed_forward_pe(tmp_store, "NVDA", 16.5)
    _seed_forward_pe(tmp_store, "META", 15.9)
    _seed_forward_pe(tmp_store, "MSFT", 19.6)

    client = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status_code=200)),
        aclose=AsyncMock(),
        post=AsyncMock(
            side_effect=[
                _model_response(
                    tool_calls=[
                        _tool_call(
                            "query_facts",
                            {"metric": "forward_pe", "order": "asc", "limit": 1},
                        )
                    ]
                ),
                _model_response("META has the lowest forward P/E."),
            ]
        ),
    )
    with TestClient(app) as test_client:
        monkeypatch.setattr(middleware_app, "store", tmp_store)
        monkeypatch.setattr(middleware_app, "config", _config(enable_tools=True))
        monkeypatch.setattr(middleware_app, "model_client", client)
        monkeypatch.setattr(
            middleware_app,
            "retriever",
            SimpleNamespace(
                retrieve=lambda **_kwargs: {
                    "facts": [],
                    "documents": [],
                    "retrieval_strategy": "test",
                }
            ),
        )
        response = test_client.post(
            "/query",
            json={
                "question": "Which stock has the lowest forward P/E?",
                "refresh": False,
            },
        )

    assert response.status_code == 200
    assert "META" in response.json()["answer"]
    second_payload = client.post.await_args_list[1].kwargs["json"]
    tool_messages = [m for m in second_payload["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    tool_result = json.loads(tool_messages[0]["content"])
    assert tool_result["metric"] == "forward_pe"
    assert tool_result["results"][0]["ticker"] == "META"


def test_tools_off_regression(tmp_store, monkeypatch):
    client = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status_code=200)),
        aclose=AsyncMock(),
        post=AsyncMock(return_value=_model_response("plain answer")),
    )
    with TestClient(app) as test_client:
        monkeypatch.setattr(middleware_app, "store", tmp_store)
        monkeypatch.setattr(middleware_app, "config", _config(enable_tools=False))
        monkeypatch.setattr(middleware_app, "model_client", client)
        monkeypatch.setattr(
            middleware_app,
            "retriever",
            SimpleNamespace(
                retrieve=lambda **_kwargs: {
                    "facts": [],
                    "documents": [],
                    "retrieval_strategy": "test",
                }
            ),
        )
        response = test_client.post(
            "/query",
            json={"question": "What is stored?", "refresh": False},
        )

    assert response.status_code == 200
    assert client.post.await_count == 1
    payload = client.post.await_args.kwargs["json"]
    assert set(payload) == {"model", "messages", "temperature", "max_tokens"}
    assert "tools" not in payload

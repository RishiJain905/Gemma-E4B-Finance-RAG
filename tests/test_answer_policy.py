"""
tests/test_answer_policy.py
Offline tests for the graded grounding answer policy.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.middleware import app as middleware_app
from src.middleware.models import QueryRequest


def _response(content: str):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
    }
    return response


def _config(**overrides):
    values = {
        "model_name": "tracealchemy",
        "llama_endpoint": "http://test/v1/chat/completions",
        "default_temperature": 0.3,
        "max_tokens": 256,
        "top_k_documents": 5,
        "top_k_facts": 10,
        "enable_tools": False,
        "enable_fetch_on_miss": False,
        "answer_policy": "graded",
        "allow_general_fallback": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_grounding_level():
    assert middleware_app._grounding_level({"facts": [1, 2], "documents": [3]}) == "grounded"
    assert middleware_app._grounding_level({"facts": [1], "documents": []}) == "partial"
    assert middleware_app._grounding_level({"facts": [], "documents": []}) == "none"


@pytest.mark.asyncio
async def test_strict_mode_unchanged(monkeypatch):
    client = SimpleNamespace(post=AsyncMock(return_value=_response("answer")))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config(answer_policy="strict"))

    await middleware_app._call_model(
        "prompt",
        0.1,
        100,
        intent={"question_type": "fact_lookup"},
        grounding_level="grounded",
    )

    payload = client.post.await_args.kwargs["json"]
    assert payload["messages"][0]["content"] == middleware_app.SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_graded_prompt_includes_mode(monkeypatch):
    client = SimpleNamespace(post=AsyncMock(return_value=_response("answer")))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config())

    await middleware_app._call_model(
        "prompt",
        0.1,
        100,
        intent={"question_type": "news"},
        grounding_level="partial",
    )

    system_prompt = client.post.await_args.kwargs["json"]["messages"][0]["content"]
    assert "Intent: news" in system_prompt
    assert "Grounding level: partial" in system_prompt
    assert "never invent specific numbers" in system_prompt.lower()


@pytest.mark.asyncio
async def test_general_fallback_labeled(monkeypatch):
    client = SimpleNamespace(post=AsyncMock(return_value=_response("Apple sells consumer devices.")))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config(allow_general_fallback=True))

    answer, _citations = await middleware_app._call_model(
        "prompt",
        0.1,
        100,
        intent={"question_type": "general"},
        grounding_level="none",
    )

    assert answer.startswith("Not from your data")
    assert "verify against a primary source" in answer

    client = SimpleNamespace(post=AsyncMock(return_value=_response("Apple sells consumer devices.")))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config(allow_general_fallback=False))

    answer, _citations = await middleware_app._call_model(
        "prompt",
        0.1,
        100,
        intent={"question_type": "general"},
        grounding_level="none",
    )

    assert "Apple sells consumer devices" not in answer
    assert "don't have enough data" in answer


@pytest.mark.asyncio
async def test_response_reports_grounding(monkeypatch):
    parser = MagicMock()
    parser.parse.return_value = {
        "ticker": None,
        "ticker_confidence": 0.0,
        "question_type": "general",
    }
    monkeypatch.setattr(
        "src.middleware.intent_parser.IntentParser",
        MagicMock(return_value=parser),
    )
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(
        middleware_app,
        "retriever",
        SimpleNamespace(
            retrieve=lambda **_kwargs: {
                "facts": [],
                "documents": [],
                "retrieval_strategy": "broad",
            }
        ),
    )
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(middleware_app, "_evaluate_and_refresh", MagicMock(return_value={}))
    monkeypatch.setattr(
        middleware_app,
        "_call_model",
        AsyncMock(return_value=("Not from your data - general knowledge: background.", [])),
    )

    response = await middleware_app.query(QueryRequest(question="What is Apple?"))

    assert response.grounding == "general"


@pytest.mark.asyncio
async def test_no_fabricated_numbers_rule_present(monkeypatch):
    client = SimpleNamespace(post=AsyncMock(return_value=_response("answer")))
    monkeypatch.setattr(middleware_app, "model_client", client)
    monkeypatch.setattr(middleware_app, "config", _config())

    await middleware_app._call_model(
        "prompt",
        0.1,
        100,
        intent={"question_type": "fact_lookup"},
        grounding_level="grounded",
    )

    system_prompt = client.post.await_args.kwargs["json"]["messages"][0]["content"]
    assert "never invent specific numbers" in system_prompt.lower()
    assert "specific figures must come from context or tools" in system_prompt.lower()

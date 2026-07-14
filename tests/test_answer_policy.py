"""
tests/test_answer_policy.py
Offline tests for the graded grounding answer policy.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.middleware import app as middleware_app
from src.middleware import prompt_policy
from src.middleware.evidence_grader import (
    CorrectiveAction,
    CoverageResult,
    SufficiencyResult,
    SufficiencyStatus,
)
from src.middleware.models import QueryRequest, QueryResponse
from src.middleware.prompt_augmenter import PromptAugmenter


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
    grounded = {
        "facts": [{"metric": "total_revenue", "value": 1.0},
                  {"metric": "gross_margin", "value": 2.0}],
        "documents": [{"document": "some retrieved text"}],
    }
    partial = {"facts": [{"metric": "total_revenue", "value": 1.0}], "documents": []}
    none = {"facts": [], "documents": []}
    assert middleware_app._grounding_level(grounded) == "grounded"
    assert middleware_app._grounding_level(partial) == "partial"
    assert middleware_app._grounding_level(none) == "none"


def test_blank_document_does_not_count_as_grounding():
    """A document row with metadata but a blank/whitespace body must not
    inflate grounding (2.2.1.1 — the confirmed document/text P0 bug)."""
    retrieval = {
        "facts": [],
        "documents": [
            {"id": "doc-1", "document": "   ", "metadata": {"ticker": "NVDA"}},
        ],
    }
    assert middleware_app._grounding_level(retrieval) == "none"

    from src.middleware.evidence import evidence_counts
    assert evidence_counts(retrieval) == (0, 0)


def test_zero_value_fact_is_usable():
    """A zero-valued fact remains valid evidence; a None-valued one does not."""
    from src.middleware.evidence import evidence_counts, usable_facts

    retrieval = {
        "facts": [
            {"metric": "net_income", "value": 0, "ticker": "NVDA"},
            {"metric": "eps", "value": None, "ticker": "NVDA"},
        ],
        "documents": [],
    }
    usable = usable_facts(retrieval)
    assert len(usable) == 1
    assert usable[0]["metric"] == "net_income"
    assert evidence_counts(retrieval) == (1, 0)
    assert middleware_app._grounding_level(retrieval) == "partial"


def _sufficiency(status, *, covered=(), missing=("sq0",), reasons=("missing_metric",)):
    rows = [
        CoverageResult(subquery_id, ("fact:NVDA",), (), ("f1",), ())
        for subquery_id in covered
    ]
    rows.extend(
        CoverageResult(subquery_id, (), ("fact:NVDA",), (), reasons)
        for subquery_id in missing
    )
    return SufficiencyResult(
        status=status, overall_score=1.0 if status is SufficiencyStatus.SUFFICIENT else 0.0,
        reason_codes=reasons, coverage=tuple(rows), conflicts=(),
        allowed_action=CorrectiveAction.NONE,
    )


def test_sufficiency_maps_to_grounded_partial_general_or_refused():
    sufficient = _sufficiency(
        SufficiencyStatus.SUFFICIENT, covered=("sq0",), missing=())
    assert middleware_app._answer_mode_from_sufficiency(
        sufficient, requires_specific_figures=True,
    ) == "grounded"
    partial = SufficiencyResult(
        status=SufficiencyStatus.BORDERLINE, overall_score=0.5,
        reason_codes=("missing_qualitative_evidence",),
        coverage=(CoverageResult("sq0", ("fact:NVDA",), ("document:NVDA",),
                                 ("f1",), ("missing_qualitative_evidence",)),),
        conflicts=(), allowed_action=CorrectiveAction.ALTERNATE_INTERNAL_MODALITY,
    )
    assert middleware_app._answer_mode_from_sufficiency(
        partial, requires_specific_figures=True,
    ) == "partial"
    missing = _sufficiency(SufficiencyStatus.MISSING)
    assert middleware_app._answer_mode_from_sufficiency(
        missing, requires_specific_figures=False,
    ) == "general"
    assert middleware_app._answer_mode_from_sufficiency(
        missing, requires_specific_figures=True,
    ) == "refused"


def test_prompt_receives_covered_and_missing_obligations():
    result = SufficiencyResult(
        status=SufficiencyStatus.BORDERLINE, overall_score=0.5,
        reason_codes=("missing_qualitative_evidence",),
        coverage=(CoverageResult("sq0", ("fact:NVDA",), ("document:NVDA",),
                                 ("f1",), ("missing_qualitative_evidence",)),),
        conflicts=(), allowed_action=CorrectiveAction.ALTERNATE_INTERNAL_MODALITY,
    )

    prompt = PromptAugmenter(config=SimpleNamespace(answer_policy="graded")).build_prompt(
        question="Why did revenue change?",
        intent={"ticker": "NVDA", "question_type": "explanation"},
        retrieval={}, grounding_level="partial",
        preselected={"facts": [], "documents": []},
        evidence_sufficiency=result,
    )

    assert "## Evidence Coverage" in prompt
    assert "Covered obligations: fact:NVDA" in prompt
    assert "Missing obligations: document:NVDA" in prompt
    assert "missing_qualitative_evidence" in prompt


def test_graded_prompt_supports_explicit_general_and_refused_modes():
    general = prompt_policy.build_system_prompt(
        answer_policy="graded", allow_general_fallback=True,
        intent={"question_type": "general"}, grounding_level="general",
    )
    refused = prompt_policy.build_system_prompt(
        answer_policy="graded", allow_general_fallback=True,
        intent={"question_type": "fact_lookup"}, grounding_level="refused",
    )

    assert "Mode: general fallback" in general
    assert "Mode: refuse" in refused


def test_flag_off_response_omits_optional_sufficiency_metadata():
    response = QueryResponse(answer="legacy")

    assert "evidence_sufficiency" not in response.model_dump()


def test_response_sufficiency_metadata_uses_answer_status_and_retry_action():
    result = SimpleNamespace(
        sufficiency=_sufficiency(SufficiencyStatus.MISSING),
        corrective_action=CorrectiveAction.ALTERNATE_INTERNAL_MODALITY,
        retry_performed=True,
    )

    metadata = middleware_app._sufficiency_metadata(result, "refused")

    assert metadata == {
        "status": "refused",
        "reason_codes": ["missing_metric"],
        "covered_subqueries": [],
        "missing_subqueries": ["sq0"],
        "corrective_action": "alternate_internal_modality",
        "retry_performed": True,
    }


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


def test_answer_validation_config_default_and_normalization(monkeypatch, tmp_path):
    """MiddlewareConfig ships answer_validation=report; unknown values -> off."""
    from src.middleware.config import MiddlewareConfig

    # Default from the committed configs/middleware.yaml.
    assert MiddlewareConfig().answer_validation == "report"

    # An env override with an unknown value normalizes to off (fail-safe).
    monkeypatch.setenv("ANSWER_VALIDATION", "nonsense")
    assert MiddlewareConfig().answer_validation == "off"

    monkeypatch.setenv("ANSWER_VALIDATION", "enforce")
    assert MiddlewareConfig().answer_validation == "enforce"


def test_answer_validation_mode_helper(monkeypatch):
    monkeypatch.setattr(middleware_app, "config", _config(answer_validation="enforce"))
    assert middleware_app._answer_validation_mode() == "enforce"
    assert middleware_app._evidence_ids_enabled() is True

    monkeypatch.setattr(middleware_app, "config", _config(answer_validation="off"))
    assert middleware_app._answer_validation_mode() == "off"
    assert middleware_app._evidence_ids_enabled() is False

    # A config lacking the attribute (older SimpleNamespace) falls back to off.
    monkeypatch.setattr(middleware_app, "config", SimpleNamespace())
    assert middleware_app._answer_validation_mode() == "off"


@pytest.mark.asyncio
async def test_query_exposes_validation_metadata_in_report_mode(monkeypatch):
    """report mode attaches validation metadata + evidence citations without
    changing the answer text or grounding."""
    parser = MagicMock()
    parser.parse.return_value = {
        "ticker": "NVDA", "ticker_confidence": 1.0, "question_type": "fact_lookup",
    }
    monkeypatch.setattr(
        "src.middleware.intent_parser.IntentParser", MagicMock(return_value=parser))
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "config", _config(answer_validation="report"))
    monkeypatch.setattr(
        middleware_app, "retriever",
        SimpleNamespace(retrieve=lambda **_kwargs: {
            "facts": [{"metric": "total_revenue", "value": 26.0,
                       "unit": "billion_usd", "ticker": "NVDA", "period": "2026-Q1",
                       "source_type": "sec_10q"}],
            "documents": [], "retrieval_strategy": "facts_only",
        }))
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(middleware_app, "_evaluate_and_refresh", MagicMock(return_value={}))
    monkeypatch.setattr(
        middleware_app, "_call_model",
        AsyncMock(return_value=("Revenue was $26.0 billion [E1].", [])))

    response = await middleware_app.query(QueryRequest(question="NVDA revenue?"))

    assert response.answer == "Revenue was $26.0 billion [E1]."
    assert response.answer_validation["validation_status"] == "supported"
    assert response.answer_validation["numeric_claims_supported"] == 1
    assert response.answer_validation["enforcement"] == "none"
    assert response.evidence_citations[0].evidence_id == "E1"
    assert response.grounding in ("grounded", "partial")


@pytest.mark.asyncio
async def test_query_off_mode_omits_validation_metadata(monkeypatch):
    """off mode leaves the response free of any 2.2.4.3 validation fields."""
    parser = MagicMock()
    parser.parse.return_value = {
        "ticker": "NVDA", "ticker_confidence": 1.0, "question_type": "fact_lookup",
    }
    monkeypatch.setattr(
        "src.middleware.intent_parser.IntentParser", MagicMock(return_value=parser))
    monkeypatch.setattr(middleware_app, "store", object())
    monkeypatch.setattr(middleware_app, "config", _config(answer_validation="off"))
    monkeypatch.setattr(
        middleware_app, "retriever",
        SimpleNamespace(retrieve=lambda **_kwargs: {
            "facts": [{"metric": "total_revenue", "value": 26.0,
                       "unit": "billion_usd", "ticker": "NVDA"}],
            "documents": [], "retrieval_strategy": "facts_only",
        }))
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    monkeypatch.setattr(middleware_app, "_evaluate_and_refresh", MagicMock(return_value={}))
    monkeypatch.setattr(
        middleware_app, "_call_model",
        AsyncMock(return_value=("Revenue was $26.0 billion.", [])))

    response = await middleware_app.query(QueryRequest(question="NVDA revenue?"))
    dumped = response.model_dump()
    assert "answer_validation" not in dumped
    assert "evidence_citations" not in dumped


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


def test_strict_policy_is_unchanged():
    """The strict system prompt is pinned byte-for-byte to the pre-2.2.1.1
    middleware SYSTEM_PROMPT constant, now owned by prompt_policy.py."""
    from src.middleware import prompt_policy

    expected = (
        "You are a financial research assistant. Answer the user's "
        "question using ONLY the provided context. If the context "
        "doesn't contain enough information, say so. "
        "Cite sources inline using [Source: type/ticker] notation."
    )
    assert prompt_policy.STRICT_SYSTEM_PROMPT == expected
    assert middleware_app.SYSTEM_PROMPT == expected
    assert prompt_policy.build_system_prompt(
        answer_policy="strict",
        allow_general_fallback=True,
        intent={"question_type": "fact_lookup"},
        grounding_level="grounded",
        tools_enabled=False,
    ) == expected


def test_all_model_paths_share_prompt_policy(monkeypatch):
    """Plain, streaming, tool, and direct-eval paths build an identical
    system prompt for equivalent policy/intent/grounding inputs — all four
    route through prompt_policy.build_system_prompt (2.2.1.1)."""
    import httpx

    from eval import run_eval as R
    from src.middleware import prompt_policy

    monkeypatch.setattr(middleware_app, "config", _config(
        answer_policy="graded", allow_general_fallback=True,
    ))
    intent = {"question_type": "fact_lookup"}

    expected = prompt_policy.build_system_prompt(
        answer_policy="graded", allow_general_fallback=True,
        intent=intent, grounding_level="partial", tools_enabled=False,
    )
    expected_tools = prompt_policy.build_system_prompt(
        answer_policy="graded", allow_general_fallback=True,
        intent=intent, grounding_level="partial", tools_enabled=True,
    )

    # Plain call (also used verbatim by the streaming call site).
    assert middleware_app._system_prompt_for_request(
        intent, "partial", tools_enabled=False,
    ) == expected
    # Tool-loop call site.
    assert middleware_app._system_prompt_for_request(
        intent, "partial", tools_enabled=True,
    ) == expected_tools

    # Direct-eval path (eval/run_eval.py) — capture the payload sent to the model.
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    eval_config = SimpleNamespace(
        model_name="tracealchemy", llama_endpoint="http://test/v1/chat/completions",
        default_temperature=0.3, max_tokens=256,
        answer_policy="graded", allow_general_fallback=True,
    )
    R._call_model_sync(eval_config, "prompt body", intent=intent, grounding_level="partial")
    assert captured["payload"]["messages"][0]["content"] == expected

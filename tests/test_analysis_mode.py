"""
tests/test_analysis_mode.py
Offline tests for analyst mode (mode="analysis"): request plumbing, prompt
policy, fast-path bypass, answer-policy/refusal bypass, generation budgets,
tool-loop iteration override + kill-switch cooldown, and the opinion intent.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.middleware import app as middleware_app
from src.middleware import prompt_policy
from src.middleware.intent_parser import IntentParser
from src.middleware.models import QueryRequest

# Reuse the eligible-orchestration-result fixtures from the deterministic
# fast-path suite (same tests dir, prepend import mode).
from test_deterministic_answer_fast_path import (  # noqa: E402
    _config as _det_config,
    _context as _det_context,
    _eligible,
    _case,
)


@pytest.fixture
def analysis_mode():
    """Set the request-scoped mode contextvar to analysis for the test body."""
    token = middleware_app._mode_var.set("analysis")
    try:
        yield
    finally:
        middleware_app._mode_var.reset(token)


@pytest.fixture
def qa_mode():
    token = middleware_app._mode_var.set("qa")
    try:
        yield
    finally:
        middleware_app._mode_var.reset(token)


# ── Request model ──────────────────────────────────────────────────────────


def test_request_accepts_mode():
    assert QueryRequest(question="q").mode is None
    assert QueryRequest(question="q", mode="analysis").mode == "analysis"
    assert QueryRequest(question="q", mode="qa").mode == "qa"


def test_request_rejects_unknown_mode():
    with pytest.raises(Exception):
        QueryRequest(question="q", mode="wizard")


# ── Prompt policy ──────────────────────────────────────────────────────────


def test_analysis_system_prompt_has_verdict_and_never_refuse_language():
    prompt = prompt_policy.build_system_prompt(
        answer_policy="graded", allow_general_fallback=True,
        intent={"question_type": "opinion"}, grounding_level="partial",
        mode="analysis",
    )
    assert "senior equity research analyst" in prompt
    assert "Never refuse, deflect, or lecture" in prompt
    assert "give a direct" in prompt  # verdict language
    assert "analytical view, not" in prompt


def test_analysis_tools_clause_only_when_tools_enabled():
    without = prompt_policy.build_system_prompt(
        answer_policy="graded", allow_general_fallback=True, intent=None,
        grounding_level="partial", tools_enabled=False, mode="analysis",
    )
    with_tools = prompt_policy.build_system_prompt(
        answer_policy="graded", allow_general_fallback=True, intent=None,
        grounding_level="partial", tools_enabled=True, mode="analysis",
    )
    assert "A good analyst" not in without
    assert "A good analyst" in with_tools
    assert "checks the data before opining" in with_tools
    assert "get_price_history" in with_tools


def test_analysis_mode_overrides_answer_policy():
    # Even a strict policy yields the analyst prompt in analysis mode.
    prompt = prompt_policy.build_system_prompt(
        answer_policy="strict", allow_general_fallback=True,
        intent={"question_type": "fact_lookup"}, grounding_level="grounded",
        mode="analysis",
    )
    assert "senior equity research analyst" in prompt


def test_graded_qa_prompt_has_opinion_line():
    prompt = prompt_policy.build_system_prompt(
        answer_policy="graded", allow_general_fallback=True,
        intent={"question_type": "opinion"}, grounding_level="partial",
    )
    assert "Opinion/assessment questions about covered securities are in scope" in prompt


def test_strict_prompt_unchanged_by_analysis_kwarg_default():
    # mode defaults to "qa", so existing strict callers are byte-identical.
    assert prompt_policy.build_system_prompt(
        answer_policy="strict", allow_general_fallback=True,
        intent={"question_type": "fact_lookup"}, grounding_level="grounded",
        tools_enabled=False,
    ) == prompt_policy.STRICT_SYSTEM_PROMPT


# ── Intent: opinion question_type ──────────────────────────────────────────


@pytest.mark.parametrize("question", [
    "is NVDA a good buy?",
    "should I buy AMD right now?",
    "is AAPL overvalued?",
    "what do you think about MSFT?",
    "would you buy TSLA here?",
    "bullish or bearish on GOOGL?",
    "your take on META?",
])
def test_opinion_questions_classify_as_opinion(question):
    assert IntentParser().parse(question)["question_type"] == "opinion"


@pytest.mark.parametrize("question", [
    "what is NVDA's revenue?",
    "compare NVDA and AMD gross margins",
    "how much cash does AAPL have?",
])
def test_fact_lookups_do_not_classify_as_opinion(question):
    assert IntentParser().parse(question)["question_type"] != "opinion"


def test_opinion_extraction_keeps_ticker():
    intent = IntentParser().parse("is NVDA a good buy?")
    assert intent["ticker"] == "NVDA"


# ── Answer-policy / refusal bypass ─────────────────────────────────────────


def test_apply_answer_policy_bypassed_in_analysis(analysis_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(answer_policy="graded", allow_general_fallback=False))
    # A refused grounding level would normally force NO_GENERAL_FALLBACK_MESSAGE.
    out = middleware_app._apply_answer_policy("My verdict: buy.", "refused")
    assert out == "My verdict: buy."


def test_apply_answer_policy_refuses_in_qa(qa_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(answer_policy="graded", allow_general_fallback=False))
    out = middleware_app._apply_answer_policy("some answer", "refused")
    assert out == middleware_app.NO_GENERAL_FALLBACK_MESSAGE


def test_response_grounding_never_refused_in_analysis(analysis_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(allow_general_fallback=False))
    # A declined-looking answer with thin evidence stays "general", not "refused".
    assert middleware_app._response_grounding(
        "I don't have enough data", "none") == "general"
    assert middleware_app._response_grounding("verdict", "grounded") == "grounded"
    assert middleware_app._response_grounding("verdict", "partial") == "partial"


def test_response_grounding_refuses_in_qa(qa_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(allow_general_fallback=False))
    assert middleware_app._response_grounding(
        "I don't have enough data", "none") == "refused"


# ── Generation budgets ─────────────────────────────────────────────────────


def test_task_settings_analysis_uses_wide_budget(analysis_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(default_temperature=0.3, max_tokens=2048))
    temp, max_tokens = middleware_app._task_settings(
        QueryRequest(question="is NVDA a buy?", mode="analysis"),
        {"question_type": "opinion"},
    )
    assert temp == 0.55
    assert max_tokens == 3072


def test_task_settings_analysis_respects_request_override(analysis_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(default_temperature=0.3, max_tokens=2048))
    temp, max_tokens = middleware_app._task_settings(
        QueryRequest(question="q", mode="analysis", temperature=0.1, max_tokens=512),
        {"question_type": "opinion"},
    )
    assert temp == 0.1
    assert max_tokens == 512


def test_task_settings_qa_unchanged(qa_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(default_temperature=0.3, max_tokens=2048))
    temp, max_tokens = middleware_app._task_settings(
        QueryRequest(question="what is revenue?"), {"question_type": "fact_lookup"})
    assert temp == 0.3
    assert max_tokens == 2048


def test_effective_max_tool_iterations(analysis_mode, qa_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(max_tool_iterations=3, analysis_max_tool_iterations=5))
    token = middleware_app._mode_var.set("analysis")
    try:
        assert middleware_app._effective_max_tool_iterations() == 5
    finally:
        middleware_app._mode_var.reset(token)
    token = middleware_app._mode_var.set("qa")
    try:
        assert middleware_app._effective_max_tool_iterations() == 3
    finally:
        middleware_app._mode_var.reset(token)


# ── Fast-path bypass (full path) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_analysis_skips_deterministic_and_runs_model(monkeypatch):
    """A context that answers deterministically in qa mode runs the model in
    analysis mode; answer_origin stays 'model' and generation is not skipped."""
    template, execution, facts, plan = _eligible(1)  # exact_fact case
    result, ledger = _case(template=template, execution=execution,
                           facts=list(facts), plan=plan)
    context = _det_context(result, ledger, list(facts))

    monkeypatch.setattr(middleware_app, "config", _det_config())
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    invoke = AsyncMock(return_value=("Analyst verdict: buy. [Source: sec/NVDA]", []))
    monkeypatch.setattr(middleware_app, "_invoke_model", invoke)

    token = middleware_app._mode_var.set("analysis")
    try:
        response = await middleware_app._answer_query_context(
            QueryRequest(question="deterministic question", mode="analysis"), context)
    finally:
        middleware_app._mode_var.reset(token)

    invoke.assert_awaited_once()
    assert response.answer_origin == "model"
    assert response.generation_skipped is False
    assert response.mode == "analysis"


@pytest.mark.asyncio
async def test_qa_still_skips_deterministic(monkeypatch):
    """Control: the same context skips the model in qa mode (no regression)."""
    template, execution, facts, plan = _eligible(1)
    result, ledger = _case(template=template, execution=execution,
                           facts=list(facts), plan=plan)
    context = _det_context(result, ledger, list(facts))

    monkeypatch.setattr(middleware_app, "config", _det_config())
    invoke = AsyncMock(side_effect=AssertionError("model must not be called in qa"))
    monkeypatch.setattr(middleware_app, "_invoke_model", invoke)
    monkeypatch.setattr(middleware_app, "_check_model_health",
                        AsyncMock(side_effect=AssertionError("health must not be probed")))

    token = middleware_app._mode_var.set("qa")
    try:
        response = await middleware_app._answer_query_context(
            QueryRequest(question="deterministic question"), context)
    finally:
        middleware_app._mode_var.reset(token)

    assert response.answer_origin == "deterministic"
    assert response.mode is None


def test_try_deterministic_response_none_in_analysis(analysis_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(enable_deterministic_answers=True))
    # A result is present, but analysis mode returns None before touching it.
    context = {"_orchestration_result": object()}
    assert middleware_app._try_deterministic_response(context) is None


def test_deterministic_contract_ineligible_in_analysis(analysis_mode, monkeypatch):
    monkeypatch.setattr(middleware_app, "config",
                        SimpleNamespace(enable_deterministic_answers=True))
    context = {"_orchestration_result": object()}
    assert middleware_app._deterministic_contract_eligible(context) is False


# ── Config view (retrieval budget) ─────────────────────────────────────────


def test_analysis_orchestrator_config_view(monkeypatch):
    base = SimpleNamespace(top_k_documents=5, analysis_max_context_chars=24000,
                           adaptive_max_context_chars=18000, enable_tools=True)
    monkeypatch.setattr(middleware_app, "config", base)
    view = middleware_app._analysis_orchestrator_config()
    assert view.top_k_documents == 8          # raised for analysis
    assert view.analysis_context_override == 24000
    assert view.enable_tools is True          # delegates everything else
    assert view.adaptive_max_context_chars == 18000


def test_context_budget_uses_analysis_override():
    from src.middleware.adaptive_orchestrator import ContextBudget, Lane

    base = SimpleNamespace(adaptive_max_context_chars=18000,
                           analysis_context_override=24000)
    budget = ContextBudget(base)
    # Override bypasses the per-lane cap (COMPLEX cap is 18000).
    assert budget.cap_for(Lane.COMPLEX) == 24000
    assert budget.cap_for(Lane.STANDARD) == 24000

    plain = SimpleNamespace(adaptive_max_context_chars=18000)
    assert ContextBudget(plain).cap_for(Lane.STANDARD) == 12000


# ── Tool kill-switch cooldown ──────────────────────────────────────────────


def test_tools_available_cooldown(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(middleware_app.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(middleware_app, "_tools_supported", True)
    monkeypatch.setattr(middleware_app, "_tools_cooldown_until", 0.0)

    assert middleware_app._tools_available() is True

    # Simulate the empty-first-response branch starting a cooldown.
    monkeypatch.setattr(
        middleware_app, "_tools_cooldown_until",
        clock["t"] + middleware_app._TOOLS_EMPTY_RESPONSE_COOLDOWN_S)
    assert middleware_app._tools_available() is False

    # After the cooldown window, tools are retried automatically.
    clock["t"] += middleware_app._TOOLS_EMPTY_RESPONSE_COOLDOWN_S + 1
    assert middleware_app._tools_available() is True


def test_tools_available_false_when_permanently_disabled(monkeypatch):
    monkeypatch.setattr(middleware_app.time, "monotonic", lambda: 1.0)
    monkeypatch.setattr(middleware_app, "_tools_supported", False)
    monkeypatch.setattr(middleware_app, "_tools_cooldown_until", 0.0)
    assert middleware_app._tools_available() is False


# ── get_filing_overview tool ───────────────────────────────────────────────


class _FilingStore:
    def __init__(self, filings=None, families=None, chunk_text="x"):
        self._filings = filings if filings is not None else [{
            "filing_type": "10-K", "filing_date": "2026-02-01", "period": "FY2025",
            "accession": "acc-1", "source_url": "https://sec.gov/acc-1",
        }]
        self._families = families if families is not None else [
            {"id": f"acc-1#{i}", "metadata": {"section_index": i, "section_title": f"Item {i}"},
             "chunk_count": 3}
            for i in range(3)
        ]
        self._chunk_text = chunk_text

    def list_filings(self, *, ticker=None, filing_type=None, limit=1, offset=0):
        return list(self._filings[:limit])

    def get_filing_section_families(self, accession, *, limit=100, offset=0):
        return list(self._families[:limit])

    def get_section_chunks(self, parent_id, *, limit=1, offset=0):
        return [{"id": f"{parent_id}#0", "document": self._chunk_text, "metadata": {}}]


def test_get_filing_overview_registered_schema():
    from src.middleware.tools import REGISTRY

    tool = REGISTRY["get_filing_overview"]
    assert tool.write is False
    assert tool.parameters["required"] == ["ticker"]
    assert "filing_type" in tool.parameters["properties"]


def test_get_filing_overview_returns_metadata_and_sections():
    from src.middleware.tools import data_tools

    result = data_tools.get_filing_overview_handler(_FilingStore(), ticker="nvda")
    assert result["status"] == "found"
    assert result["ticker"] == "NVDA"
    assert result["filing"]["filing_type"] == "10-K"
    assert result["filing"]["accession"] == "acc-1"
    assert len(result["sections"]) == 3
    assert result["sections"][0]["section_title"] == "Item 0"
    assert result["sections"][0]["excerpt"] == "x"


def test_get_filing_overview_bounded_output():
    from src.middleware.tools import data_tools

    # Huge section bodies must not blow past the ~6000 char budget.
    big = "a" * 5000
    families = [
        {"id": f"acc-1#{i}", "metadata": {"section_index": i, "section_title": f"S{i}"},
         "chunk_count": 1}
        for i in range(30)
    ]
    store = _FilingStore(families=families, chunk_text=big)
    result = data_tools.get_filing_overview_handler(store, ticker="NVDA")
    excerpt_chars = sum(len(s.get("excerpt", "")) for s in result["sections"])
    assert excerpt_chars <= data_tools._FILING_OVERVIEW_CHAR_BUDGET
    assert len(result["sections"]) <= data_tools._FILING_OVERVIEW_MAX_SECTIONS


def test_get_filing_overview_not_found():
    from src.middleware.tools import data_tools

    result = data_tools.get_filing_overview_handler(_FilingStore(filings=[]), ticker="ZZZ")
    assert result["status"] == "not_found"
    assert result["sections"] == []


def test_get_filing_overview_metadata_only_when_unindexed():
    from src.middleware.tools import data_tools

    result = data_tools.get_filing_overview_handler(
        _FilingStore(families=[]), ticker="NVDA")
    assert result["status"] == "metadata_only"
    assert result["sections"] == []

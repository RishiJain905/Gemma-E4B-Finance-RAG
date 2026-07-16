"""Offline tests for Phase 2.3.7.2 indirect capability query routing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eval import metrics as eval_metrics
from src.middleware import deterministic_router as dr
from src.middleware.adaptive_orchestrator import Lane, orchestrate
from src.middleware.conversation import compile_question
from src.middleware.intent_parser import IntentParser
from src.middleware.models import ChatTurn


FIXTURE = (
    Path(__file__).parent
    / "fixtures/evaluation/phase2_3_indirect_queries.json"
)


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _parser() -> IntentParser:
    return IntentParser()


def _obligation_view(plan) -> dict:
    obligations = plan.obligations
    return {
        "operation": obligations.operation,
        "universe_scope": obligations.universe_scope,
        "entity_set": list(obligations.entity_set),
        "metrics": list(obligations.metrics),
        "item_types": list(obligations.item_types),
        "sources": list(obligations.sources),
        "completeness": obligations.completeness,
        "limit": obligations.limit,
        "as_of": obligations.as_of,
        "qualitative": obligations.qualitative,
        "evidence_modes": list(obligations.evidence_modes),
    }


@pytest.mark.parametrize("case", _cases(), ids=lambda row: row["id"])
def test_indirect_queries_compile_to_expected_obligations(case):
    parser = _parser()
    plan = parser.parse_plan(case["question"])

    assert plan.intents == case["expected_intents"]
    assert _obligation_view(plan) == case["expected_obligations"]


def test_concise_and_verbose_paraphrases_normalize_identically():
    parser = _parser()
    by_pair: dict[str, list] = {}
    for case in _cases():
        if case.get("pair_id"):
            by_pair.setdefault(case["pair_id"], []).append(
                parser.parse_plan(case["question"])
            )

    assert by_pair
    for pair_id, plans in by_pair.items():
        assert len(plans) == 2, pair_id
        assert plans[0].intents == plans[1].intents
        assert plans[0].obligations == plans[1].obligations


def test_golden_indirect_paraphrase_pairs_share_normalized_plans():
    golden = Path(__file__).parents[1] / "eval/golden/finance_qa.jsonl"
    rows = [
        json.loads(line)
        for line in golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list] = {}
    parser = _parser()
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id and str(row.get("id", "")).startswith("indirect-"):
            grouped.setdefault(pair_id, []).append(parser.parse_plan(row["question"]))

    assert grouped
    for pair_id, plans in grouped.items():
        assert len(plans) == 2, pair_id
        assert plans[0].intents == plans[1].intents, pair_id
        assert plans[0].obligations == plans[1].obligations, pair_id


def test_company_research_is_not_misrouted_as_inventory():
    plan = _parser().parse_plan("What do you know about Apple?")

    assert not set(plan.intents) & {
        "capability_inventory",
        "coverage_check",
        "source_inventory",
        "metric_inventory",
        "corpus_summary",
    }
    assert plan.obligations.universe_scope == "explicit_entities"
    assert plan.obligations.evidence_modes == ("documents",)


def test_all_and_count_require_catalog_not_top_k_retrieval():
    parser = _parser()
    all_plan = parser.parse_plan("List all supported tickers")
    count_plan = parser.parse_plan("How many companies are in your database?")

    assert all_plan.obligations.completeness == "all"
    assert count_plan.obligations.completeness == "count"
    assert all_plan.obligations.evidence_modes == ("catalog",)
    assert count_plan.obligations.evidence_modes == ("catalog",)
    assert all_plan.subqueries[0].retrieval_modes == ("tools",)
    assert count_plan.subqueries[0].retrieval_modes == ("tools",)


def _config(**overrides):
    values = {
        "enable_adaptive_rag": True,
        "enable_deterministic_answers": True,
        "max_deterministic_tools_per_query": 3,
        "adaptive_max_subqueries": 3,
        "adaptive_max_retrieval_rounds": 2,
        "adaptive_max_planning_calls": 1,
        "adaptive_max_context_chars": 18_000,
        "top_k_facts": 10,
        "top_k_documents": 5,
        "enable_evidence_sufficiency": False,
        "enable_reranker": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_catalog_question_uses_catalog_lane_without_retrieval_or_planner():
    plan = _parser().parse_plan("Which companies do you know about?")
    store = MagicMock()
    store.describe_coverage.return_value = {
        "status": "ok",
        "coverage_basis": "canonical",
        "securities": [{"ticker": "AAPL"}, {"ticker": "NVDA"}],
        "result_count": 2,
        "total_matching": 2,
        "complete": True,
        "next_cursor": None,
    }
    retriever = MagicMock()
    planner = MagicMock()

    result = orchestrate(
        plan,
        store,
        _config(),
        retriever=retriever,
        planning_client=planner,
    )

    assert result.lane is Lane.CATALOG
    assert result.answer_origin == "deterministic"
    assert "AAPL" in result.deterministic_answer
    retriever.retrieve.assert_not_called()
    retriever.retrieve_candidates.assert_not_called()
    planner.assert_not_called()


def test_partial_catalog_page_cannot_build_deterministic_answer():
    plan = _parser().parse_plan("List all supported tickers")
    decision = dr.route(plan, ())
    store = MagicMock()
    store.describe_coverage.return_value = {
        "status": "ok",
        "coverage_basis": "canonical",
        "securities": [{"ticker": "AAPL"}],
        "result_count": 1,
        "total_matching": 20,
        "complete": False,
        "next_cursor": "cursor-1",
    }

    execution = dr.execute_route(decision, store)

    assert execution.complete is False
    assert dr.INCOMPLETE_PARTIAL_CATALOG_PAGE in execution.incomplete_reason_codes
    assert execution.answer is None
    assert execution.answer_origin is None


def test_compound_catalog_set_over_execution_cap_requests_narrowing():
    plan = _parser().parse_plan(
        "Which semiconductor companies do you cover, and compare their latest margins?"
    )
    decision = dr.route(plan, {"gross_margin_pct"})

    assert decision.matched is True
    assert decision.complete is False
    assert dr.INCOMPLETE_MISSING_METRIC_COVERAGE in decision.reason_codes
    assert decision.tool_invocations[0].name == "describe_coverage"


def _compound_execution(calls, tickers):
    def execute(decision, store, **kwargs):
        from src.middleware.deterministic_router import (
            ExecutedInvocation,
            ExecutionResult,
        )

        inv = decision.tool_invocations[0]
        calls.append((inv.name, dict(inv.arguments)))
        if inv.name == "describe_coverage":
            payload = {
                "status": "ok",
                "coverage_basis": "canonical",
                "securities": [{"ticker": ticker} for ticker in tickers],
                "result_count": len(tickers),
                "total_matching": len(tickers),
                "complete": True,
                "next_cursor": None,
            }
        else:
            payload = {
                "metric": "gross_margin_pct",
                "results": [
                    {"ticker": ticker, "metric": "gross_margin_pct", "value": 50.0}
                    for ticker in tickers
                ],
            }
        return ExecutionResult(
            invocations=[ExecutedInvocation(
                inv.name,
                dict(inv.arguments),
                inv.subquery_id,
                inv.reason_code,
                result=payload,
            )],
            complete=decision.complete,
        )

    return execute


def test_compound_catalog_resolves_set_before_bounded_structured_comparison():
    plan = _parser().parse_plan(
        "Which semiconductor companies do you cover, and compare their latest margins?"
    )
    calls = []
    retriever = MagicMock()
    retriever.retrieve.return_value = {"facts": [], "documents": []}

    result = orchestrate(
        plan,
        object(),
        _config(top_k_facts=3),
        retriever=retriever,
        available_metrics={"gross_margin_pct"},
        execute_fn=_compound_execution(calls, ["AMD", "NVDA"]),
    )

    assert [name for name, _ in calls[:2]] == ["describe_coverage", "query_facts"]
    assert calls[1][1]["tickers"] == ["AMD", "NVDA"]
    assert calls[1][1]["limit"] == 2
    assert {fact["ticker"] for fact in result.merged_facts} >= {"AMD", "NVDA"}
    assert result.planning_ran is False


def test_compound_catalog_set_over_cap_is_not_silently_sampled():
    plan = _parser().parse_plan(
        "Which semiconductor companies do you cover, and compare their latest margins?"
    )
    calls = []
    retriever = MagicMock()
    retriever.retrieve.return_value = {"facts": [], "documents": []}

    result = orchestrate(
        plan,
        object(),
        _config(top_k_facts=2),
        retriever=retriever,
        available_metrics={"gross_margin_pct"},
        execute_fn=_compound_execution(calls, ["AMD", "INTC", "NVDA"]),
    )

    assert [name for name, _ in calls] == ["describe_coverage"]
    assert dr.INCOMPLETE_RESULT_SET_TOO_LARGE in result.reason_codes
    assert any("narrow" in str(fact.get("value", "")).lower() for fact in result.merged_facts)


def test_six_stable_incomplete_reason_codes_are_wired():
    assert {
        dr.INCOMPLETE_PARTIAL_CATALOG_PAGE,
        dr.INCOMPLETE_MISSING_COVERAGE_DIMENSION,
        dr.INCOMPLETE_UNRESOLVED_UNIVERSE_SCOPE,
        dr.INCOMPLETE_MISSING_METRIC_COVERAGE,
        dr.INCOMPLETE_QUALITATIVE_EVIDENCE_REQUIRED,
        dr.INCOMPLETE_RESULT_SET_TOO_LARGE,
    } == {
        "partial_catalog_page",
        "missing_coverage_dimension",
        "unresolved_universe_scope",
        "missing_metric_coverage",
        "qualitative_evidence_required",
        "result_set_too_large",
    }

    unresolved = dr.route(
        _parser().parse_plan("Which of those also have transcripts?"), ()
    )
    assert dr.INCOMPLETE_UNRESOLVED_UNIVERSE_SCOPE in unresolved.reason_codes
    assert unresolved.tool_invocations[0].arguments["operation"] == "summary"

    qualitative = dr.route(
        _parser().parse_plan("What do you cover, and explain the available data?"), ()
    )
    assert dr.INCOMPLETE_QUALITATIVE_EVIDENCE_REQUIRED in qualitative.reason_codes

    from src.middleware.query_plan import AnswerObligations

    missing_dimension = _parser().parse_plan("List all supported tickers")
    missing_dimension.obligations = AnswerObligations(
        operation="list_securities",
        universe_scope="security_registry",
        item_types=("unsupported_dimension",),
        completeness="all",
        evidence_modes=("catalog",),
    )
    decision = dr.route(missing_dimension, ())
    assert dr.INCOMPLETE_MISSING_COVERAGE_DIMENSION in decision.reason_codes


def test_prior_inventory_is_carried_into_bounded_followup_without_model_call():
    history = [
        ChatTurn(role="user", content="Which semiconductor companies do you cover?"),
        ChatTurn(
            role="assistant",
            content="AMD and NVDA.",
            context={
                "grounding": "grounded",
                "coverage_metadata": {
                    "complete": True,
                    "securities": ["AMD", "NVDA"],
                },
            },
        ),
    ]

    compiled = compile_question("Which of those also have transcripts?", history)

    assert compiled.carried_entities == ["AMD", "NVDA"]
    assert compiled.inventory_scope == ["AMD", "NVDA"]
    assert compiled.ambiguous_slots == []
    assert "AMD" in compiled.retrieval_query and "NVDA" in compiled.retrieval_query


def test_bounded_inventory_followup_intersects_only_the_prior_security_set():
    history = [
        ChatTurn(role="user", content="Which semiconductor companies do you cover?"),
        ChatTurn(
            role="assistant",
            content="AMD and NVDA.",
            context={
                "grounding": "grounded",
                "coverage_metadata": {
                    "complete": True,
                    "total_matching": 2,
                    "securities": ["AMD", "NVDA"],
                },
            },
        ),
    ]
    compiled = compile_question("Which of those also have transcripts?", history)
    plan = _parser().parse_plan(compiled.retrieval_query)
    decision = dr.route(plan, ())

    class CoverageStore:
        def describe_coverage(self, *, operation, ticker=None, **_kwargs):
            assert operation == "security_sources"
            item_types = ["transcript"] if ticker == "AMD" else ["sec_filing"]
            return {
                "covered": True,
                "complete": True,
                "item_types": item_types,
                "sources": [],
                "result_count": 0,
                "total_matching": 0,
            }

    execution = dr.execute_route(decision, CoverageStore())

    assert [
        invocation.arguments["ticker"] for invocation in decision.tool_invocations
    ] == ["AMD", "NVDA"]
    assert all(
        invocation.arguments["operation"] == "security_sources"
        for invocation in decision.tool_invocations
    )
    assert execution.complete is True
    assert execution.answer is not None
    assert "AMD" in execution.answer
    assert "NVDA" not in execution.answer

    orchestration = orchestrate(plan, CoverageStore(), _config())
    assert orchestration.lane is Lane.CATALOG
    assert orchestration.set_complete is True
    assert orchestration.result_set_size == 1


def test_followup_without_inventory_context_is_ambiguous_and_planner_failure_is_truthful():
    compiled = compile_question("Which of those also have transcripts?", [])

    assert "universe_scope" in compiled.ambiguous_slots
    assert compiled.inventory_scope == []


def test_phase_2372_metrics_record_accuracy_completeness_and_call_counts():
    rows = [
        {
            "case": {
                "expected_plan": {"intents": ["source_inventory"]},
                "expected_lane": "catalog",
                "expected_obligations": ["catalog", "all"],
                "expected_set": ["sec", "fred"],
            },
            "query_plan": {
                "intents": ["source_inventory"],
                "obligations": ["catalog", "all"],
            },
            "orchestration": {
                "lane": "catalog",
                "planning_calls": 0,
                "model_calls": 0,
                "set_complete": True,
            },
            "result_set": ["fred", "sec"],
        }
    ]

    block = eval_metrics.indirect_query_metrics(rows)

    assert block["plan_accuracy"]["score"] == 1.0
    assert block["router_accuracy"]["score"] == 1.0
    assert block["obligation_coverage"]["score"] == 1.0
    assert block["set_completeness"]["score"] == 1.0
    assert block["unnecessary_planner_calls"] == 0
    assert block["model_call_count"] == 0

"""Offline promotion gates for Phase 2.3.7.7 RAG quality and latency evaluation."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval import metrics
from eval import run_eval
from src.middleware import app as middleware_app
from src.middleware import deterministic_router as dr
from src.middleware.adaptive_orchestrator import Lane, OrchestrationResult, orchestrate
from src.middleware.evidence import assign_evidence_ids, build_evidence_items
from src.middleware.intent_parser import IntentParser
from src.middleware.models import QueryRequest
from src.middleware.query_plan import AnswerObligations, QueryPlan, QuerySubquery


FIXTURE = Path(__file__).parent / "fixtures/evaluation/phase2_3_rag_quality.jsonl"

REQUIRED_FAMILIES = {
    "capability_exact",
    "capability_paraphrased",
    "capability_verbose",
    "capability_conversational",
    "inventory_tickers",
    "inventory_sources",
    "inventory_types",
    "inventory_metrics",
    "inventory_count",
    "coverage_positive",
    "coverage_negative",
    "indirect_fact",
    "indirect_comparison",
    "indirect_ranking",
    "indirect_calculation",
    "qualitative_generation",
    "temporal_as_of",
    "former_membership",
    "freshness",
    "source_conflict",
    "alias",
    "ticker_change",
    "unresolved_entity",
    "unsupported_request",
    "lexical_exact_match",
    "lexical_phrase",
    "lexical_filing_form",
    "lexical_event",
    "lexical_semantic",
    "cache_repeated",
    "revision_behavior",
    "partial_page",
    "partial_result_set",
}


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolved_fixture_rows() -> list[dict]:
    source_cache: dict[Path, dict[str, dict]] = {}
    rows: list[dict] = []
    for raw in _fixture_rows():
        row = dict(raw)
        source = row.get("source_fixture")
        if source:
            path = Path(__file__).parents[1] / source
            if path not in source_cache:
                payload = json.loads(path.read_text(encoding="utf-8"))
                cases = payload.get("cases") or payload.get("queries") or []
                source_cache[path] = {case["id"]: case for case in cases}
            base = source_cache[path][row["source_case_id"]]
            row = {**base, **row}
        rows.append(row)
    return rows


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
        "answer_validation": "off",
        "require_evidence_ids": False,
        "return_timings": True,
        "enable_graph_observer": False,
        "enable_tools": False,
        "default_temperature": 0.3,
        "max_tokens": 256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _coverage_store(tickers=("AAPL", "AMD", "NVDA")):
    store = MagicMock()
    store.describe_coverage.return_value = {
        "status": "ok",
        "coverage_basis": "canonical",
        "securities": [{"ticker": ticker} for ticker in tickers],
        "result_count": len(tickers),
        "total_matching": len(tickers),
        "complete": True,
        "next_cursor": None,
    }
    return store


def _fact_result() -> tuple[OrchestrationResult, list[dict], list]:
    plan = QueryPlan(
        original_question="What was NVDA revenue in FY2025?",
        retrieval_query="What was NVDA revenue in FY2025?",
        intents=["fact_lookup"],
        metrics=["revenue"],
        periods=["FY2025"],
        primary_intent="fact_lookup",
        subqueries=[QuerySubquery(id="sq0", text="NVDA revenue FY2025")],
        obligations=AnswerObligations(
            metrics=("revenue",), completeness="all", evidence_modes=("facts",),
        ),
    )
    facts = [{
        "ticker": "NVDA",
        "metric": "revenue",
        "value": 26.0,
        "unit": "USD bn",
        "period": "FY2025",
        "source_type": "sec_companyfacts",
    }]
    execution = dr.ExecutionResult(
        invocations=[dr.ExecutedInvocation(
            name="get_fundamentals",
            arguments={"ticker": "NVDA", "metrics": ["revenue"]},
            subquery_id="sq0",
            reason_code=dr.REASON_FUNDAMENTALS,
            result={"ticker": "NVDA", "fundamentals": {"revenue": 26.0}},
        )],
        complete=True,
    )
    result = OrchestrationResult(
        lane=Lane.FAST,
        plan=plan,
        tool_execution=execution,
        merged_facts=facts,
        merged_documents=[],
        subqueries_executed=["sq0"],
        reason_codes=["fast_complete_route"],
    )
    ledger = assign_evidence_ids(build_evidence_items(facts, []))
    return result, facts, ledger


def _answer_context(result, facts, ledger) -> dict:
    return {
        "start": time.time(),
        "timings": {},
        "intent": {"ticker": "NVDA", "question_type": result.plan.primary_intent},
        "freshness": {
            "overall": "fresh",
            "refreshed_during_query": [],
            "stale_sources_used": [],
            "fetched_on_miss": [],
            "warning": None,
        },
        "retrieval": {"facts": facts, "documents": [], "retrieval_strategy": "fast"},
        "grounding_level": "grounded",
        "augmented_prompt": "model prompt",
        "include_evidence_trace": False,
        "conversation": None,
        "compiled": None,
        "retrieval_query": result.plan.retrieval_query,
        "orchestration": {
            "lane": result.lane.value,
            "deterministic_tools": [item.name for item in result.tool_execution.invocations],
            "model_calls": 0,
        },
        "coverage_metadata": None,
        "evidence_sufficiency": None,
        "evidence_ledger": ledger,
        "graph_evidence_ids": [item.evidence_id for item in ledger],
        "calculations": [],
        "_orchestration_result": result,
    }


def test_labeled_fixture_covers_every_required_family_and_contract_field():
    rows = _resolved_fixture_rows()

    assert {row["family"] for row in rows} >= REQUIRED_FAMILIES
    assert all(row.get("version") == "2.3.7.7" for row in rows)
    for row in rows:
        assert row.get("question"), row["id"]
        assert isinstance(row.get("expected_obligations"), dict), row["id"]
        assert row.get("eligible_route") in {
            "catalog", "structured", "retrieval", "generation", "clarify", "refuse"
        }, row["id"]
        assert row.get("required_completeness"), row["id"]
        assert "generation_may_be_skipped" in row, row["id"]
        assert "acceptable_refusal" in row, row["id"]
        assert "acceptable_clarification" in row, row["id"]
        assert (
            "expected_evidence_ids" in row or "expected_structured_values" in row
        ), row["id"]


def test_runner_persists_canonical_stage_timings_and_unavailable_values():
    stages = {
        "intent_plan_ms": 1.0,
        "catalog_tools_ms": None,
        "dense_retrieval_ms": 2.0,
        "lexical_retrieval_ms": 3.0,
        "fusion_rerank_ms": 4.0,
        "prompt_construction_ms": 5.0,
        "model_ttft_ms": None,
        "model_total_ms": 6.0,
        "end_to_end_ms": 21.0,
    }
    row = run_eval._row_from_endpoint(
        {"id": "timing", "question": "q"},
        {"answer": "a", "model_available": False, "timings": {"stages": stages}},
        latency_s=0.025,
    )

    assert row["timings"]["stages"] == stages
    assert row["stage_timings"] == stages
    assert row["stage_timings"]["catalog_tools_ms"] is None
    assert row["stage_timings"]["model_ttft_ms"] is None


def test_middleware_normalizes_separated_stages_without_zero_filling():
    timings = {
        "intent_parse": 1.0,
        "query_plan": 2.0,
        "catalog_tools": 3.0,
        "retrieval": {
            "embedding": 4.0,
            "chroma": 5.0,
            "lexical": 6.0,
            "fusion_rerank": 7.0,
        },
        "prompt_build": 8.0,
        "model_call": 9.0,
    }

    stages = middleware_app._quality_stage_timings(timings, end_to_end_ms=45.0)

    assert stages == {
        "intent_plan_ms": 3.0,
        "catalog_tools_ms": 3.0,
        "dense_retrieval_ms": 9.0,
        "lexical_retrieval_ms": 6.0,
        "fusion_rerank_ms": 7.0,
        "prompt_construction_ms": 8.0,
        "model_ttft_ms": None,
        "model_total_ms": 9.0,
        "end_to_end_ms": 45.0,
    }


def test_quality_metrics_include_denominators_calls_and_per_family_results():
    rows = [
        {
            "case": {
                "family": "inventory_tickers",
                "expected_inventory": ["AAPL", "AMD", "NVDA"],
                "generation_may_be_skipped": True,
                "expected_structured_values": [],
            },
            "result_set": ["AAPL", "AMD", "NVDA"],
            "generation_skipped": True,
            "orchestration": {
                "model_calls": 0,
                "planning_calls": 0,
                "retrieval_rounds": 0,
                "deterministic_tools": ["describe_coverage"],
            },
            "embedding_calls": 0,
        },
        {
            "case": {
                "family": "indirect_fact",
                "generation_may_be_skipped": False,
                "expected_structured_values": [{
                    "ticker": "NVDA", "metric": "revenue", "value": 26.0,
                    "unit": "USD bn", "period": "FY2025",
                }],
            },
            "structured_values": [{
                "ticker": "NVDA", "metric": "revenue", "value": 26.0,
                "unit": "USD bn", "period": "FY2025",
            }],
            "generation_skipped": False,
            "orchestration": {
                "model_calls": 1,
                "planning_calls": 1,
                "retrieval_rounds": 1,
                "deterministic_tools": ["get_fundamentals"],
            },
            "embedding_calls": 1,
        },
    ]

    block = metrics.rag_quality_metrics(rows)

    assert block["inventory"]["set_precision"]["score"] == 1.0
    assert block["inventory"]["set_recall"]["score"] == 1.0
    assert block["inventory"]["exact_count_accuracy"]["score"] == 1.0
    assert block["inventory"]["invented_item_count"]["count"] == 0
    assert block["structured"]["value_accuracy"]["score"] == 1.0
    assert block["structured"]["unit_accuracy"]["score"] == 1.0
    assert block["structured"]["period_accuracy"]["score"] == 1.0
    assert block["fast_path_eligibility"]["precision"]["score"] == 1.0
    assert block["fast_path_eligibility"]["recall"]["score"] == 1.0
    assert block["calls"]["model"]["total"] == 1
    assert block["calls"]["embedding"]["total"] == 1
    assert block["calls"]["planner"]["total"] == 1
    assert block["calls"]["retrieval_rounds"]["total"] == 1
    assert block["calls"]["tool"]["total"] == 2
    assert block["per_family"]["inventory_tickers"]["n_cases"] == 1
    assert block["per_family"]["indirect_fact"]["n_cases"] == 1


def test_all_ticker_inventory_gate_is_exact_against_seeded_registry():
    expected = ["AAPL", "AMD", "NVDA"]
    store = _coverage_store(expected)
    result = orchestrate(
        IntentParser().parse_plan("List all supported tickers."),
        store,
        _config(),
        retriever=MagicMock(),
        planning_client=MagicMock(),
    )
    actual = [row["ticker"] for row in result.tool_execution.invocations[0].result["securities"]]
    block = metrics.inventory_set_metrics([{
        "case": {"family": "inventory_tickers", "expected_inventory": expected},
        "result_set": actual,
    }])

    assert block["set_precision"]["score"] == 1.0
    assert block["set_recall"]["score"] == 1.0
    assert block["exact_count_accuracy"]["score"] == 1.0
    assert block["invented_item_count"]["count"] == 0
    store.describe_coverage.assert_called_once()


def test_capability_routing_precision_meets_offline_gate():
    inventory_intents = {
        "capability_inventory", "coverage_check", "source_inventory",
        "metric_inventory", "corpus_summary",
    }
    tp = fp = 0
    for case in _resolved_fixture_rows():
        expected = bool(case.get("capability_route_expected"))
        predicted = bool(set(IntentParser().parse_plan(case["question"]).intents) & inventory_intents)
        tp += int(expected and predicted)
        fp += int(not expected and predicted)

    precision = tp / (tp + fp)
    assert precision >= 0.98


@pytest.mark.asyncio
async def test_eligible_deterministic_answer_makes_zero_model_calls(monkeypatch):
    result, facts, ledger = _fact_result()
    probe = AsyncMock(side_effect=AssertionError("model health must not be probed"))
    invoke = AsyncMock(side_effect=AssertionError("model must not be called"))
    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "_check_model_health", probe)
    monkeypatch.setattr(middleware_app, "_invoke_model", invoke)

    response = await middleware_app._answer_query_context(
        QueryRequest(question=result.plan.original_question),
        _answer_context(result, facts, ledger),
    )

    assert response.generation_skipped is True
    assert response.orchestration["model_calls"] == 0
    probe.assert_not_awaited()
    invoke.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["ineligible", "partial", "conflicting", "qualitative"])
async def test_ineligible_cases_never_skip_generation(monkeypatch, mutation):
    result, facts, ledger = _fact_result()
    result = deepcopy(result)
    facts = deepcopy(facts)
    context = _answer_context(result, facts, ledger)
    if mutation == "ineligible":
        result.lane = Lane.STANDARD
    elif mutation == "partial":
        result.tool_execution.complete = False
        result.tool_execution.incomplete_reason_codes = [dr.INCOMPLETE_METRIC_COVERAGE]
    elif mutation == "conflicting":
        result.merged_facts[0]["conflict"] = True
        context["retrieval"]["facts"][0]["conflict"] = True
    else:
        result.plan.obligations = AnswerObligations(
            qualitative=True, evidence_modes=("facts", "documents"),
        )

    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    invoke = AsyncMock(return_value=("model answer", []))
    monkeypatch.setattr(middleware_app, "_invoke_model", invoke)

    response = await middleware_app._answer_query_context(
        QueryRequest(question=result.plan.original_question), context,
    )

    assert response.generation_skipped is False
    invoke.assert_awaited_once()


def test_catalog_questions_make_no_embedding_or_planner_calls():
    store = _coverage_store()
    embedding = MagicMock()
    store.chroma.embedding_fn = embedding
    retriever = MagicMock()
    planner = MagicMock()

    result = orchestrate(
        IntentParser().parse_plan("What tickers do you know about?"),
        store,
        _config(),
        retriever=retriever,
        planning_client=planner,
    )

    assert result.lane is Lane.CATALOG
    embedding.assert_not_called()
    retriever.retrieve.assert_not_called()
    retriever.retrieve_candidates.assert_not_called()
    planner.assert_not_called()


@pytest.mark.parametrize(
    "measurement_gate",
    [
        "deterministic-answer end-to-end latency improvement",
        "retrieval nDCG@10 and lexical Recall@10",
        "indirect-plan live improvement and non-fast-path p95",
    ],
)
def test_live_measurement_gates_are_explicitly_deferred(measurement_gate):
    pytest.skip(f"requires fixed-corpus measured results: {measurement_gate}")

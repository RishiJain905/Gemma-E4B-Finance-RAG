"""Offline contract tests for Phase 2.3.7.3 deterministic final answers."""

from __future__ import annotations

import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.middleware import app as middleware_app
from src.middleware import deterministic_answers as da
from src.middleware import deterministic_router as dr
from src.middleware.adaptive_orchestrator import Lane, OrchestrationResult
from src.middleware.evidence import assign_evidence_ids, build_evidence_items
from src.middleware.models import QueryRequest
from src.middleware.query_plan import AnswerObligations, QueryPlan, QuerySubquery


def _plan(
    *,
    intents: tuple[str, ...] = ("fact_lookup",),
    metrics: tuple[str, ...] = ("revenue",),
    periods: tuple[str, ...] = (),
    evidence_modes: tuple[str, ...] = ("facts",),
    qualitative: bool = False,
) -> QueryPlan:
    return QueryPlan(
        original_question="deterministic question",
        retrieval_query="deterministic question",
        intents=list(intents),
        metrics=list(metrics),
        periods=list(periods),
        primary_intent=intents[0] if intents else "general",
        subqueries=[QuerySubquery(id="sq0", text="deterministic question")],
        obligations=AnswerObligations(
            metrics=metrics,
            completeness="all",
            qualitative=qualitative,
            evidence_modes=evidence_modes,
        ),
    )


def _execution(
    name: str,
    reason: str,
    result: dict,
    *,
    arguments: dict | None = None,
    calculations: list[dict] | None = None,
    complete: bool = True,
    error: bool = False,
    invocation_error: str | None = None,
) -> dr.ExecutionResult:
    return dr.ExecutionResult(
        invocations=[dr.ExecutedInvocation(
            name=name,
            arguments=arguments or {},
            subquery_id="sq0",
            reason_code=reason,
            result=result,
            error=invocation_error,
        )],
        calculations=list(calculations or []),
        complete=complete,
        error=error,
    )


def _ledger(facts: list[dict]):
    return assign_evidence_ids(build_evidence_items(facts, []))


def _case(
    *,
    template: da.TemplateClass,
    execution: dr.ExecutionResult,
    facts: list[dict],
    plan: QueryPlan | None = None,
    lane: Lane = Lane.FAST,
) -> tuple[OrchestrationResult, list]:
    plan = plan or _plan()
    result = OrchestrationResult(
        lane=lane,
        plan=plan,
        tool_execution=execution,
        merged_facts=facts,
        merged_documents=[],
        subqueries_executed=["sq0"],
    )
    result.reason_codes = ["fast_complete_route"]
    if template is da.TemplateClass.COVERAGE:
        result.set_complete = True
        result.result_set_size = len(execution.invocations[0].result.get("securities", []))
    return result, _ledger(facts)


ELIGIBLE_CASES = [
    pytest.param(
        da.TemplateClass.COVERAGE,
        _execution(
            "describe_coverage",
            dr.REASON_COVERAGE,
            {
                "status": "ok",
                "coverage_basis": "canonical",
                "securities": [{"ticker": "AAPL"}, {"ticker": "NVDA"}],
                "result_count": 2,
                "total_matching": 2,
                "complete": True,
                "next_cursor": None,
            },
            arguments={"operation": "list_securities", "ticker_only": True},
        ),
        [{
            "ticker": "CATALOG", "metric": "coverage_list_securities",
            "value": "AAPL, NVDA", "source_type": "catalog",
        }],
        _plan(
            intents=("capability_inventory",), metrics=(),
            evidence_modes=("catalog",),
        ),
        id="coverage",
    ),
    pytest.param(
        da.TemplateClass.FACT,
        _execution(
            "get_fundamentals",
            dr.REASON_FUNDAMENTALS,
            {"ticker": "NVDA", "fundamentals": {"revenue": 26.0}},
            arguments={"ticker": "NVDA", "metrics": ["revenue"]},
        ),
        [{
            "ticker": "NVDA", "metric": "revenue", "value": 26.0,
            "unit": "USD bn", "period": "FY2025", "source_type": "sec_companyfacts",
        }],
        _plan(periods=("FY2025",)),
        id="exact_fact",
    ),
    pytest.param(
        da.TemplateClass.BOUNDED_SET,
        _execution(
            "query_facts",
            dr.REASON_RANK,
            {"metric": "revenue", "results": [
                {"ticker": "NVDA", "metric": "revenue", "value": 26.0,
                 "unit": "USD bn", "period": "FY2025", "source_type": "sec"},
                {"ticker": "AMD", "metric": "revenue", "value": 20.0,
                 "unit": "USD bn", "period": "FY2025", "source_type": "sec"},
            ]},
            arguments={"metric": "revenue", "order": "desc", "limit": 2},
        ),
        [
            {"ticker": ticker, "metric": "revenue", "value": value,
             "unit": "USD bn", "period": "FY2025", "source_type": "sec"}
            for ticker, value in (("NVDA", 26.0), ("AMD", 20.0))
        ],
        _plan(intents=("comparison",)),
        id="bounded_ranking",
    ),
    pytest.param(
        da.TemplateClass.CALCULATION,
        _execution(
            "query_facts",
            dr.REASON_COMPARE,
            {"metric": "revenue", "results": [
                {"ticker": "NVDA", "metric": "revenue", "value": 26.0,
                 "unit": "USD bn", "period": "FY2025", "source_type": "sec"},
                {"ticker": "AMD", "metric": "revenue", "value": 20.0,
                 "unit": "USD bn", "period": "FY2025", "source_type": "sec"},
            ]},
            arguments={"metric": "revenue", "tickers": ["NVDA", "AMD"]},
            calculations=[{"operation": "difference", "result": "6.0", "unit": "USD bn"}],
        ),
        [
            {"ticker": ticker, "metric": "revenue", "value": value,
             "unit": "USD bn", "period": "FY2025", "source_type": "sec"}
            for ticker, value in (("NVDA", 26.0), ("AMD", 20.0))
        ],
        _plan(intents=("comparison",)),
        id="decimal_calculation",
    ),
    pytest.param(
        da.TemplateClass.ESTIMATE,
        _execution(
            "get_estimates",
            dr.REASON_ESTIMATES,
            {"ticker": "NVDA", "estimates": {
                "estimate_eps_next_y": {"value": 5.25, "period": "FY2026", "unit": "USD"},
            }},
            arguments={"ticker": "NVDA", "horizon": "year"},
        ),
        [{
            "ticker": "NVDA", "metric": "estimate_eps_next_y", "value": 5.25,
            "unit": "USD", "period": "FY2026", "source_type": "estimates",
        }],
        _plan(intents=("projection",), metrics=("estimate_eps_next_y",)),
        id="estimate",
    ),
    pytest.param(
        da.TemplateClass.TARGET,
        _execution(
            "get_price_targets",
            dr.REASON_PRICE_TARGETS,
            {"ticker": "NVDA", "price_targets": {
                "price_target_mean": {"value": 150.0, "period": "2026-07-15", "unit": "USD"},
            }},
            arguments={"ticker": "NVDA"},
        ),
        [{
            "ticker": "NVDA", "metric": "price_target_mean", "value": 150.0,
            "unit": "USD", "period": "2026-07-15", "source_type": "estimates",
        }],
        _plan(intents=("projection",), metrics=("price_target_mean",)),
        id="price_target",
    ),
    pytest.param(
        da.TemplateClass.GUIDANCE,
        _execution(
            "get_guidance",
            dr.REASON_GUIDANCE,
            {"ticker": "NVDA", "status": "found", "guidance": {
                "revenue_low": {"value": 27.0, "unit": "USD bn", "period": "Q3 FY2026"},
            }},
            arguments={"ticker": "NVDA"},
        ),
        [{
            "ticker": "NVDA", "metric": "guidance_revenue_low", "value": 27.0,
            "unit": "USD bn", "period": "Q3 FY2026", "source_type": "guidance",
        }],
        _plan(intents=("projection",), metrics=()),
        id="guidance",
    ),
    pytest.param(
        da.TemplateClass.MACRO,
        _execution(
            "get_macro_snapshot",
            dr.REASON_MACRO,
            {"macro": {"fed_rate": {"value": 4.5, "unit": "%", "period": "2026-07-01"}}},
        ),
        [{
            "ticker": "MACRO", "metric": "fed_rate", "value": 4.5,
            "unit": "%", "period": "2026-07-01", "source_type": "fred",
        }],
        _plan(intents=("fact_lookup",), metrics=()),
        id="macro",
    ),
    pytest.param(
        da.TemplateClass.FRESHNESS,
        _execution(
            "check_freshness",
            dr.REASON_FRESHNESS,
            {"ticker": "NVDA", "overall": "fresh", "sources": {
                "fundamentals": {"status": "fresh", "as_of": "2026-07-16"},
            }, "stale_sources": []},
            arguments={"ticker": "NVDA"},
        ),
        [{
            "ticker": "NVDA", "metric": "freshness_fundamentals", "value": "fresh",
            "period": "2026-07-16", "source_type": "freshness",
        }],
        _plan(intents=("fact_lookup",), metrics=()),
        id="freshness",
    ),
]


def _eligible(index: int):
    return deepcopy(ELIGIBLE_CASES[index].values[:4])


@pytest.mark.parametrize("template,execution,facts,plan", ELIGIBLE_CASES)
def test_typed_templates_preserve_values_provenance_and_caveats(
    template, execution, facts, plan,
):
    result, ledger = _case(
        template=template, execution=execution, facts=facts, plan=plan,
        lane=Lane.CATALOG if template is da.TemplateClass.COVERAGE else Lane.FAST,
    )

    contract = da.check_final_answer_contract(
        result, evidence_ledger=ledger, freshness={"overall": "fresh"},
    )
    rendered = da.render_deterministic_answer(result, evidence_ledger=ledger)

    assert contract.eligible is True
    assert contract.template is template
    assert rendered.template is template
    assert len(rendered.text) <= da.MAX_DETERMINISTIC_ANSWER_CHARS
    if facts:
        assert "[E1]" in rendered.text
    if template in {da.TemplateClass.ESTIMATE, da.TemplateClass.TARGET, da.TemplateClass.GUIDANCE}:
        assert "not guarantees" in rendered.text


def _config(**overrides):
    values = {
        "enable_deterministic_answers": True,
        "answer_validation": "off",
        "require_evidence_ids": False,
        "return_timings": True,
        "enable_graph_observer": False,
        "enable_tools": False,
        "enable_streaming": True,
        "enable_tool_final_streaming": False,
        "enable_stream_progress_events": True,
        "stream_progress_include_counts": True,
        "default_temperature": 0.3,
        "max_tokens": 256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _context(result: OrchestrationResult, ledger: list, facts: list[dict]) -> dict:
    return {
        "start": time.time(),
        "timings": {},
        "intent": {"ticker": "NVDA", "question_type": result.plan.primary_intent},
        "freshness": {
            "overall": "fresh", "refreshed_during_query": [],
            "stale_sources_used": [], "fetched_on_miss": [], "warning": None,
        },
        "retrieval": {
            "facts": facts, "documents": [], "retrieval_strategy": "fast",
        },
        "grounding_level": "grounded",
        "augmented_prompt": "model prompt",
        "include_evidence_trace": False,
        "conversation": None,
        "compiled": None,
        "retrieval_query": result.plan.retrieval_query,
        "orchestration": {
            "lane": result.lane.value,
            "deterministic_tools": [i.name for i in result.tool_execution.invocations],
            "model_calls": 0,
        },
        "coverage_metadata": None,
        "evidence_sufficiency": None,
        "evidence_ledger": ledger,
        "graph_evidence_ids": [item.evidence_id for item in ledger],
        "calculations": list(result.tool_execution.calculations),
        "_orchestration_result": result,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("template,execution,facts,plan", ELIGIBLE_CASES)
async def test_every_eligible_class_skips_all_model_calls(
    monkeypatch, template, execution, facts, plan,
):
    result, ledger = _case(
        template=template, execution=execution, facts=facts, plan=plan,
        lane=Lane.CATALOG if template is da.TemplateClass.COVERAGE else Lane.FAST,
    )
    monkeypatch.setattr(middleware_app, "config", _config())
    probe = AsyncMock(side_effect=AssertionError("model health must not be probed"))
    invoke = AsyncMock(side_effect=AssertionError("model must not be called"))
    monkeypatch.setattr(middleware_app, "_check_model_health", probe)
    monkeypatch.setattr(middleware_app, "_invoke_model", invoke)

    response = await middleware_app._answer_query_context(
        QueryRequest(question="deterministic question"),
        _context(result, ledger, facts),
    )

    assert response.answer_origin == "deterministic"
    assert response.generation_skipped is True
    assert response.generation_skip_reason == "complete_deterministic_route"
    assert response.orchestration["model_calls"] == 0
    assert response.answer_validation["numeric_claims_unsupported"] == 0
    probe.assert_not_awaited()
    invoke.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "qualitative", "partial", "failed_tool", "conflict", "stale_policy",
        "write", "write_attempted", "performed_write", "sentiment",
    ],
)
async def test_ineligible_classes_call_model_exactly_once(monkeypatch, mutation):
    template, execution, facts, plan = _eligible(1)
    result, ledger = _case(template=template, execution=execution, facts=list(facts), plan=plan)
    context = _context(result, ledger, list(facts))
    if mutation == "qualitative":
        result.plan.obligations = AnswerObligations(
            qualitative=True, evidence_modes=("facts", "documents"),
        )
    elif mutation == "partial":
        result.tool_execution.complete = False
        result.tool_execution.incomplete_reason_codes = [dr.INCOMPLETE_METRIC_COVERAGE]
    elif mutation == "failed_tool":
        result.tool_execution.error = True
        result.tool_execution.invocations[0].error = "failed"
    elif mutation == "conflict":
        context["retrieval"]["facts"][0]["conflict"] = True
        result.merged_facts[0]["conflict"] = True
    elif mutation == "stale_policy":
        context["freshness"]["overall"] = "stale"
        context["freshness"]["stale_sources_used"] = ["fundamentals"]
    elif mutation == "write":
        result.tool_execution.invocations[0].name = "refresh_data"
    elif mutation == "write_attempted":
        context["_write_requested"] = True
    elif mutation == "performed_write":
        context["freshness"]["refreshed_during_query"] = ["fundamentals"]
    elif mutation == "sentiment":
        result.tool_execution.invocations[0].reason_code = dr.REASON_SENTIMENT

    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    invoke = AsyncMock(return_value=("model answer", []))
    monkeypatch.setattr(middleware_app, "_invoke_model", invoke)

    response = await middleware_app._answer_query_context(
        QueryRequest(question="deterministic question"), context,
    )

    assert response.answer == "model answer"
    assert response.answer_origin == "model"
    assert response.generation_skipped is False
    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_renderer_failure_falls_back_to_one_model_call(monkeypatch):
    template, execution, facts, plan = _eligible(1)
    result, ledger = _case(template=template, execution=execution, facts=facts, plan=plan)
    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    invoke = AsyncMock(return_value=("single fallback", []))
    monkeypatch.setattr(middleware_app, "_invoke_model", invoke)
    monkeypatch.setattr(
        da, "render_deterministic_answer", MagicMock(side_effect=RuntimeError("renderer broke")),
    )

    response = await middleware_app._answer_query_context(
        QueryRequest(question="deterministic question"),
        _context(result, ledger, facts),
    )

    assert response.answer == "single fallback"
    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_feature_off_preserves_model_path_metadata(monkeypatch):
    template, execution, facts, plan = _eligible(1)
    result, ledger = _case(template=template, execution=execution, facts=facts, plan=plan)
    monkeypatch.setattr(middleware_app, "config", _config(enable_deterministic_answers=False))
    monkeypatch.setattr(middleware_app, "_check_model_health", AsyncMock(return_value=True))
    invoke = AsyncMock(return_value=("legacy model answer", []))
    monkeypatch.setattr(middleware_app, "_invoke_model", invoke)

    response = await middleware_app._answer_query_context(
        QueryRequest(question="deterministic question"),
        _context(result, ledger, facts),
    )

    assert response.answer == "legacy model answer"
    assert response.answer_origin is None
    assert response.generation_skipped is None
    assert response.generation_skip_reason is None
    invoke.assert_awaited_once()


def test_stream_emits_one_bounded_content_event_then_one_terminal_metadata(monkeypatch):
    template, execution, facts, plan = _eligible(1)
    result, ledger = _case(template=template, execution=execution, facts=facts, plan=plan)
    context = _context(result, ledger, facts)

    async def build_context(_request):
        emitter = middleware_app._stream_emitter()
        emitter.tool_started("get_fundamentals", subquery_id="sq0")
        emitter.tool_completed("get_fundamentals", "ok", count=1, subquery_id="sq0")
        return context

    monkeypatch.setattr(middleware_app, "config", _config())
    monkeypatch.setattr(middleware_app, "_build_query_context", build_context)
    monkeypatch.setattr(
        middleware_app, "_check_model_health",
        AsyncMock(side_effect=AssertionError("model must not be probed")),
    )

    response = TestClient(middleware_app.app).post(
        "/query/stream", json={"question": "deterministic question"},
    )

    assert response.status_code == 200
    assert response.text.count("event: token\n") == 1
    assert response.text.count("event: metadata\n") == 1
    assert response.text.index("event: tool_completed\n") < response.text.index("event: token\n")
    assert response.text.index("event: token\n") < response.text.index("event: metadata\n")
    assert '"answer_origin": "deterministic"' in response.text
    assert '"generation_skipped": true' in response.text


@pytest.mark.asyncio
async def test_graph_and_evidence_trace_record_truthful_generation_skip(monkeypatch):
    from src.middleware.evidence_trace import EvidenceTraceCollector
    from src.middleware.graph_observer import TraceHub

    template, execution, facts, plan = _eligible(1)
    result, ledger = _case(template=template, execution=execution, facts=facts, plan=plan)
    config = _config(enable_graph_observer=True)
    hub = TraceHub()
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "graph_hub", hub)
    request = QueryRequest(
        question="deterministic question", include_evidence_trace=True,
    )
    emitter = middleware_app._install_query_emitter(request, chat_events=False)
    context = _context(result, ledger, facts)
    context["include_evidence_trace"] = True
    collector = EvidenceTraceCollector(
        answer_policy="graded",
        grounding_level="grounded",
        raw_question=request.question,
        retrieval_query=request.question,
        facts=facts,
        documents=[],
    )
    collector.record_evidence_ledger(ledger)
    collector.record_orchestration({"lane": "fast", "model_calls": 0})
    middleware_app._evidence_trace_var.set(collector)
    middleware_app._emit_graph_evidence(context["retrieval"], ledger)

    response = await middleware_app._answer_query_context(request, context)

    trace = response.evidence_trace
    assert trace["answer_origin"] == "deterministic"
    assert trace["generation_skipped"] is True
    assert trace["deterministic_template"] == "fact"
    assert trace["system_prompt"] == trace["user_prompt"] == ""
    snapshot = hub.snapshot(emitter.query_id)
    answer = next(node for node in snapshot["nodes"] if node["kind"] == "answer")
    assert answer["metadata"]["answer_origin"] == "deterministic"
    assert answer["metadata"]["generation_skipped"] is True
    assert snapshot["complete"] is True


def test_empty_complete_result_has_explicit_deterministic_answer():
    execution = _execution(
        "query_facts", dr.REASON_THRESHOLD,
        {"metric": "revenue", "results": []},
        arguments={"metric": "revenue", "op": "gt", "value": 1000},
    )
    result, ledger = _case(
        template=da.TemplateClass.BOUNDED_SET,
        execution=execution,
        facts=[],
        plan=_plan(intents=("comparison",)),
    )

    contract = da.check_final_answer_contract(
        result, evidence_ledger=ledger, freshness={"overall": "fresh"},
    )
    rendered = da.render_deterministic_answer(result, evidence_ledger=ledger)

    assert contract.eligible is True
    assert "No stored results matched" in rendered.text


def test_threshold_result_at_bound_is_not_claimed_complete_without_tool_proof():
    rows = [
        {"ticker": f"T{i}", "metric": "revenue", "value": i, "source_type": "tool"}
        for i in range(10)
    ]
    execution = _execution(
        "query_facts", dr.REASON_THRESHOLD,
        {"metric": "revenue", "results": rows},
        arguments={"metric": "revenue", "op": "gt", "value": 0},
    )
    result, ledger = _case(
        template=da.TemplateClass.BOUNDED_SET,
        execution=execution,
        facts=rows,
        plan=_plan(intents=("comparison",)),
    )

    contract = da.check_final_answer_contract(
        result, evidence_ledger=ledger, freshness={"overall": "fresh"},
    )

    assert contract.eligible is False
    assert contract.reason == "result_completeness_unknown"


def test_registered_write_tool_is_ineligible_even_under_read_reason_code():
    from src.middleware.tools.base import REGISTRY, Tool

    tool = Tool(
        name="custom_write_for_contract_test",
        description="test only",
        parameters={"type": "object", "properties": {}},
        handler=lambda _store: {},
        write=True,
    )
    REGISTRY[tool.name] = tool
    try:
        execution = _execution(
            tool.name,
            dr.REASON_FUNDAMENTALS,
            {"ticker": "NVDA", "fundamentals": {"revenue": 26.0}},
            arguments={"ticker": "NVDA", "metrics": ["revenue"]},
        )
        facts = [{
            "ticker": "NVDA", "metric": "revenue", "value": 26.0,
            "source_type": "tool",
        }]
        result, ledger = _case(
            template=da.TemplateClass.FACT, execution=execution, facts=facts,
        )

        contract = da.check_final_answer_contract(
            result, evidence_ledger=ledger, freshness={"overall": "fresh"},
        )
    finally:
        REGISTRY.pop(tool.name, None)

    assert contract.eligible is False
    assert contract.reason == "write_route"


def test_complete_coverage_list_requires_exact_count_and_list_fidelity():
    template, execution, facts, plan = _eligible(0)
    execution.invocations[0].result["total_matching"] = 3
    result, ledger = _case(
        template=template,
        execution=execution,
        facts=facts,
        plan=plan,
        lane=Lane.CATALOG,
    )

    contract = da.check_final_answer_contract(
        result, evidence_ledger=ledger, freshness={"overall": "fresh"},
    )

    assert contract.eligible is False
    assert contract.reason == "result_completeness_unknown"


def test_multi_entity_coverage_renders_and_cites_every_invocation():
    from src.middleware import adaptive_orchestrator as ao
    from src.middleware.answer_validator import validate_deterministic_answer

    invocations = [
        dr.ExecutedInvocation(
            name="describe_coverage",
            arguments={
                "operation": "security_sources", "ticker": ticker,
                "filters": {"item_type": "transcript"},
            },
            subquery_id="sq0",
            reason_code=dr.REASON_COVERAGE,
            result={
                "status": "ok", "complete": True, "covered": True,
                "item_types": item_types, "sources": [],
            },
        )
        for ticker, item_types in (
            ("AMD", ["transcript"]), ("NVDA", ["sec_filing"]),
        )
    ]
    execution = dr.ExecutionResult(invocations=invocations, complete=True)
    facts = ao._merge_facts(ao._normalize_tool_facts(execution), [])
    plan = _plan(
        intents=("coverage_check",), metrics=(), evidence_modes=("catalog",),
    )
    result, ledger = _case(
        template=da.TemplateClass.COVERAGE,
        execution=execution,
        facts=facts,
        plan=plan,
        lane=Lane.CATALOG,
    )
    result.set_complete = True
    result.result_set_size = 1

    contract = da.check_final_answer_contract(
        result, evidence_ledger=ledger, freshness={"overall": "fresh"},
    )
    rendered = da.render_deterministic_answer(result, evidence_ledger=ledger)
    validation = validate_deterministic_answer(
        rendered.text, ledger, calculations=list(rendered.validation_values),
    )

    assert contract.eligible is True
    assert "AMD: present [E1]" in rendered.text
    assert "NVDA: not present [E2]" in rendered.text
    assert "1 of 2" in rendered.text
    assert validation.numeric_claims_unsupported == 0


def test_fact_route_cannot_skip_when_any_requested_metric_is_missing():
    execution = _execution(
        "get_fundamentals",
        dr.REASON_FUNDAMENTALS,
        {"ticker": "NVDA", "fundamentals": {"revenue": 26.0}},
        arguments={"ticker": "NVDA", "metrics": ["revenue", "gross_margin"]},
    )
    facts = [{
        "ticker": "NVDA", "metric": "revenue", "value": 26.0,
        "source_type": "tool",
    }]
    result, ledger = _case(
        template=da.TemplateClass.FACT,
        execution=execution,
        facts=facts,
        plan=_plan(metrics=("revenue", "gross_margin")),
    )

    contract = da.check_final_answer_contract(
        result, evidence_ledger=ledger, freshness={"overall": "fresh"},
    )

    assert contract.eligible is False
    assert contract.reason == "result_completeness_unknown"

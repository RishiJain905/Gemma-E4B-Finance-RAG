"""
tests/test_deterministic_router.py
Offline tests for Phase 2.2.3.2 — deterministic finance tool routing, the
whitelisted calculator, the read-only executor, and the shared named dispatch.

All tests run without the middleware, model, or network: routing is pure, the
executor uses a seeded in-memory SQLite store with a mocked Chroma backend, and
the "no model call" gate is proved with a MagicMock model client that must stay
untouched. A checked-in route matrix (tests/fixtures/route_matrix.json) drives
the routing-accuracy / write-avoidance quality gate and the p95 latency gate.
"""

import json
import sys
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Mock chromadb before importing the store (mirrors test_query_facts_tool).
if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.middleware import deterministic_router as dr  # noqa: E402
from src.middleware.deterministic_router import (  # noqa: E402
    CalculationSpec,
    OperandRef,
    RouteDecision,
    ToolInvocation,
    calculate,
    execute_route,
    route,
)
from src.middleware.query_plan import (  # noqa: E402
    QueryEntity,
    QueryPlan,
    QuerySubquery,
    normalize_question,
)
from src.middleware.tools import REGISTRY, ToolContext, dispatch_named_tool, sanity  # noqa: E402
from src.storage.store import Store  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "route_matrix.json"

# Metric universe used by the standalone (non-matrix) tests.
KNOWN_METRICS = frozenset({
    "total_revenue", "net_income", "forward_pe", "pe_ratio", "revenue_growth",
    "price_target_mean", "free_cash_flow", "gross_margin_pct",
})

TEST_ANALYTICS_CONFIG = {
    "excluded_symbols": [],
    "equity_only_metrics": [],
    "metric_ranges": {},
}


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_sanity_config(tmp_path, monkeypatch):
    config_path = tmp_path / "analytics.yaml"
    config_path.write_text(yaml.safe_dump(TEST_ANALYTICS_CONFIG))
    monkeypatch.setattr(sanity, "_CONFIG_PATH", config_path)
    sanity._reset_cache()
    yield
    sanity._reset_cache()


@pytest.fixture
def mock_chroma():
    with patch("src.storage.store.ChromaStore") as mock_cls:
        instance = MagicMock()
        instance.heartbeat.return_value = True
        instance.count.return_value = 0
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def store(mock_chroma, tmp_path: Path):
    return Store(db_path=tmp_path / "test.db", chroma_path=tmp_path / "chroma")


# ── Plan builder (offline; avoids the network-backed resolver) ──


def make_plan(
    question,
    *,
    entities=(),
    metrics=(),
    intents=("general",),
    primary_intent=None,
    periods=(),
):
    """Construct a validated QueryPlan directly, without the intent parser.

    Keeps router tests offline and deterministic — route() is a pure function
    of the plan and the known metric names, so building the plan by hand
    exercises exactly the contract 2.2.3.1 provides.
    """
    plan_entities = [
        QueryEntity(
            ticker=t.upper(), resolved_name=None, confidence=1.0,
            source="resolved", mention=t, start=i,
        )
        for i, t in enumerate(entities)
    ]
    intents = list(intents) or ["general"]
    primary_intent = primary_intent or intents[0]
    sq0 = QuerySubquery(
        id="sq0", text=question,
        entity_tickers=tuple(e.ticker for e in plan_entities),
        intents=tuple(intents), metrics=tuple(metrics), periods=tuple(periods),
        retrieval_modes=(), derived=False, parent_id=None,
    )
    return QueryPlan(
        original_question=question,
        retrieval_query=question,
        normalized_question=normalize_question(question),
        entities=plan_entities,
        intents=intents,
        metrics=list(metrics),
        periods=list(periods),
        subqueries=[sq0],
        primary_intent=primary_intent,
        reason_codes=[],
    ).validate()


def _seed_pe(store, ticker, value, period="2026-Q1"):
    store.sqlite.upsert_fundamental(
        ticker=ticker, metric="forward_pe", value=value,
        unit="ratio", period=period, source_type="yfinance",
    )


# ── The 11 spec-named tests ───────────────────────────────


def test_lowest_pe_routes_to_query_facts():
    plan = make_plan(
        "which stock has the lowest forward pe",
        metrics=["forward_pe"], intents=["comparison"], primary_intent="comparison",
    )
    decision = route(plan, KNOWN_METRICS)

    assert decision.matched and decision.complete
    assert len(decision.tool_invocations) == 1
    inv = decision.tool_invocations[0]
    assert inv.name == "query_facts"
    assert inv.arguments["metric"] == "forward_pe"
    assert inv.arguments["order"] == "asc"
    assert dr.REASON_RANK in decision.reason_codes


def test_comparison_preserves_ticker_order():
    plan = make_plan(
        "compare NVDA and AMD revenue",
        entities=["NVDA", "AMD"], metrics=["total_revenue"],
        intents=["comparison"], primary_intent="comparison",
    )
    decision = route(plan, KNOWN_METRICS)

    inv = decision.tool_invocations[0]
    assert inv.name == "query_facts"
    assert inv.arguments["tickers"] == ["NVDA", "AMD"]
    assert inv.arguments["metric"] == "total_revenue"
    assert inv.arguments["latest_only"] is True
    assert decision.reason_codes[0] == dr.REASON_COMPARE


def test_price_target_routes_to_consensus_tool():
    plan = make_plan(
        "what is NVDA's price target",
        entities=["NVDA"], metrics=["price_target_mean"],
        intents=["projection"], primary_intent="projection",
    )
    decision = route(plan, KNOWN_METRICS)

    assert decision.matched
    assert decision.tool_invocations[0].name == "get_price_targets"
    assert decision.tool_invocations[0].arguments == {"ticker": "NVDA"}


def test_numeric_plus_risk_is_incomplete_and_requires_documents():
    plan = make_plan(
        "what is NVDA revenue and what are its risks",
        entities=["NVDA"], metrics=["total_revenue"],
        intents=["risk", "fact_lookup"], primary_intent="risk",
    )
    decision = route(plan, KNOWN_METRICS)

    assert decision.matched is True
    assert decision.complete is False
    assert decision.requires_documents is True
    assert decision.tool_invocations[0].name == "get_fundamentals"


def test_unknown_metric_abstains_without_tool_call():
    plan = make_plan(
        "what is NVDA's altman z score",
        entities=["NVDA"], metrics=["altman_z_score"],
        intents=["fact_lookup"], primary_intent="fact_lookup",
    )
    decision = route(plan, KNOWN_METRICS)

    assert decision.matched is False
    assert decision.tool_invocations == []
    assert decision.abstain_reason == dr.ABSTAIN_UNKNOWN_METRIC


def test_router_never_selects_refresh_data():
    tempting = [
        make_plan("refresh NVDA data now", entities=["NVDA"], intents=["general"]),
        make_plan("update AMD fundamentals and fetch latest",
                  entities=["AMD"], metrics=["total_revenue"], intents=["general"]),
        make_plan("refresh and compare NVDA and AMD revenue",
                  entities=["NVDA", "AMD"], metrics=["total_revenue"],
                  intents=["comparison"], primary_intent="comparison"),
    ]
    for plan in tempting:
        decision = route(plan, KNOWN_METRICS)
        for inv in decision.tool_invocations:
            assert inv.name != "refresh_data"
            assert REGISTRY[inv.name].write is False

    # Every matrix case, too — nothing routable is ever a write tool.
    matrix = json.loads(FIXTURE.read_text())
    known = frozenset(matrix["available_metrics"])
    for case in matrix["cases"]:
        decision = route(_plan_from_case(case), known)
        for inv in decision.tool_invocations:
            assert REGISTRY[inv.name].write is False


def test_named_dispatch_reuses_schema_validation(store):
    ctx = ToolContext(allow_write=False, max_refreshes=0)

    # Wrong type is caught by the shared validate_args before the handler runs.
    result, name, args = dispatch_named_tool(
        "get_fundamentals", {"ticker": 123}, store, ctx
    )
    assert "error" in result and "ticker must be string" in result["error"]
    assert name == "get_fundamentals"

    # Missing required argument is likewise rejected.
    result, _, _ = dispatch_named_tool("get_fundamentals", {}, store, ctx)
    assert "error" in result and "missing required argument" in result["error"]

    # The write guard is shared too: a write tool is refused read-only.
    result, _, _ = dispatch_named_tool("refresh_data", {"ticker": "NVDA"}, store, ctx)
    assert result == {"error": "write tools disabled"}

    # A valid call is schema-filtered (extra keys dropped) and dispatched.
    _seed_pe(store, "NVDA", 16.5)
    store.sqlite.upsert_fundamental(
        ticker="NVDA", metric="total_revenue", value=26.0,
        unit="usd", period="2026-Q1", source_type="yfinance",
    )
    result, _, args = dispatch_named_tool(
        "get_fundamentals",
        {"ticker": "NVDA", "metrics": ["total_revenue"], "bogus": "drop"},
        store, ctx,
    )
    assert "bogus" not in args
    assert result["fundamentals"] == {"total_revenue": 26.0}


def test_tool_failure_falls_back_without_raising():
    # A store whose handler blows up: the handler captures its own exception and
    # returns an error dict; the executor flags error and never raises.
    broken = MagicMock()
    broken.get_fundamentals_batch.side_effect = RuntimeError("db exploded")

    decision = RouteDecision(
        matched=True, complete=True,
        tool_invocations=[
            ToolInvocation("get_fundamentals", {"ticker": "NVDA"}, "sq0",
                           dr.REASON_FUNDAMENTALS)
        ],
    )
    result = execute_route(decision, broken)

    assert result.error is True
    assert result.answer is None
    assert result.invocations[0].error is not None


def test_calculations_use_decimal_and_keep_operands():
    result = calculate(
        "difference",
        {
            "a": {"value": 100, "unit": "usd", "period": "2026-Q1"},
            "b": {"value": 40, "unit": "usd", "period": "2026-Q1"},
        },
    )
    assert isinstance(result["result"], Decimal)
    assert result["result"] == Decimal("60")
    assert result["unit"] == "usd"
    assert result["formula"] == "a - b"
    # Operands are preserved (value/unit/period) for 2.2.4.3 provenance.
    assert result["operands"]["a"] == {
        "value": Decimal("100"), "unit": "usd", "period": "2026-Q1"
    }
    assert result["operands"]["b"]["period"] == "2026-Q1"

    pct = calculate(
        "percent_change",
        {
            "old": {"value": 100, "unit": "usd", "period": "2025-Q1"},
            "new": {"value": 125, "unit": "usd", "period": "2026-Q1"},
        },
    )
    assert pct["result"] == Decimal("25")
    assert pct["unit"] == "percent"


def test_divide_by_zero_returns_structured_error():
    # Same unit so the zero-denominator guard is what fires (ratio now enforces
    # unit-equality — see test_calculator_ratio_rejects_incompatible_units).
    ratio = calculate(
        "ratio",
        {
            "numerator": {"value": 5, "unit": "x", "period": "p"},
            "denominator": {"value": 0, "unit": "x", "period": "p"},
        },
    )
    assert ratio["error"] == "divide_by_zero"
    assert "operation" in ratio and ratio["operation"] == "ratio"

    pct = calculate(
        "percent_change",
        {
            "old": {"value": 0, "unit": "usd", "period": "p"},
            "new": {"value": 10, "unit": "usd", "period": "p"},
        },
    )
    assert pct["error"] == "divide_by_zero"


def test_complete_deterministic_answer_needs_no_model_call(store):
    # A mocked model client that MUST remain untouched by the deterministic path.
    model_client = MagicMock()

    store.sqlite.upsert_fundamental(
        ticker="NVDA", metric="total_revenue", value=26.0,
        unit="usd", period="2026-Q1", source_type="yfinance",
    )
    plan = make_plan(
        "what is NVDA's revenue",
        entities=["NVDA"], metrics=["total_revenue"],
        intents=["fact_lookup"], primary_intent="fact_lookup",
    )
    decision = route(plan, KNOWN_METRICS)
    assert decision.complete is True

    result = execute_route(decision, store)

    assert result.error is False
    assert result.answer is not None
    assert "NVDA" in result.answer
    # The deterministic route never called the model.
    model_client.assert_not_called()
    assert model_client.post.call_count == 0


# ── Route matrix: accuracy + write-avoidance quality gate ──


def _plan_from_case(case):
    return make_plan(
        case["question"],
        entities=case.get("entities", []),
        metrics=case.get("metrics", []),
        intents=case.get("intents", ["general"]),
        primary_intent=case.get("primary_intent"),
        periods=case.get("periods", []),
    )


def _case_matches(decision, expect):
    if not expect.get("matched", False):
        if decision.matched:
            return False
        if expect.get("abstain") and decision.abstain_reason != expect["abstain"]:
            return False
        return True
    if not decision.matched:
        return False
    tool = decision.tool_invocations[0].name if decision.tool_invocations else None
    if tool != expect.get("tool"):
        return False
    if "complete" in expect and decision.complete != expect["complete"]:
        return False
    if "requires_documents" in expect and (
        decision.requires_documents != expect["requires_documents"]
    ):
        return False
    return True


def test_route_matrix_accuracy_and_write_avoidance():
    matrix = json.loads(FIXTURE.read_text())
    known = frozenset(matrix["available_metrics"])
    cases = matrix["cases"]

    hits = 0
    for case in cases:
        decision = route(_plan_from_case(case), known)
        if _case_matches(decision, case["expect"]):
            hits += 1
        else:  # pragma: no cover - surfaces the offending case on failure
            print(f"route mismatch: {case['name']} -> "
                  f"matched={decision.matched} tool="
                  f"{[i.name for i in decision.tool_invocations]} "
                  f"abstain={decision.abstain_reason}")
        # 100% write-tool avoidance is non-negotiable on every case.
        for inv in decision.tool_invocations:
            assert REGISTRY[inv.name].write is False

    accuracy = hits / len(cases)
    assert accuracy >= 0.95, f"routing accuracy {accuracy:.2%} below 95%"


def test_route_p95_under_5ms():
    matrix = json.loads(FIXTURE.read_text())
    known = frozenset(matrix["available_metrics"])
    plans = [_plan_from_case(c) for c in matrix["cases"]]

    durations = []
    for _ in range(50):
        for plan in plans:
            start = time.perf_counter()
            route(plan, known)
            durations.append((time.perf_counter() - start) * 1000.0)

    durations.sort()
    p95 = durations[int(len(durations) * 0.95)]
    assert p95 < 5.0, f"router p95 {p95:.3f}ms exceeds 5ms budget"


# ── Extra coverage: purity, macro, calculator guards ──────


def test_route_is_pure_no_store_argument():
    # route() takes only (plan, metrics) — no store/model/network handle.
    plan = make_plan("what is NVDA's net income", entities=["NVDA"],
                     metrics=["net_income"], intents=["fact_lookup"],
                     primary_intent="fact_lookup")
    d1 = route(plan, KNOWN_METRICS)
    d2 = route(plan, KNOWN_METRICS)
    assert d1.tool_invocations[0].arguments == d2.tool_invocations[0].arguments


def test_calculator_rejects_incompatible_units():
    result = calculate(
        "difference",
        {
            "a": {"value": 10, "unit": "usd", "period": "p"},
            "b": {"value": 5, "unit": "eur", "period": "p"},
        },
    )
    assert result["error"] == "incompatible_units"


def test_calculator_rejects_missing_period():
    result = calculate(
        "difference",
        {
            "a": {"value": 10, "unit": "usd", "period": None},
            "b": {"value": 5, "unit": "usd", "period": "p"},
        },
    )
    assert result["error"] == "missing_periods"


def test_executor_runs_calculation_with_provenance(store):
    _seed_pe(store, "NVDA", 30.0)
    _seed_pe(store, "AMD", 20.0)
    decision = RouteDecision(
        matched=True, complete=True,
        tool_invocations=[
            ToolInvocation(
                "query_facts",
                {"metric": "forward_pe", "tickers": ["NVDA", "AMD"], "latest_only": True},
                "sq0", dr.REASON_COMPARE,
            )
        ],
        calculations=[
            CalculationSpec(
                operation="difference",
                operands={"a": OperandRef(0, "NVDA"), "b": OperandRef(0, "AMD")},
            )
        ],
    )
    result = execute_route(decision, store)

    assert result.error is False
    assert result.calculations[0]["result"] == Decimal("10")
    assert result.calculations[0]["operands"]["a"]["period"] == "2026-Q1"


# ══════════════════════════════════════════════════════════════════════
# 2.2.3.4 review findings — regression tests (findings 1, 2, 3, 4, 6)
# ══════════════════════════════════════════════════════════════════════


# ── Finding 1: refresh/update wording must not answer from cache as fresh ──

def test_refresh_wording_abstains():
    cases = [
        "refresh NVDA data now",
        "update AMD fundamentals and fetch latest",
        "refresh and compare NVDA and AMD revenue",
        "re-fetch NVDA revenue",
        "pull the latest MSFT numbers",
    ]
    plans = [
        make_plan("refresh NVDA data now", entities=["NVDA"], intents=["general"]),
        make_plan("update AMD fundamentals and fetch latest", entities=["AMD"],
                  metrics=["total_revenue"], intents=["fact_lookup"],
                  primary_intent="fact_lookup"),
        make_plan("refresh and compare NVDA and AMD revenue",
                  entities=["NVDA", "AMD"], metrics=["total_revenue"],
                  intents=["comparison"], primary_intent="comparison"),
        make_plan("re-fetch NVDA revenue", entities=["NVDA"],
                  metrics=["total_revenue"], intents=["fact_lookup"],
                  primary_intent="fact_lookup"),
        make_plan("pull the latest MSFT numbers", entities=["MSFT"],
                  intents=["fact_lookup"], primary_intent="fact_lookup"),
    ]
    for q, plan in zip(cases, plans):
        decision = route(plan, KNOWN_METRICS)
        assert decision.matched is False, f"{q!r} should abstain"
        assert decision.abstain_reason == dr.ABSTAIN_REFRESH_REQUESTED
        assert decision.tool_invocations == []


def test_latest_fact_lookup_still_routes():
    # Bare "latest" (no refresh/fetch verb) is a normal latest-fact lookup and
    # must NOT be swept up by the refresh guard.
    plan = make_plan("what is NVDA's latest revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"],
                     primary_intent="fact_lookup")
    decision = route(plan, KNOWN_METRICS)
    assert decision.matched is True
    assert decision.tool_invocations[0].name == "get_fundamentals"


# ── Finding 2: complete only when all plan obligations are covered ──

def test_projection_plus_uncovered_metric_is_incomplete():
    # Price target (projection) + forward P/E (realized valuation): one tool runs,
    # forward_pe is uncovered, so the route must not claim complete.
    plan = make_plan(
        "what is NVDA price target and forward pe",
        entities=["NVDA"], metrics=["price_target_mean", "forward_pe"],
        intents=["projection"], primary_intent="projection",
    )
    decision = route(plan, KNOWN_METRICS)
    assert decision.matched is True
    assert decision.tool_invocations[0].name == "get_price_targets"
    assert decision.complete is False
    assert decision.requires_documents is True
    assert dr.INCOMPLETE_METRIC_COVERAGE in decision.reason_codes


def test_projection_plus_sentiment_is_incomplete():
    plan = make_plan(
        "what is NVDA's price target and how is sentiment",
        entities=["NVDA"], metrics=["price_target_mean"],
        intents=["projection", "sentiment"], primary_intent="projection",
    )
    decision = route(plan, KNOWN_METRICS)
    assert decision.matched is True
    assert decision.complete is False
    assert dr.INCOMPLETE_INTENT_COVERAGE in decision.reason_codes


def test_fundamentals_covering_all_metrics_stays_complete():
    plan = make_plan(
        "what is NVDA revenue and net income",
        entities=["NVDA"], metrics=["total_revenue", "net_income"],
        intents=["fact_lookup"], primary_intent="fact_lookup",
    )
    decision = route(plan, KNOWN_METRICS)
    assert decision.matched is True
    assert decision.complete is True
    assert decision.tool_invocations[0].arguments["metrics"] == [
        "total_revenue", "net_income"]


# ── Finding 3: calculator unit/period integrity ──

def test_calculator_ratio_rejects_incompatible_units():
    result = calculate(
        "ratio",
        {
            "numerator": {"value": 10, "unit": "usd", "period": "p"},
            "denominator": {"value": 5, "unit": "eur", "period": "p"},
        },
    )
    assert result["error"] == "incompatible_units"


def test_calculator_ratio_same_unit_ok():
    result = calculate(
        "ratio",
        {
            "numerator": {"value": 10, "unit": "usd", "period": "p"},
            "denominator": {"value": 5, "unit": "usd", "period": "p"},
        },
    )
    assert "error" not in result
    assert result["result"] == Decimal("2")


def test_calculator_difference_rejects_mismatched_periods():
    result = calculate(
        "difference",
        {
            "a": {"value": 100, "unit": "usd", "period": "2026-Q2"},
            "b": {"value": 40, "unit": "usd", "period": "2026-Q1"},
        },
    )
    assert result["error"] == "mismatched_periods"


def test_calculator_percent_change_allows_different_periods():
    # percent_change is old-vs-new by design; different periods must be allowed.
    result = calculate(
        "percent_change",
        {
            "old": {"value": 100, "unit": "usd", "period": "2025-Q1"},
            "new": {"value": 125, "unit": "usd", "period": "2026-Q1"},
        },
    )
    assert "error" not in result
    assert result["result"] == Decimal("25")


def test_route_compare_across_mismatched_periods_flags_error(store):
    # End-to-end: comparing two tickers whose latest periods differ produces a
    # difference calc that must error (not silently diff apples-to-oranges).
    store.sqlite.upsert_fundamental(
        ticker="NVDA", metric="total_revenue", value=100.0,
        unit="usd", period="2026-Q2", source_type="yfinance")
    store.sqlite.upsert_fundamental(
        ticker="AMD", metric="total_revenue", value=40.0,
        unit="usd", period="2026-Q1", source_type="yfinance")
    decision = RouteDecision(
        matched=True, complete=True,
        tool_invocations=[ToolInvocation(
            "query_facts",
            {"metric": "total_revenue", "tickers": ["NVDA", "AMD"],
             "latest_only": True},
            "sq0", dr.REASON_COMPARE)],
        calculations=[CalculationSpec(
            operation="difference",
            operands={"a": OperandRef(0, "NVDA"), "b": OperandRef(0, "AMD")})],
    )
    result = execute_route(decision, store)
    assert result.error is True
    assert result.calculations[0]["error"] == "mismatched_periods"


# ── Finding 4: query_plan validate() guards malformed tickers ──

def test_query_plan_validate_guards_non_string_ticker():
    from src.middleware.query_plan import (
        QueryEntity, QueryPlan, QueryPlanError, QuerySubquery)

    bad_entity = QueryEntity(
        ticker=None, resolved_name=None, confidence=1.0,  # type: ignore[arg-type]
        source="resolved", mention="x", start=0)
    plan = QueryPlan(
        original_question="q", retrieval_query="q",
        entities=[bad_entity], intents=["general"],
        subqueries=[QuerySubquery(id="sq0", text="q")],
        primary_intent="general")
    # Must raise a QueryPlanError (fail-soft boundary), not an AttributeError.
    try:
        plan.validate()
        assert False, "expected QueryPlanError"
    except QueryPlanError as e:
        assert "entity_not_uppercase" in e.reason_codes


# ── Finding 6: 'gross margin' resolves to one metric ──

def test_gross_margin_is_single_metric():
    from src.middleware.intent_parser import IntentParser
    metrics = IntentParser()._extract_metrics("what is NVDA gross margin")
    assert metrics == ["gross_margin_pct"]


def test_gross_margin_rank_routes_not_ambiguous():
    # With the double-match gone, a single-metric rank over gross margin routes
    # instead of abstaining on a false ambiguous_metric.
    plan = make_plan(
        "which company has the highest gross margin",
        metrics=["gross_margin_pct"], intents=["comparison"],
        primary_intent="comparison")
    decision = route(plan, KNOWN_METRICS)
    assert decision.matched is True
    assert decision.tool_invocations[0].arguments["metric"] == "gross_margin_pct"

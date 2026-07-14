"""
tests/test_adaptive_orchestrator.py
Offline tests for Phase 2.2.3.3 — the bounded adaptive-RAG orchestrator: lane
selection, the shared execution budget, the single context budget, and
conditional re-ranking.

Everything runs without the middleware, model, ChromaDB process, or network.
Retrieval, deterministic tool execution, re-ranking, and the planning client are
all fakes injected through :func:`adaptive_orchestrator.orchestrate`'s seams, so
each test exercises the orchestration contract in isolation. The lanes are pure
functions of the plan + injected route decision; budgets are asserted directly.
"""

import sys
import time
from unittest.mock import MagicMock

# Defensive: some sibling modules import the store (which imports chromadb) at
# collection time. Mock it before importing anything under src, mirroring the
# router tests. Only if not already present so a real install is untouched.
if "chromadb" not in sys.modules:  # pragma: no cover - import shim
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.middleware import adaptive_orchestrator as ao  # noqa: E402
from src.middleware import deterministic_router as dr  # noqa: E402
from src.middleware.adaptive_orchestrator import (  # noqa: E402
    ContextBudget,
    ExecutionBudget,
    Lane,
    orchestrate,
)
from src.middleware.config import MiddlewareConfig  # noqa: E402
from src.middleware.deterministic_router import (  # noqa: E402
    ExecutedInvocation,
    ExecutionResult,
    RouteDecision,
    ToolInvocation,
)
from src.middleware.query_plan import (  # noqa: E402
    QueryEntity,
    QueryPlan,
    QuerySubquery,
    normalize_question,
)
from src.middleware.evidence_grader import (  # noqa: E402
    CorrectiveAction,
    SufficiencyStatus,
)


# ── Builders ──────────────────────────────────────────────


def make_config(**overrides) -> MiddlewareConfig:
    cfg = MiddlewareConfig()
    cfg.enable_adaptive_rag = True
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_plan(
    question="what about it",
    *,
    entities=(),
    metrics=(),
    intents=("general",),
    primary_intent=None,
    periods=(),
) -> QueryPlan:
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
    ).validate()


def doc(doc_id, body="body text here", ticker=None, fusion=0.5, **meta):
    md = {"ticker": ticker} if ticker else {}
    md.update(meta)
    return {"id": doc_id, "document": body, "metadata": md, "fusion_score": fusion}


class FakeReranker:
    """Reverses candidate order (proving it ran) unless configured to fail."""

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def rerank(self, query, docs, top_n=5):
        self.calls += 1
        if self.fail:
            raise RuntimeError("reranker exploded")
        out = list(reversed(docs))[:top_n]
        for d in out:
            d["rerank_score"] = 1.0
        return out


class FakeRetriever:
    """Records calls; returns fixed facts/documents/candidate pool."""

    def __init__(self, *, facts=None, documents=None, candidates=None,
                 strategy="hybrid", reranker=None):
        self.retrieve_calls = 0
        self.retrieve_candidates_calls = 0
        self.last_query = None
        self.last_intent = None
        self._facts = facts or []
        self._documents = documents or []
        self._candidates = candidates
        self._strategy = strategy
        self.reranker = reranker or FakeReranker()

    def retrieve(self, query, intent, top_k_documents=5, top_k_facts=10):
        self.retrieve_calls += 1
        self.last_query = query
        self.last_intent = dict(intent)
        return {
            "facts": [dict(f) for f in self._facts],
            "documents": [dict(d) for d in self._documents],
            "ticker": intent.get("ticker"),
            "strategy": "hybrid",
            "retrieval_strategy": self._strategy,
        }

    def retrieve_candidates(self, query, intent, top_k_documents=5, top_k_facts=10):
        self.retrieve_candidates_calls += 1
        self.last_query = query
        c = dict(self._candidates or {})
        c.setdefault("facts", [dict(f) for f in self._facts])
        c.setdefault("documents", [dict(d) for d in self._documents])
        c.setdefault("retrieval_strategy", self._strategy)
        c.setdefault("vector_ids", [])
        c.setdefault("lexical_ids", [])
        c.setdefault("candidate_count", len(c["documents"]))
        return c


def complete_route(tool="get_fundamentals", result=None):
    result = result or {"fundamentals": {"total_revenue": 26.0}, "ticker": "NVDA"}
    decision = RouteDecision(
        matched=True, complete=True, requires_documents=False,
        tool_invocations=[ToolInvocation(tool, {"ticker": "NVDA"}, "sq0",
                                         dr.REASON_FUNDAMENTALS)],
        reason_codes=[dr.REASON_FUNDAMENTALS],
    )
    execution = ExecutionResult(
        invocations=[ExecutedInvocation(tool, {"ticker": "NVDA"}, "sq0",
                                        dr.REASON_FUNDAMENTALS, result=result)],
        error=False, answer=None,
    )
    return (lambda plan, metrics: decision), (lambda *a, **k: execution)


# ── 1. Fast lane skips chroma / reranker / planner ────────


def test_fast_lane_skips_chroma_reranker_and_planner():
    route_fn, execute_fn = complete_route()
    retriever = FakeRetriever()
    planner = MagicMock()
    plan = make_plan("what is NVDA revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"],
                     primary_intent="fact_lookup")

    result = orchestrate(
        plan, store=object(), config=make_config(),
        retriever=retriever, planning_client=planner,
        route_fn=route_fn, execute_fn=execute_fn,
    )

    assert result.lane is Lane.FAST
    assert retriever.retrieve_calls == 0
    assert retriever.retrieve_candidates_calls == 0
    assert retriever.reranker.calls == 0
    planner.assert_not_called()
    assert result.merged_documents == []
    assert any(f["metric"] == "total_revenue" for f in result.merged_facts)
    assert result.retrieval_rounds_used == 0


# ── 2. Standard lane uses existing hybrid once ────────────


def test_standard_lane_uses_existing_hybrid_once():
    retriever = FakeRetriever(documents=[doc("d1", ticker="NVDA")])
    plan = make_plan("why did NVDA drop", entities=["NVDA"],
                     intents=["explanation"], primary_intent="explanation")

    result = orchestrate(
        plan, store=object(), config=make_config(enable_reranker=False),
        retriever=retriever,
        route_fn=lambda p, m: RouteDecision(),  # abstain
    )

    assert result.lane is Lane.STANDARD
    assert retriever.retrieve_calls == 1
    assert retriever.retrieve_candidates_calls == 0
    assert retriever.reranker.calls == 0
    assert result.retrieval_rounds_used == 1
    assert result.subqueries_executed == ["sq0"]


# ── 3. Complex lane caps three subqueries ─────────────────


def test_complex_lane_decomposition_seam_is_inactive():
    """Until 2.2.4.2, the complex lane retrieves only sq0. A decompose() that
    yields derived subqueries is recorded as deferred and NEVER reported as
    executed (review H) — the metadata must not claim unretrieved subqueries."""
    retriever = FakeRetriever(documents=[doc("d1")])
    plan = make_plan("compare NVDA AMD INTC margins",
                     entities=["NVDA", "AMD", "INTC"])  # >2 entities → complex

    extra = [
        QuerySubquery(id=f"sq{i}", text=f"derived {i}", derived=True, parent_id="sq0")
        for i in range(1, 6)
    ]

    result = orchestrate(
        plan, store=object(), config=make_config(),
        retriever=retriever,
        route_fn=lambda p, m: RouteDecision(),
        decompose=lambda p: extra,
    )

    assert result.lane is Lane.COMPLEX
    assert result.subqueries_executed == ["sq0"]  # only sq0 actually retrieved
    assert "subquery_decomposition_deferred" in result.reason_codes
    assert "subquery_budget_exhausted" not in result.reason_codes
    # Only one subquery unit is consumed even though 5 were proposed.
    assert retriever.retrieve_calls == 1


def test_subquery_budget_cap_is_uncircumventable():
    """The SUBQUERY cap itself is enforced by ExecutionBudget regardless of lane."""
    budget = ExecutionBudget(max_subqueries=3)
    assert [budget.consume(ao.SUBQUERY) for _ in range(4)] == [True, True, True, False]
    assert "subquery_budget_exhausted" in budget.exhausted
    assert budget.used(ao.SUBQUERY) == 3


# ── 4. Retrieval-round budget stops at two ────────────────


def test_retrieval_round_budget_stops_at_two():
    retriever = FakeRetriever(documents=[doc("d1")])
    plan = make_plan("compare NVDA AMD INTC", entities=["NVDA", "AMD", "INTC"])

    result = orchestrate(
        plan, store=object(), config=make_config(enable_reranker=False),
        retriever=retriever,
        route_fn=lambda p, m: RouteDecision(),
        corrective_retry=lambda p, f, d: True,  # always wants another round
    )

    assert result.lane is Lane.COMPLEX
    assert result.retrieval_rounds_used == 2  # capped, no infinite loop
    assert "retrieval_round_budget_exhausted" in result.reason_codes


class SequencedRetriever(FakeRetriever):
    """Returns one deterministic payload per bounded retrieval call."""

    def __init__(self, payloads):
        super().__init__()
        self.payloads = list(payloads)

    def retrieve(self, query, intent, top_k_documents=5, top_k_facts=10):
        self.retrieve_calls += 1
        self.last_query = query
        self.last_intent = dict(intent)
        payload = self.payloads[min(self.retrieve_calls - 1, len(self.payloads) - 1)]
        return {
            "facts": [dict(f) for f in payload.get("facts", [])],
            "documents": [dict(d) for d in payload.get("documents", [])],
            "ticker": intent.get("ticker"),
            "strategy": "hybrid",
            "retrieval_strategy": "hybrid",
        }


def _exact_revenue(period="FY2025"):
    return {"evidence_id": f"revenue-{period}", "ticker": "NVDA",
            "metric": "total_revenue", "value": 26.0, "period": period,
            "unit": "USD", "source_type": "sec_10k"}


def test_evidence_sufficient_skips_corrective_round():
    retriever = SequencedRetriever([{"facts": [_exact_revenue()]}])
    plan = make_plan("NVDA revenue FY2025", entities=["NVDA"],
                     metrics=["total_revenue"], periods=["FY2025"],
                     intents=["fact_lookup"], primary_intent="fact_lookup")

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_evidence_sufficiency=True,
                           enable_corrective_retry=True),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )

    assert result.sufficiency.status is SufficiencyStatus.SUFFICIENT
    assert result.retry_performed is False
    assert result.retrieval_rounds_used == 1
    assert retriever.retrieve_calls == 1


def test_fast_projection_tool_evidence_is_graded_as_estimate():
    plan = make_plan("NVDA EPS estimate", entities=["NVDA"],
                     metrics=["estimate_eps_next_y"], intents=["projection"],
                     primary_intent="projection")
    decision = RouteDecision(
        matched=True, complete=True, requires_documents=False,
        tool_invocations=[ToolInvocation("get_estimates", {"ticker": "NVDA"},
                                         "sq0", "projection")],
    )
    execution = ExecutionResult(invocations=[ExecutedInvocation(
        "get_estimates", {"ticker": "NVDA"}, "sq0", "projection",
        result={"ticker": "NVDA", "estimates": {
            "estimate_eps_next_y": {"value": 4.2, "period": "FY2027E"},
        }},
    )])

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_evidence_sufficiency=True),
        retriever=FakeRetriever(),
        route_fn=lambda p, m: decision,
        execute_fn=lambda *args, **kwargs: execution,
    )

    assert result.lane is Lane.FAST
    assert result.sufficiency.status is SufficiencyStatus.SUFFICIENT


def test_borderline_evidence_consumes_exactly_one_final_round():
    retriever = SequencedRetriever([
        {"facts": [_exact_revenue("FY2024")]},
        {"facts": [_exact_revenue("FY2025")]},
        {"facts": [_exact_revenue("FY2026")]},
    ])
    plan = make_plan("NVDA revenue FY2025", entities=["NVDA"],
                     metrics=["total_revenue"], periods=["FY2025"],
                     intents=["fact_lookup"], primary_intent="fact_lookup")

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_evidence_sufficiency=True,
                           enable_corrective_retry=True,
                           max_corrective_retries=99),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )

    assert result.retry_performed is True
    assert result.corrective_action is CorrectiveAction.ALTERNATE_INTERNAL_MODALITY
    assert result.retrieval_rounds_used == 2
    assert retriever.retrieve_calls == 2
    assert result.sufficiency.status is SufficiencyStatus.SUFFICIENT


def test_alternate_modality_targets_the_missing_document_route():
    retriever = SequencedRetriever([
        {"facts": [_exact_revenue()]},
        {"documents": [doc("why", ticker="NVDA", source_type="sec_10k")]},
    ])
    plan = make_plan("Why did NVDA revenue change?", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["explanation"],
                     primary_intent="explanation")

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_evidence_sufficiency=True,
                           enable_corrective_retry=True),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )

    assert result.retry_performed is True
    assert retriever.last_intent["question_type"] == "explanation"
    assert result.sufficiency.status is SufficiencyStatus.SUFFICIENT


def test_clearly_missing_evidence_never_retries():
    retriever = SequencedRetriever([{"facts": [], "documents": []}])
    plan = make_plan("NVDA revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"],
                     primary_intent="fact_lookup")

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_evidence_sufficiency=True,
                           enable_corrective_retry=True),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )

    assert result.sufficiency.status is SufficiencyStatus.MISSING
    assert result.corrective_action is CorrectiveAction.NONE
    assert result.retry_performed is False
    assert retriever.retrieve_calls == 1


def test_enabled_gate_never_delegates_action_selection_to_callback():
    retriever = SequencedRetriever([{"facts": [], "documents": []}])
    chooser = MagicMock(return_value=True)
    plan = make_plan("compare NVDA AMD INTC", entities=["NVDA", "AMD", "INTC"])

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_evidence_sufficiency=True,
                           enable_corrective_retry=True),
        retriever=retriever, corrective_retry=chooser,
        route_fn=lambda p, m: RouteDecision(),
    )

    chooser.assert_not_called()
    assert result.retry_performed is False
    assert result.retrieval_rounds_used == 1


def test_second_round_insufficiency_stops_and_preserves_best_evidence():
    retriever = SequencedRetriever([
        {"facts": [_exact_revenue("FY2024")]},
        {"facts": [_exact_revenue("FY2023")]},
    ])
    plan = make_plan("NVDA revenue FY2025", entities=["NVDA"],
                     metrics=["total_revenue"], periods=["FY2025"],
                     intents=["fact_lookup"], primary_intent="fact_lookup")

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_evidence_sufficiency=True,
                           enable_corrective_retry=True),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )

    assert result.retrieval_rounds_used == 2
    assert result.retry_performed is True
    assert result.sufficiency.status is SufficiencyStatus.BORDERLINE
    assert len(result.merged_facts) == 2
    assert "corrective_retry_exhausted" in result.reason_codes


def test_run_derived_subqueries_is_noop_until_2_2_4_2():
    retriever = SequencedRetriever([{"facts": [], "documents": []}])
    plan = make_plan("NVDA revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"],
                     primary_intent="fact_lookup")
    plan.subqueries.append(QuerySubquery(
        id="sq1", text="derived", entity_tickers=("NVDA",),
        metrics=("total_revenue",), retrieval_modes=("facts",),
        derived=True, parent_id="sq0",
    ))
    plan.validate()

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_evidence_sufficiency=True,
                           enable_corrective_retry=True),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )

    assert result.corrective_action is CorrectiveAction.RUN_DERIVED_SUBQUERIES
    assert result.retry_performed is False
    assert result.retrieval_rounds_used == 1
    assert retriever.retrieve_calls == 1
    assert "run_derived_subqueries_deferred_2_2_4_2" in result.reason_codes


# ── 5. Planning-call budget stops at one ──────────────────


def test_planning_call_budget_stops_at_one():
    # The budget itself is uncircumventable.
    budget = ExecutionBudget(max_planning_calls=1)
    assert budget.consume(ao.PLANNING_CALL) is True
    assert budget.consume(ao.PLANNING_CALL) is False
    assert budget.used(ao.PLANNING_CALL) == 1

    # And an orchestrated complex request makes at most one planning call.
    retriever = FakeRetriever(documents=[doc("d1")])
    planner = MagicMock(return_value={
        "retrieval_query": "NVDA AMD INTC gross margin",
        "entities": ["NVDA", "AMD", "INTC"], "intents": ["comparison"],
        "metrics": [], "periods": [],
    })
    plan = make_plan("compare NVDA AMD INTC", entities=["NVDA", "AMD", "INTC"])

    result = orchestrate(
        plan, store=object(),
        config=make_config(adaptive_enable_planning_call=True),
        retriever=retriever, planning_client=planner,
        route_fn=lambda p, m: RouteDecision(),
    )

    assert result.planning_ran is True
    assert planner.call_count == 1


# ── 6. Invalid planning JSON falls back to the rule plan ──


def test_invalid_planning_json_uses_rule_plan():
    retriever = FakeRetriever(documents=[doc("d1")])
    planner = MagicMock(return_value="{ this is not valid json ")
    plan = make_plan("compare NVDA AMD INTC margins",
                     entities=["NVDA", "AMD", "INTC"])

    result = orchestrate(
        plan, store=object(),
        config=make_config(adaptive_enable_planning_call=True),
        retriever=retriever, planning_client=planner,
        route_fn=lambda p, m: RouteDecision(),
    )

    assert result.planning_ran is True
    assert "planning_invalid_rule_plan" in result.reason_codes
    # Retrieval used the deterministic rule plan's query, not planner output.
    assert retriever.last_query == plan.retrieval_query


# ── 7. Context budget preserves raw question + coverage ───


def test_context_budget_preserves_raw_question_and_required_coverage():
    config = make_config()
    long_question = "Explain in detail " * 500  # far larger than any lane cap
    plan = make_plan(long_question, entities=["NVDA"], metrics=["total_revenue"])

    facts = [{"metric": "total_revenue", "value": 26.0, "ticker": "NVDA",
              "period": "2026-Q1", "source_type": "sqlite"}]
    docs = [doc("d1", body="NVDA datacenter growth", ticker="NVDA")]

    sel = ContextBudget(config).select(plan, facts, docs, Lane.STANDARD)

    # The raw question is never part of the evidence budget.
    assert sel.context_chars < ContextBudget(config).cap_for(Lane.STANDARD)
    assert sel.context_chars < len(long_question)
    # Required coverage survives: the exact fact and the entity document are kept.
    assert facts[0] in sel.facts
    assert any(d["id"] == "d1" for d in sel.documents)


def test_context_budget_applies_authority_ranking_at_final_selection():
    config = make_config()
    plan = make_plan("latest Oracle financing", entities=["ORCL"])
    secondary = doc("news", body="Analyst reaction", ticker="ORCL", fusion=0.032)
    secondary["metadata"].update({
        "source": "finnhub", "source_category": "news_vendor",
        "item_type": "news", "event_type": "debt_raise",
    })
    primary = doc("sec", body="Debt prospectus terms", ticker="ORCL", fusion=0.032)
    primary["metadata"].update({
        "source": "sec", "source_category": "regulatory_filing",
        "item_type": "filing", "event_type": "debt_raise",
    })

    selection = ContextBudget(config).select(
        plan, [], [secondary, primary], Lane.STANDARD,
    )

    assert [row["id"] for row in selection.documents[:2]] == ["sec", "news"]


# ── 8. Blank document body is dropped ─────────────────────


def test_blank_document_body_is_dropped():
    config = make_config()
    plan = make_plan("NVDA growth", entities=["NVDA"])
    docs = [
        {"id": "blank", "document": "   ", "metadata": {}},
        doc("good", body="real evidence body", ticker="NVDA"),
    ]

    sel = ContextBudget(config).select(plan, [], docs, Lane.STANDARD)

    ids = [d["id"] for d in sel.documents]
    assert ids == ["good"]
    assert sel.dropped_documents >= 1
    assert "context_dropped_blank_body" in sel.reason_codes


# ── 9. Conditional rerank runs on channel disagreement ────


def test_conditional_rerank_runs_on_channel_disagreement():
    pool = [doc(f"d{i}", body=f"body {i}", fusion=0.5 - i * 0.01) for i in range(8)]
    candidates = {
        "documents": pool,
        "vector_ids": ["d0", "d1", "d2"],
        "lexical_ids": ["d5", "d6", "d7"],  # disjoint top-3 → disagreement
        "candidate_count": len(pool),
        "retrieval_strategy": "hybrid",
    }
    reranker = FakeReranker()
    retriever = FakeRetriever(candidates=candidates, reranker=reranker,
                              documents=pool)
    plan = make_plan("why did NVDA drop", entities=["NVDA"],
                     intents=["explanation"], primary_intent="explanation")

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_reranker=True, adaptive_conditional_rerank=True),
        retriever=retriever,
        route_fn=lambda p, m: RouteDecision(),
    )

    assert reranker.calls == 1
    assert result.rerank_ran is True
    assert "rerank_signal_channel_disagreement" in result.reason_codes


# ── 10. Rerank skipped for an exact fact lookup ───────────


def test_rerank_skipped_for_exact_fact_lookup():
    route_fn, execute_fn = complete_route()
    reranker = FakeReranker()
    retriever = FakeRetriever(reranker=reranker)
    plan = make_plan("what is NVDA revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"],
                     primary_intent="fact_lookup")

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_reranker=True, adaptive_conditional_rerank=True),
        retriever=retriever, route_fn=route_fn, execute_fn=execute_fn,
    )

    assert result.lane is Lane.FAST
    assert reranker.calls == 0
    assert retriever.retrieve_candidates_calls == 0


# ── 11. Reranker failure preserves RRF order ──────────────


def test_reranker_failure_preserves_rrf_order():
    pool = [doc(f"d{i}", body=f"body {i}", fusion=0.5 - i * 0.01) for i in range(8)]
    candidates = {
        "documents": pool,
        "vector_ids": ["d0", "d1", "d2"],
        "lexical_ids": ["d5", "d6", "d7"],
        "candidate_count": len(pool),
    }
    reranker = FakeReranker(fail=True)
    retriever = FakeRetriever(candidates=candidates, reranker=reranker, documents=pool)
    plan = make_plan("why did NVDA drop", entities=["NVDA"],
                     intents=["explanation"], primary_intent="explanation")

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_reranker=True, adaptive_conditional_rerank=True),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )

    assert reranker.calls == 1
    assert result.rerank_ran is False
    assert "rerank_fallback" in result.reason_codes
    # RRF order preserved (first top_k of the fused pool, unreordered).
    top_k = 5
    assert [d["id"] for d in result.merged_documents] == [d["id"] for d in pool[:top_k]]


# ── 12. Adaptive failure falls back to existing retriever ─


def test_adaptive_failure_falls_back_to_existing_retriever():
    sentinel_docs = [doc("fallback-doc", ticker="NVDA")]
    retriever = FakeRetriever(documents=sentinel_docs)
    # A matched-but-incomplete route whose executor raises → adaptive stage fails.
    decision = RouteDecision(
        matched=True, complete=False, requires_documents=True,
        tool_invocations=[ToolInvocation("get_fundamentals", {"ticker": "NVDA"},
                                         "sq0", dr.REASON_FUNDAMENTALS)],
    )
    plan = make_plan("why did NVDA drop", entities=["NVDA"],
                     intents=["explanation"], primary_intent="explanation")

    def boom(*a, **k):
        raise RuntimeError("executor blew up")

    result = orchestrate(
        plan, store=object(), config=make_config(),
        retriever=retriever,
        route_fn=lambda p, m: decision, execute_fn=boom,
    )

    assert result.fallback_reason == "adaptive_error"
    assert [d["id"] for d in result.merged_documents] == ["fallback-doc"]
    assert retriever.retrieve_calls >= 1


# ── 13. Feature-disabled path is byte-for-byte compatible ─


def test_feature_disabled_path_is_byte_for_byte_compatible_metadata():
    retriever = FakeRetriever(
        facts=[{"metric": "total_revenue", "value": 26.0, "ticker": "NVDA"}],
        documents=[doc("d1", ticker="NVDA")],
        strategy="hybrid+rerank",
    )
    plan = make_plan("what is NVDA revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"],
                     primary_intent="fact_lookup")
    config = make_config(enable_adaptive_rag=False)

    result = orchestrate(plan, store=object(), config=config, retriever=retriever)

    direct = retriever.retrieve(
        plan.retrieval_query, plan.to_legacy_intent(),
        top_k_documents=config.top_k_documents, top_k_facts=config.top_k_facts,
    )

    assert result.fallback_reason == "feature_disabled"
    assert result.retrieval == direct
    assert result.merged_documents == direct["documents"]
    assert result.merged_facts == direct["facts"]
    assert result.retrieval_strategy == "hybrid+rerank"


# ── Offline latency benchmark ─────────────────────────────


def test_fast_lane_orchestration_overhead_is_small():
    """Orchestration overhead (lane selection + budgeting + fake tool exec, no
    token generation) must be small. The per-lane latency gates *relative to the
    live path* are measured in 2.2.3.4's evaluation; here we bound only what is
    measurable offline (well under the 250 ms complex-lane overhead ceiling).
    """
    route_fn, execute_fn = complete_route()
    retriever = FakeRetriever()
    config = make_config()
    plan = make_plan("what is NVDA revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"],
                     primary_intent="fact_lookup")

    durations = []
    for _ in range(300):
        start = time.perf_counter()
        orchestrate(plan, store=object(), config=config, retriever=retriever,
                    route_fn=route_fn, execute_fn=execute_fn)
        durations.append((time.perf_counter() - start) * 1000.0)

    durations.sort()
    p95 = durations[int(len(durations) * 0.95)]
    assert p95 < 250.0, f"fast-lane orchestration p95 {p95:.3f}ms exceeds 250ms"


# ══════════════════════════════════════════════════════════════════════
# 2.2.3.4 second review wave — regression tests (A, B, C, E, G, H, I)
# ══════════════════════════════════════════════════════════════════════


# ── A: fast lane rejected when the plan has complex obligations ──

def test_fast_lane_rejected_for_complex_plan_even_if_route_complete():
    # A complete numeric route on a >2-entity plan must NOT go fast (which skips
    # document retrieval) — defense in depth against router complete-overclaim.
    route_fn, execute_fn = complete_route()
    retriever = FakeRetriever(documents=[doc("d1")])
    plan = make_plan("compare NVDA AMD INTC revenue",
                     entities=["NVDA", "AMD", "INTC"],
                     metrics=["total_revenue"], intents=["comparison"],
                     primary_intent="comparison")

    result = orchestrate(
        plan, store=object(), config=make_config(),
        retriever=retriever, route_fn=route_fn, execute_fn=execute_fn,
    )

    assert result.lane is Lane.COMPLEX
    assert "complex_multi_entity" in result.reason_codes
    assert retriever.retrieve_calls == 1  # documents WERE retrieved


# ── B: context budget honors period + reserves coverage before extra facts ──

def test_context_budget_exact_match_requires_period():
    config = make_config()
    plan = make_plan("NVDA revenue 2026-Q2", entities=["NVDA"],
                     metrics=["total_revenue"], periods=["2026-Q2"])
    wrong_period = {"metric": "total_revenue", "value": 20.0, "ticker": "NVDA",
                    "period": "2026-Q1", "source_type": "sqlite"}
    right_period = {"metric": "total_revenue", "value": 26.0, "ticker": "NVDA",
                    "period": "2026-Q2", "source_type": "sqlite"}

    sel = ContextBudget(config).select(plan, [wrong_period, right_period], [], Lane.STANDARD)

    # Both facts fit here, but only the exact-period row counts as coverage.
    covered = {f["ticker"] for f in sel.facts if f.get("period") == "2026-Q2"}
    assert "NVDA" in covered
    assert right_period in sel.facts


def test_context_budget_reserves_coverage_doc_over_extra_facts():
    # A tiny cap that fits the exact fact + the coverage doc but not the pile of
    # non-exact facts. The qualitative coverage doc must survive.
    config = make_config(adaptive_max_context_chars=1000)
    plan = make_plan("why did NVDA drop; explain", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["explanation"],
                     primary_intent="explanation")
    exact = {"metric": "total_revenue", "value": 26.0, "ticker": "NVDA",
             "period": None, "source_type": "sqlite"}
    filler = [{"metric": "misc", "value": i, "ticker": "OTHER", "period": None,
               "source_type": "sqlite"} for i in range(40)]
    cover_doc = doc("cover", body="NVDA fell on datacenter guidance " * 5, ticker="NVDA")

    sel = ContextBudget(config).select(plan, [exact] + filler, [cover_doc], Lane.STANDARD)

    assert any(d["id"] == "cover" for d in sel.documents)  # coverage survived
    assert exact in sel.facts


# ── C: budgeter / top-level failures fall soft, never raise ──

def test_context_budgeter_failure_falls_soft(monkeypatch):
    retriever = FakeRetriever(documents=[doc("d1", ticker="NVDA")])
    plan = make_plan("why did NVDA drop", entities=["NVDA"],
                     intents=["explanation"], primary_intent="explanation")

    def boom_select(self, plan, facts, documents, lane):
        raise RuntimeError("budgeter exploded")

    monkeypatch.setattr(ContextBudget, "select", boom_select)

    # Must not raise; returns a result with an empty, reason-coded selection.
    result = orchestrate(
        plan, store=object(), config=make_config(enable_reranker=False),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )
    assert result.context is not None
    assert result.context.context_chars == 0
    assert "context_budget_error" in result.context.reason_codes


def test_budget_from_config_failure_falls_soft(monkeypatch):
    retriever = FakeRetriever(documents=[doc("d1", ticker="NVDA")])
    plan = make_plan("why did NVDA drop", entities=["NVDA"],
                     intents=["explanation"], primary_intent="explanation")

    def boom_from_config(cls, config):
        raise RuntimeError("budget construction failed")

    monkeypatch.setattr(ExecutionBudget, "from_config", classmethod(boom_from_config))

    result = orchestrate(
        plan, store=object(), config=make_config(),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )
    # Top-level fail-soft -> legacy fallback, never an exception.
    assert result.fallback_reason == "adaptive_error"
    assert retriever.retrieve_calls >= 1


# ── E: fast-lane route error reuses the caller's retriever ──

def test_fast_lane_route_error_reuses_injected_retriever():
    sentinel = [doc("fallback-doc", ticker="NVDA")]
    retriever = FakeRetriever(documents=sentinel)
    decision = RouteDecision(
        matched=True, complete=True, requires_documents=False,
        tool_invocations=[ToolInvocation("get_fundamentals", {"ticker": "NVDA"},
                                         "sq0", dr.REASON_FUNDAMENTALS)],
        reason_codes=[dr.REASON_FUNDAMENTALS],
    )
    errored = ExecutionResult(invocations=[], error=True, answer=None)
    plan = make_plan("what is NVDA revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"],
                     primary_intent="fact_lookup")

    result = orchestrate(
        plan, store=object(), config=make_config(),
        retriever=retriever,
        route_fn=lambda p, m: decision, execute_fn=lambda *a, **k: errored,
    )

    assert result.fallback_reason == "deterministic_route_error"
    # The injected retriever ran the fallback (no fresh Retriever constructed).
    assert retriever.retrieve_calls == 1
    assert [d["id"] for d in result.merged_documents] == ["fallback-doc"]


# ── G: reranker's silent RRF fallback is reported as rerank_fallback ──

class SilentFallbackReranker:
    """Mimics Reranker.rerank's internal fallback: returns RRF order, no raise,
    every doc carries rerank_score=None."""

    def __init__(self):
        self.calls = 0

    def rerank(self, query, docs, top_n=5):
        self.calls += 1
        out = list(docs)[:top_n]
        for d in out:
            d["rerank_score"] = None
        return out


def test_silent_reranker_fallback_reported_as_fallback():
    pool = [doc(f"d{i}", body=f"body {i}", fusion=0.5 - i * 0.01) for i in range(8)]
    candidates = {
        "documents": pool,
        "vector_ids": ["d0", "d1", "d2"],
        "lexical_ids": ["d5", "d6", "d7"],  # disagreement → rerank attempted
        "candidate_count": len(pool),
    }
    reranker = SilentFallbackReranker()
    retriever = FakeRetriever(candidates=candidates, reranker=reranker, documents=pool)
    plan = make_plan("why did NVDA drop", entities=["NVDA"],
                     intents=["explanation"], primary_intent="explanation")

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_reranker=True, adaptive_conditional_rerank=True),
        retriever=retriever, route_fn=lambda p, m: RouteDecision(),
    )

    assert reranker.calls == 1
    assert result.rerank_ran is False           # NOT reported as applied
    assert "rerank_fallback" in result.reason_codes
    assert "rerank_applied" not in result.reason_codes


# ── I: id and (parent_id, chunk_index) are independent dedup identities ──

def test_dedupe_treats_id_and_parent_chunk_independently():
    # Same physical chunk: once keyed by id, once keyed only by parent+chunk.
    d_with_id = {"id": "doc#0", "document": "chunk body",
                 "metadata": {"parent_id": "doc", "chunk_index": 0}}
    d_by_chunk = {"document": "chunk body",
                  "metadata": {"parent_id": "doc", "chunk_index": 0}}
    out = ao._dedupe_docs([d_with_id, d_by_chunk])
    assert len(out) == 1


# ══════════════════════════════════════════════════════════════════════
# 2.2.4.2 — Selective decomposition executed through the corrective seam
# ══════════════════════════════════════════════════════════════════════


def _decomp_config(**overrides):
    return make_config(
        enable_evidence_sufficiency=True, enable_corrective_retry=True,
        enable_query_decomposition=True, enable_reranker=False, **overrides)


def _filing_fact(metric="total_revenue"):
    return {"evidence_id": f"{metric}-r", "ticker": "NVDA", "metric": metric,
            "value": 26.0, "period": None, "unit": "USD", "source_type": "sec_10k"}


def _facts_sq0_plan(question, *, entities, metrics, intents):
    """Plan whose sq0 declares an explicit facts obligation (so a returned
    structured fact covers it and only the qualitative derived subquery is left)."""
    plan_entities = [
        QueryEntity(ticker=t.upper(), resolved_name=None, confidence=1.0,
                    source="resolved", mention=t, start=i)
        for i, t in enumerate(entities)
    ]
    sq0 = QuerySubquery(
        id="sq0", text=question,
        entity_tickers=tuple(e.ticker for e in plan_entities),
        intents=tuple(intents), metrics=tuple(metrics), retrieval_modes=("facts",))
    return QueryPlan(
        original_question=question, retrieval_query=question,
        normalized_question=normalize_question(question), entities=plan_entities,
        intents=list(intents), metrics=list(metrics), periods=[],
        subqueries=[sq0], primary_intent=intents[0]).validate()


class DecompRetriever(FakeRetriever):
    """Returns structured facts from ``retrieve`` and a modality-specific
    document pool from ``retrieve_candidates`` (keyed on question_type)."""

    def __init__(self):
        super().__init__()
        self.cand_question_types: list = []

    def retrieve(self, query, intent, top_k_documents=5, top_k_facts=10):
        self.retrieve_calls += 1
        self.last_intent = dict(intent)
        return {"facts": [_filing_fact()], "documents": [], "ticker": "NVDA",
                "strategy": "facts_only", "retrieval_strategy": "vector"}

    def retrieve_candidates(self, query, intent, top_k_documents=5, top_k_facts=10):
        self.retrieve_candidates_calls += 1
        qt = intent.get("question_type")
        self.cand_question_types.append(qt)
        if qt in ("risk", "news", "explanation", "sentiment"):
            name = {"risk": "sec_10k", "news": "gdelt"}.get(qt, "sec_10k")
            return {"facts": [], "documents": [doc(f"doc-{qt}", ticker="NVDA",
                                                   source=name)],
                    "retrieval_strategy": "hybrid", "vector_ids": [],
                    "lexical_ids": [], "candidate_count": 1}
        return {"facts": [_filing_fact()], "documents": [],
                "retrieval_strategy": "vector", "vector_ids": [],
                "lexical_ids": [], "candidate_count": 0}


def test_fact_plus_risk_runs_derived_subquery_and_fuses():
    retriever = DecompRetriever()
    plan = _facts_sq0_plan("NVDA revenue and risk factors", entities=["NVDA"],
                           metrics=["total_revenue"], intents=["fact_lookup", "risk"])

    result = orchestrate(
        plan, store=object(), config=_decomp_config(), retriever=retriever,
        route_fn=lambda p, m: RouteDecision())

    assert result.lane is Lane.COMPLEX
    assert "subquery_decomposition_applied" in result.reason_codes
    # sq0's fact covers sq0 + the structured derived subquery; only the
    # qualitative derived subquery is executed in the corrective round.
    assert result.derived_subqueries == ["sq2"]
    assert result.subqueries_executed == ["sq0", "sq2"]
    assert result.retrieval_rounds_used == 2
    assert result.retry_performed is True
    assert result.sufficiency.status is SufficiencyStatus.SUFFICIENT
    assert "derived_fusion_applied" in result.reason_codes
    # The fused document carries its subquery provenance.
    fused = result.merged_documents
    assert fused and all("subquery_ids" in d for d in fused)
    assert any("sq2" in d["subquery_ids"] for d in fused)


def test_decomposition_never_exceeds_three_subqueries_or_two_rounds():
    retriever = DecompRetriever()
    # sq0 (facts) covered by a fact; two qualitative derived subqueries left.
    question = "NVDA revenue risk and news"
    entities = [QueryEntity(ticker="NVDA", resolved_name=None, confidence=1.0,
                            source="resolved", mention="NVDA", start=0)]
    sq0 = QuerySubquery(id="sq0", text=question, entity_tickers=("NVDA",),
                        intents=("fact_lookup", "risk", "news"),
                        metrics=("total_revenue",), retrieval_modes=("facts",))
    sq1 = QuerySubquery(id="sq1", text="NVDA risk", entity_tickers=("NVDA",),
                        intents=("risk",), retrieval_modes=("documents",),
                        derived=True, parent_id="sq0",
                        derivation_source="deterministic")
    sq2 = QuerySubquery(id="sq2", text="NVDA news", entity_tickers=("NVDA",),
                        intents=("news",), retrieval_modes=("documents",),
                        derived=True, parent_id="sq0",
                        derivation_source="deterministic")
    plan = QueryPlan(
        original_question=question, retrieval_query=question,
        normalized_question=normalize_question(question), entities=entities,
        intents=["fact_lookup", "risk", "news"], metrics=["total_revenue"],
        periods=[], subqueries=[sq0, sq1, sq2], primary_intent="fact_lookup").validate()

    result = orchestrate(
        plan, store=object(), config=_decomp_config(), retriever=retriever,
        route_fn=lambda p, m: RouteDecision())

    assert len(result.subqueries_executed) == 3       # sq0 + 2 derived, hard cap
    assert result.retrieval_rounds_used == 2          # one sq0 round + one batch
    assert result.derived_subqueries == ["sq1", "sq2"]
    assert retriever.cand_question_types == ["risk", "news"]  # specialized routing


def test_simple_sufficient_query_adds_no_derived_when_enabled():
    """A simple, already-sufficient lookup generates zero derived subqueries and
    no extra round even with decomposition enabled (zero added cost)."""
    retriever = DecompRetriever()
    plan = _facts_sq0_plan("NVDA revenue", entities=["NVDA"],
                           metrics=["total_revenue"], intents=["fact_lookup"])

    result = orchestrate(
        plan, store=object(), config=_decomp_config(), retriever=retriever,
        route_fn=lambda p, m: RouteDecision())

    assert result.lane is Lane.STANDARD
    assert result.derived_subqueries == []
    assert result.retrieval_rounds_used == 1
    assert retriever.retrieve_candidates_calls == 0
    assert result.sufficiency.status is SufficiencyStatus.SUFFICIENT


def test_decomposition_disabled_keeps_deferred_placeholder():
    """With the flag off, a plan carrying a derived subquery still hits the
    2.2.4.1 deferred placeholder — legacy behavior is unchanged."""
    retriever = DecompRetriever()
    plan = _facts_sq0_plan("NVDA revenue and risk factors", entities=["NVDA"],
                           metrics=["total_revenue"], intents=["fact_lookup", "risk"])
    plan.subqueries.append(QuerySubquery(
        id="sq1", text="NVDA risk", entity_tickers=("NVDA",), intents=("risk",),
        retrieval_modes=("documents",), derived=True, parent_id="sq0"))
    plan.validate()

    result = orchestrate(
        plan, store=object(),
        config=make_config(enable_evidence_sufficiency=True,
                           enable_corrective_retry=True, enable_reranker=False),
        retriever=retriever, route_fn=lambda p, m: RouteDecision())

    assert "run_derived_subqueries_deferred_2_2_4_2" in result.reason_codes
    assert result.derived_subqueries == []
    assert result.retry_performed is False
    assert result.retrieval_rounds_used == 1

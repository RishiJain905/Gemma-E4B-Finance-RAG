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
        self._facts = facts or []
        self._documents = documents or []
        self._candidates = candidates
        self._strategy = strategy
        self.reranker = reranker or FakeReranker()

    def retrieve(self, query, intent, top_k_documents=5, top_k_facts=10):
        self.retrieve_calls += 1
        self.last_query = query
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


def test_complex_lane_caps_three_subqueries():
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
    assert len(result.subqueries_executed) == 3  # sq0 + 2 derived, cap = 3
    assert "subquery_budget_exhausted" in result.reason_codes


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

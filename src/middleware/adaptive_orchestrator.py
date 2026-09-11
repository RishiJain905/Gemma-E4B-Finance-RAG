"""
src/middleware/adaptive_orchestrator.py
Bounded adaptive-RAG orchestrator (Phase 2.2.3.3).

Selects a fast, standard, or complex lane from a validated
:class:`~src.middleware.query_plan.QueryPlan`, enforces one shared, uncircum-
ventable :class:`ExecutionBudget`, assembles only useful context under one
:class:`ContextBudget`, and invokes the existing re-ranker only when documented
ambiguity signals make it plausibly useful.

This module is self-contained: it never mutates ``/query`` and never fails a
request. Any exception in a lane stage, the router, the planner, or budgeting is
logged and demoted to the existing single-query ``Retriever.retrieve()`` path
(see :func:`orchestrate`). The request-path switch that decides *whether* to call
this orchestrator lives in 2.2.3.4, not here.

Design contract (docs/phase2.2/2.2.3-adaptive-rag-orchestration/
2.2.3.3-adaptive-lanes-context-budget-and-conditional-reranking.md and
docs/phase2.2/ARCHITECTURE-DECISION.md "Route contracts" / "Context and latency
policy"):

- Every counter increments through :meth:`ExecutionBudget.consume`; callers
  cannot bypass the caps (3 subqueries incl. sq0, 2 retrieval rounds, 1 planning
  call, 1 re-rank call, deterministic-tool cap). Budget exhaustion returns the
  best evidence already collected plus a partial reason code — never a loop.
- Lane selection is deterministic and observable (stable ``reason_codes`` for
  identical inputs).
- The raw ``original_question`` is preserved in full, outside the context
  budget; document bodies are read through :func:`evidence.document_body`.
"""

from __future__ import annotations

import json
import logging
import time
from time import perf_counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from . import deterministic_router as dr
from .evidence import document_body
from .evidence_grader import (
    CorrectiveAction,
    SufficiencyResult,
    SufficiencyStatus,
    grade_evidence,
    validated_metric_aliases,
)
from .query_plan import (
    QueryEntity,
    QueryPlan,
    QuerySubquery,
    attach_derived_subqueries,
    decompose_plan,
    normalize_question,
)

if TYPE_CHECKING:
    from .config import MiddlewareConfig
    from .deterministic_router import ExecutionResult, RouteDecision
    from .retriever import Retriever

logger = logging.getLogger(__name__)


# ── Lanes ─────────────────────────────────────────────────


class Lane(str, Enum):
    """Catalog-first routing plus the three adaptive RAG cost tiers."""

    CATALOG = "catalog"
    FAST = "fast"
    STANDARD = "standard"
    COMPLEX = "complex"


# Qualitative document intents (mirror deterministic_router._DOCUMENT_INTENTS).
_QUALITATIVE_INTENTS = frozenset({"explanation", "news", "risk"})
# Intents that imply a structured/numeric obligation.
_STRUCTURED_INTENTS = frozenset({"fact_lookup", "comparison", "projection"})


# ── Execution budget (the single, uncircumventable counter) ──


# consume() kinds.
SUBQUERY = "subquery"
RETRIEVAL_ROUND = "retrieval_round"
PLANNING_CALL = "planning_call"
RERANK_CALL = "rerank_call"
DETERMINISTIC_TOOL = "deterministic_tool"


@dataclass
class ExecutionBudget:
    """One shared budget for a single orchestrated request.

    The per-kind counters are private; :meth:`consume` is the *only* mutator, so
    calling code cannot exceed a cap by touching state directly. ``consume``
    returns ``True`` when a unit was granted (and increments), ``False`` when the
    cap is already reached (and records a stable exhaustion reason). Every hard
    cap comes from the clamped :class:`~src.middleware.config.MiddlewareConfig`.
    """

    max_subqueries: int = 3
    max_retrieval_rounds: int = 2
    max_planning_calls: int = 1
    max_rerank_calls: int = 1
    max_deterministic_tools: int = 3

    _counts: dict = field(default_factory=dict)
    exhausted: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._counts = {
            SUBQUERY: 0,
            RETRIEVAL_ROUND: 0,
            PLANNING_CALL: 0,
            RERANK_CALL: 0,
            DETERMINISTIC_TOOL: 0,
        }

    @classmethod
    def from_config(cls, config: "MiddlewareConfig") -> "ExecutionBudget":
        """Build a budget from the clamped adaptive config limits."""
        return cls(
            max_subqueries=int(getattr(config, "adaptive_max_subqueries", 3)),
            max_retrieval_rounds=int(getattr(config, "adaptive_max_retrieval_rounds", 2)),
            max_planning_calls=int(getattr(config, "adaptive_max_planning_calls", 1)),
            max_rerank_calls=1,
            max_deterministic_tools=int(
                getattr(config, "max_deterministic_tools_per_query", 3)
            ),
        )

    def _limit(self, kind: str) -> int:
        return {
            SUBQUERY: self.max_subqueries,
            RETRIEVAL_ROUND: self.max_retrieval_rounds,
            PLANNING_CALL: self.max_planning_calls,
            RERANK_CALL: self.max_rerank_calls,
            DETERMINISTIC_TOOL: self.max_deterministic_tools,
        }[kind]

    def consume(self, kind: str) -> bool:
        """Grant one unit of ``kind`` if the cap allows; else record exhaustion.

        Returns ``True`` and increments when under the cap, ``False`` otherwise.
        A ``False`` result must make the caller *stop* — never retry in a loop.
        """
        if kind not in self._counts:
            raise ValueError(f"unknown budget kind: {kind!r}")
        if self._counts[kind] >= self._limit(kind):
            reason = f"{kind}_budget_exhausted"
            if reason not in self.exhausted:
                self.exhausted.append(reason)
            return False
        self._counts[kind] += 1
        return True

    def used(self, kind: str) -> int:
        """Units of ``kind`` consumed so far (read-only telemetry)."""
        return self._counts.get(kind, 0)


# ── Context budget ────────────────────────────────────────


# Char caps per lane; the effective cap is min(lane cap, adaptive_max_context_chars).
_LANE_CONTEXT_CAPS = {
    Lane.CATALOG: 4000,
    Lane.FAST: 4000,
    Lane.STANDARD: 12000,
    Lane.COMPLEX: 18000,
}

# Rough per-item overheads so char accounting approximates the rendered prompt.
_FACT_OVERHEAD = 60
_DOC_HEADER = 80
# Below this many leftover chars we do not bother truncating a final chunk.
_MIN_TRUNC_CHARS = 400
# Estimated-token divisor (4 chars ≈ 1 token) — telemetry only, no tokenizer.
_CHARS_PER_TOKEN = 4


@dataclass
class ContextSelection:
    """The budgeted evidence set plus drop telemetry.

    ``facts``/``documents`` are the packed, in-budget evidence (the adaptive
    path's single budget owner). ``truncated`` is True when the final document
    chunk was cut to fit; ``reason_codes`` records observable drop/keep rules.
    """

    facts: list[dict] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)
    context_chars: int = 0
    estimated_tokens: int = 0
    dropped_facts: int = 0
    dropped_documents: int = 0
    reason_codes: list[str] = field(default_factory=list)
    truncated: bool = False


def _fact_chars(fact: dict) -> int:
    return _FACT_OVERHEAD + sum(
        len(str(fact.get(k, "")))
        for k in ("metric", "value", "unit", "period", "ticker", "source_type")
    )


def _doc_chars(doc: dict) -> int:
    return _DOC_HEADER + len(document_body(doc))


def _doc_key(doc: dict) -> tuple:
    """Stable de-dup identity: (stable id, parent id, chunk position)."""
    meta = doc.get("metadata") or {}
    doc_id = doc.get("id") or meta.get("id")
    parent = meta.get("parent_id") or doc.get("parent_id")
    chunk = meta.get("chunk_index", meta.get("chunk"))
    if doc_id is not None:
        return ("id", doc_id)
    return ("body", parent, chunk, document_body(doc))


def _doc_identities(doc: dict) -> list[tuple]:
    """All independent de-dup identities a doc carries (2.2.3.4 review I).

    A stable id and a (parent_id, chunk_index) pair are INDEPENDENT identities:
    the same physical chunk can surface once keyed by id and once keyed by its
    parent+chunk position, and either match means duplicate. A doc is a duplicate
    when ANY of its identities was already seen. Bodyless-and-idless docs fall
    back to their rendered body.
    """
    meta = doc.get("metadata") or {}
    identities: list[tuple] = []
    doc_id = doc.get("id") or meta.get("id")
    if doc_id is not None:
        identities.append(("id", doc_id))
    parent = meta.get("parent_id") or doc.get("parent_id")
    chunk = meta.get("chunk_index", meta.get("chunk"))
    if parent is not None and chunk is not None:
        identities.append(("chunk", parent, chunk))
    if not identities:
        identities.append(("body", document_body(doc)))
    return identities


def _doc_score(doc: dict) -> float:
    score = doc.get("ranking_score")
    if score is None:
        score = doc.get("rerank_score")
    if score is None:
        score = doc.get("fusion_score")
    try:
        return float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _is_low_authority(doc: dict) -> bool:
    """Deprioritize stale / low-authority / scoreless chunks to the last bucket."""
    meta = doc.get("metadata") or {}
    if meta.get("stale") or doc.get("stale"):
        return True
    if doc.get("rerank_score") is None and doc.get("fusion_score") is None:
        return True
    return False


class ContextBudget:
    """Selects and packs evidence into one lane-bounded character budget.

    Selection order (spec Step 3): exact facts/tool results first, then one best
    non-blank item per uncovered entity, then remaining documents by score, then
    low-authority/stale/duplicate chunks last. Facts are never cut mid-record;
    whole document chunks are preferred and only the final chunk may be
    truncated (``truncated=True``). The raw question is never part of the budget.
    """

    def __init__(self, config: "MiddlewareConfig"):
        self.config = config

    def cap_for(self, lane: Lane) -> int:
        lane_cap = _LANE_CONTEXT_CAPS.get(lane, _LANE_CONTEXT_CAPS[Lane.COMPLEX])
        hard = int(getattr(self.config, "adaptive_max_context_chars", 18000))
        return max(0, min(lane_cap, hard))

    def select(
        self,
        plan: QueryPlan,
        facts: list[dict],
        documents: list[dict],
        lane: Lane,
    ) -> ContextSelection:
        cap = self.cap_for(lane)
        sel = ContextSelection()
        used = 0

        requested_entities = set(plan.tickers)
        requested_metrics = set(plan.metrics)
        requested_periods = set(plan.periods)

        # ── Facts: an EXACT slot matches entity AND metric AND (when the plan
        # requested periods) period, so a same-metric row from the wrong period
        # is not treated as the answer (2.2.3.4 review B). ──
        def _is_exact(f: dict) -> bool:
            metric_ok = (not requested_metrics) or f.get("metric") in requested_metrics
            entity_ok = (not requested_entities) or f.get("ticker") in requested_entities
            period_ok = (not requested_periods) or f.get("period") in requested_periods
            return bool(metric_ok and entity_ok and period_ok)

        def _pack_fact(f: dict) -> bool:
            nonlocal used
            cost = _fact_chars(f)
            if used + cost <= cap:
                sel.facts.append(f)
                used += cost
                return True
            sel.dropped_facts += 1
            if "context_dropped_over_budget" not in sel.reason_codes:
                sel.reason_codes.append("context_dropped_over_budget")
            return False

        usable_fact_rows = [
            f for f in facts
            if isinstance(f, dict) and f.get("metric") not in (None, "")
            and f.get("value") is not None
        ]
        sel.dropped_facts += len(facts) - len(usable_fact_rows)
        exact_facts = [f for f in usable_fact_rows if _is_exact(f)]
        other_facts = [f for f in usable_fact_rows if not _is_exact(f)]

        # 1. Exact facts / tool results first (never cut mid-record).
        for f in exact_facts:
            _pack_fact(f)

        # ── Documents: dedupe (any independent identity), drop blanks ──
        # Apply policy at the final context boundary. Candidate retrieval stays
        # complete and untruncated for adaptive merge and obligation coverage.
        policy_documents = list(documents)
        try:
            from .evidence_taxonomy import normalize_evidence, pack_event_coverage, rank_evidence

            taxonomy_enabled = bool(getattr(self.config, "enable_evidence_taxonomy", True))
            ranking_enabled = bool(getattr(self.config, "enable_authority_ranking", True))
            packing_enabled = bool(
                getattr(self.config, "enable_duplicate_coverage_packing", True)
            )
            if taxonomy_enabled or ranking_enabled or packing_enabled:
                policy_documents = [normalize_evidence(row) for row in policy_documents]
            if ranking_enabled:
                policy_documents = rank_evidence(
                    policy_documents,
                    query=plan.retrieval_query,
                    filters=dict(getattr(plan, "evidence_filters", {}) or {}),
                    authority_max_boost=max(0.0, min(0.025, float(
                        getattr(self.config, "authority_max_boost", 0.025)
                    ))),
                )
            if packing_enabled:
                policy_documents = pack_event_coverage(
                    policy_documents,
                    limit=len(policy_documents),
                    max_secondary_per_event=int(
                        getattr(self.config, "max_secondary_per_event", 2)
                    ),
                )
        except Exception:  # noqa: BLE001 - evidence policy is fail-soft
            logger.warning("Final evidence policy failed; preserving merged pool", exc_info=True)

        seen: set = set()
        deduped: list[dict] = []
        for d in policy_documents:
            if not isinstance(d, dict):
                continue
            if not document_body(d):
                sel.dropped_documents += 1
                if "context_dropped_blank_body" not in sel.reason_codes:
                    sel.reason_codes.append("context_dropped_blank_body")
                continue
            idents = _doc_identities(d)
            if any(i in seen for i in idents):
                sel.dropped_documents += 1
                if "context_dropped_duplicate" not in sel.reason_codes:
                    sel.reason_codes.append("context_dropped_duplicate")
                continue
            seen.update(idents)
            deduped.append(d)

        by_score = sorted(deduped, key=lambda d: (-_doc_score(d), str(d.get("id"))))

        # 2. Coverage documents are RESERVED before non-exact facts (review B):
        # one best doc per plan entity whose obligation isn't already satisfied,
        # so a qualitative obligation's only document can't be starved by extra
        # same-entity facts. When the plan carries a qualitative/document intent,
        # NO entity is treated as fact-covered — a document is required even if a
        # fact exists (the previous "any fact = covered" rule dropped the only
        # document for a why/risk/news obligation). For a purely numeric plan, an
        # EXACT fact covers the entity and no document is force-reserved.
        plan_needs_documents = bool(set(plan.intents) & _QUALITATIVE_INTENTS)
        covered_entities = set() if plan_needs_documents else {
            f.get("ticker") for f in sel.facts
        }
        claimed: set = set()
        for entity in [e for e in plan.tickers if e not in covered_entities]:
            best = next(
                (
                    d for d in by_score
                    if id(d) not in claimed
                    and (d.get("metadata") or {}).get("ticker") == entity
                ),
                None,
            )
            if best is None:
                continue
            claimed.add(id(best))
            cost = _doc_chars(best)
            if used + cost <= cap:
                sel.documents.append(best)
                used += cost
            else:
                sel.dropped_documents += 1
                if "context_dropped_over_budget" not in sel.reason_codes:
                    sel.reason_codes.append("context_dropped_over_budget")

        # 3. Non-exact facts next (after coverage is reserved).
        for f in other_facts:
            _pack_fact(f)

        # 4. Remaining documents by score, low-authority/stale last.
        low_authority = [d for d in by_score if id(d) not in claimed and _is_low_authority(d)]
        remaining_docs = [
            d for d in by_score if id(d) not in claimed and not _is_low_authority(d)
        ]
        ordered_docs = remaining_docs + low_authority
        overflow: list[dict] = []
        for d in ordered_docs:
            cost = _doc_chars(d)
            if used + cost <= cap:
                sel.documents.append(d)
                used += cost
            else:
                overflow.append(d)

        # Last resort: truncate the single best overflow chunk into leftover space.
        leftover = cap - used
        if overflow and leftover >= _MIN_TRUNC_CHARS:
            victim = overflow[0]
            body = document_body(victim)
            keep = max(0, leftover - _DOC_HEADER)
            if keep > 0 and keep < len(body):
                clone = dict(victim)
                clone["document"] = body[:keep]
                clone.pop("text", None)
                clone.pop("content", None)
                clone["truncated"] = True
                sel.documents.append(clone)
                used += _DOC_HEADER + keep
                sel.truncated = True
                sel.reason_codes.append("context_truncated_final_chunk")
                overflow = overflow[1:]

        if overflow:
            sel.dropped_documents += len(overflow)
            if "context_dropped_over_budget" not in sel.reason_codes:
                sel.reason_codes.append("context_dropped_over_budget")

        if not sel.reason_codes:
            sel.reason_codes.append("context_within_budget")
        sel.context_chars = used
        sel.estimated_tokens = used // _CHARS_PER_TOKEN
        return sel


# ── Orchestration result ──────────────────────────────────


@dataclass
class OrchestrationResult:
    """The full, observable outcome of one orchestrated request.

    ``retrieval`` mirrors the legacy ``Retriever.retrieve()`` dict shape so
    2.2.3.4 can feed either the budgeted ``context`` (preferred) or the raw
    merged evidence into the prompt builder. ``fallback_reason`` is set only when
    the adaptive layer demoted to the existing single-query path.
    """

    lane: Lane
    plan: QueryPlan
    reason_codes: list[str] = field(default_factory=list)
    merged_facts: list[dict] = field(default_factory=list)
    merged_documents: list[dict] = field(default_factory=list)
    tool_execution: Optional["ExecutionResult"] = None
    calculations: list[dict] = field(default_factory=list)
    deterministic_answer: Optional[str] = None
    answer_origin: Optional[str] = None
    answer_metadata: Optional[dict] = None
    subqueries_executed: list[str] = field(default_factory=list)
    retrieval_rounds_used: int = 0
    planning_ran: bool = False
    rerank_ran: bool = False
    retrieval_strategy: Optional[str] = None
    context: Optional[ContextSelection] = None
    context_size: int = 0
    estimated_tokens: int = 0
    fallback_reason: Optional[str] = None
    retrieval: Optional[dict] = None
    sufficiency: Optional[SufficiencyResult] = None
    corrective_action: CorrectiveAction = CorrectiveAction.NONE
    retry_performed: bool = False
    # 2.2.4.2 decomposition telemetry. ``derived_subqueries`` are the ids
    # actually retrieved and fused this request (never declared-but-unrun);
    # ``drift_reason_codes`` records why any proposed derived subquery was
    # rejected by the drift validator. ``proposed_subqueries`` counts every
    # candidate the decomposer offered (accepted + rejected).
    derived_subqueries: list[str] = field(default_factory=list)
    drift_reason_codes: list[str] = field(default_factory=list)
    proposed_subqueries: int = 0
    set_complete: Optional[bool] = None
    result_set_size: Optional[int] = None

    def add_reason(self, code: str) -> None:
        if code not in self.reason_codes:
            self.reason_codes.append(code)

    def graph_trace_metadata(self) -> dict:
        """Return safe counters from this executed orchestration result."""
        dropped = 0
        if self.context is not None:
            dropped = self.context.dropped_facts + self.context.dropped_documents
        metadata = {
            "lane": self.lane.value,
            "retrieval_rounds": self.retrieval_rounds_used,
            "planning_calls": 1 if self.planning_ran else 0,
            "reranker_calls": 1 if self.rerank_ran else 0,
            "context_chars": self.context_size,
            "evidence_dropped": dropped,
            "fallback_reason": self.fallback_reason,
            "retrieval_strategy": self.retrieval_strategy,
        }
        if self.sufficiency is not None:
            metadata.update(self.sufficiency.graph_trace_metadata())
        return metadata


# ── Public entry point ────────────────────────────────────

# Signatures for the injectable seams (all optional; real defaults are the
# deterministic router functions). Tests substitute fakes here.
RouteFn = Callable[[QueryPlan, Iterable[str]], "RouteDecision"]
ExecuteFn = Callable[..., "ExecutionResult"]
PlanningClient = Callable[[QueryPlan], Any]
DecomposeFn = Callable[[QueryPlan], list[QuerySubquery]]
CorrectiveFn = Callable[[QueryPlan, list[dict], list[dict]], bool]


def orchestrate(
    plan: QueryPlan,
    store: Any,
    config: "MiddlewareConfig",
    *,
    retriever: Optional["Retriever"] = None,
    available_metrics: Iterable[str] = (),
    planning_client: Optional[PlanningClient] = None,
    decompose: Optional[DecomposeFn] = None,
    corrective_retry: Optional[CorrectiveFn] = None,
    route_fn: Optional[RouteFn] = None,
    execute_fn: Optional[ExecuteFn] = None,
    retrieval_cache: Optional[Any] = None,
    refresh: bool = False,
) -> OrchestrationResult:
    """Run one bounded adaptive-RAG request over ``plan``.

    Deterministically selects a lane, executes it under a single
    :class:`ExecutionBudget`, packs evidence under one :class:`ContextBudget`,
    and returns an :class:`OrchestrationResult`. When ``config.enable_adaptive_rag``
    is False — or any adaptive stage raises — this falls back to the existing
    single-query ``Retriever.retrieve()`` result and never raises.

    ``available_metrics`` is passed to the deterministic router; ``planning_client``
    / ``decompose`` / ``corrective_retry`` are optional bounded hooks. Planning is
    disabled by default. Selective decomposition (2.2.4.2) is gated by
    ``enable_query_decomposition``: when on, the complex lane declares ≤2
    drift-validated derived subqueries on the plan and the 2.2.4.1 corrective seam
    retrieves + fuses them; when off, ``decompose`` proposals are recorded as
    deferred and the ``RUN_DERIVED_SUBQUERIES`` action stays a no-op placeholder.
    """
    if not getattr(config, "enable_adaptive_rag", False):
        return _fallback(plan, retriever, store, config, reason="feature_disabled")

    # One outer fail-soft boundary (2.2.3.4 review C): budget construction, lane
    # selection, and context finalization all sit inside it, so ANY raise —
    # including from ExecutionBudget.from_config or the context budgeter — demotes
    # to the single-query fallback instead of escaping. The fallback's own
    # budgeter call is made re-entrancy-safe by _apply_context_budget, which can
    # never re-raise, so the last resort cannot loop back into a failing stage.
    try:
        return _run_adaptive(
            plan, store, config,
            retriever=retriever, available_metrics=available_metrics,
            planning_client=planning_client, decompose=decompose,
            corrective_retry=corrective_retry, route_fn=route_fn, execute_fn=execute_fn,
            retrieval_cache=retrieval_cache, refresh=refresh,
        )
    except Exception:  # noqa: BLE001 - orchestrate() must never raise into /query
        logger.exception("Adaptive orchestration failed; falling back to retrieve()")
        return _fallback(plan, retriever, store, config, reason="adaptive_error")


def _run_adaptive(
    plan: QueryPlan,
    store: Any,
    config: "MiddlewareConfig",
    *,
    retriever: Optional["Retriever"],
    available_metrics: Iterable[str],
    planning_client: Optional[PlanningClient],
    decompose: Optional[DecomposeFn],
    corrective_retry: Optional[CorrectiveFn],
    route_fn: Optional[RouteFn],
    execute_fn: Optional[ExecuteFn],
    retrieval_cache: Optional[Any] = None,
    refresh: bool = False,
) -> OrchestrationResult:
    """Adaptive body under one shared budget. Wrapped by :func:`orchestrate`'s
    fail-soft boundary; the inner lane try/except keeps the specific
    ``adaptive_error`` demotion for a lane-stage failure."""
    route_fn = route_fn or dr.route
    execute_fn = execute_fn or dr.execute_route
    budget = ExecutionBudget.from_config(config)
    available_metrics = tuple(available_metrics)  # materialize (may feed key + router)

    try:
        decision = route_fn(plan, available_metrics)
    except Exception:  # noqa: BLE001 - router must never sink a request
        logger.exception("Adaptive router raised; treating as abstain")
        decision = dr.RouteDecision()

    lane, lane_reasons = _select_lane(plan, decision, config)
    result = OrchestrationResult(lane=lane, plan=plan)
    for code in lane_reasons:
        result.add_reason(code)

    # ── Versioned retrieval-cache probe (2.2.6.2) ──
    # Look up ONLY normalized pre-prompt evidence (never a final answer), keyed on
    # the plan + config/model fingerprint + store revision. A hit reuses the
    # cached evidence and skips all retrieval; the cheap deterministic tail
    # (re-grade + context budget) still runs so hit and miss produce the same
    # response shape. An explicit refresh never looks up (but still stores the
    # fresh result). Any cache/revision error is a miss, never a failure.
    coverage_route = bool(
        decision.tool_invocations
        and decision.tool_invocations[0].reason_code == dr.REASON_COVERAGE
    )
    cache_key = None if coverage_route else _retrieval_cache_key(
        retrieval_cache, store, plan, config, lane, available_metrics)
    if cache_key is not None and not refresh:
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            _load_cached_evidence(result, cached)
            result.add_reason("retrieval_cache_hit")
            if getattr(config, "enable_evidence_sufficiency", False):
                _grade_cached_evidence(result, result.plan)
            _apply_context_budget(result, result.plan, config, lane)
            result.retrieval = _as_retrieval_dict(result.plan, result)
            return result
        result.add_reason("retrieval_cache_miss")

    try:
        if lane is Lane.CATALOG:
            fell_back = _execute_catalog(
                result, plan, decision, store, config, budget, execute_fn, retriever
            )
            if fell_back is not None:
                return fell_back
        elif lane is Lane.FAST:
            fell_back = _execute_fast(
                result, plan, decision, store, config, budget, execute_fn, retriever
            )
            if fell_back is not None:
                return fell_back
        elif lane is Lane.STANDARD:
            _execute_standard(
                result, plan, decision, store, config, budget, execute_fn, retriever
            )
        else:
            _execute_complex(
                result, plan, decision, store, config, budget, execute_fn,
                retriever, planning_client, decompose, corrective_retry,
            )
    except Exception:  # noqa: BLE001 - a lane failure must fall soft, never raise
        logger.exception("Adaptive lane %s failed; falling back", lane.value)
        return _fallback(plan, retriever, store, config, reason="adaptive_error")

    # Grade (and, when enabled, decompose) against ``result.plan`` — the complex
    # lane may have attached drift-validated derived subqueries to it (2.2.4.2);
    # for every other path ``result.plan is plan`` so behavior is unchanged.
    if getattr(config, "enable_evidence_sufficiency", False):
        _apply_evidence_sufficiency(
            result, result.plan, store, config, budget, retriever)

    # Store the FINAL evidence (post any corrective round) so a later hit reuses
    # exactly what this request retrieved, without re-running any retrieval.
    if cache_key is not None:
        _store_cached_evidence(retrieval_cache, cache_key, result)

    _apply_context_budget(result, result.plan, config, lane)
    result.retrieval = _as_retrieval_dict(result.plan, result)
    return result


# ── Lane selection (deterministic, observable) ────────────


def _select_lane(
    plan: QueryPlan, decision: "RouteDecision", config: "MiddlewareConfig"
) -> tuple[Lane, list[str]]:
    """Pick the cheapest sufficient lane; return (lane, stable reason codes)."""
    complex_reasons = _complex_reasons(plan)
    catalog_route = bool(
        decision.matched
        and decision.tool_invocations
        and decision.tool_invocations[0].reason_code == dr.REASON_COVERAGE
    )

    if catalog_route and set(plan.obligations.evidence_modes) <= {"catalog"}:
        reasons = ["lane_catalog", "catalog_route", *decision.reason_codes]
        return Lane.CATALOG, _dedup(reasons)

    # Fast: a complete deterministic route with no qualitative doc obligation AND
    # no independent multi-obligation signal in the plan. The `not complex_reasons`
    # clause is defense in depth against a router that overclaims completeness
    # (2.2.3.4 review A): a plan with multiple entities / periods / mixed
    # modalities never skips document retrieval via the fast lane, even if the
    # deterministic route reported complete.
    if (
        decision.matched and decision.complete
        and not decision.requires_documents and not complex_reasons
    ):
        reasons = ["lane_fast", "fast_complete_route"]
        reasons.extend(decision.reason_codes)
        return Lane.FAST, _dedup(reasons)

    if complex_reasons:
        return Lane.COMPLEX, _dedup(["lane_complex", *complex_reasons])

    return Lane.STANDARD, ["lane_standard"]


def _complex_reasons(plan: QueryPlan) -> list[str]:
    """Signals that the plan has independently answerable obligations."""
    reasons: list[str] = []
    intents = set(plan.intents)
    non_general = [i for i in plan.intents if i != "general"]
    qualitative = intents & _QUALITATIVE_INTENTS
    structured = bool(plan.metrics) or bool(intents & _STRUCTURED_INTENTS)

    if len(plan.entities) > 2:
        reasons.append("complex_multi_entity")
    if len(plan.periods) > 1:
        reasons.append("complex_multi_period")
    if qualitative and structured:
        reasons.append("complex_structured_plus_qualitative")
    if len(set(non_general)) >= 2:
        reasons.append("complex_multi_intent")
    return reasons


def _dedup(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


# ── Fast lane ─────────────────────────────────────────────


def _execute_catalog(
    result: OrchestrationResult,
    plan: QueryPlan,
    decision: "RouteDecision",
    store: Any,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    execute_fn: ExecuteFn,
    retriever: Optional["Retriever"] = None,
) -> Optional[OrchestrationResult]:
    """Execute only the read-only coverage catalog; never create a retriever."""
    return _execute_fast(
        result, plan, decision, store, config, budget, execute_fn, retriever
    )


def _execute_fast(
    result: OrchestrationResult,
    plan: QueryPlan,
    decision: "RouteDecision",
    store: Any,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    execute_fn: ExecuteFn,
    retriever: Optional["Retriever"] = None,
) -> Optional[OrchestrationResult]:
    """Deterministic SQLite/tools/calculations only. No Chroma/BM25/rerank/plan.

    Returns a fallback :class:`OrchestrationResult` when the deterministic route
    errors (2.2.3.2 ``ExecutionResult.error`` semantics); otherwise ``None`` and
    ``result`` is populated in place.
    """
    for code in decision.reason_codes:
        result.add_reason(code)

    allowed = 0
    for _ in decision.tool_invocations:
        if budget.consume(DETERMINISTIC_TOOL):
            allowed += 1
        else:
            break

    execution = execute_fn(
        decision,
        store,
        max_tools=allowed,
        build_answer=bool(getattr(config, "enable_deterministic_answers", False)),
    )
    result.tool_execution = execution
    result.calculations = list(execution.calculations)
    for code in execution.incomplete_reason_codes:
        result.add_reason(code)
    if decision.tool_invocations and decision.tool_invocations[0].reason_code == dr.REASON_COVERAGE:
        if execution.invocations:
            payload = execution.answer_metadata or execution.invocations[0].result
            result.set_complete = bool(payload.get("complete", False))
            result.result_set_size = int(
                payload.get("total_matching") or payload.get("result_count") or 0
            )

    if execution.error:
        result.add_reason("deterministic_route_error")
        # Reuse the caller's shared Retriever (review E) rather than constructing
        # a fresh one, and note that the single fallback retrieve() is
        # intentionally NOT metered by the ExecutionBudget.
        return _fallback(
            plan, retriever, store, config, reason="deterministic_route_error"
        )

    result.deterministic_answer = execution.answer
    result.answer_origin = execution.answer_origin
    result.answer_metadata = execution.answer_metadata
    result.merged_facts = _normalize_tool_facts(execution)
    result.merged_documents = []  # fast lane never retrieves documents
    result.subqueries_executed = _sq0_ids(plan)
    result.retrieval_rounds_used = 0
    if budget.consume(SUBQUERY):
        result.add_reason("fast_executed_sq0")
    return None


# ── Standard lane ─────────────────────────────────────────


def _execute_standard(
    result: OrchestrationResult,
    plan: QueryPlan,
    decision: "RouteDecision",
    store: Any,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    execute_fn: ExecuteFn,
    retriever: Optional["Retriever"],
) -> None:
    """One coherent topic: existing hybrid retrieve() once, plus safe tool facts."""
    tool_facts = _catalog_or_partial_route_facts(
        result, plan, decision, store, config, budget, execute_fn
    )

    r = _get_retriever(retriever, store, config)
    facts: list[dict] = []
    docs: list[dict] = []
    if budget.consume(SUBQUERY) and budget.consume(RETRIEVAL_ROUND):
        facts, docs = _retrieve_round(r, plan, config, budget, Lane.STANDARD, result)
        result.retrieval_rounds_used += 1
    else:
        result.add_reason("standard_retrieval_skipped_budget")

    result.subqueries_executed = _sq0_ids(plan)
    result.merged_facts = _merge_facts(tool_facts, facts)
    result.merged_documents = _dedupe_docs(docs)


# ── Complex lane ──────────────────────────────────────────


def _execute_complex(
    result: OrchestrationResult,
    plan: QueryPlan,
    decision: "RouteDecision",
    store: Any,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    execute_fn: ExecuteFn,
    retriever: Optional["Retriever"],
    planning_client: Optional[PlanningClient],
    decompose: Optional[DecomposeFn],
    corrective_retry: Optional[CorrectiveFn],
) -> None:
    """Bounded multi-obligation lane. sq0 now; derived subqueries land in 2.2.4.2."""
    active_plan = plan

    # Optional pre-answer planning call — disabled by default, one budget unit,
    # only when deterministic parsing left retrieval modes unassigned.
    if (
        getattr(config, "adaptive_enable_planning_call", False)
        and planning_client is not None
        and _needs_planning(plan)
    ):
        if budget.consume(PLANNING_CALL):
            result.planning_ran = True
            revised = _try_planning(planning_client, plan)
            if revised is not None:
                active_plan = revised
                result.add_reason("planning_applied")
            else:
                result.add_reason("planning_invalid_rule_plan")
        else:
            result.add_reason("planning_budget_exhausted")

    tool_facts = _catalog_or_partial_route_facts(
        result, active_plan, decision, store, config, budget, execute_fn
    )

    # Selective decomposition (2.2.4.2): when enabled, a genuinely compound plan
    # is split into ≤2 drift-validated derived subqueries that are DECLARED on
    # the plan now (so the grader builds their obligations) but retrieved only in
    # the bounded corrective seam. When disabled — or when no derived subquery
    # survives validation — the seam is inert and a proposed decomposition is
    # recorded as deferred, exactly as before. sq0 is the only subquery retrieved
    # in this round, so ``subqueries_executed`` never claims an un-retrieved
    # subquery (2.2.3.4 review H).
    extra = _derive_subqueries(active_plan, decompose, config)
    if extra:
        if getattr(config, "enable_query_decomposition", False):
            attached = attach_derived_subqueries(active_plan, extra)
            if len(attached.subqueries) > len(active_plan.subqueries):
                active_plan = attached
                result.plan = active_plan
                result.add_reason("subquery_decomposition_applied")
                for subquery in active_plan.subqueries:
                    if subquery.derived and subquery.reason_code:
                        result.add_reason(subquery.reason_code)
            else:
                result.add_reason("subquery_decomposition_rejected")
        else:
            result.add_reason("subquery_decomposition_deferred")

    sq0 = active_plan.subqueries[0] if active_plan.subqueries else _synthetic_sq0(active_plan)
    executed: list[QuerySubquery] = []
    if budget.consume(SUBQUERY):
        executed.append(sq0)
    else:
        result.add_reason("subquery_budget_exhausted")

    result.subqueries_executed = [sq.id for sq in executed]

    r = _get_retriever(retriever, store, config)
    facts: list[dict] = []
    docs: list[dict] = []

    # All executed subqueries run as ONE bounded retrieval round.
    if executed and budget.consume(RETRIEVAL_ROUND):
        result.retrieval_rounds_used += 1
        rf, rd = _retrieve_round(r, active_plan, config, budget, Lane.COMPLEX, result)
        facts.extend(rf)
        docs.extend(rd)
    else:
        result.add_reason("complex_retrieval_skipped_budget")

    # Corrective second round hook (2.2.4.1). Bounded strictly by the round
    # budget: consume() returns False after the cap, so this can never loop.
    if (
        corrective_retry is not None
        and not getattr(config, "enable_evidence_sufficiency", False)
    ):
        while _wants_retry(corrective_retry, active_plan, facts, docs):
            if not budget.consume(RETRIEVAL_ROUND):
                result.add_reason("retrieval_round_budget_exhausted")
                break
            result.retrieval_rounds_used += 1
            result.add_reason("corrective_retry_round")
            rf, rd = _retrieve_round(r, active_plan, config, budget, Lane.COMPLEX, result)
            facts.extend(rf)
            docs.extend(rd)

    result.merged_facts = _merge_facts(tool_facts, facts)
    result.merged_documents = _dedupe_docs(docs)


def _wants_retry(
    corrective_retry: CorrectiveFn, plan: QueryPlan, facts: list[dict], docs: list[dict]
) -> bool:
    try:
        return bool(corrective_retry(plan, facts, docs))
    except Exception:  # noqa: BLE001 - a broken hook must not fail the request
        logger.warning("Corrective-retry hook failed; stopping", exc_info=True)
        return False


def _apply_evidence_sufficiency(
    result: OrchestrationResult,
    plan: QueryPlan,
    store: Any,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    retriever: Optional["Retriever"],
) -> None:
    """Grade once and execute at most one allowlisted internal correction."""
    retrieval = {
        "facts": result.merged_facts,
        "documents": result.merged_documents,
    }
    first = grade_evidence(plan, retrieval)
    result.sufficiency = first
    result.corrective_action = first.allowed_action
    result.add_reason(f"evidence_{first.status.value}")
    for code in first.reason_codes:
        result.add_reason(code)

    retries = max(0, min(1, int(getattr(config, "max_corrective_retries", 1))))
    if (
        first.status is not SufficiencyStatus.BORDERLINE
        or first.allowed_action is CorrectiveAction.NONE
        or not getattr(config, "enable_corrective_retry", False)
        or retries == 0
    ):
        return

    from .stream_events import current_emitter

    emitter = current_emitter()
    correction_started = perf_counter()
    if emitter is not None:
        emitter.stage("correct", "started", reason=first.allowed_action.value)

    if first.allowed_action is CorrectiveAction.RUN_DERIVED_SUBQUERIES:
        # 2.2.4.2 wires this reserved 2.2.4.1 seam: execute the plan's
        # drift-validated derived subqueries through their specialized modalities
        # and fuse. Gated by ``enable_query_decomposition`` so the seam stays the
        # 2.2.4.1 deferred placeholder (legacy behavior) until promotion.
        if getattr(config, "enable_query_decomposition", False):
            _run_derived_subqueries(
                result, plan, store, config, budget, retriever, first)
        else:
            result.add_reason("run_derived_subqueries_deferred_2_2_4_2")
        if emitter is not None:
            emitter.stage(
                "correct", "completed",
                elapsed_ms=(perf_counter() - correction_started) * 1000,
                reason=first.allowed_action.value,
            )
        return
    if not budget.consume(RETRIEVAL_ROUND):
        result.add_reason("retrieval_round_budget_exhausted")
        if emitter is not None:
            emitter.stage("correct", "fallback", reason="retrieval_round_budget_exhausted")
        return

    result.retry_performed = True
    result.retrieval_rounds_used += 1
    result.add_reason("corrective_retry_round")
    result.add_reason(f"corrective_action_{first.allowed_action.value}")
    r = _get_retriever(retriever, store, config)
    if first.allowed_action is CorrectiveAction.EXPAND_PARENT_SECTION:
        facts: list[dict] = []
        if getattr(config, "enable_hierarchical_retrieval", False):
            # Bounded filing → section → child expansion (2.2.5.3). Returns the
            # annotated root hits plus their in-budget neighbors, so it REPLACES
            # the document set (annotations must win over the un-annotated roots).
            expanded = r.hierarchical_expand(
                result.merged_documents, plan, missing_local_context=True)
            result.merged_facts = _merge_facts(result.merged_facts, facts)
            result.merged_documents = _dedupe_docs(expanded)
            result.add_reason("hierarchical_expansion_applied")
        else:
            documents = r.expand_parent_sections(result.merged_documents)
            result.merged_facts = _merge_facts(result.merged_facts, facts)
            result.merged_documents = _dedupe_docs(
                [*result.merged_documents, *documents])
    else:
        facts, documents = _retrieve_round(
            r, plan, config, budget, result.lane, result,
            corrective_action=first.allowed_action,
            corrective_missing_fields=tuple(
                field for row in first.coverage for field in row.missing_fields
            ),
        )
        result.merged_facts = _merge_facts(result.merged_facts, facts)
        result.merged_documents = _dedupe_docs(
            [*result.merged_documents, *documents])
    result.sufficiency = grade_evidence(plan, {
        "facts": result.merged_facts,
        "documents": result.merged_documents,
    })
    if result.sufficiency.status is not SufficiencyStatus.SUFFICIENT:
        result.add_reason("corrective_retry_exhausted")
    if emitter is not None:
        emitter.stage(
            "correct", "completed",
            elapsed_ms=(perf_counter() - correction_started) * 1000,
            reason=first.allowed_action.value,
        )


# ── Selective decomposition + weighted fusion (2.2.4.2) ────


def _derive_subqueries(
    plan: QueryPlan, decompose: Optional[DecomposeFn], config: "MiddlewareConfig"
) -> list[QuerySubquery]:
    """Proposed derived subqueries for a compound plan (never ``sq0``).

    Uses the injected ``decompose`` hook when supplied (tests substitute a fake),
    else the deterministic :func:`~src.middleware.query_plan.decompose_plan` when
    decomposition is enabled, else none. Best-effort — a raising decomposer
    yields ``[]`` so a broken hook can never fail the request."""
    try:
        if decompose is not None:
            proposed = decompose(plan) or []
        elif getattr(config, "enable_query_decomposition", False):
            proposed = decompose_plan(plan) or []
        else:
            return []
    except Exception:  # noqa: BLE001 - decomposition is a best-effort hook
        logger.warning("Subquery decomposition failed; using rule plan", exc_info=True)
        return []
    return [s for s in proposed if getattr(s, "id", None) != "sq0"]


def _subquery_legacy_intent(plan: QueryPlan, sq: QuerySubquery) -> dict:
    """Project one derived subquery onto a legacy-intent dict so the existing
    retriever routes it to the right specialized modality (facts/tools → SQLite,
    qualitative → hybrid, news/risk/sentiment → document channel with the
    ticker metadata filter, macro → FRED, projection → estimates)."""
    intent = dict(plan.to_legacy_intent())
    intent["ticker"] = sq.entity_tickers[0] if sq.entity_tickers else None
    intent["metrics"] = list(sq.metrics)
    if sq.periods:
        intent["timeframe"] = sq.periods[0]
    modes = set(sq.retrieval_modes)
    sub_intents = set(sq.intents) or set(plan.intents)
    if "macro" in modes:
        intent["question_type"] = "general"
        intent["ticker"] = None
    elif "documents" in modes and "facts" not in modes:
        if "news" in sub_intents:
            intent["question_type"] = "news"
        elif "risk" in sub_intents:
            intent["question_type"] = "risk"
        elif "sentiment" in sub_intents:
            intent["question_type"] = "sentiment"
        else:
            intent["question_type"] = "explanation"
    elif "facts" in modes and "documents" not in modes:
        intent["question_type"] = (
            "projection" if "projection" in sub_intents else "fact_lookup")
    else:
        if "projection" in sub_intents:
            intent["question_type"] = "projection"
        elif "comparison" in sub_intents:
            intent["question_type"] = "comparison"
        else:
            intent["question_type"] = "fact_lookup"
    return intent


def _retrieve_derived_subquery(
    r: "Retriever", plan: QueryPlan, sq: QuerySubquery, config: "MiddlewareConfig"
) -> tuple[list[dict], list[dict]]:
    """Retrieve one derived subquery's specialized evidence (un-reranked pool).

    Uses ``retrieve_candidates`` so documents come back before any re-rank —
    fusion owns the single conditional re-rank. Facts are tagged with their
    subquery id for provenance."""
    intent = _subquery_legacy_intent(plan, sq)
    cand = r.retrieve_candidates(
        query=sq.text, intent=intent,
        top_k_documents=int(getattr(config, "top_k_documents", 5)),
        top_k_facts=int(getattr(config, "top_k_facts", 10)),
    )
    facts = [dict(f, subquery_id=sq.id) for f in cand.get("facts", []) if isinstance(f, dict)]
    docs = [d for d in cand.get("documents", []) if isinstance(d, dict)]
    return facts, docs


def _timed_rerank(r: "Retriever", query: str, docs: list[dict], *, top_n: int):
    """Run one adaptive rerank and expose its duration through Retriever timings."""
    started = time.perf_counter()
    try:
        return r.reranker.rerank(query, docs, top_n=top_n)
    finally:
        timings = getattr(r, "_timings", None)
        if not isinstance(timings, dict):
            timings = {}
            r._timings = timings
        timings["fusion_rerank"] = float(timings.get("fusion_rerank") or 0.0) + (
            time.perf_counter() - started
        ) * 1000


def _maybe_rerank_fused_documents(
    r: "Retriever",
    plan: QueryPlan,
    fused: list[dict],
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    result: OrchestrationResult,
) -> list[dict]:
    """Pass the fused candidates through the conditional re-ranker at most once.

    Bounded by the shared single RERANK_CALL unit — if the sq0 round already
    used it, this skips. Never raises: a reranker failure (or its silent RRF
    fallback) keeps the fused order and records ``rerank_fallback``."""
    top_k = int(getattr(config, "top_k_documents", 5))
    if len(fused) < 2:
        return fused
    if not (getattr(config, "enable_reranker", False)
            and getattr(config, "adaptive_conditional_rerank", True)):
        return fused
    if not budget.consume(RERANK_CALL):
        result.add_reason("rerank_budget_exhausted")
        return fused[:top_k]
    try:
        reranked = _timed_rerank(
            r, plan.retrieval_query, list(fused), top_n=top_k,
        )
    except Exception:  # noqa: BLE001 - reranker should not raise
        logger.warning("Fused rerank failed; keeping fusion order", exc_info=True)
        result.add_reason("rerank_fallback")
        return fused[:top_k]
    if reranked and all(d.get("rerank_score") is None for d in reranked):
        result.add_reason("rerank_fallback")
        return list(reranked)
    result.rerank_ran = True
    result.add_reason("rerank_applied_derived_fusion")
    return list(reranked)


def _run_derived_subqueries(
    result: OrchestrationResult,
    plan: QueryPlan,
    store: Any,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    retriever: Optional["Retriever"],
    first: SufficiencyResult,
) -> None:
    """Execute the plan's drift-validated derived subqueries (2.2.4.2 Steps 3-5).

    One bounded retrieval round: each uncovered derived subquery is retrieved
    through its specialized modality, its documents are fused with the original
    ``sq0`` documents (weighted RRF — ``sq0`` strongest), the fused set is
    conditionally re-ranked at most once, derived facts are merged onto their
    obligations, and sufficiency is re-graded. The SUBQUERY / RETRIEVAL_ROUND /
    RERANK caps make this uncircumventable; fail-soft keeps the pre-derived
    evidence on any error."""
    missing = set(first.missing_subqueries)
    derived = [sq for sq in plan.subqueries if sq.derived and sq.id in missing]
    result.proposed_subqueries = sum(1 for sq in plan.subqueries if sq.derived)
    if not derived:
        result.add_reason("no_derived_subqueries_to_run")
        return
    if not budget.consume(RETRIEVAL_ROUND):
        result.add_reason("retrieval_round_budget_exhausted")
        return

    result.retry_performed = True
    result.retrieval_rounds_used += 1
    result.add_reason("corrective_retry_round")
    result.add_reason("corrective_action_run_derived_subqueries")

    try:
        r = _get_retriever(retriever, store, config)
        channels: list[dict] = [
            {"subquery_id": "sq0", "weight": 1.0,
             "documents": list(result.merged_documents)},
        ]
        derived_facts: list[dict] = []
        for sq in derived:
            if not budget.consume(SUBQUERY):
                result.add_reason("subquery_budget_exhausted")
                break
            sq_facts, sq_docs = _retrieve_derived_subquery(r, plan, sq, config)
            weight = 0.6 if sq.derivation_source == "planner" else 0.8
            channels.append(
                {"subquery_id": sq.id, "weight": weight, "documents": sq_docs})
            derived_facts.extend(sq_facts)
            result.subqueries_executed.append(sq.id)
            result.derived_subqueries.append(sq.id)

        from .retriever import fuse_weighted_subqueries
        fused = fuse_weighted_subqueries(
            channels, top_k=int(getattr(config, "top_k_documents", 5)),
            k=int(getattr(config, "rrf_k", 60)))
        fused = _maybe_rerank_fused_documents(r, plan, fused, config, budget, result)
        result.merged_documents = fused
        result.merged_facts = _merge_facts(result.merged_facts, derived_facts)
        result.retrieval_strategy = "decomposed_fusion"
        result.add_reason("derived_fusion_applied")
    except Exception:  # noqa: BLE001 - decomposition must fall soft, never raise
        logger.warning(
            "Derived subquery execution failed; keeping sq0 evidence", exc_info=True)
        result.add_reason("derived_subqueries_error")

    result.sufficiency = grade_evidence(plan, {
        "facts": result.merged_facts,
        "documents": result.merged_documents,
    })
    if result.sufficiency.status is not SufficiencyStatus.SUFFICIENT:
        result.add_reason("corrective_retry_exhausted")


# ── Shared retrieval + conditional rerank ─────────────────


def _catalog_or_partial_route_facts(
    result: OrchestrationResult,
    plan: QueryPlan,
    decision: "RouteDecision",
    store: Any,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    execute_fn: ExecuteFn,
) -> list[dict]:
    """Resolve a catalog universe before one bounded structured comparison."""
    is_catalog = bool(
        decision.matched
        and decision.tool_invocations
        and decision.tool_invocations[0].reason_code == dr.REASON_COVERAGE
    )
    if not is_catalog or "facts" not in plan.obligations.evidence_modes:
        return _partial_route_facts(
            result, decision, store, config, budget, execute_fn
        )

    if not budget.consume(DETERMINISTIC_TOOL):
        result.add_reason("deterministic_tool_budget_exhausted")
        return []
    catalog = execute_fn(decision, store, max_tools=1, build_answer=False)
    result.tool_execution = catalog
    for code in catalog.incomplete_reason_codes:
        if code != dr.INCOMPLETE_MISSING_METRIC_COVERAGE:
            result.add_reason(code)
    if catalog.error or not catalog.invocations:
        result.add_reason("partial_route_error_ignored")
        return []

    facts = _normalize_tool_facts(catalog)
    payload = catalog.invocations[0].result
    tickers = [
        str(row.get("ticker"))
        for row in payload.get("securities") or []
        if isinstance(row, dict) and row.get("ticker")
    ]
    total = int(payload.get("total_matching") or len(tickers))
    result.result_set_size = total
    result.set_complete = bool(payload.get("complete", False))
    result.add_reason("includes_deterministic_tool_evidence")

    blocking_codes = {
        dr.INCOMPLETE_PARTIAL_CATALOG_PAGE,
        dr.INCOMPLETE_MISSING_COVERAGE_DIMENSION,
        dr.INCOMPLETE_UNRESOLVED_UNIVERSE_SCOPE,
        dr.INCOMPLETE_QUALITATIVE_EVIDENCE_REQUIRED,
    }
    if not result.set_complete or blocking_codes & set(catalog.incomplete_reason_codes):
        return facts

    cap = max(1, min(int(getattr(config, "top_k_facts", 10)), 100))
    if total > cap or len(tickers) > cap:
        result.set_complete = False
        result.add_reason(dr.INCOMPLETE_RESULT_SET_TOO_LARGE)
        facts.append({
            "metric": "coverage_request_to_narrow",
            "value": (
                f"The catalog matched {total} securities, above the {cap}-security "
                "comparison cap. Narrow by sector, index, coverage tier, or ticker."
            ),
            "ticker": "CATALOG",
            "source_type": "catalog",
            "kind": "tool_result",
            "complete": False,
            "total_matching": total,
        })
        return facts

    metrics = list(plan.obligations.metrics or plan.metrics)
    if len(metrics) != 1 or not tickers:
        result.set_complete = False
        result.add_reason(dr.INCOMPLETE_MISSING_METRIC_COVERAGE)
        return facts
    if not budget.consume(DETERMINISTIC_TOOL):
        result.set_complete = False
        result.add_reason("deterministic_tool_budget_exhausted")
        result.add_reason(dr.INCOMPLETE_RESULT_SET_TOO_LARGE)
        return facts

    arguments = {
        "metric": metrics[0],
        "tickers": tickers,
        "latest_only": plan.obligations.as_of in {None, "latest"},
        "limit": len(tickers),
    }
    if plan.obligations.completeness == "top_n":
        arguments.update(order="desc", limit=plan.obligations.limit or len(tickers))
    elif plan.obligations.completeness == "bottom_n":
        arguments.update(order="asc", limit=plan.obligations.limit or len(tickers))
    invocation = dr.ToolInvocation(
        "query_facts", arguments,
        plan.subqueries[0].id if plan.subqueries else "sq0",
        dr.REASON_COMPARE,
    )
    structured_decision = dr.RouteDecision(
        matched=True,
        complete=True,
        tool_invocations=[invocation],
        reason_codes=[dr.REASON_COMPARE],
    )
    structured = execute_fn(
        structured_decision, store, max_tools=1, build_answer=False
    )
    catalog.invocations.extend(structured.invocations)
    catalog.calculations.extend(structured.calculations)
    catalog.error = catalog.error or structured.error
    catalog.complete = catalog.complete and structured.complete
    catalog.incomplete_reason_codes.extend(
        code for code in structured.incomplete_reason_codes
        if code not in catalog.incomplete_reason_codes
    )
    result.tool_execution = catalog
    if structured.error:
        result.set_complete = False
        result.add_reason("partial_route_error_ignored")
        return facts

    structured_facts = _normalize_tool_facts(structured)
    covered = {str(fact.get("ticker")) for fact in structured_facts if fact.get("ticker")}
    if not set(tickers) <= covered:
        result.set_complete = False
        result.add_reason(dr.INCOMPLETE_MISSING_METRIC_COVERAGE)
    return [*facts, *structured_facts]


def _partial_route_facts(
    result: OrchestrationResult,
    decision: "RouteDecision",
    store: Any,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    execute_fn: ExecuteFn,
) -> list[dict]:
    """Run a safe *partial* deterministic route (matched but incomplete) and
    return its tool facts, or ``[]``. A complete route belongs to the fast lane;
    an abstain contributes nothing here."""
    if not (decision.matched and not decision.complete and decision.tool_invocations):
        return []
    allowed = 0
    for _ in decision.tool_invocations:
        if budget.consume(DETERMINISTIC_TOOL):
            allowed += 1
        else:
            break
    if allowed == 0:
        return []
    execution = execute_fn(decision, store, max_tools=allowed, build_answer=False)
    if execution.error:
        result.add_reason("partial_route_error_ignored")
        return []
    for code in execution.incomplete_reason_codes:
        result.add_reason(code)
    result.tool_execution = execution
    result.calculations = list(execution.calculations)
    result.add_reason("includes_deterministic_tool_evidence")
    return _normalize_tool_facts(execution)


def _retrieve_round(
    r: "Retriever",
    plan: QueryPlan,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    lane: Lane,
    result: OrchestrationResult,
    corrective_action: CorrectiveAction = CorrectiveAction.NONE,
    corrective_missing_fields: tuple[str, ...] = (),
) -> tuple[list[dict], list[dict]]:
    """Perform one bounded retrieval for ``plan`` (sq0). Uses the existing
    hybrid ``retrieve()`` unless conditional re-ranking is engaged, in which case
    it pulls the candidate pool via ``retrieve_candidates()`` and re-ranks only on
    a documented ambiguity signal."""
    query = plan.retrieval_query
    intent = plan.to_legacy_intent()
    if corrective_action is CorrectiveAction.BROADEN_TICKER_FILTER:
        intent["ticker"] = None
    elif corrective_action is CorrectiveAction.APPLY_VALIDATED_ALIAS:
        intent["metrics"] = list(validated_metric_aliases(intent.get("metrics", [])))
    elif corrective_action is CorrectiveAction.ALTERNATE_INTERNAL_MODALITY:
        missing_documents = any(
            field.startswith(("document:", "news:", "filing:"))
            for field in corrective_missing_fields
        )
        if missing_documents:
            intent["question_type"] = "explanation"
        else:
            intent["question_type"] = "fact_lookup"
    top_k = int(getattr(config, "top_k_documents", 5))
    top_f = int(getattr(config, "top_k_facts", 10))

    use_conditional = (
        bool(getattr(config, "enable_reranker", False))
        and bool(getattr(config, "adaptive_conditional_rerank", True))
        and lane in (Lane.STANDARD, Lane.COMPLEX)
    )

    if not use_conditional:
        base = r.retrieve(
            query=query, intent=intent, top_k_documents=top_k, top_k_facts=top_f
        )
        result.retrieval_strategy = base.get("retrieval_strategy", "vector")
        return list(base.get("facts", [])), list(base.get("documents", []))

    cand = r.retrieve_candidates(
        query=query, intent=intent, top_k_documents=top_k, top_k_facts=top_f
    )
    result.retrieval_strategy = cand.get("retrieval_strategy") or cand.get("strategy") or "hybrid"
    facts = list(cand.get("facts", []))
    docs = _conditional_rerank(r, plan, cand, config, budget, result)
    return facts, docs


def _conditional_rerank(
    r: "Retriever",
    plan: QueryPlan,
    cand: dict,
    config: "MiddlewareConfig",
    budget: ExecutionBudget,
    result: OrchestrationResult,
) -> list[dict]:
    """Re-rank the candidate pool only on a documented ambiguity signal; else
    keep RRF order truncated to ``top_k_documents``. Never raises: a reranker
    failure keeps the fused order and records ``rerank_fallback``."""
    top_k = int(getattr(config, "top_k_documents", 5))
    pool = list(cand.get("documents", []))
    candidate_count = int(cand.get("candidate_count", len(pool)))

    if candidate_count <= top_k:
        result.add_reason("rerank_skipped_candidates_at_limit")
        return pool[:top_k]

    signals = _rerank_signals(cand, plan)
    if not signals:
        result.add_reason("rerank_skipped_no_ambiguity")
        return pool[:top_k]
    for s in signals:
        result.add_reason(f"rerank_signal_{s}")

    if not budget.consume(RERANK_CALL):
        result.add_reason("rerank_budget_exhausted")
        return pool[:top_k]

    try:
        reranked = _timed_rerank(r, plan.retrieval_query, pool, top_n=top_k)
    except Exception:  # noqa: BLE001 - defensive: reranker should not raise
        logger.warning("Conditional rerank failed; keeping RRF order", exc_info=True)
        result.add_reason("rerank_fallback")
        return pool[:top_k]

    # Reranker.rerank NEVER raises: on an internal load/score failure it silently
    # returns the RRF order with rerank_score=None on every doc. Detect that so
    # we report rerank_fallback, not rerank_applied (2.2.3.4 review G).
    if reranked and all(d.get("rerank_score") is None for d in reranked):
        result.add_reason("rerank_fallback")
        return list(reranked)

    result.rerank_ran = True
    result.add_reason("rerank_applied")
    return list(reranked)


# Two top-3 fusion scores within this spread count as "tightly clustered".
_CLUSTER_EPS = 1e-3


def _rerank_signals(cand: dict, plan: QueryPlan) -> list[str]:
    """Documented ambiguity signals that justify one re-rank call."""
    signals: list[str] = []

    vec = list(cand.get("vector_ids", []))[:3]
    lex = list(cand.get("lexical_ids", []))[:3]
    if vec and lex and len(set(vec) & set(lex)) <= 1:
        signals.append("channel_disagreement")

    if len(plan.entities) >= 2 or len(plan.subqueries) >= 2:
        signals.append("multi_entity_competition")

    docs = cand.get("documents", [])
    scores = [
        d.get("fusion_score") for d in docs[:3]
        if isinstance(d, dict) and d.get("fusion_score") is not None
    ]
    if len(scores) >= 2 and (max(scores) - min(scores)) <= _CLUSTER_EPS:
        signals.append("clustered_scores")

    qualitative = set(plan.intents) & _QUALITATIVE_INTENTS
    if qualitative and (plan.metrics or plan.entities):
        signals.append("qualitative_plus_anchor")

    return _dedup(signals)


# ── Planning (optional, disabled by default) ──────────────


def _needs_planning(plan: QueryPlan) -> bool:
    """True when deterministic parsing left any subquery without retrieval modes."""
    if not plan.subqueries:
        return True
    return any(not sq.retrieval_modes for sq in plan.subqueries)


def _try_planning(
    planning_client: PlanningClient, plan: QueryPlan
) -> Optional[QueryPlan]:
    """Call the planner, parse strict JSON into a validated QueryPlan, or None.

    The raw ``original_question`` is always preserved from ``plan`` — a planner
    can propose a retrieval query and structured slots, never rewrite the user's
    request. Invalid output returns ``None`` so the caller keeps the rule plan.
    """
    try:
        raw = planning_client(plan)
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        if not isinstance(data, dict):
            return None
        return _plan_from_json(data, plan).validate()
    except Exception:  # noqa: BLE001 - invalid plan → rule plan, never raise
        logger.warning("Planning output invalid; keeping rule plan", exc_info=True)
        return None


def _plan_from_json(data: dict, base: QueryPlan) -> QueryPlan:
    """Build a QueryPlan from planner JSON, preserving the raw question."""
    retrieval_query = str(data.get("retrieval_query") or base.retrieval_query).strip()
    entities: list[QueryEntity] = []
    for i, raw_entity in enumerate(data.get("entities", []) or []):
        ticker = str(raw_entity).strip().upper()
        if ticker:
            entities.append(
                QueryEntity(
                    ticker=ticker, resolved_name=None, confidence=1.0,
                    source="planner", mention=ticker, start=-1,
                )
            )
    metrics = _dedup(str(m) for m in (data.get("metrics", []) or []))
    intents = _dedup(str(i) for i in (data.get("intents", []) or [])) or ["general"]
    periods = _dedup(str(p) for p in (data.get("periods", []) or []))
    sq0 = QuerySubquery(
        id="sq0", text=retrieval_query,
        entity_tickers=tuple(e.ticker for e in entities),
        intents=tuple(intents), metrics=tuple(metrics), periods=tuple(periods),
        retrieval_modes=(), derived=False, parent_id=None,
    )
    return QueryPlan(
        original_question=base.original_question,
        retrieval_query=retrieval_query,
        normalized_question=normalize_question(retrieval_query),
        entities=entities,
        intents=intents,
        metrics=metrics,
        periods=periods,
        subqueries=[sq0],
        primary_intent=intents[0],
        evidence_topic=base.evidence_topic,
        evidence_filters=dict(base.evidence_filters),
        obligations=base.obligations,
        reason_codes=["planner"],
    )


# ── Fallback + helpers ────────────────────────────────────


def _fallback(
    plan: QueryPlan,
    retriever: Optional["Retriever"],
    store: Any,
    config: "MiddlewareConfig",
    *,
    reason: str,
) -> OrchestrationResult:
    """Demote to the existing single-query ``Retriever.retrieve()`` path.

    Produces a standard-lane result whose ``merged_facts``/``merged_documents``
    and ``retrieval`` mirror ``retrieve()`` byte-for-byte, so a feature-disabled
    or failed adaptive request is indistinguishable from the current pipeline.

    The single ``retrieve()`` here is intentionally NOT metered by the
    :class:`ExecutionBudget` (2.2.3.4 review E): it is the legacy fallback path,
    not an additional adaptive retrieval round, so it must run even when the
    adaptive budget is already exhausted.
    """
    result = OrchestrationResult(lane=Lane.STANDARD, plan=plan, fallback_reason=reason)
    result.add_reason(f"fallback_{reason}")
    try:
        r = _get_retriever(retriever, store, config)
        base = r.retrieve(
            query=plan.retrieval_query,
            intent=plan.to_legacy_intent(),
            top_k_documents=int(getattr(config, "top_k_documents", 5)),
            top_k_facts=int(getattr(config, "top_k_facts", 10)),
        )
        result.retrieval = base
        result.merged_facts = list(base.get("facts", []))
        result.merged_documents = list(base.get("documents", []))
        result.retrieval_strategy = base.get("retrieval_strategy")
        result.subqueries_executed = _sq0_ids(plan)
        result.retrieval_rounds_used = 1
    except Exception:  # noqa: BLE001 - even the fallback retrieve must not raise
        logger.exception("Fallback retrieve() failed; returning empty evidence")
        result.add_reason("fallback_retrieve_error")
        result.retrieval = {"facts": [], "documents": [], "ticker": None}

    _apply_context_budget(result, plan, config, Lane.STANDARD)
    return result


def _apply_context_budget(
    result: OrchestrationResult, plan: QueryPlan, config: "MiddlewareConfig", lane: Lane
) -> None:
    # Never raises (2.2.3.4 review C): a budgeter failure yields an empty
    # selection with a reason code, so even the last-resort fallback that calls
    # this cannot re-enter a raising budgeter and loop/escape.
    try:
        selection = ContextBudget(config).select(
            plan, result.merged_facts, result.merged_documents, lane
        )
    except Exception:  # noqa: BLE001 - budgeter must fail soft to empty selection
        logger.exception("Context budgeting failed; using empty selection")
        selection = ContextSelection(reason_codes=["context_budget_error"])
    result.context = selection
    result.context_size = selection.context_chars
    result.estimated_tokens = selection.estimated_tokens


# ── Versioned retrieval cache seam (2.2.6.2) ──────────────


def _retrieval_cache_key(
    cache: Optional[Any],
    store: Any,
    plan: QueryPlan,
    config: "MiddlewareConfig",
    lane: Lane,
    available_metrics: tuple,
) -> Optional[str]:
    """Build this request's retrieval-cache key, or ``None`` to bypass the cache.

    Bypasses (returns ``None``) when the cache is absent/disabled, the store
    exposes no revision (or reading it raised), or the key cannot be built — any
    of which means "do not cache", never an error. Reading the revision here binds
    the key to the exact data version, so an ingestion write between two identical
    requests forces a fresh retrieval.
    """
    if cache is None or not getattr(config, "enable_retrieval_cache", False):
        return None
    try:
        from .retrieval_cache import build_cache_key, config_fingerprint

        revision_fn = getattr(store, "retrieval_revision", None)
        if not callable(revision_fn):
            return None  # store has no revision -> cannot invalidate safely
        # Drop the whole cache if the config/model fingerprint changed since the
        # last request (a redeploy that swapped the model or a retrieval toggle).
        cache.check_fingerprint(config_fingerprint(config))
        revision = int(revision_fn())
        corpus_revision_fn = getattr(store, "corpus_revision", None)
        corpus_revision = (
            int(corpus_revision_fn()) if callable(corpus_revision_fn) else revision
        )
        return build_cache_key(
            plan=plan, config=config, lane=lane, revision=revision,
            corpus_revision=corpus_revision,
            available_metrics=available_metrics,
            as_of=getattr(plan, "as_of", None),
        )
    except Exception:  # noqa: BLE001 - inability to key means bypass, never fail
        logger.warning("Retrieval-cache key build failed; bypassing cache", exc_info=True)
        return None


def _load_cached_evidence(result: OrchestrationResult, cached: dict) -> None:
    """Populate ``result`` from a cached pre-prompt evidence snapshot (2.2.6.2)."""
    result.merged_facts = list(cached.get("merged_facts", []))
    result.merged_documents = list(cached.get("merged_documents", []))
    result.retrieval_strategy = cached.get("retrieval_strategy")
    result.subqueries_executed = list(cached.get("subqueries_executed", []))
    result.retrieval_rounds_used = int(cached.get("retrieval_rounds_used", 0))
    result.derived_subqueries = list(cached.get("derived_subqueries", []))
    for code in cached.get("reason_codes", []):
        result.add_reason(code)


def _store_cached_evidence(
    cache: Any, cache_key: str, result: OrchestrationResult
) -> None:
    """Cache ONLY the normalized pre-prompt evidence + retrieval metadata.

    Never stores a final answer, the context selection, or grader objects — just
    the merged facts/documents and the retrieval telemetry needed to reproduce
    the same response shape on a hit. Fail-soft: a store failure is ignored.
    """
    try:
        snapshot = {
            "merged_facts": result.merged_facts,
            "merged_documents": result.merged_documents,
            "retrieval_strategy": result.retrieval_strategy,
            "subqueries_executed": list(result.subqueries_executed),
            "retrieval_rounds_used": int(result.retrieval_rounds_used),
            "derived_subqueries": list(result.derived_subqueries),
            # Exclude this-request cache-status codes so a later hit does not
            # replay a stale "miss" marker alongside its own "hit".
            "reason_codes": [
                c for c in result.reason_codes if not c.startswith("retrieval_cache_")
            ],
        }
        cache.set(cache_key, snapshot)
    except Exception:  # noqa: BLE001 - caching must never fail a request
        logger.warning("Retrieval-cache store failed; continuing", exc_info=True)


def _grade_cached_evidence(result: OrchestrationResult, plan: QueryPlan) -> None:
    """Re-grade cached evidence so a hit reports the same sufficiency as a miss.

    Grading is a pure function of (plan, evidence); the cached evidence is the
    final set the original request produced, so this reproduces the verdict
    without any (expensive) corrective retrieval. NOTE: with query decomposition
    enabled the miss path may grade against a decomposed plan while a hit grades
    the original plan — an accepted, documented minor divergence for an opt-in
    cache; with decomposition off (default) the verdicts are identical.
    """
    verdict = grade_evidence(plan, {
        "facts": result.merged_facts,
        "documents": result.merged_documents,
    })
    result.sufficiency = verdict
    result.corrective_action = verdict.allowed_action
    result.add_reason(f"evidence_{verdict.status.value}")
    for code in verdict.reason_codes:
        result.add_reason(code)


def _get_retriever(
    retriever: Optional["Retriever"], store: Any, config: "MiddlewareConfig"
) -> "Retriever":
    if retriever is not None:
        return retriever
    from .retriever import Retriever  # lazy: heavy import kept off the fast lane
    return Retriever(store=store, config=config)


def _normalize_tool_facts(execution: "ExecutionResult") -> list[dict]:
    """Project deterministic tool results into fact-like evidence rows.

    Covers the dominant read-tool shapes: ``get_fundamentals`` (metric→value
    map), ``query_facts`` (result rows), and ``estimates``/``price_targets``
    (metric→{value,period} maps). Unhandled shapes are simply skipped — the raw
    results remain available on ``OrchestrationResult.tool_execution``.
    """
    facts: list[dict] = []
    for inv in execution.invocations:
        res = inv.result if isinstance(inv.result, dict) else {}
        if inv.error:
            continue

        if inv.name == "describe_coverage":
            operation = str(inv.arguments.get("operation") or "summary")
            values: list[str] = []
            for key in ("securities", "sources", "item_types", "metrics"):
                for value in res.get(key) or []:
                    if isinstance(value, dict):
                        label = value.get("ticker") or value.get("source")
                    else:
                        label = value
                    if label is not None and str(label) not in values:
                        values.append(str(label))
            display = ", ".join(values) if values else str(
                res.get("message") or res.get("total_matching") or 0
            )
            if res.get("complete") is False and res.get("status") != "unavailable":
                display = (
                    f"{display}. Returned {res.get('result_count', len(values))} of "
                    f"{res.get('total_matching', len(values))}; narrow with a sector, "
                    "index, coverage tier, source, item type, or ticker filter."
                )
            facts.append({
                "metric": f"coverage_{operation}",
                "value": display,
                "ticker": (
                    inv.arguments.get("ticker")
                    if operation == "security_sources"
                    else "CATALOG"
                ),
                "period": res.get("universe_snapshot_at"),
                "source_type": "catalog",
                "kind": "tool_result",
                "complete": bool(res.get("complete", False)),
                "total_matching": res.get("total_matching"),
            })
            continue

        fundamentals = res.get("fundamentals")
        if isinstance(fundamentals, dict):
            ticker = res.get("ticker")
            for metric, value in fundamentals.items():
                if value is not None:
                    facts.append({
                        "metric": metric, "value": value, "ticker": ticker,
                        "period": None, "source_type": "tool",
                    })

        rows = res.get("results")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("value") is not None:
                    facts.append({**row, "source_type": row.get("source_type", "tool")})

        for key in ("estimates", "price_targets"):
            mapping = res.get(key)
            if isinstance(mapping, dict):
                for metric, fact in mapping.items():
                    if isinstance(fact, dict) and fact.get("value") is not None:
                        facts.append({
                            "metric": metric, "value": fact.get("value"),
                            "unit": fact.get("unit"),
                            "period": fact.get("period"), "ticker": res.get("ticker"),
                            "source_type": "estimates",
                        })

        guidance = res.get("guidance")
        if isinstance(guidance, dict):
            for metric, fact in guidance.items():
                if isinstance(fact, dict) and fact.get("value") is not None:
                    facts.append({
                        "metric": f"guidance_{metric}", "value": fact.get("value"),
                        "unit": fact.get("unit"), "period": fact.get("period"),
                        "ticker": res.get("ticker"), "source_type": "guidance",
                    })

        macro = res.get("macro")
        if isinstance(macro, dict):
            for metric, fact in macro.items():
                if isinstance(fact, dict) and fact.get("value") is not None:
                    facts.append({
                        "metric": metric, "value": fact.get("value"),
                        "unit": fact.get("unit"),
                        "period": fact.get("period") or fact.get("as_of"),
                        "ticker": "MACRO", "source_type": fact.get("source_type", "fred"),
                    })
                elif isinstance(fact, (int, float)):
                    facts.append({
                        "metric": metric, "value": fact, "ticker": "MACRO",
                        "source_type": "fred",
                    })

        if inv.name == "classify_trade_bias":
            ticker = res.get("ticker") or inv.arguments.get("ticker")
            facts.append({
                "metric": "trade_bias",
                "value": res.get("bias"),
                "ticker": ticker,
                "source_type": "tool",
                "kind": "tool_result",
            })
            for signal in res.get("signals") or []:
                if not isinstance(signal, dict) or not signal.get("name"):
                    continue
                facts.append({
                    "metric": str(signal["name"]),
                    "value": signal.get("value"),
                    "ticker": ticker,
                    "period": signal.get("period"),
                    "source_type": "tool",
                    "kind": "tool_result",
                })

        if inv.name == "check_freshness":
            ticker = res.get("ticker") or inv.arguments.get("ticker")
            for source, detail in (res.get("sources") or {}).items():
                if isinstance(detail, dict):
                    status = detail.get("status") or detail.get("freshness") or "unknown"
                    period = detail.get("as_of") or detail.get("last_fetched")
                else:
                    status, period = detail, None
                facts.append({
                    "metric": f"freshness_{source}", "value": status,
                    "period": period, "ticker": ticker, "source_type": "freshness",
                })
    return facts


def _merge_facts(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Merge two fact lists, de-duplicating by (ticker, metric, period)."""
    merged: list[dict] = []
    seen: set = set()
    for f in list(primary) + list(secondary):
        if not isinstance(f, dict):
            continue
        key = (f.get("ticker"), f.get("metric"), f.get("period"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(f)
    return merged


def _dedupe_docs(docs: list[dict]) -> list[dict]:
    """De-duplicate documents by ANY independent identity (id OR parent+chunk)."""
    out: list[dict] = []
    seen: set = set()
    for d in docs:
        if not isinstance(d, dict):
            continue
        idents = _doc_identities(d)
        if any(i in seen for i in idents):
            continue
        seen.update(idents)
        out.append(d)
    return out


def _sq0_ids(plan: QueryPlan) -> list[str]:
    return [plan.subqueries[0].id] if plan.subqueries else ["sq0"]


def _synthetic_sq0(plan: QueryPlan) -> QuerySubquery:
    """A stand-in sq0 when a plan somehow arrives without subqueries."""
    return QuerySubquery(
        id="sq0", text=plan.retrieval_query,
        entity_tickers=tuple(plan.tickers), intents=tuple(plan.intents),
        metrics=tuple(plan.metrics), periods=tuple(plan.periods),
    )


def _as_retrieval_dict(plan: QueryPlan, result: OrchestrationResult) -> dict:
    """Legacy ``Retriever.retrieve()``-compatible view of merged evidence."""
    primary = plan.primary_entity
    return {
        "facts": result.merged_facts,
        "documents": result.merged_documents,
        "ticker": primary.ticker if primary else None,
        "strategy": result.lane.value,
        "retrieval_strategy": result.retrieval_strategy or "vector",
    }

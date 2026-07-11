"""
src/middleware/query_plan.py
Bounded multi-entity query-plan contract (2.2.3.1).

Replaces the single-ticker / single-intent parse result with an explicit,
validated plan that preserves the user's raw question and represents every
resolved entity, intent, metric, and period. The legacy ``IntentParser.parse``
dictionary remains available through :meth:`QueryPlan.to_legacy_intent` so this
can ship behind a feature flag without touching the current query path.

Text-form contract (see docs/phase2.2/ARCHITECTURE-DECISION.md "Query contract"):
- ``original_question`` is the exact request text — never normalized, rewritten,
  truncated, or replaced. Prompt display and answer generation use it.
- ``retrieval_query`` is a *separate* field: the validated standalone query
  compiled by 2.2.2, or the raw question when no conversation rewrite is needed.
  Embedding / lexical retrieval use it (and subquery text).
- ``normalized_question`` is a matching-only representation; classification may
  use it.
A rejected plan raises :class:`QueryPlanError` with machine-readable
``reason_codes`` so a caller can fall back to ``IntentParser.parse`` without ever
failing the user request. Plan construction never mutates the raw input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Retrieval modes a subquery may request (advisory hint for the 2.2.3.2 router).
RETRIEVAL_MODES = frozenset({"facts", "documents", "macro", "tools"})


class QueryPlanError(ValueError):
    """Raised by :meth:`QueryPlan.validate` when a plan violates its invariants.

    Carries ``reason_codes`` (stable, machine-readable) so a caller can log the
    exact rule that failed and fall back to the legacy parse path.
    """

    def __init__(self, reason_codes: list[str]) -> None:
        self.reason_codes = list(reason_codes)
        super().__init__(", ".join(reason_codes) or "invalid_query_plan")


@dataclass(frozen=True)
class QueryEntity:
    """One resolved entity with the mention that produced it.

    ``mention``/``start`` come from :meth:`SymbolResolver.resolve_all` and keep
    multi-entity ordering deterministic (first appearance in the retrieval
    query). An override entity has ``source="override"`` and ``start=-1``.
    """

    ticker: str
    resolved_name: Optional[str]
    confidence: float
    source: str
    mention: str
    start: int


@dataclass(frozen=True)
class QuerySubquery:
    """A bounded, request-local unit of retrieval work.

    ``id`` is a stable request-local id (``sq0``, ``sq1``, ...). ``sq0`` is
    always the validated standalone retrieval query and is never derived.
    Selective decomposition into further subqueries belongs to 2.2.4.2; this
    task only ever emits ``sq0``.
    """

    id: str
    text: str
    entity_tickers: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    retrieval_modes: tuple[str, ...] = ()
    derived: bool = False
    parent_id: Optional[str] = None


@dataclass
class QueryPlan:
    """The explicit, validated plan for one request.

    Ordered ``entities``/``intents``/``metrics``/``periods`` preserve user
    mention order and are deduplicated (never sorted away). ``primary_intent``
    and ``primary_period`` exist solely for backward-compatible legacy strategy
    selection via :meth:`to_legacy_intent`; they do not replace the ordered
    lists. ``reason_codes`` records observable rule matches.
    """

    original_question: str
    retrieval_query: str
    normalized_question: str = ""
    entities: list[QueryEntity] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    subqueries: list[QuerySubquery] = field(default_factory=list)
    primary_intent: str = "general"
    # Legacy-compat scalars: the pattern-order-first timeframe and its type, so
    # to_legacy_intent() reproduces IntentParser.parse() exactly.
    primary_period: Optional[str] = None
    primary_period_type: Optional[str] = None
    reason_codes: list[str] = field(default_factory=list)

    # ── Ordered accessors ──────────────────────────────
    @property
    def tickers(self) -> list[str]:
        """Entity tickers in first-mention order."""
        return [entity.ticker for entity in self.entities]

    @property
    def primary_entity(self) -> Optional[QueryEntity]:
        """The single most salient entity (first ordered), or ``None``."""
        return self.entities[0] if self.entities else None

    # ── Validation ─────────────────────────────────────
    def validate(self) -> "QueryPlan":
        """Enforce the plan invariants; raise :class:`QueryPlanError` on any break.

        Records the failing rule(s) in ``reason_codes`` before raising so a
        caller can observe why the plan was rejected and fall back cleanly.
        Returns ``self`` when valid to allow ``plan = QueryPlan(...).validate()``.
        """
        reasons: list[str] = []

        if not self.original_question or not self.original_question.strip():
            reasons.append("blank_original_question")
        if not self.retrieval_query or not self.retrieval_query.strip():
            reasons.append("blank_retrieval_query")

        tickers = [entity.ticker for entity in self.entities]
        # Type-guard before .upper() so a malformed (None / non-str) ticker
        # becomes a QueryPlanError, not an AttributeError that escapes the
        # fail-soft boundary (2.2.3.4 review finding 4).
        if any(not isinstance(t, str) or t != t.upper() for t in tickers):
            reasons.append("entity_not_uppercase")
        if len(tickers) != len(set(tickers)):
            reasons.append("entity_duplicate")
        # Non-override entities must appear in first-mention (start) order.
        located = [
            entity.start
            for entity in self.entities
            if entity.source != "override" and entity.start >= 0
        ]
        if located != sorted(located):
            reasons.append("entity_out_of_mention_order")

        if len(self.intents) != len(set(self.intents)):
            reasons.append("intent_duplicate")
        if len(self.metrics) != len(set(self.metrics)):
            reasons.append("metric_duplicate")
        if len(self.periods) != len(set(self.periods)):
            reasons.append("period_duplicate")

        if not (1 <= len(self.subqueries) <= 3):
            reasons.append("subquery_count_out_of_bounds")
        if self.subqueries:
            head = self.subqueries[0]
            if head.id != "sq0" or head.derived or head.text != self.retrieval_query:
                reasons.append("sq0_not_retrieval_query")

        plan_entities = set(tickers)
        plan_metrics = set(self.metrics)
        plan_periods = set(self.periods)
        for sub in self.subqueries:
            if any(mode not in RETRIEVAL_MODES for mode in sub.retrieval_modes):
                reasons.append("invalid_retrieval_mode")
            if not sub.derived:
                continue
            # Derived subqueries may narrow, never introduce new plan elements.
            if not set(sub.entity_tickers) <= plan_entities:
                reasons.append("derived_entity_drift")
            if not set(sub.metrics) <= plan_metrics:
                reasons.append("derived_metric_drift")
            if not set(sub.periods) <= plan_periods:
                reasons.append("derived_period_drift")

        if reasons:
            deduped = list(dict.fromkeys(reasons))
            self.reason_codes = list(dict.fromkeys(self.reason_codes + deduped))
            raise QueryPlanError(deduped)
        return self

    # ── Legacy adapter ─────────────────────────────────
    def to_legacy_intent(self) -> dict:
        """Project the plan onto the current ``IntentParser.parse`` dict shape.

        Uses the first ordered entity and ``primary_intent``/``primary_period``
        so feature-disabled callers keep the exact keys they read today. Does
        not mutate the plan.
        """
        primary = self.primary_entity
        return {
            "ticker": primary.ticker if primary else None,
            "metrics": list(self.metrics),
            "question_type": self.primary_intent,
            "timeframe": self.primary_period,
            "timeframe_type": self.primary_period_type,
            "original_question": self.original_question,
            "ticker_confidence": primary.confidence if primary else 0.0,
            "resolved_name": primary.resolved_name if primary else None,
            "ticker_source": primary.source if primary else "none",
        }


def normalize_question(text: str) -> str:
    """Matching-only normalization: lowercase, punctuation-to-space, collapsed."""
    lowered = (text or "").lower()
    collapsed = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()

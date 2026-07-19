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
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

# Retrieval modes a subquery may request (advisory hint for the 2.2.3.2 router).
RETRIEVAL_MODES = frozenset({"facts", "documents", "macro", "tools"})
COMPLETENESS_REQUIREMENTS = frozenset(
    {"all", "count", "top_n", "bottom_n", "existence", "sample"}
)
EVIDENCE_MODES = frozenset({"catalog", "facts", "documents"})


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
class AnswerObligations:
    """Normalized requirements that determine whether a route is complete."""

    entity_set: tuple[str, ...] = ()
    universe_scope: Optional[str] = None
    operation: Optional[str] = None
    metrics: tuple[str, ...] = ()
    item_types: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    completeness: str = "sample"
    limit: Optional[int] = None
    as_of: Optional[str] = None
    qualitative: bool = False
    evidence_modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuerySubquery:
    """A bounded, request-local unit of retrieval work.

    ``id`` is a stable request-local id (``sq0``, ``sq1``, ...). ``sq0`` is
    always the validated standalone retrieval query and is never derived.
    Selective decomposition into ``sq1``/``sq2`` belongs to 2.2.4.2
    (:func:`decompose_plan`); ``sq0`` is emitted by the parser.

    2.2.4.2 fields (all optional, defaulted, so every pre-2.2.4.2 construction
    and :meth:`QueryPlan.validate` are unaffected): ``derivation_source`` is
    ``"deterministic"`` (rule decomposition) or ``"planner"`` (2.2.2.2 planner
    proposal) — it drives the fusion weight; ``reason_code`` is the stable
    ``decompose_*`` rule that produced the subquery; ``covers_obligations`` is
    the ordered evidence-obligation slot signature the subquery is meant to
    supply.
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
    derivation_source: Optional[str] = None
    reason_code: Optional[str] = None
    covers_obligations: tuple[str, ...] = ()


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
    evidence_topic: Optional[str] = None
    evidence_filters: dict = field(default_factory=dict)
    obligations: AnswerObligations = field(default_factory=AnswerObligations)
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

        obligation_entities = list(self.obligations.entity_set)
        if any(
            not isinstance(ticker, str) or ticker != ticker.upper()
            for ticker in obligation_entities
        ):
            reasons.append("obligation_entity_not_uppercase")
        if len(obligation_entities) != len(set(obligation_entities)):
            reasons.append("obligation_entity_duplicate")
        if self.obligations.completeness not in COMPLETENESS_REQUIREMENTS:
            reasons.append("invalid_completeness_requirement")
        if any(mode not in EVIDENCE_MODES for mode in self.obligations.evidence_modes):
            reasons.append("invalid_evidence_mode")
        if self.obligations.limit is not None and self.obligations.limit < 1:
            reasons.append("invalid_obligation_limit")

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


# ── Selective query decomposition + drift validation (2.2.4.2) ──────────────
#
# Deterministic decomposition of a genuinely compound / low-coverage plan into
# at most two derived, finance-specific subqueries (``sq1``/``sq2``) that only
# ever narrow the validated parent plan — never introduce a new entity, metric,
# period, operation, threshold, or broader topic. The same
# :func:`validate_derived_subquery` / :func:`select_derived_subqueries` pair
# validates both the deterministic drafts here and the optional planner-proposed
# subqueries in ``query_rewriter``, so every derived query flows through one
# drift gate. Simple plans yield zero derived subqueries (and thus zero added
# cost); the whole plan stays capped at three subqueries including ``sq0``.

_STRUCTURED_INTENTS = frozenset({"fact_lookup", "comparison", "projection"})
_QUALITATIVE_INTENTS = frozenset({"explanation", "news", "risk", "sentiment"})
# Macro keywords that mark a request as (also) macro-economic. Kept local so
# this module stays self-contained (mirror of the retriever's MACRO_KEYWORDS).
_MACRO_TOKENS = frozenset({
    "gdp", "cpi", "inflation", "unemployment", "treasury", "interest", "rate",
    "rates", "fed", "federal", "funds", "yield", "macro", "economy", "economic",
    "recession", "payroll", "payrolls", "nonfarm", "ppi",
})
# Stopwords that cannot on their own make a derived query "non-empty".
_DERIVED_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is", "are",
    "was", "were", "be", "what", "how", "why", "about", "with", "its", "their",
    "s", "vs", "versus",
})
_DERIVED_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DERIVED_TICKERISH_RE = re.compile(r"\b[A-Z]{1,5}\b")


def _content_tokens(text: str) -> list[str]:
    """Non-stopword alphanumeric tokens of a derived query's text."""
    return [
        token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token and token not in _DERIVED_STOPWORDS
    ]


def _allowed_metric_set(plan: "QueryPlan") -> set[str]:
    """Plan metrics expanded by the grader's audited alias catalog (lowercased).

    Imported lazily so ``query_plan`` never imports ``evidence_grader`` at module
    load (which would form a cycle — ``evidence_grader`` imports this module)."""
    allowed = {str(m).strip().lower() for m in plan.metrics if str(m).strip()}
    try:
        from .evidence_grader import validated_metric_aliases
        allowed.update(validated_metric_aliases(plan.metrics))
    except Exception:  # noqa: BLE001 - alias expansion is best-effort
        pass
    return allowed


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _DERIVED_NUMBER_RE.findall(text or "")}


def _plan_number_pool(plan: "QueryPlan") -> set[str]:
    pool: set[str] = set()
    pool |= _numbers(plan.original_question)
    pool |= _numbers(plan.retrieval_query)
    for period in plan.periods:
        pool |= _numbers(str(period))
    for metric in plan.metrics:
        pool |= _numbers(str(metric))
    return pool


def _known_tickers() -> frozenset[str]:
    try:
        from .intent_parser import IntentParser
        return frozenset(IntentParser.KNOWN_TICKERS)
    except Exception:  # noqa: BLE001 - catalog is advisory for the free-text guard
        return frozenset()


def _text_introduces_foreign_ticker(text: str, allowed: set[str]) -> bool:
    """True when the derived text names a *known* ticker absent from the plan."""
    known = _known_tickers()
    if not known:
        return False
    for token in _DERIVED_TICKERISH_RE.findall(text or ""):
        upper = token.upper()
        if upper in allowed:
            continue
        if upper in known:
            return True
    return False


def _obligation_slots(subquery: "QuerySubquery") -> tuple[str, ...]:
    """Ordered, deduplicated evidence-obligation signature for a subquery."""
    slots: list[str] = []
    for mode in subquery.retrieval_modes:
        slots.append(f"mode:{mode}")
    for entity in subquery.entity_tickers:
        slots.append(f"entity:{entity}")
    for metric in subquery.metrics:
        slots.append(f"metric:{metric}")
    for period in subquery.periods:
        slots.append(f"period:{period}")
    return tuple(dict.fromkeys(slots))


def validate_derived_subquery(
    plan: "QueryPlan", subquery: "QuerySubquery"
) -> tuple[bool, list[str]]:
    """Validate one candidate derived subquery against its parent plan.

    A derived query is valid only when every entity, metric, period, operation,
    and retrieval mode comes from the validated parent plan (or the grader's
    known metric aliases), its text is non-empty / not stopword-only, and it
    introduces no new numeric threshold or date. Returns ``(is_valid, reasons)``
    where ``reasons`` are stable ``drift_*`` codes recorded in the evidence
    trace on rejection.
    """
    reasons: list[str] = []
    allowed_entities = {str(t).upper() for t in plan.tickers}
    allowed_metrics = _allowed_metric_set(plan)
    allowed_periods = {str(p) for p in plan.periods}
    allowed_intents = set(plan.intents)

    if not {str(t).upper() for t in subquery.entity_tickers} <= allowed_entities:
        reasons.append("drift_invented_entity")
    if not {str(m).strip().lower() for m in subquery.metrics if str(m).strip()} <= allowed_metrics:
        reasons.append("drift_new_metric")
    if not {str(p) for p in subquery.periods} <= allowed_periods:
        reasons.append("drift_new_period")
    if subquery.intents and not set(subquery.intents) <= allowed_intents:
        reasons.append("drift_broader_topic")
    if any(mode not in RETRIEVAL_MODES for mode in subquery.retrieval_modes):
        reasons.append("drift_invalid_mode")
    if not _content_tokens(subquery.text):
        reasons.append("drift_empty_query")
    if _numbers(subquery.text) - _plan_number_pool(plan):
        reasons.append("drift_new_number")
    if _text_introduces_foreign_ticker(subquery.text, allowed_entities):
        if "drift_invented_entity" not in reasons:
            reasons.append("drift_invented_entity")

    return (not reasons), list(dict.fromkeys(reasons))


def select_derived_subqueries(
    plan: "QueryPlan",
    candidates: Iterable["QuerySubquery"],
    *,
    source: str,
    limit: int = 2,
) -> tuple[list["QuerySubquery"], list[str]]:
    """Validate, dedupe, and cap candidate derived subqueries.

    Runs every candidate through :func:`validate_derived_subquery`, drops
    duplicated paraphrases (a candidate whose obligation slots are already fully
    covered by an accepted one adds no distinct coverage), keeps at most
    ``limit`` and re-ids the survivors ``sq1``/``sq2`` with parent ``sq0``, the
    given ``derivation_source``, and their obligation signature. Returns
    ``(accepted, drift_reason_codes)``; the reason codes are the union of every
    rejection so the caller can record them in the evidence trace.
    """
    accepted: list["QuerySubquery"] = []
    reasons: list[str] = []
    covered: set[str] = set()
    for candidate in candidates:
        ok, drift = validate_derived_subquery(plan, candidate)
        if not ok:
            reasons.extend(drift)
            continue
        slots = frozenset(_obligation_slots(candidate))
        if slots and slots <= covered:
            reasons.append("drift_duplicate_obligations")
            continue
        covered |= slots
        accepted.append(candidate)
        if len(accepted) >= max(0, limit):
            break

    finalized: list["QuerySubquery"] = []
    for index, candidate in enumerate(accepted, start=1):
        finalized.append(replace(
            candidate,
            id=f"sq{index}",
            derived=True,
            parent_id="sq0",
            derivation_source=source,
            covers_obligations=_obligation_slots(candidate),
        ))
    return finalized, list(dict.fromkeys(reasons))


def _plan_is_macro(plan: "QueryPlan") -> bool:
    tokens = set((plan.normalized_question or normalize_question(plan.retrieval_query)).split())
    return bool(tokens & _MACRO_TOKENS)


def _decomposition_signals(plan: "QueryPlan") -> set[str]:
    """Structural signals that a plan carries independently-answerable clauses."""
    intents = set(plan.intents)
    structured = bool(plan.metrics) or bool(intents & _STRUCTURED_INTENTS)
    qualitative = bool(intents & _QUALITATIVE_INTENTS)
    signals: set[str] = set()
    if len(plan.entities) >= 2:
        signals.add("multi_entity")
    if len(plan.periods) > 1:
        signals.add("multi_period")
    if structured and qualitative:
        signals.add("structured_plus_qualitative")
    if _plan_is_macro(plan) and plan.entities:
        signals.add("macro_plus_company")
    non_general = [i for i in plan.intents if i != "general"]
    if len(set(non_general)) >= 2:
        signals.add("multi_intent")
    return signals


def _draft_text(
    entities: Iterable[str],
    metrics: Iterable[str],
    periods: Iterable[str],
    focus: Iterable[str],
) -> str:
    parts: list[str] = []
    parts.extend(str(e) for e in entities)
    parts.extend(str(m).replace("_", " ") for m in metrics)
    parts.extend(str(p) for p in periods)
    parts.extend(str(f) for f in focus)
    return " ".join(part for part in parts if part).strip()


def _make_draft(
    plan: "QueryPlan",
    *,
    entities: Iterable[str] = (),
    metrics: Iterable[str] = (),
    periods: Iterable[str] = (),
    modes: Iterable[str] = (),
    reason: str,
    focus: Iterable[str] = (),
) -> "QuerySubquery":
    entities = tuple(entities)
    metrics = tuple(metrics)
    periods = tuple(periods)
    modes = tuple(modes)
    text = _draft_text(entities, metrics, periods, focus) or plan.retrieval_query
    draft = QuerySubquery(
        id="sq?", text=text, entity_tickers=entities, intents=(),
        metrics=metrics, periods=periods, retrieval_modes=modes,
        derived=True, parent_id="sq0", reason_code=reason,
    )
    return replace(draft, covers_obligations=_obligation_slots(draft))


def _qualitative_focus(plan: "QueryPlan") -> tuple[str, ...]:
    intents = set(plan.intents)
    for intent in ("risk", "news", "sentiment", "explanation"):
        if intent in intents:
            return (intent,)
    return ("context",)


def _candidate_drafts(plan: "QueryPlan") -> list["QuerySubquery"]:
    """Deterministic candidate derived subqueries for one compound plan.

    Exactly one splitting strategy is chosen per plan (most-specific first) so
    the two derived slots always describe a single coherent decomposition.
    """
    signals = _decomposition_signals(plan)
    if not signals:
        return []
    intents = set(plan.intents)
    structured = bool(plan.metrics) or bool(intents & _STRUCTURED_INTENTS)
    qualitative = bool(intents & _QUALITATIVE_INTENTS)

    if "macro_plus_company" in signals:
        company_modes = ["facts"] + (["documents"] if qualitative else [])
        return [
            _make_draft(plan, modes=["macro"], reason="decompose_macro",
                        focus=("macro",)),
            _make_draft(plan, entities=plan.tickers, metrics=plan.metrics,
                        periods=plan.periods, modes=company_modes,
                        reason="decompose_company"),
        ]

    if "multi_entity" in signals:
        entity_modes = (["facts"] if structured else []) + (
            ["documents"] if qualitative else [])
        if not entity_modes:
            entity_modes = ["facts"]
        return [
            _make_draft(plan, entities=[entity], metrics=plan.metrics,
                        periods=plan.periods, modes=entity_modes,
                        reason="decompose_entity")
            for entity in plan.tickers
        ]

    if "structured_plus_qualitative" in signals or "multi_intent" in signals:
        drafts: list["QuerySubquery"] = []
        if structured:
            drafts.append(_make_draft(
                plan, entities=plan.tickers, metrics=plan.metrics,
                periods=plan.periods, modes=["facts"], reason="decompose_structured"))
        if qualitative:
            drafts.append(_make_draft(
                plan, entities=plan.tickers, modes=["documents"],
                reason="decompose_qualitative", focus=_qualitative_focus(plan)))
        return drafts

    if "multi_period" in signals:
        return [
            _make_draft(plan, entities=plan.tickers, metrics=plan.metrics,
                        periods=[period], modes=["facts"], reason="decompose_period")
            for period in plan.periods
        ]

    return []


def decompose_plan(
    plan: "QueryPlan", missing_obligations: Optional[Iterable[str]] = None
) -> list["QuerySubquery"]:
    """Decompose a compound / low-coverage plan into ≤2 derived subqueries.

    Returns the derived subqueries (``sq1``/``sq2``) to attach to ``plan`` —
    never ``sq0`` — or ``[]`` for a single coherent lookup, one qualitative
    topic, or any plan already covered by the standard lane. ``plan`` is never
    mutated. ``missing_obligations`` (the grader's uncovered subquery/obligation
    ids) lowers the bar so a grader-requested decomposition still fires when the
    structural signals are weak; simple plans still yield zero derived
    subqueries and add no cost. The result is validated, deduplicated, and
    capped so the full plan never exceeds three subqueries.
    """
    try:
        plan.validate()
    except QueryPlanError:
        return []
    if any(subquery.derived for subquery in plan.subqueries):
        return []  # already decomposed — idempotent

    signals = _decomposition_signals(plan)
    if not signals and not missing_obligations:
        return []

    drafts = _candidate_drafts(plan)
    accepted, _reasons = select_derived_subqueries(
        plan, drafts, source="deterministic", limit=2)
    return accepted


def attach_derived_subqueries(
    plan: "QueryPlan", derived: Iterable["QuerySubquery"]
) -> "QueryPlan":
    """Return a copy of ``plan`` with ≤2 validated derived subqueries appended.

    Fails soft: if appending the derived subqueries would break plan validation
    (or none survive re-validation), the original ``plan`` is returned unchanged
    so decomposition can never turn a valid request into an invalid one.
    """
    extras = [subquery for subquery in derived if subquery.id != "sq0"][:2]
    if not extras:
        return plan
    candidate = QueryPlan(
        original_question=plan.original_question,
        retrieval_query=plan.retrieval_query,
        normalized_question=plan.normalized_question,
        entities=list(plan.entities),
        intents=list(plan.intents),
        metrics=list(plan.metrics),
        periods=list(plan.periods),
        subqueries=[plan.subqueries[0], *extras] if plan.subqueries else list(extras),
        primary_intent=plan.primary_intent,
        primary_period=plan.primary_period,
        primary_period_type=plan.primary_period_type,
        evidence_topic=plan.evidence_topic,
        evidence_filters=dict(plan.evidence_filters),
        obligations=plan.obligations,
        reason_codes=list(plan.reason_codes),
    )
    try:
        candidate.validate()
    except QueryPlanError:
        return plan
    return candidate

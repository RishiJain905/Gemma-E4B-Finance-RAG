"""
src/middleware/deterministic_router.py
Deterministic finance tool routing, whitelisted calculation, and a read-only
executor (Phase 2.2.3.2).

Given a validated :class:`~src.middleware.query_plan.QueryPlan` and the set of
metric names that actually exist in the store, :func:`route` decides — with
pure, explicit rules and no model call — whether a request can be answered by
the existing read-only finance tools. Safe analytical, comparison, projection,
and bounded-calculation requests map onto ``get_fundamentals`` / ``query_facts``
/ ``get_estimates`` / ``get_price_targets`` / ``get_guidance`` /
``get_macro_snapshot`` / ``get_sentiment`` / ``classify_trade_bias``.
Anything ambiguous, qualitative, or
write-requiring makes the router *abstain* so the caller keeps the existing
hybrid retrieval + model path.

``route`` never touches the network, the model, or storage. Side effects live
only in :func:`execute_route`, which dispatches the routed tools through the
shared guarded dispatcher (``tools.base.dispatch_named_tool``) with a read-only
:class:`~src.middleware.tools.base.ToolContext`, runs any whitelisted
:class:`CalculationSpec` with :func:`calculate` (``decimal.Decimal`` only), and
— when every plan obligation is covered — builds a short deterministic answer
that lets the caller skip model generation. Handler errors are attached to the
invocation and cause the caller to fall back to normal retrieval; they never
raise out of the executor.

The two feature flags that gate this path
(``enable_deterministic_tool_routing`` and ``enable_deterministic_answers``)
default off; 2.2.3.4 owns wiring the router into ``/query``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from .query_plan import QueryPlan
    from .tools.base import ToolContext

logger = logging.getLogger(__name__)

# ── Stable reason / abstain codes ─────────────────────────
# Route reason codes (why a tool was selected) and abstain reasons (why the
# router declined) are stable strings so 2.2.3.3/2.2.3.4 and telemetry can key
# off them without parsing prose.

REASON_FUNDAMENTALS = "route_get_fundamentals"
REASON_RANK = "route_query_facts_rank"
REASON_THRESHOLD = "route_query_facts_threshold"
REASON_COMPARE = "route_query_facts_compare"
REASON_ESTIMATES = "route_get_estimates"
REASON_PRICE_TARGETS = "route_get_price_targets"
REASON_GUIDANCE = "route_get_guidance"
REASON_MACRO = "route_get_macro_snapshot"
REASON_FRESHNESS = "route_check_freshness"
REASON_SENTIMENT = "route_get_sentiment"
REASON_COVERAGE = "route_describe_coverage"
REASON_TRADE_BIAS = "route_classify_trade_bias"

ABSTAIN_AMBIGUOUS_ENTITY = "ambiguous_entity"
ABSTAIN_AMBIGUOUS_METRIC = "ambiguous_metric"
ABSTAIN_UNKNOWN_METRIC = "unknown_metric"
ABSTAIN_AMBIGUOUS_THRESHOLD = "ambiguous_threshold"
ABSTAIN_AMBIGUOUS_SORT = "ambiguous_sort"
ABSTAIN_QUALITATIVE = "qualitative_evidence_required"
ABSTAIN_WRITE_REQUIRED = "write_tool_required"
ABSTAIN_REFRESH_REQUESTED = "refresh_requested"
ABSTAIN_UNROUTABLE = "unroutable"

# Reason codes for a matched-but-incomplete route: the safe tool ran, but the
# plan carries obligations the single dispatched tool did not cover, so the
# caller must NOT treat the route as a complete answer (2.2.3.4 review finding 2).
INCOMPLETE_METRIC_COVERAGE = "partial_metric_coverage"
INCOMPLETE_INTENT_COVERAGE = "partial_intent_coverage"
INCOMPLETE_PARTIAL_CATALOG_PAGE = "partial_catalog_page"
INCOMPLETE_MISSING_COVERAGE_DIMENSION = "missing_coverage_dimension"
INCOMPLETE_UNRESOLVED_UNIVERSE_SCOPE = "unresolved_universe_scope"
INCOMPLETE_MISSING_METRIC_COVERAGE = "missing_metric_coverage"
INCOMPLETE_QUALITATIVE_EVIDENCE_REQUIRED = "qualitative_evidence_required"
INCOMPLETE_RESULT_SET_TOO_LARGE = "result_set_too_large"

_STABLE_INCOMPLETE_CODES = frozenset({
    INCOMPLETE_PARTIAL_CATALOG_PAGE,
    INCOMPLETE_MISSING_COVERAGE_DIMENSION,
    INCOMPLETE_UNRESOLVED_UNIVERSE_SCOPE,
    INCOMPLETE_MISSING_METRIC_COVERAGE,
    INCOMPLETE_QUALITATIVE_EVIDENCE_REQUIRED,
    INCOMPLETE_RESULT_SET_TOO_LARGE,
})

_MAX_SCOPED_CATALOG_ENTITIES = 3

# Intents whose evidence lives in documents (why / risk / news / filings), plus
# trend which needs a time series the read tools do not return. When one of
# these co-occurs with a routable numeric core the route is executed but marked
# incomplete + requires_documents (the compound rule).
_DOCUMENT_INTENTS = frozenset({"explanation", "news", "risk"})
_TIMESERIES_INTENTS = frozenset({"trend"})

# Intents whose evidence a single read tool cannot supply — each needs its own
# retrieval/tool. When a secondary such intent co-occurs with the served
# numeric/projection core, the route stays incomplete so the whole obligation
# isn't answered from one tool's slice (2.2.3.4 review finding 2).
_SELF_EVIDENCE_INTENTS = frozenset(
    {"projection", "sentiment", "explanation", "news", "risk", "trend"}
)
_PROJECTION_REASONS = frozenset({REASON_ESTIMATES, REASON_PRICE_TARGETS, REASON_GUIDANCE})
_NUMERIC_REASONS = frozenset(
    {REASON_FUNDAMENTALS, REASON_RANK, REASON_THRESHOLD, REASON_COMPARE}
)

# The single write tool must never be reachable from the router.
_WRITE_TOOLS = frozenset({"refresh_data"})

# Ranking direction cues. "best"/"worst" are intentionally excluded — their
# direction depends on the metric (a low P/E is "best"), so they signal a rank
# intent without a safe direction and force an abstain.
_RANK_ASC = frozenset({
    "lowest", "cheapest", "smallest", "least", "fewest", "bottom",
    "min", "minimum", "smallest-valued",
})
_RANK_DESC = frozenset({
    "highest", "largest", "biggest", "greatest", "most", "top",
    "max", "maximum", "priciest",
})
_RANK_AMBIGUOUS = frozenset({"best", "worst", "rank", "ranked", "ranking", "order"})

# Threshold operator cues mapped to query_facts `op` values.
_THRESHOLD_CUES: tuple[tuple[str, str], ...] = (
    ("at least", "gte"),
    ("no less than", "gte"),
    ("no more than", "lte"),
    ("at most", "lte"),
    ("greater than or equal", "gte"),
    ("less than or equal", "lte"),
    ("greater than", "gt"),
    ("more than", "gt"),
    ("less than", "lt"),
    ("higher than", "gt"),
    ("lower than", "lt"),
    ("above", "gt"),
    ("below", "lt"),
    ("over", "gt"),
    ("under", "lt"),
    ("exceeds", "gt"),
    (">=", "gte"),
    ("<=", "lte"),
    (">", "gt"),
    ("<", "lt"),
)

# Macro series answerable without a company entity (mirror of the intent
# parser's advisory hint, kept local so route() stays self-contained).
_MACRO_RE = re.compile(
    r"\b(gdp|cpi|inflation|unemployment|treasury|interest rate|fed funds"
    r"|federal funds|yield curve|ppi|nonfarm|payroll|jobs report"
    r"|consumer confidence|retail sales)\b",
    re.IGNORECASE,
)

# Wording that explicitly requests fresh/re-ingested data. The deterministic
# route only ever reads cached data, so a request to refresh/update MUST NOT be
# answered from cache and presented as current (2.2.3.4 review finding 1): the
# router abstains and the caller runs the normal freshness-aware retrieval +
# model path. Bare "latest"/"current" is intentionally NOT matched (it is common
# in legitimate latest-fact lookups); only an explicit refresh verb, an
# update/refetch/reload imperative, or a "<fetch verb> ... latest/fresh" phrase.
_REFRESH_RE = re.compile(
    r"\b(?:refresh|re-?fetch|refetch|reload|re-?load|update[sd]?)\b"
    r"|\b(?:fetch|pull|grab|download|re-?pull)\s+(?:the\s+|me\s+)?"
    r"(?:latest|newest|current|fresh|live|updated|new)\b",
    re.IGNORECASE,
)

_PRICE_TARGET_RE = re.compile(r"price target", re.IGNORECASE)
_GUIDANCE_RE = re.compile(r"\bguidance\b", re.IGNORECASE)
_QUARTER_HORIZON_RE = re.compile(r"\bnext quarter\b|\bthis quarter\b", re.IGNORECASE)
_YEAR_HORIZON_RE = re.compile(r"\bnext year\b|\bthis year\b|\bfull year\b", re.IGNORECASE)
_DAYS_RE = re.compile(r"\b(\d{1,3})\s*(?:day|days)\b", re.IGNORECASE)
# A bare number (optionally %, optional decimals). Magnitude suffixes such as
# "b"/"m"/"billion" are deliberately NOT matched so a scale-ambiguous threshold
# ("above 20B") abstains instead of guessing a unit.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LIMIT_NEAR_RANK_RE = re.compile(r"\b(?:top|bottom)\s+(\d{1,3})\b", re.IGNORECASE)

_DEFAULT_RANK_LIMIT = 5
_DEFAULT_SENTIMENT_DAYS = 7

_COVERAGE_SIGNAL_RE = re.compile(
    r"\b(?:cover(?:age|ed)?|know about|answer questions about|available "
    r"sources?|sources?|data types?|item types?|metric families?|all tickers?|"
    r"every ticker|which companies|which securities)\b",
    re.IGNORECASE,
)

_FRESHNESS_STATUS_RE = re.compile(
    r"\b(?:freshness|data freshness|source status|sources? stale|sources? fresh|"
    r"how (?:fresh|current|recent)|when (?:was|were).*(?:updated|fetched))\b",
    re.IGNORECASE,
)
_TRADE_BIAS_RE = re.compile(
    r"\b(?:long or short|short or long|long vs\.? short|short vs\.? long|"
    r"long/short|buy or sell|sell or buy|overweight or underweight|"
    r"trade bias|directional (?:call|bias)|should I (?:buy|sell|short|long)|"
    r"bullish or bearish)\b",
    re.IGNORECASE,
)


def is_trade_bias_question(text: str) -> bool:
    """True when the ask requires classify_trade_bias (long vs short)."""
    return bool(_TRADE_BIAS_RE.search(text or ""))


_ALL_TICKERS_RE = re.compile(
    r"\b(?:all|every)\s+(?:active\s+)?tickers?\b|\bwhat tickers\b|"
    r"\bwhich companies can you answer questions about\b",
    re.IGNORECASE,
)
_COVERAGE_SOURCE_RE = re.compile(r"\b(?:source|sources|data types?)\b", re.IGNORECASE)
_COVERAGE_ITEM_TYPE_RE = re.compile(r"\bitem types?\b", re.IGNORECASE)
_COVERAGE_METRIC_RE = re.compile(r"\b(?:metrics?|metric families?)\b", re.IGNORECASE)
_COVERAGE_CONTAINS_RE = re.compile(
    r"\b(?:do you cover|is .* covered|covered .*|coverage for)\b",
    re.IGNORECASE,
)


# ── Route data structures ─────────────────────────────────


@dataclass(frozen=True)
class ToolInvocation:
    """One resolved, validated tool call the executor should run.

    ``arguments`` are already shaped for the target tool's JSON schema (the
    guarded dispatcher re-validates them). ``subquery_id`` ties the invocation
    back to the plan subquery it serves; ``reason_code`` is a stable
    ``route_*`` string explaining why this tool was chosen.
    """

    name: str
    arguments: dict
    subquery_id: str
    reason_code: str


@dataclass(frozen=True)
class OperandRef:
    """A pointer from a calculation operand to a field in a tool result.

    Structured (never an expression string) so operand resolution can never
    evaluate arbitrary code. ``invocation_index`` selects a tool result;
    ``selector`` picks a row (a ticker for ``query_facts`` rows, a metric key
    for a fundamentals map); ``field`` is the value field to read.
    """

    invocation_index: int
    selector: str
    field: str = "value"


@dataclass(frozen=True)
class CalculationSpec:
    """A whitelisted derived-number request over tool results.

    ``operation`` is one of the five supported operations; ``operands`` maps
    calculation-local operand names (e.g. ``"a"``/``"b"``) to :class:`OperandRef`
    pointers into the tool results; ``display_unit`` and ``precision`` control
    the rendered result. Executed by :func:`calculate` with ``Decimal`` values.
    """

    operation: str
    operands: dict[str, OperandRef]
    display_unit: Optional[str] = None
    precision: int = 6
    reason_code: str = ""


@dataclass
class RouteDecision:
    """The router's verdict for one request.

    ``matched`` is whether any deterministic tool route applied. ``complete``
    is whether the routed evidence answers the *whole* request (no model
    generation needed). ``requires_documents`` flags that document/qualitative
    retrieval is still needed for the uncovered part. ``abstain_reason`` is set
    (and ``matched`` False) when the router declined.
    """

    matched: bool = False
    complete: bool = False
    requires_documents: bool = False
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    calculations: list[CalculationSpec] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    abstain_reason: Optional[str] = None


# ── Signal detection (pure text helpers) ──────────────────


@dataclass(frozen=True)
class _RankSignal:
    order: Optional[str]  # "asc" | "desc" | None (ambiguous)
    limit: int
    ambiguous: bool


def _detect_rank(text: str, tokens: set[str]) -> Optional[_RankSignal]:
    """Detect a ranking request and its direction, or ``None`` if not a rank.

    Returns a signal with ``ambiguous=True`` when a rank cue is present but no
    safe direction can be inferred (e.g. "best"), so the router abstains rather
    than guessing which way to sort.
    """
    asc = bool(tokens & _RANK_ASC)
    desc = bool(tokens & _RANK_DESC)
    ambiguous_cue = bool(tokens & _RANK_AMBIGUOUS)

    if not (asc or desc or ambiguous_cue):
        return None

    limit_match = _LIMIT_NEAR_RANK_RE.search(text)
    limit = int(limit_match.group(1)) if limit_match else _DEFAULT_RANK_LIMIT
    limit = max(1, min(limit, 100))

    if asc and not desc:
        return _RankSignal(order="asc", limit=limit, ambiguous=False)
    if desc and not asc:
        return _RankSignal(order="desc", limit=limit, ambiguous=False)
    # Conflicting or only-ambiguous cue → no safe direction.
    return _RankSignal(order=None, limit=limit, ambiguous=True)


@dataclass(frozen=True)
class _ThresholdSignal:
    op: Optional[str]
    value: Optional[float]
    ambiguous: bool


def _detect_threshold(text: str) -> Optional[_ThresholdSignal]:
    """Detect a numeric threshold filter (op + value), or ``None`` if absent.

    ``ambiguous=True`` when a threshold cue is present but no parseable numeric
    value follows it (or the number carries a scale suffix we will not guess),
    so the router abstains.
    """
    lowered = text.lower()
    for cue, op in _THRESHOLD_CUES:
        idx = lowered.find(cue)
        if idx == -1:
            continue
        tail = text[idx + len(cue):]
        match = _NUMBER_RE.search(tail)
        if not match:
            return _ThresholdSignal(op=None, value=None, ambiguous=True)
        # Reject a magnitude suffix immediately after the number (scale unknown).
        suffix = tail[match.end():match.end() + 12].lower().lstrip()
        if suffix[:1] in {"b", "m", "t", "k"} or suffix.startswith(
            ("billion", "million", "trillion", "thousand")
        ):
            return _ThresholdSignal(op=None, value=None, ambiguous=True)
        try:
            return _ThresholdSignal(op=op, value=float(match.group(0)), ambiguous=False)
        except ValueError:
            return _ThresholdSignal(op=None, value=None, ambiguous=True)
    return None


def _horizon(text: str) -> Optional[str]:
    if _QUARTER_HORIZON_RE.search(text):
        return "quarter"
    if _YEAR_HORIZON_RE.search(text):
        return "year"
    return None


def _sentiment_days(text: str) -> int:
    match = _DAYS_RE.search(text)
    if not match:
        return _DEFAULT_SENTIMENT_DAYS
    return max(1, min(int(match.group(1)), 90))


# ── The router ────────────────────────────────────────────


def _abstain(reason: str, reason_codes: list[str]) -> RouteDecision:
    return RouteDecision(
        matched=False,
        complete=False,
        requires_documents=reason == ABSTAIN_QUALITATIVE,
        reason_codes=reason_codes,
        abstain_reason=reason,
    )


def _coverage_filters(text: str) -> dict:
    """Extract only explicit, lossless coverage filters from a safe question."""
    lowered = text.lower()
    filters: dict[str, str] = {}
    if re.search(r"\bs\s*&\s*p\s*500\b|\bsp500\b|\bs&p500\b", lowered):
        filters["index"] = "sp500"
    elif re.search(r"\bnasdaq\s*100\b|\bnasdaq100\b", lowered):
        filters["index"] = "nasdaq100"

    sector_aliases = (
        ("health care", "Health Care"),
        ("healthcare", "Health Care"),
        ("technology", "Information Technology"),
        ("information technology", "Information Technology"),
        ("automotive", "Automotive"),
        ("industrials", "Industrials"),
    )
    for alias, sector in sector_aliases:
        if alias in lowered:
            filters["sector"] = sector
            break
    for tier in ("broad", "deep", "sector"):
        if re.search(rf"\b{tier}\s+coverage\b|\b{tier}\s+securities\b", lowered):
            filters["coverage_tier"] = tier
            break
    if re.search(r"\bsemiconductors?\b", lowered):
        filters["industry"] = "Semiconductors"
    if re.search(r"\bsec\b|\bsec filings?\b|\b10-[kq]\b", lowered):
        filters["source_category"] = "sec_filing"
    if re.search(r"\bnews\b", lowered):
        filters["item_type"] = "news"
    elif re.search(r"\btranscripts?\b", lowered):
        filters["item_type"] = "transcript"
    elif re.search(r"\bmarket data\b|\bmarket bars?\b", lowered):
        filters["item_type"] = "market_bar"
    return filters


def _coverage_invocation(plan: "QueryPlan") -> Optional[ToolInvocation]:
    """Map an explicit capability question to the read-only coverage tool."""
    text = plan.retrieval_query or plan.original_question or ""
    obligations = plan.obligations
    if (
        "catalog" not in obligations.evidence_modes
        and not _COVERAGE_SIGNAL_RE.search(text)
    ):
        return None
    # A referential follow-up without a resolved prior set must not silently
    # broaden into the global catalog. A summary is safe supporting evidence;
    # the unresolved-universe reason below prevents it becoming a final answer.
    if obligations.universe_scope == "unresolved":
        subquery_id = plan.subqueries[0].id if plan.subqueries else "sq0"
        return ToolInvocation(
            "describe_coverage", {"operation": "summary"}, subquery_id, REASON_COVERAGE
        )
    lowered = text.lower()
    ticker = plan.tickers[0] if len(plan.tickers) == 1 else None
    filters = _coverage_filters(text)
    subquery_id = plan.subqueries[0].id if plan.subqueries else "sq0"

    if obligations.operation == "summary":
        return ToolInvocation(
            "describe_coverage", {"operation": "summary"}, subquery_id, REASON_COVERAGE
        )
    if obligations.operation == "list_sources":
        args = {"operation": "list_sources"}
        if filters:
            args["filters"] = filters
        return ToolInvocation("describe_coverage", args, subquery_id, REASON_COVERAGE)
    if obligations.operation == "list_metrics":
        args = {"operation": "list_metrics"}
        if ticker:
            args["ticker"] = ticker
        if filters:
            args["filters"] = filters
        return ToolInvocation("describe_coverage", args, subquery_id, REASON_COVERAGE)
    if obligations.operation == "contains_security" and ticker:
        return ToolInvocation(
            "describe_coverage",
            {"operation": "contains_security", "ticker": ticker},
            subquery_id,
            REASON_COVERAGE,
        )
    if _ALL_TICKERS_RE.search(text) or obligations.operation in {"list_securities", "compare"}:
        args: dict[str, Any] = {
            "operation": "list_securities",
            "ticker_only": obligations.completeness in {"all", "count"},
        }
        if filters:
            args["filters"] = filters
        return ToolInvocation(
            "describe_coverage",
            args,
            subquery_id,
            REASON_COVERAGE,
        )
    if ticker and _COVERAGE_CONTAINS_RE.search(text):
        return ToolInvocation(
            "describe_coverage",
            {"operation": "contains_security", "ticker": ticker},
            subquery_id,
            REASON_COVERAGE,
        )
    if ticker and _COVERAGE_SOURCE_RE.search(text):
        return ToolInvocation(
            "describe_coverage",
            {"operation": "security_sources", "ticker": ticker},
            subquery_id,
            REASON_COVERAGE,
        )
    if _COVERAGE_ITEM_TYPE_RE.search(text):
        args = {"operation": "list_item_types"}
        if ticker:
            args["ticker"] = ticker
        if filters:
            args["filters"] = filters
        return ToolInvocation("describe_coverage", args, subquery_id, REASON_COVERAGE)
    if _COVERAGE_METRIC_RE.search(text):
        args = {"operation": "list_metrics"}
        if ticker:
            args["ticker"] = ticker
        if filters:
            args["filters"] = filters
        return ToolInvocation("describe_coverage", args, subquery_id, REASON_COVERAGE)
    if _COVERAGE_SOURCE_RE.search(text):
        args = {"operation": "list_sources"}
        if filters:
            args["filters"] = filters
        return ToolInvocation("describe_coverage", args, subquery_id, REASON_COVERAGE)
    if filters or "securities" in lowered or "companies" in lowered:
        args = {"operation": "list_securities"}
        if filters:
            args["filters"] = filters
        return ToolInvocation("describe_coverage", args, subquery_id, REASON_COVERAGE)
    return ToolInvocation(
        "describe_coverage", {"operation": "summary"}, subquery_id, REASON_COVERAGE
    )


def _scoped_coverage_invocations(
    plan: "QueryPlan", fallback: ToolInvocation
) -> tuple[list[ToolInvocation], list[str]]:
    """Compile a bounded prior/explicit entity set into exact catalog checks."""
    obligations = plan.obligations
    entities = list(obligations.entity_set)
    dimensions_requested = bool(obligations.item_types or obligations.sources)
    if obligations.universe_scope != "explicit_entities" or not dimensions_requested:
        return [fallback], []
    if not entities:
        return [fallback], [INCOMPLETE_UNRESOLVED_UNIVERSE_SCOPE]
    if len(entities) > _MAX_SCOPED_CATALOG_ENTITIES:
        summary = ToolInvocation(
            "describe_coverage",
            {"operation": "summary"},
            fallback.subquery_id,
            REASON_COVERAGE,
        )
        return [summary], [INCOMPLETE_RESULT_SET_TOO_LARGE]

    filters = _coverage_filters(plan.retrieval_query or plan.original_question or "")
    invocations = []
    for ticker in entities:
        args: dict[str, Any] = {"operation": "security_sources", "ticker": ticker}
        if filters:
            args["filters"] = filters
        invocations.append(
            ToolInvocation(
                "describe_coverage",
                args,
                fallback.subquery_id,
                REASON_COVERAGE,
            )
        )
    return invocations, []


def route(plan: "QueryPlan", available_metrics: Iterable[str]) -> RouteDecision:
    """Deterministically route a query plan onto safe read-only finance tools.

    Pure function: inspects only ``plan`` and the known metric names. Returns a
    :class:`RouteDecision` describing the ordered tool invocations (and optional
    calculations) to run, or an abstention with a stable ``abstain_reason`` when
    the request is ambiguous, qualitative-only, or would require a write tool.
    """
    known = frozenset(available_metrics or ())
    text = (plan.retrieval_query or plan.original_question or "")
    tokens = set(re.findall(r"[a-z]+", text.lower()))

    entities = list(plan.tickers)
    metrics = list(plan.metrics)
    intents = set(plan.intents)
    primary = plan.primary_intent

    reason_codes: list[str] = []

    # Guard: an explicit refresh/update request must never be answered from the
    # read-only cache and presented as current — abstain so the caller takes the
    # freshness-aware retrieval + model path (2.2.3.4 review finding 1).
    if _REFRESH_RE.search(text):
        return _abstain(ABSTAIN_REFRESH_REQUESTED, [ABSTAIN_REFRESH_REQUESTED])

    # Guard: a requested metric that does not exist → abstain, no tool call.
    # A freshness/source-status lookup is read-only and reports the cached
    # status itself. It is distinct from an imperative refresh request above.
    if _FRESHNESS_STATUS_RE.search(text):
        if len(entities) != 1:
            return _abstain(ABSTAIN_AMBIGUOUS_ENTITY, [ABSTAIN_AMBIGUOUS_ENTITY])
        subquery_id = plan.subqueries[0].id if plan.subqueries else "sq0"
        return RouteDecision(
            matched=True,
            complete=True,
            tool_invocations=[ToolInvocation(
                "check_freshness",
                {"ticker": entities[0]},
                subquery_id,
                REASON_FRESHNESS,
            )],
            reason_codes=[REASON_FRESHNESS],
        )

    if is_trade_bias_question(text):
        if len(entities) != 1:
            return _abstain(ABSTAIN_AMBIGUOUS_ENTITY, [ABSTAIN_AMBIGUOUS_ENTITY])
        subquery_id = plan.subqueries[0].id if plan.subqueries else "sq0"
        return RouteDecision(
            matched=True,
            complete=True,
            tool_invocations=[ToolInvocation(
                "classify_trade_bias",
                {"ticker": entities[0]},
                subquery_id,
                REASON_TRADE_BIAS,
            )],
            reason_codes=[REASON_TRADE_BIAS],
        )

    coverage = _coverage_invocation(plan)
    if coverage is not None:
        incomplete: list[str] = []
        coverage_invocations, scoped_incomplete = _scoped_coverage_invocations(
            plan, coverage
        )
        incomplete.extend(scoped_incomplete)
        modes = set(plan.obligations.evidence_modes)
        if plan.obligations.universe_scope == "unresolved":
            incomplete.append(INCOMPLETE_UNRESOLVED_UNIVERSE_SCOPE)
        if "facts" in modes:
            incomplete.append(INCOMPLETE_MISSING_METRIC_COVERAGE)
        if "documents" in modes or plan.obligations.qualitative:
            incomplete.append(INCOMPLETE_QUALITATIVE_EVIDENCE_REQUIRED)
        if plan.obligations.item_types and not (
            coverage_invocations[0].arguments.get("filters", {}).get("item_type")
            or coverage_invocations[0].arguments.get("operation") == "security_sources"
        ):
            incomplete.append(INCOMPLETE_MISSING_COVERAGE_DIMENSION)
        return RouteDecision(
            matched=True,
            complete=not incomplete,
            requires_documents="documents" in modes,
            tool_invocations=coverage_invocations,
            reason_codes=[REASON_COVERAGE, *incomplete],
        )

    unknown = [m for m in metrics if m not in known]
    if unknown:
        return _abstain(
            ABSTAIN_UNKNOWN_METRIC,
            [ABSTAIN_UNKNOWN_METRIC, f"metric:{unknown[0]}"],
        )
    known_metrics = metrics  # all present in `known` at this point

    document_intents = intents & _DOCUMENT_INTENTS
    timeseries_intents = intents & _TIMESERIES_INTENTS
    macro_request = (not entities) and bool(_MACRO_RE.search(text))

    rank = _detect_rank(text, tokens)
    threshold = _detect_threshold(text)

    invocation: Optional[ToolInvocation] = None
    calculations: list[CalculationSpec] = []
    subquery_id = plan.subqueries[0].id if plan.subqueries else "sq0"

    is_projection = primary == "projection" or "projection" in intents
    is_sentiment = primary == "sentiment"

    # 1. Projection: sourced consensus only (estimates / price targets /
    #    guidance). Requires exactly one entity — the projection tools are
    #    single-ticker and multi-entity projection is not in the routing table.
    if is_projection:
        if len(entities) != 1:
            return _abstain(ABSTAIN_AMBIGUOUS_ENTITY, [ABSTAIN_AMBIGUOUS_ENTITY])
        ticker = entities[0]
        if _PRICE_TARGET_RE.search(text) or "price_target_mean" in known_metrics:
            invocation = ToolInvocation(
                "get_price_targets", {"ticker": ticker}, subquery_id, REASON_PRICE_TARGETS
            )
        elif _GUIDANCE_RE.search(text):
            invocation = ToolInvocation(
                "get_guidance", {"ticker": ticker}, subquery_id, REASON_GUIDANCE
            )
        else:
            args: dict[str, Any] = {"ticker": ticker}
            horizon = _horizon(text)
            if horizon:
                args["horizon"] = horizon
            invocation = ToolInvocation(
                "get_estimates", args, subquery_id, REASON_ESTIMATES
            )

    # 2. Macro snapshot: no company entity, names a macro series.
    elif macro_request:
        invocation = ToolInvocation(
            "get_macro_snapshot", {}, subquery_id, REASON_MACRO
        )

    # 3. Sentiment summary for one ticker.
    elif is_sentiment:
        if len(entities) != 1:
            return _abstain(ABSTAIN_AMBIGUOUS_ENTITY, [ABSTAIN_AMBIGUOUS_ENTITY])
        invocation = ToolInvocation(
            "get_sentiment",
            {"ticker": entities[0], "days": _sentiment_days(text)},
            subquery_id,
            REASON_SENTIMENT,
        )

    # 4. Rank / threshold over a single known metric via query_facts.
    elif rank is not None or threshold is not None:
        if rank is not None and rank.ambiguous:
            return _abstain(ABSTAIN_AMBIGUOUS_SORT, [ABSTAIN_AMBIGUOUS_SORT])
        if threshold is not None and threshold.ambiguous:
            return _abstain(ABSTAIN_AMBIGUOUS_THRESHOLD, [ABSTAIN_AMBIGUOUS_THRESHOLD])
        if len(known_metrics) != 1:
            return _abstain(ABSTAIN_AMBIGUOUS_METRIC, [ABSTAIN_AMBIGUOUS_METRIC])
        args = {"metric": known_metrics[0], "latest_only": True}
        codes: list[str] = []
        if rank is not None:
            args["order"] = rank.order
            args["limit"] = rank.limit
            codes.append(REASON_RANK)
        if threshold is not None:
            args["op"] = threshold.op
            args["value"] = threshold.value
            codes.append(REASON_THRESHOLD)
        if entities:
            args["tickers"] = list(entities)
        invocation = ToolInvocation(
            "query_facts", args, subquery_id, codes[0]
        )
        reason_codes.extend(codes)

    # 5. Compare tickers on exactly one known metric (ordered) via query_facts.
    elif len(entities) >= 2:
        if len(known_metrics) != 1:
            return _abstain(ABSTAIN_AMBIGUOUS_METRIC, [ABSTAIN_AMBIGUOUS_METRIC])
        invocation = ToolInvocation(
            "query_facts",
            {
                "metric": known_metrics[0],
                "tickers": list(entities),
                "latest_only": True,
                "limit": max(len(entities), 10),
            },
            subquery_id,
            REASON_COMPARE,
        )
        # A "difference between A and B" comparison also yields a derived
        # number; record it as provenance-ready calculation over the two rows.
        if "difference" in tokens and len(entities) == 2:
            calculations.append(
                CalculationSpec(
                    operation="difference",
                    operands={
                        "a": OperandRef(0, entities[0], "value"),
                        "b": OperandRef(0, entities[1], "value"),
                    },
                    display_unit=None,
                    precision=6,
                    reason_code="calc_difference",
                )
            )

    # 6. Single-ticker fundamentals lookup for one or more known metrics.
    elif len(entities) == 1 and known_metrics:
        invocation = ToolInvocation(
            "get_fundamentals",
            {"ticker": entities[0], "metrics": list(known_metrics)},
            subquery_id,
            REASON_FUNDAMENTALS,
        )

    # No safe numeric route applied.
    if invocation is None:
        if document_intents or timeseries_intents:
            return _abstain(ABSTAIN_QUALITATIVE, [ABSTAIN_QUALITATIVE])
        return _abstain(ABSTAIN_UNROUTABLE, [ABSTAIN_UNROUTABLE])

    reason_codes.insert(0, invocation.reason_code)

    # Full plan-obligation coverage (review finding 2): a route is complete only
    # when the single dispatched tool covers every requested metric AND no
    # secondary self-evidence intent (a second projection/sentiment/qualitative
    # obligation) is left unserved. Any gap keeps the route matched-but-incomplete
    # so the caller retrieves the rest instead of presenting a partial slice as
    # the whole answer. This subsumes the original document/time-series rule.
    covered = _covered_metrics(invocation, known_metrics)
    uncovered_metrics = [m for m in known_metrics if m not in covered]
    served = _served_intents(invocation.reason_code)
    uncovered_intents = sorted((intents & _SELF_EVIDENCE_INTENTS) - served)

    needs_documents = (
        bool(document_intents)
        or bool(timeseries_intents)
        or len(plan.periods) > 1
        or bool(uncovered_metrics)
        or bool(uncovered_intents)
    )
    if document_intents:
        reason_codes.append("compound_requires_documents")
    if timeseries_intents or len(plan.periods) > 1:
        reason_codes.append("timeseries_incomplete")
    if uncovered_metrics:
        reason_codes.append(INCOMPLETE_METRIC_COVERAGE)
    if uncovered_intents:
        reason_codes.append(INCOMPLETE_INTENT_COVERAGE)

    return RouteDecision(
        matched=True,
        complete=not needs_documents,
        requires_documents=needs_documents,
        tool_invocations=[invocation],
        calculations=calculations,
        reason_codes=reason_codes,
        abstain_reason=None,
    )


def _covered_metrics(invocation: ToolInvocation, requested: list[str]) -> set:
    """The requested metrics the single dispatched tool actually returns.

    ``get_fundamentals`` covers every metric in its ``metrics`` arg; ``query_facts``
    covers its single ``metric``; projection tools cover only their own metric
    family (price-target / estimate metrics), so a realized valuation metric asked
    for alongside a price target is left uncovered.
    """
    args = invocation.arguments
    if "metrics" in args:
        return set(args.get("metrics") or [])
    if args.get("metric") is not None:
        return {args["metric"]}
    if invocation.reason_code == REASON_PRICE_TARGETS:
        return {m for m in requested if "price_target" in m}
    if invocation.reason_code == REASON_ESTIMATES:
        return {m for m in requested if m.startswith("estimate")}
    return set()


def _served_intents(reason_code: str) -> frozenset:
    """The self-evidence intent(s) the dispatched tool satisfies."""
    if reason_code in _PROJECTION_REASONS:
        return frozenset({"projection"})
    if reason_code == REASON_SENTIMENT:
        return frozenset({"sentiment"})
    if reason_code == REASON_MACRO:
        return frozenset()
    return frozenset({"fact_lookup", "comparison"})


# ── Whitelisted calculator ────────────────────────────────

_BINARY_OPERANDS: dict[str, tuple[str, str]] = {
    "difference": ("a", "b"),
    "spread": ("a", "b"),
    "ratio": ("numerator", "denominator"),
    "percent_change": ("old", "new"),
}
# Operations whose operands must share a unit. ``ratio`` is included (2.2.3.4
# review finding 3): a ratio across incompatible units (e.g. usd/eur) is
# meaningless and must error rather than silently return a bare number; a
# same-unit ratio (P/E, debt/equity) is dimensionless and valid.
_UNIT_EQUAL_OPS = frozenset({"difference", "spread", "percent_change", "ratio"})
# Operations whose operands must share a reporting period. ``percent_change`` is
# deliberately excluded — comparing an old period to a new one is its purpose —
# and ``ratio`` is excluded (a cross-period ratio can be intentional). A
# ``difference``/``spread`` across mismatched periods (NVDA Q2 vs AMD Q1) is a
# silent apples-to-oranges error and must be rejected (review finding 3).
_PERIOD_EQUAL_OPS = frozenset({"difference", "spread"})
_SUPPORTED_OPERATIONS = frozenset(set(_BINARY_OPERANDS) | {"rank"})


def _calc_error(code: str, message: str, operation: str, operands: Any) -> dict:
    return {
        "error": code,
        "message": message,
        "operation": operation,
        "operands": operands,
    }


def _to_decimal(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _round(value: Decimal, precision: int) -> Decimal:
    quantum = Decimal(1).scaleb(-max(0, int(precision)))
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def calculate(
    operation: str,
    operands: dict,
    *,
    display_unit: Optional[str] = None,
    precision: int = 6,
) -> dict:
    """Evaluate one whitelisted finance calculation over ``Decimal`` operands.

    ``operands`` maps operand names to ``{"value", "unit", "period"}`` records.
    Only the five whitelisted operations are supported; there is no expression
    parsing and :func:`eval` is never called. Divide-by-zero, missing units,
    incompatible units, and missing periods return a structured
    ``{"error": <code>, ...}`` dict instead of raising. On success the returned
    dict preserves every operand (value/unit/period), the ``formula`` string,
    the result ``unit``, and the ``Decimal`` ``result`` so 2.2.4.3 can build
    provenance for the derived number.
    """
    if operation not in _SUPPORTED_OPERATIONS:
        return _calc_error(
            "unsupported_operation",
            f"operation must be one of {sorted(_SUPPORTED_OPERATIONS)}",
            operation,
            operands,
        )

    if operation == "rank":
        return _calculate_rank(operands, display_unit, precision)

    required = _BINARY_OPERANDS[operation]
    resolved: dict[str, dict] = {}
    for name in required:
        record = operands.get(name)
        if not isinstance(record, dict) or record.get("value") is None:
            return _calc_error(
                "missing_operands",
                f"operand '{name}' is required with a numeric value",
                operation,
                operands,
            )
        dec = _to_decimal(record.get("value"))
        if dec is None:
            return _calc_error(
                "missing_operands",
                f"operand '{name}' value is not numeric",
                operation,
                operands,
            )
        if not record.get("period"):
            return _calc_error(
                "missing_periods",
                f"operand '{name}' has no reporting period",
                operation,
                operands,
            )
        unit = record.get("unit")
        if not unit:
            return _calc_error(
                "missing_units",
                f"operand '{name}' has no unit",
                operation,
                operands,
            )
        resolved[name] = {"value": dec, "unit": str(unit), "period": record.get("period")}

    if operation in _UNIT_EQUAL_OPS:
        units = {resolved[name]["unit"].lower() for name in required}
        if len(units) != 1:
            return _calc_error(
                "incompatible_units",
                f"operands must share a unit for {operation}: {sorted(units)}",
                operation,
                operands,
            )

    if operation in _PERIOD_EQUAL_OPS:
        periods = {str(resolved[name]["period"]) for name in required}
        if len(periods) != 1:
            return _calc_error(
                "mismatched_periods",
                f"operands must share a reporting period for {operation}: "
                f"{sorted(periods)}",
                operation,
                operands,
            )

    formula, result, unit = _apply_binary(operation, required, resolved, display_unit)
    if isinstance(result, dict):  # divide-by-zero surfaced as structured error
        return result

    preserved = {
        name: {
            "value": resolved[name]["value"],
            "unit": resolved[name]["unit"],
            "period": resolved[name]["period"],
        }
        for name in required
    }
    return {
        "operation": operation,
        "operands": preserved,
        "unit": unit,
        "precision": precision,
        "formula": formula,
        "result": _round(result, precision),
    }


def _apply_binary(operation, required, resolved, display_unit):
    a_name, b_name = required
    a = resolved[a_name]["value"]
    b = resolved[b_name]["value"]

    if operation == "difference":
        return "a - b", a - b, resolved[a_name]["unit"]
    if operation == "spread":
        return "a - b", a - b, resolved[a_name]["unit"]
    if operation == "ratio":
        if b == 0:
            return "", _calc_error(
                "divide_by_zero", "denominator is zero", operation, resolved
            ), None
        unit = display_unit or "ratio"
        return "numerator / denominator", a / b, unit
    if operation == "percent_change":
        if a == 0:
            return "", _calc_error(
                "divide_by_zero", "baseline (old) is zero", operation, resolved
            ), None
        return "(new - old) / old * 100", (b - a) / a * Decimal(100), "percent"
    # Unreachable: guarded by _SUPPORTED_OPERATIONS.
    return "", _calc_error(
        "unsupported_operation", operation, operation, resolved
    ), None


def _calculate_rank(operands, display_unit, precision):
    """Rank named operands by value; provenance-only ordering artifact."""
    entries = []
    for name, record in operands.items():
        if not isinstance(record, dict) or record.get("value") is None:
            return _calc_error(
                "missing_operands",
                f"operand '{name}' is required with a numeric value",
                "rank",
                operands,
            )
        dec = _to_decimal(record.get("value"))
        if dec is None:
            return _calc_error(
                "missing_operands",
                f"operand '{name}' value is not numeric",
                "rank",
                operands,
            )
        entries.append({
            "name": name,
            "value": dec,
            "unit": record.get("unit"),
            "period": record.get("period"),
        })

    descending = (display_unit or "desc").lower() != "asc"
    entries.sort(key=lambda e: e["value"], reverse=descending)
    for position, entry in enumerate(entries, start=1):
        entry["rank"] = position
    return {
        "operation": "rank",
        "operands": {e["name"]: {
            "value": e["value"], "unit": e["unit"], "period": e["period"]
        } for e in entries},
        "order": "desc" if descending else "asc",
        "formula": "rank(values)",
        "result": entries,
    }


# ── Read-only executor ────────────────────────────────────


@dataclass
class ExecutedInvocation:
    """A tool invocation after execution: the validated args and its result."""

    name: str
    arguments: dict
    subquery_id: str
    reason_code: str
    result: dict
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Outcome of executing a :class:`RouteDecision`.

    ``error`` is True if any tool or calculation failed; the caller then falls
    back to normal retrieval. ``answer`` is a deterministic answer string only
    when the route was complete and every step succeeded (else ``None``, so the
    caller passes the structured results to the model generator).
    """

    invocations: list[ExecutedInvocation] = field(default_factory=list)
    calculations: list[dict] = field(default_factory=list)
    error: bool = False
    answer: Optional[str] = None
    answer_origin: Optional[str] = None
    answer_metadata: Optional[dict] = None
    complete: bool = True
    incomplete_reason_codes: list[str] = field(default_factory=list)


def _make_readonly_context() -> "ToolContext":
    from .tools.base import ToolContext

    # allow_write=False + max_refreshes=0: even if a write tool were somehow
    # routed (it never is), the guarded dispatcher would reject it.
    return ToolContext(allow_write=False, max_refreshes=0)


def _resolve_operand(ref: OperandRef, results: list[dict]) -> Optional[dict]:
    """Resolve one :class:`OperandRef` to a ``{value, unit, period}`` record.

    Handles both ``query_facts`` (``{"results": [rows]}`` selected by ticker)
    and ``get_fundamentals`` (``{"fundamentals": {metric: value}}`` selected by
    metric). Returns ``None`` when the target field is absent.
    """
    if ref.invocation_index < 0 or ref.invocation_index >= len(results):
        return None
    result = results[ref.invocation_index]
    if not isinstance(result, dict) or "error" in result:
        return None

    rows = result.get("results")
    if isinstance(rows, list):
        for row in rows:
            if str(row.get("ticker")) == str(ref.selector):
                return {
                    "value": row.get(ref.field),
                    "unit": row.get("unit"),
                    "period": row.get("period"),
                }
        return None

    fundamentals = result.get("fundamentals")
    if isinstance(fundamentals, dict) and ref.selector in fundamentals:
        return {"value": fundamentals[ref.selector], "unit": None, "period": None}
    return None


def execute_route(
    decision: RouteDecision,
    store,
    *,
    max_tools: int = 3,
    build_answer: bool = True,
) -> ExecutionResult:
    """Execute a matched :class:`RouteDecision` against the store, read-only.

    Runs at most ``max_tools`` invocations through the shared guarded dispatcher
    with a read-only context, resolves and evaluates any calculations, and — for
    a complete route with no failures — builds a short deterministic answer.
    Never raises: a handler error is captured on the invocation and flips
    ``error`` so the caller falls back to the normal retrieval lane.
    """
    from .tools.base import dispatch_named_tool

    result = ExecutionResult()
    if not decision.matched or not decision.tool_invocations:
        result.error = True
        return result
    result.complete = bool(decision.complete)
    result.incomplete_reason_codes = [
        code for code in decision.reason_codes if code in _STABLE_INCOMPLETE_CODES
    ]

    ctx = _make_readonly_context()
    raw_results: list[dict] = []
    try:
        tool_limit = max(0, int(max_tools))
        for inv in decision.tool_invocations[:tool_limit]:
            tool_result, name, args = dispatch_named_tool(
                inv.name, dict(inv.arguments), store, ctx,
                subquery_id=inv.subquery_id,
            )
            err = tool_result.get("error") if isinstance(tool_result, dict) else None
            if err:
                result.error = True
            raw_results.append(tool_result if isinstance(tool_result, dict) else {})
            result.invocations.append(
                ExecutedInvocation(
                    name=name or inv.name,
                    arguments=args,
                    subquery_id=inv.subquery_id,
                    reason_code=inv.reason_code,
                    result=tool_result if isinstance(tool_result, dict) else {},
                    error=err,
                )
            )

        if len(decision.tool_invocations) > tool_limit:
            result.complete = False
            if INCOMPLETE_RESULT_SET_TOO_LARGE not in result.incomplete_reason_codes:
                result.incomplete_reason_codes.append(INCOMPLETE_RESULT_SET_TOO_LARGE)

        for inv, raw in zip(decision.tool_invocations, raw_results):
            if inv.name != "describe_coverage" or not isinstance(raw, dict):
                continue
            if raw.get("status") == "unavailable":
                continue  # 2.3.7.1 structured unavailable answer is preserved.
            if raw.get("complete") is False:
                result.complete = False
                code = (
                    INCOMPLETE_PARTIAL_CATALOG_PAGE
                    if raw.get("next_cursor")
                    or int(raw.get("total_matching") or 0) > int(raw.get("result_count") or 0)
                    else INCOMPLETE_MISSING_COVERAGE_DIMENSION
                )
                if code not in result.incomplete_reason_codes:
                    result.incomplete_reason_codes.append(code)

        for spec in decision.calculations:
            operands: dict[str, dict] = {}
            resolvable = True
            for name, ref in spec.operands.items():
                record = _resolve_operand(ref, raw_results)
                if record is None:
                    resolvable = False
                    break
                operands[name] = record
            if not resolvable:
                result.error = True
                result.calculations.append(
                    {"error": "unresolved_operand", "operation": spec.operation}
                )
                continue
            calc = calculate(
                spec.operation,
                operands,
                display_unit=spec.display_unit,
                precision=spec.precision,
            )
            if "error" in calc:
                result.error = True
            result.calculations.append(calc)
    except Exception:  # noqa: BLE001 — executor must never raise into /query
        logger.exception("Deterministic route execution failed")
        result.error = True
        return result

    if build_answer and result.complete and not result.error:
        result.answer = build_deterministic_answer(decision, result)
        if result.answer is not None:
            result.answer_origin = "deterministic"
        if decision.tool_invocations and decision.tool_invocations[0].reason_code == REASON_COVERAGE:
            if _is_filtered_security_source_set(result.invocations):
                result.answer_metadata = _coverage_set_answer_metadata(result.invocations)
            else:
                result.answer_metadata = _coverage_answer_metadata(
                    result.invocations[0].result if result.invocations else {}
                )
    return result


# ── Deterministic answer templates ────────────────────────

_PROJECTION_CAVEAT = (
    "These are analyst estimates, not guarantees, and not financial advice."
)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_deterministic_answer(
    decision: RouteDecision, execution: ExecutionResult
) -> Optional[str]:
    """Render a short, deterministic answer for a complete route.

    Returns ``None`` (so the caller uses the model generator) when the route is
    incomplete, any step errored, or the result shape is unexpected. Projection
    answers always append the estimates/not-guarantees caveat.
    """
    if not decision.complete or execution.error or not execution.invocations:
        return None

    inv = execution.invocations[0]
    result = inv.result
    reason = inv.reason_code

    try:
        if reason == REASON_FUNDAMENTALS:
            return _answer_fundamentals(result)
        if reason in (REASON_RANK, REASON_THRESHOLD):
            return _answer_query_facts(result, inv.arguments)
        if reason == REASON_COMPARE:
            return _answer_comparison(result, execution.calculations)
        if reason == REASON_PRICE_TARGETS:
            return _answer_price_targets(result)
        if reason == REASON_ESTIMATES:
            return _answer_estimates(result)
        if reason == REASON_GUIDANCE:
            return _answer_guidance(result)
        if reason == REASON_MACRO:
            return _answer_macro(result)
        if reason == REASON_SENTIMENT:
            return _answer_sentiment(result, inv.arguments)
        if reason == REASON_TRADE_BIAS:
            return _answer_trade_bias(result)
        if reason == REASON_COVERAGE:
            if _is_filtered_security_source_set(execution.invocations):
                return _answer_coverage_set(execution.invocations)
            return _answer_coverage(result, inv.arguments)
    except Exception:  # noqa: BLE001 — a template glitch must not fail /query
        logger.exception("Deterministic answer rendering failed")
        return None
    return None


def _answer_fundamentals(result: dict) -> Optional[str]:
    ticker = result.get("ticker")
    fundamentals = result.get("fundamentals") or {}
    if not fundamentals:
        return None
    parts = [f"{metric.replace('_', ' ')} is {_fmt(value)}"
             for metric, value in fundamentals.items()]
    return f"For {ticker}, " + "; ".join(parts) + "."


def _answer_query_facts(result: dict, args: dict) -> Optional[str]:
    metric = result.get("metric")
    rows = result.get("results") or []
    if not rows:
        return None
    listed = ", ".join(f"{r.get('ticker')} ({_fmt(r.get('value'))})" for r in rows)
    order = args.get("order")
    if order == "asc":
        return f"Ranked by {metric} (lowest first): {listed}."
    if order == "desc":
        return f"Ranked by {metric} (highest first): {listed}."
    if args.get("op") is not None:
        return f"Stocks with {metric} {args.get('op')} {_fmt(args.get('value'))}: {listed}."
    return f"{metric}: {listed}."


def _answer_comparison(result: dict, calculations: list[dict]) -> Optional[str]:
    metric = result.get("metric")
    rows = result.get("results") or []
    if not rows:
        return None
    listed = "; ".join(
        f"{r.get('ticker')}: {_fmt(r.get('value'))}"
        + (f" ({r.get('period')})" if r.get("period") else "")
        for r in rows
    )
    answer = f"{metric} — {listed}."
    for calc in calculations:
        if calc.get("operation") == "difference" and "result" in calc:
            answer += f" Difference: {_fmt(calc['result'])} {calc.get('unit') or ''}".rstrip()
            answer += "."
    return answer


def _answer_price_targets(result: dict) -> Optional[str]:
    ticker = result.get("ticker")
    targets = result.get("price_targets") or {}
    if not targets:
        return None
    parts = [f"{metric.replace('_', ' ')} {_fmt(fact.get('value'))} ({fact.get('period')})"
             for metric, fact in targets.items()]
    return f"Analyst consensus price targets for {ticker}: " + "; ".join(parts) + ". " + _PROJECTION_CAVEAT


def _answer_estimates(result: dict) -> Optional[str]:
    ticker = result.get("ticker")
    estimates = result.get("estimates") or {}
    if not estimates:
        return None
    parts = [f"{metric.replace('_', ' ')} {_fmt(fact.get('value'))} ({fact.get('period')})"
             for metric, fact in estimates.items()]
    return f"Analyst consensus estimates for {ticker}: " + "; ".join(parts) + ". " + _PROJECTION_CAVEAT


def _answer_guidance(result: dict) -> Optional[str]:
    ticker = result.get("ticker")
    guidance = result.get("guidance") or {}
    if not guidance or result.get("status") == "not_found":
        return None
    return f"Latest management guidance for {ticker}: {guidance}. " + _PROJECTION_CAVEAT


def _answer_macro(result: dict) -> Optional[str]:
    macro = result.get("macro")
    if not macro:
        return None
    return f"Current macro snapshot: {macro}."


def _answer_sentiment(result: dict, args: dict) -> Optional[str]:
    if not result or result.get("error"):
        return None
    ticker = args.get("ticker")
    days = args.get("days")
    return f"News sentiment summary for {ticker} over the last {days} days: {result}."


def _answer_trade_bias(result: dict) -> Optional[str]:
    """Render the required long/short classification from indexed evidence."""
    if not result or result.get("error"):
        return None
    ticker = result.get("ticker")
    bias = result.get("bias")
    if not ticker or not bias:
        return None
    confidence = result.get("confidence")
    message = result.get("message") or f"RAG trade bias for {ticker} is {bias}."
    caveat = " This is not financial advice."
    if result.get("evidence_status") == "miss":
        return f"{message}{caveat}"
    conf = f" (confidence {confidence})" if confidence is not None else ""
    return f"{ticker} trade bias is {bias}{conf}. {message}{caveat}"


def _coverage_answer_metadata(result: dict) -> dict:
    """Expose the Store revision and truthful page metadata beside the answer."""
    securities = [
        str(row.get("ticker"))
        for row in result.get("securities") or []
        if isinstance(row, dict) and row.get("ticker")
    ]
    return {
        "coverage_basis": result.get("coverage_basis"),
        "data_revision": result.get("data_revision"),
        "universe_snapshot_at": result.get("universe_snapshot_at"),
        "result_count": result.get("result_count", 0),
        "total_matching": result.get("total_matching", 0),
        "complete": bool(result.get("complete", False)),
        "next_cursor": result.get("next_cursor"),
        "filters_applied": result.get("filters_applied") or {},
        "securities": securities,
    }


def _is_filtered_security_source_set(
    invocations: list[ExecutedInvocation],
) -> bool:
    """Return whether invocations are exact per-security dimension checks."""
    return bool(invocations) and all(
        invocation.arguments.get("operation") == "security_sources"
        and bool(invocation.arguments.get("filters"))
        for invocation in invocations
    )


def _coverage_set_match(invocation: ExecutedInvocation) -> bool:
    """Evaluate one exact security_sources result against its requested filter."""
    result = invocation.result
    if result.get("status") == "unavailable" or not result.get("covered"):
        return False
    filters = invocation.arguments.get("filters") or {}
    item_type = filters.get("item_type")
    if item_type:
        return str(item_type) in {
            str(value) for value in result.get("item_types") or []
        }
    source = filters.get("source") or filters.get("source_category")
    if source:
        expected = str(source)
        for row in result.get("sources") or []:
            if not isinstance(row, dict) or not row.get("has_evidence"):
                continue
            values = {
                str(row.get("source") or ""),
                str(row.get("source_category") or ""),
                str(row.get("item_type") or ""),
            }
            if expected in values:
                return True
        return False
    return False


def _coverage_set_answer_metadata(
    invocations: list[ExecutedInvocation],
) -> dict:
    """Expose a complete, bounded intersection for safe conversation carryover."""
    matches = [
        str(invocation.arguments.get("ticker"))
        for invocation in invocations
        if _coverage_set_match(invocation)
    ]
    unavailable = any(
        invocation.result.get("status") == "unavailable" for invocation in invocations
    )
    first = invocations[0].result if invocations else {}
    return {
        "coverage_basis": first.get("coverage_basis"),
        "data_revision": first.get("data_revision"),
        "universe_snapshot_at": first.get("universe_snapshot_at"),
        "result_count": len(matches),
        "total_matching": len(matches),
        "complete": not unavailable,
        "next_cursor": None,
        "filters_applied": invocations[0].arguments.get("filters") or {},
        "securities": matches,
    }


def _answer_coverage_set(invocations: list[ExecutedInvocation]) -> str:
    """Render an exact intersection over a bounded explicit security set."""
    if any(
        invocation.result.get("status") == "unavailable" for invocation in invocations
    ):
        return (
            "Coverage metadata is unavailable. I cannot determine which securities "
            "in the requested set have the requested evidence."
        )
    filters = invocations[0].arguments.get("filters") or {}
    dimension = (
        filters.get("item_type")
        or filters.get("source")
        or filters.get("source_category")
        or "requested"
    )
    matches = [
        str(invocation.arguments.get("ticker"))
        for invocation in invocations
        if _coverage_set_match(invocation)
    ]
    listed = ", ".join(matches) if matches else "none"
    return (
        f"Within the requested security set, stored {dimension} evidence is present "
        f"for: {listed}."
    )


def _answer_coverage(result: dict, args: dict) -> str:
    """Render a deterministic answer without claiming configured data is stored."""
    if result.get("status") == "unavailable":
        return (
            "Coverage metadata is unavailable. I cannot determine security membership, "
            "stored evidence, or source capability from the current registry."
        )

    operation = args.get("operation")
    basis = result.get("coverage_basis") or "unknown"
    complete = bool(result.get("complete", False))
    if operation == "list_securities":
        rows = result.get("securities") or []
        tickers = ", ".join(str(row.get("ticker")) for row in rows if row.get("ticker"))
        total = result.get("total_matching", len(rows))
        if args.get("ticker_only") and complete:
            return f"The active security registry contains {total} securities: {tickers}."
        answer = f"Found {len(rows)} of {total} matching securities: {tickers or 'none'}."
        if not complete:
            answer += " This is a bounded page; more results are available via the next cursor."
        return answer
    if operation == "contains_security":
        ticker = args.get("ticker") or "the requested security"
        if result.get("covered"):
            return f"Yes — {ticker} is in the active security registry (basis: {basis})."
        suggestions = ", ".join(result.get("suggestions") or [])
        suffix = f" Possible matches: {suggestions}." if suggestions else ""
        return f"No — {ticker} is not in the active security registry (basis: {basis}).{suffix}"
    if operation == "security_sources":
        ticker = args.get("ticker") or "the requested security"
        if not result.get("covered"):
            return f"Coverage for {ticker} is not present in the active security registry."
        sources = result.get("sources") or []
        evidence = [row["source"] for row in sources if row.get("has_evidence")]
        capable = [
            row["source"] for row in sources
            if row.get("capability", {}).get("configured")
            and row.get("capability", {}).get("available")
        ]
        answer = f"For {ticker}, stored evidence is present from {', '.join(evidence) or 'no listed source'}"
        answer += f"; available configured capabilities are {', '.join(capable) or 'none listed'}."
        answer += " Capability and stored evidence are reported separately."
        return answer
    if operation == "list_sources":
        sources = result.get("sources") or []
        names = ", ".join(str(row.get("source")) for row in sources)
        return f"Configured source capabilities: {names or 'none listed'}."
    if operation == "list_item_types":
        values = ", ".join(str(value) for value in result.get("item_types") or [])
        return f"Stored item types: {values or 'none listed'}."
    if operation == "list_metrics":
        values = ", ".join(str(value) for value in result.get("metrics") or [])
        return f"Stored metric families: {values or 'none listed'}."
    return (
        f"Coverage summary: {result.get('active_securities', 0)} active securities, "
        f"{len(result.get('metrics') or [])} metric families, and "
        f"{len(result.get('item_types') or [])} item types (basis: {basis})."
    )

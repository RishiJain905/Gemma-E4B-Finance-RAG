"""
src/middleware/deterministic_answers.py
Final-answer contract and typed renderers for deterministic finance answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterable, Optional

from . import deterministic_router as dr

MAX_DETERMINISTIC_ANSWER_CHARS = 16_000
GENERATION_SKIP_REASON = "complete_deterministic_route"

_WRITE_TOOLS = frozenset({"refresh_data"})
_PROJECTION_CAVEAT = (
    "These figures are estimates, not guarantees, and are not financial advice."
)


class TemplateClass(str, Enum):
    """Small allowlist of deterministic final-answer shapes."""

    COVERAGE = "coverage"
    FACT = "fact"
    BOUNDED_SET = "bounded_set"
    CALCULATION = "calculation"
    FRESHNESS = "freshness"
    ESTIMATE = "estimate"
    TARGET = "target"
    GUIDANCE = "guidance"
    MACRO = "macro"
    TRADE_BIAS = "trade_bias"


@dataclass(frozen=True)
class ContractCheck:
    """Eligibility verdict for the deterministic final-answer contract."""

    eligible: bool
    reason: str
    template: Optional[TemplateClass] = None


@dataclass(frozen=True)
class RenderedAnswer:
    """Bounded deterministic text plus values used for numeric validation."""

    text: str
    template: TemplateClass
    validation_values: tuple[dict, ...] = ()


class DeterministicAnswerError(ValueError):
    """Raised when a supposedly eligible result cannot be rendered safely."""


_REASON_TO_TEMPLATE = {
    dr.REASON_COVERAGE: TemplateClass.COVERAGE,
    dr.REASON_FUNDAMENTALS: TemplateClass.FACT,
    dr.REASON_RANK: TemplateClass.BOUNDED_SET,
    dr.REASON_THRESHOLD: TemplateClass.BOUNDED_SET,
    dr.REASON_COMPARE: TemplateClass.BOUNDED_SET,
    dr.REASON_ESTIMATES: TemplateClass.ESTIMATE,
    dr.REASON_PRICE_TARGETS: TemplateClass.TARGET,
    dr.REASON_GUIDANCE: TemplateClass.GUIDANCE,
    dr.REASON_MACRO: TemplateClass.MACRO,
    dr.REASON_FRESHNESS: TemplateClass.FRESHNESS,
    dr.REASON_TRADE_BIAS: TemplateClass.TRADE_BIAS,
}


def _template_for(execution) -> Optional[TemplateClass]:
    if execution is None or not execution.invocations:
        return None
    reason = execution.invocations[0].reason_code
    template = _REASON_TO_TEMPLATE.get(reason)
    if template is TemplateClass.BOUNDED_SET and execution.calculations:
        return TemplateClass.CALCULATION
    return template


def _has_flag(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if bool(value.get(key)):
            return True
        return any(_has_flag(item, key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_flag(item, key) for item in value)
    return False


def _known_result_shape(template: TemplateClass, invocation, result) -> bool:
    raw = invocation.result
    if not isinstance(raw, dict):
        return False
    if template is TemplateClass.COVERAGE:
        complete = (
            raw.get("status") != "unavailable"
            and isinstance(raw.get("complete"), bool)
            and raw.get("complete") is True
            and not raw.get("next_cursor")
        )
        if not complete:
            return False
        list_key = {
            "list_securities": "securities",
            "list_sources": "sources",
            "list_item_types": "item_types",
            "list_metrics": "metrics",
        }.get(invocation.arguments.get("operation"))
        if list_key is not None:
            rows = raw.get(list_key)
            if not isinstance(rows, list):
                return False
            size = len(rows)
            return (
                int(raw.get("result_count", size)) == size
                and int(raw.get("total_matching", size)) == size
            )
        return True
    if template is TemplateClass.FACT:
        fundamentals = raw.get("fundamentals")
        if not isinstance(fundamentals, dict):
            return False
        requested = set(map(str, invocation.arguments.get("metrics") or ()))
        return not requested or requested <= set(map(str, fundamentals))
    if template in {TemplateClass.BOUNDED_SET, TemplateClass.CALCULATION}:
        rows = raw.get("results")
        if not isinstance(rows, list):
            return False
        if invocation.reason_code == dr.REASON_THRESHOLD:
            limit = int(invocation.arguments.get("limit") or 10)
            if len(rows) >= limit and raw.get("complete") is not True:
                return False
        requested = invocation.arguments.get("tickers") or []
        if invocation.reason_code == dr.REASON_COMPARE and requested:
            returned = {str(row.get("ticker")) for row in rows if isinstance(row, dict)}
            if not set(map(str, requested)) <= returned:
                return False
        return True
    if template is TemplateClass.ESTIMATE:
        return isinstance(raw.get("estimates"), dict)
    if template is TemplateClass.TARGET:
        return isinstance(raw.get("price_targets"), dict)
    if template is TemplateClass.GUIDANCE:
        return raw.get("status") in {"found", "not_found"} and isinstance(
            raw.get("guidance"), dict
        )
    if template is TemplateClass.MACRO:
        return isinstance(raw.get("macro"), dict)
    if template is TemplateClass.FRESHNESS:
        return "overall" in raw and isinstance(raw.get("sources", {}), dict)
    if template is TemplateClass.TRADE_BIAS:
        return raw.get("bias") in {"long", "short", "neutral"}
    return False


def _is_write_tool(name: str) -> bool:
    if name in _WRITE_TOOLS:
        return True
    try:
        from .tools.base import REGISTRY

        tool = REGISTRY.get(name)
        return bool(tool is not None and tool.write)
    except Exception:  # noqa: BLE001 - registry inspection is defense in depth
        return True


def _ledger_periods(ledger: Iterable[Any]) -> set[str]:
    return {
        str(getattr(item, "period", "") or getattr(item, "as_of", ""))
        for item in ledger
        if getattr(item, "period", None) or getattr(item, "as_of", None)
    }


def _has_material_result(template: TemplateClass, invocation) -> bool:
    raw = invocation.result
    if template is TemplateClass.COVERAGE:
        return bool(
            raw.get("securities") or raw.get("sources") or raw.get("item_types")
            or raw.get("metrics") or raw.get("covered")
        )
    if template is TemplateClass.FACT:
        return bool(raw.get("fundamentals"))
    if template in {TemplateClass.BOUNDED_SET, TemplateClass.CALCULATION}:
        return bool(raw.get("results"))
    if template is TemplateClass.ESTIMATE:
        return bool(raw.get("estimates"))
    if template is TemplateClass.TARGET:
        return bool(raw.get("price_targets"))
    if template is TemplateClass.GUIDANCE:
        return bool(raw.get("guidance"))
    if template is TemplateClass.MACRO:
        return bool(raw.get("macro"))
    if template is TemplateClass.FRESHNESS:
        return bool(raw.get("sources")) or bool(raw.get("overall"))
    if template is TemplateClass.TRADE_BIAS:
        return raw.get("evidence_status") == "hit" and bool(raw.get("signals"))
    return False


def _coverage_match(invocation) -> bool:
    """Evaluate one exact security_sources result against its filter."""
    raw = invocation.result
    if raw.get("status") == "unavailable" or not raw.get("covered"):
        return False
    filters = invocation.arguments.get("filters") or {}
    item_type = filters.get("item_type")
    if item_type:
        return str(item_type) in {str(value) for value in raw.get("item_types") or []}
    source = filters.get("source") or filters.get("source_category")
    if source:
        expected = str(source)
        return any(
            isinstance(row, dict)
            and row.get("has_evidence")
            and expected in {
                str(row.get("source") or ""),
                str(row.get("source_category") or ""),
                str(row.get("item_type") or ""),
            }
            for row in raw.get("sources") or []
        )
    return False


def check_final_answer_contract(
    result,
    *,
    evidence_ledger: Iterable[Any],
    freshness: Optional[dict],
    write_requested: bool = False,
) -> ContractCheck:
    """Enforce the Phase 2.3.7.3 final-answer contract without rendering.

    Every negative verdict is a normal model-path fallback, never an error.
    """
    execution = getattr(result, "tool_execution", None)
    template = _template_for(execution)
    if template is None:
        return ContractCheck(False, "route_not_allowlisted")
    if getattr(result, "lane", None) not in {"fast", "catalog", None}:
        lane = getattr(getattr(result, "lane", None), "value", getattr(result, "lane", None))
        if lane not in {"fast", "catalog"}:
            return ContractCheck(False, "route_not_fast", template)
    if execution is None or not execution.invocations:
        return ContractCheck(False, "route_not_matched", template)
    if not execution.complete or execution.incomplete_reason_codes:
        return ContractCheck(False, "route_incomplete", template)
    if execution.error:
        return ContractCheck(False, "tool_execution_failed", template)
    if any(
        invocation.error
        or not isinstance(invocation.result, dict)
        or bool(invocation.result.get("error"))
        for invocation in execution.invocations
    ):
        return ContractCheck(False, "tool_execution_failed", template)
    if any(_is_write_tool(invocation.name) for invocation in execution.invocations):
        return ContractCheck(False, "write_route", template)
    if write_requested:
        return ContractCheck(False, "write_requested", template)

    plan = getattr(result, "plan", None)
    obligations = getattr(plan, "obligations", None)
    evidence_modes = set(getattr(obligations, "evidence_modes", ()) or ())
    intents = set(getattr(plan, "intents", ()) or ())
    if (
        bool(getattr(obligations, "qualitative", False))
        or "documents" in evidence_modes
        or intents & {"explanation", "risk", "news", "sentiment"}
        or bool(getattr(result, "merged_documents", ()) or ())
    ):
        return ContractCheck(False, "qualitative_obligation", template)

    invocation = execution.invocations[0]
    if not _known_result_shape(template, invocation, result):
        return ContractCheck(False, "result_completeness_unknown", template)
    if template is TemplateClass.COVERAGE and any(
        not _known_result_shape(template, item, result)
        for item in execution.invocations[1:]
    ):
        return ContractCheck(False, "result_completeness_unknown", template)
    if template is TemplateClass.COVERAGE and getattr(result, "set_complete", None) is False:
        return ContractCheck(False, "result_set_incomplete", template)

    ledger = list(evidence_ledger or ())
    if _has_material_result(template, invocation) and not ledger:
        return ContractCheck(False, "provenance_missing", template)
    if len(execution.invocations) > 1 and len(ledger) < len(execution.invocations):
        return ContractCheck(False, "provenance_missing", template)
    requested_periods = set(map(str, getattr(plan, "periods", ()) or ()))
    if requested_periods and not requested_periods <= _ledger_periods(ledger):
        return ContractCheck(False, "period_unresolved", template)

    facts = list(getattr(result, "merged_facts", ()) or ())
    raw_results = [item.result for item in execution.invocations]
    if _has_flag(facts, "conflict") or _has_flag(raw_results, "conflict"):
        return ContractCheck(False, "conflict_requires_judgment", template)
    freshness = freshness or {}
    if freshness.get("refreshed_during_query") or freshness.get("fetched_on_miss"):
        return ContractCheck(False, "write_performed", template)
    if template not in {TemplateClass.FRESHNESS, TemplateClass.TRADE_BIAS}:
        if freshness.get("stale_sources_used") or str(freshness.get("overall", "")).lower() == "stale":
            return ContractCheck(False, "stale_policy_violation", template)
        if _has_flag(facts, "stale") or _has_flag(raw_results, "stale"):
            return ContractCheck(False, "stale_policy_violation", template)
    return ContractCheck(True, GENERATION_SKIP_REASON, template)


def _decimal_equal(left: Any, right: Any) -> bool:
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return str(left) == str(right)


def _find_item(
    ledger: Iterable[Any],
    *,
    ticker: Any = None,
    metric: Any = None,
    value: Any = None,
    period: Any = None,
) -> Optional[Any]:
    best = None
    best_score = -1
    for item in ledger:
        score = 0
        item_ticker = getattr(item, "ticker", None)
        item_metric = getattr(item, "metric", None)
        item_value = getattr(item, "value", None)
        item_period = getattr(item, "period", None) or getattr(item, "as_of", None)
        if ticker is not None:
            if str(item_ticker or "").upper() != str(ticker).upper():
                continue
            score += 4
        if metric is not None:
            if str(item_metric or "").lower() != str(metric).lower():
                continue
            score += 4
        if value is not None and item_value is not None:
            if not _decimal_equal(item_value, value):
                continue
            score += 2
        if period is not None and item_period is not None:
            if str(item_period) != str(period):
                continue
            score += 1
        if score > best_score:
            best, best_score = item, score
    return best


def _format_value(value: Any, unit: Any = None, period: Any = None) -> str:
    rendered = str(value)
    if unit not in (None, ""):
        rendered += f" {unit}"
    if period not in (None, ""):
        rendered += f" ({period})"
    return rendered


def _record(
    ledger: Iterable[Any],
    *,
    ticker: Any,
    metric: Any,
    value: Any,
    unit: Any = None,
    period: Any = None,
) -> tuple[str, dict]:
    item = _find_item(
        ledger, ticker=ticker, metric=metric, value=value, period=period,
    )
    if item is not None:
        unit = unit or getattr(item, "unit", None)
        period = period or getattr(item, "period", None) or getattr(item, "as_of", None)
    citation = f" [{getattr(item, 'evidence_id')}]" if item is not None else ""
    text = _format_value(value, unit, period) + citation
    return text, {"result": value, "unit": unit, "period": period}


def _mapping_records(
    mapping: dict,
    *,
    ticker: str,
    ledger: Iterable[Any],
    metric_prefix: str = "",
) -> tuple[list[str], list[dict]]:
    parts: list[str] = []
    values: list[dict] = []
    for metric, raw in mapping.items():
        full_metric = f"{metric_prefix}{metric}"
        if isinstance(raw, dict) and "value" in raw:
            value = raw.get("value")
            unit = raw.get("unit")
            period = raw.get("period") or raw.get("as_of")
        elif isinstance(raw, (str, int, float, Decimal)):
            value, unit, period = raw, None, None
        else:
            continue
        rendered, validation = _record(
            ledger, ticker=ticker, metric=full_metric, value=value,
            unit=unit, period=period,
        )
        parts.append(f"{str(metric).replace('_', ' ')}: {rendered}")
        values.append(validation)
    return parts, values


def _render_coverage(execution, ledger) -> RenderedAnswer:
    invocation = execution.invocations[0]
    if len(execution.invocations) > 1:
        filters = invocation.arguments.get("filters") or {}
        dimension = (
            filters.get("item_type") or filters.get("source")
            or filters.get("source_category") or "requested"
        )
        outcomes: list[str] = []
        matches = 0
        for index, item in enumerate(execution.invocations):
            matched = _coverage_match(item)
            matches += int(matched)
            citation = (
                f" [{ledger[index].evidence_id}]" if index < len(ledger) else ""
            )
            outcomes.append(
                f"{item.arguments.get('ticker')}: "
                f"{'present' if matched else 'not present'}{citation}"
            )
        total = len(execution.invocations)
        text = (
            f"Complete {dimension} coverage for the requested security set: "
            + "; ".join(outcomes)
            + f". {matches} of {total} requested securities match."
        )
        return RenderedAnswer(
            text=text,
            template=TemplateClass.COVERAGE,
            validation_values=(
                {"result": matches, "unit": "count"},
                {"result": total, "unit": "count"},
            ),
        )
    raw = invocation.result
    operation = invocation.arguments.get("operation")
    item = next(iter(ledger), None)
    citation = f" [{item.evidence_id}]" if item is not None else ""
    values: list[dict] = []
    if operation == "list_securities":
        tickers = [str(row.get("ticker")) for row in raw.get("securities", []) if row.get("ticker")]
        count = int(raw.get("total_matching", len(tickers)))
        values.append({"result": count, "unit": "count"})
        text = (
            f"The complete active security registry contains {count} securities: "
            f"{', '.join(tickers) if tickers else 'none'}.{citation}"
        )
    elif operation == "contains_security":
        ticker = invocation.arguments.get("ticker", "the requested security")
        answer = "Yes" if raw.get("covered") else "No"
        text = f"{answer} — {ticker} is {'present' if raw.get('covered') else 'not present'} in the active security registry.{citation}"
    elif operation == "security_sources":
        ticker = invocation.arguments.get("ticker", "the requested security")
        sources = [
            str(row.get("source")) for row in raw.get("sources", [])
            if isinstance(row, dict) and row.get("has_evidence") and row.get("source")
        ]
        text = f"Stored source coverage for {ticker}: {', '.join(sources) if sources else 'none'}.{citation}"
    else:
        key = {
            "list_sources": "sources",
            "list_item_types": "item_types",
            "list_metrics": "metrics",
        }.get(operation)
        if key:
            entries = []
            for row in raw.get(key, []):
                entries.append(str(row.get("source") if isinstance(row, dict) else row))
            count = int(raw.get("total_matching", len(entries)))
            values.append({"result": count, "unit": "count"})
            text = f"The complete {key.replace('_', ' ')} inventory contains {count} entries: {', '.join(entries) if entries else 'none'}.{citation}"
        else:
            text = f"Coverage inventory summary: {raw.get('message') or 'available'}.{citation}"
    return RenderedAnswer(text=text, template=TemplateClass.COVERAGE,
                          validation_values=tuple(values))


def _render_fact(invocation, ledger) -> RenderedAnswer:
    raw = invocation.result
    ticker = str(raw.get("ticker") or invocation.arguments.get("ticker") or "")
    parts, values = _mapping_records(raw.get("fundamentals", {}), ticker=ticker, ledger=ledger)
    text = (
        f"No stored facts matched the requested metric for {ticker}."
        if not parts else f"For {ticker}, " + "; ".join(parts) + "."
    )
    return RenderedAnswer(text, TemplateClass.FACT, tuple(values))


def _render_bounded(invocation, execution, ledger) -> RenderedAnswer:
    raw = invocation.result
    metric = raw.get("metric") or invocation.arguments.get("metric") or "metric"
    rows = raw.get("results", [])
    if not rows:
        return RenderedAnswer(
            "No stored results matched the requested bounded filter.",
            TemplateClass.BOUNDED_SET,
        )
    parts: list[str] = []
    values: list[dict] = [{"result": len(rows), "unit": "count"}]
    for row in rows:
        rendered, validation = _record(
            ledger,
            ticker=row.get("ticker"), metric=row.get("metric") or metric,
            value=row.get("value"), unit=row.get("unit"), period=row.get("period"),
        )
        parts.append(f"{row.get('ticker')}: {rendered}")
        values.append(validation)
    if invocation.reason_code == dr.REASON_COMPARE:
        prefix = f"Compared all {len(rows)} requested securities by {metric}: "
    elif invocation.reason_code == dr.REASON_RANK:
        direction = "lowest first" if invocation.arguments.get("order") == "asc" else "highest first"
        prefix = f"Complete bounded ranking by {metric} ({direction}; {len(rows)} results): "
    else:
        prefix = f"Complete bounded filter by {metric} ({len(rows)} matches): "
    text = prefix + "; ".join(parts) + "."
    template = TemplateClass.CALCULATION if execution.calculations else TemplateClass.BOUNDED_SET
    for calc in execution.calculations:
        if "result" not in calc:
            raise DeterministicAnswerError("calculation result is missing")
        rendered = _format_value(calc["result"], calc.get("unit"), calc.get("period"))
        text += f" {str(calc.get('operation') or 'calculation').replace('_', ' ').title()}: {rendered}."
        values.append({
            "result": calc["result"], "unit": calc.get("unit"),
            "period": calc.get("period"),
        })
    return RenderedAnswer(text, template, tuple(values))


def _render_projection(invocation, ledger, template: TemplateClass) -> RenderedAnswer:
    raw = invocation.result
    ticker = str(raw.get("ticker") or invocation.arguments.get("ticker") or "")
    key = "estimates" if template is TemplateClass.ESTIMATE else "price_targets"
    parts, values = _mapping_records(raw.get(key, {}), ticker=ticker, ledger=ledger)
    label = "analyst consensus estimates" if template is TemplateClass.ESTIMATE else "analyst consensus price targets"
    text = (
        f"No stored {label} are available for {ticker}."
        if not parts else f"For {ticker}, {label}: " + "; ".join(parts) + "."
    )
    return RenderedAnswer(f"{text} {_PROJECTION_CAVEAT}", template, tuple(values))


def _render_guidance(invocation, ledger) -> RenderedAnswer:
    raw = invocation.result
    ticker = str(raw.get("ticker") or invocation.arguments.get("ticker") or "")
    parts, values = _mapping_records(
        raw.get("guidance", {}), ticker=ticker, ledger=ledger,
        metric_prefix="guidance_",
    )
    text = (
        f"No stored management guidance is available for {ticker}."
        if not parts else f"Latest stored management guidance for {ticker}: " + "; ".join(parts) + "."
    )
    return RenderedAnswer(f"{text} {_PROJECTION_CAVEAT}", TemplateClass.GUIDANCE, tuple(values))


def _render_macro(invocation, ledger) -> RenderedAnswer:
    parts, values = _mapping_records(
        invocation.result.get("macro", {}), ticker="MACRO", ledger=ledger,
    )
    text = "No stored macro snapshot values are available." if not parts else "Stored macro snapshot: " + "; ".join(parts) + "."
    return RenderedAnswer(text, TemplateClass.MACRO, tuple(values))


def _render_freshness(invocation, ledger) -> RenderedAnswer:
    raw = invocation.result
    ticker = str(raw.get("ticker") or invocation.arguments.get("ticker") or "")
    parts: list[str] = []
    for source, detail in raw.get("sources", {}).items():
        if isinstance(detail, dict):
            status = detail.get("status") or detail.get("freshness") or "unknown"
            as_of = detail.get("as_of") or detail.get("last_fetched")
        else:
            status, as_of = detail, None
        item = _find_item(
            ledger, ticker=ticker, metric=f"freshness_{source}", value=status,
            period=as_of,
        )
        citation = f" [{item.evidence_id}]" if item is not None else ""
        period = f" (as of {as_of})" if as_of else ""
        parts.append(f"{source}: {status}{period}{citation}")
    stale = raw.get("stale_sources") or []
    disclosure = f" Stale sources: {', '.join(map(str, stale))}." if stale else " No stale sources are reported."
    sources = "; ".join(parts) if parts else "no per-source rows"
    text = f"Freshness for {ticker} is {raw.get('overall', 'unknown')}. Sources: {sources}.{disclosure}"
    return RenderedAnswer(text, TemplateClass.FRESHNESS)


def _render_trade_bias(invocation, ledger) -> RenderedAnswer:
    """Force a long / short / neutral answer from classify_trade_bias."""
    raw = invocation.result if isinstance(invocation.result, dict) else {}
    ticker = str(raw.get("ticker") or invocation.arguments.get("ticker") or "")
    bias = str(raw.get("bias") or "neutral")
    confidence = raw.get("confidence")
    item = _find_item(ledger, ticker=ticker, metric="trade_bias", value=bias)
    citation = f" [{getattr(item, 'evidence_id')}]" if item is not None else ""
    caveat = " This is not financial advice."
    if raw.get("evidence_status") == "miss":
        text = (
            f"No indexed long/short evidence for {ticker}; RAG miss. "
            f"Open-web fallback is allowed.{caveat}"
        )
        return RenderedAnswer(text, TemplateClass.TRADE_BIAS)
    conf = f" (confidence {confidence})" if confidence is not None else ""
    text = (
        f"{ticker} is a {bias} trade{conf}.{citation} "
        f"Answer from classify_trade_bias; do not guess.{caveat}"
    )
    return RenderedAnswer(text, TemplateClass.TRADE_BIAS, ({"result": bias},))


def render_deterministic_answer(result, *, evidence_ledger: Iterable[Any]) -> RenderedAnswer:
    """Render one allowlisted typed answer; never calls or rewrites through a model."""
    execution = getattr(result, "tool_execution", None)
    template = _template_for(execution)
    if template is None or execution is None or not execution.invocations:
        raise DeterministicAnswerError("route has no deterministic template")
    invocation = execution.invocations[0]
    ledger = list(evidence_ledger or ())
    if template is TemplateClass.COVERAGE:
        rendered = _render_coverage(execution, ledger)
    elif template is TemplateClass.FACT:
        rendered = _render_fact(invocation, ledger)
    elif template in {TemplateClass.BOUNDED_SET, TemplateClass.CALCULATION}:
        rendered = _render_bounded(invocation, execution, ledger)
    elif template in {TemplateClass.ESTIMATE, TemplateClass.TARGET}:
        rendered = _render_projection(invocation, ledger, template)
    elif template is TemplateClass.GUIDANCE:
        rendered = _render_guidance(invocation, ledger)
    elif template is TemplateClass.MACRO:
        rendered = _render_macro(invocation, ledger)
    elif template is TemplateClass.FRESHNESS:
        rendered = _render_freshness(invocation, ledger)
    elif template is TemplateClass.TRADE_BIAS:
        rendered = _render_trade_bias(invocation, ledger)
    else:  # pragma: no cover - enum + allowlist make this defensive only
        raise DeterministicAnswerError("unsupported deterministic template")
    if not rendered.text.strip():
        raise DeterministicAnswerError("deterministic answer is empty")
    if len(rendered.text) > MAX_DETERMINISTIC_ANSWER_CHARS:
        raise DeterministicAnswerError("deterministic answer exceeds the bounded limit")
    return rendered


def render_execution_answer(execution) -> Optional[str]:
    """Compatibility renderer used by the router before evidence normalization."""
    result = type("ExecutionView", (), {"tool_execution": execution})()
    try:
        return render_deterministic_answer(result, evidence_ledger=[]).text
    except DeterministicAnswerError:
        return None

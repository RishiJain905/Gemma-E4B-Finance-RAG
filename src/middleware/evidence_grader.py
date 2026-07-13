"""
src/middleware/evidence_grader.py
Deterministic, route-aware evidence sufficiency grading and corrective actions.
"""

from __future__ import annotations

import logging
from time import perf_counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .evidence import document_body, evidence_field, evidence_id
from .query_plan import QueryPlan

logger = logging.getLogger(__name__)


class SufficiencyStatus(str, Enum):
    """The three deterministic evidence-gate outcomes."""

    SUFFICIENT = "sufficient"
    BORDERLINE = "borderline"
    MISSING = "missing"


class CorrectiveAction(str, Enum):
    """The complete allowlist of bounded internal corrective actions."""

    BROADEN_TICKER_FILTER = "broaden_ticker_filter"
    APPLY_VALIDATED_ALIAS = "apply_validated_alias"
    EXPAND_PARENT_SECTION = "expand_parent_section"
    RUN_DERIVED_SUBQUERIES = "run_derived_subqueries"
    ALTERNATE_INTERNAL_MODALITY = "alternate_internal_modality"
    NONE = "none"


@dataclass(frozen=True)
class EvidenceObligation:
    """One required evidence unit derived only from a validated QueryPlan."""

    subquery_id: str
    entities: tuple[str, ...]
    metrics: tuple[str, ...]
    periods: tuple[str, ...]
    modalities: tuple[str, ...]
    operations: tuple[str, ...]
    freshness_required: bool
    minimum_source_class: str


@dataclass(frozen=True)
class CoverageResult:
    """Coverage details for one subquery obligation."""

    subquery_id: str
    covered_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    @property
    def covered(self) -> bool:
        """Whether every required field has compatible usable evidence."""
        return not self.missing_fields


@dataclass(frozen=True)
class SufficiencyResult:
    """Machine-readable output of the evidence sufficiency gate."""

    status: SufficiencyStatus
    overall_score: float
    reason_codes: tuple[str, ...]
    coverage: tuple[CoverageResult, ...]
    conflicts: tuple[dict, ...]
    allowed_action: CorrectiveAction

    @property
    def covered_subqueries(self) -> tuple[str, ...]:
        """Subquery ids whose obligations are fully covered."""
        return tuple(row.subquery_id for row in self.coverage if row.covered)

    @property
    def missing_subqueries(self) -> tuple[str, ...]:
        """Subquery ids with at least one explicit missing obligation."""
        return tuple(row.subquery_id for row in self.coverage if not row.covered)

    def to_metadata(self) -> dict:
        """Return the response/prompt-safe sufficiency metadata."""
        return {
            "sufficiency": self.status.value,
            "reason_codes": list(self.reason_codes),
            "covered_subqueries": list(self.covered_subqueries),
            "missing_subqueries": list(self.missing_subqueries),
            "corrective_action": self.allowed_action.value,
        }

    def graph_trace_metadata(self) -> dict:
        """Return bounded grader facts for graph stage instrumentation."""
        return {
            "status": self.status.value,
            "score": self.overall_score,
            "count": len(self.covered_subqueries),
            "reason": self.allowed_action.value,
        }


_MODE_MAP = {
    "facts": "fact",
    "documents": "document",
    "macro": "macro",
    "tools": "calculation",
}
_QUALITATIVE_INTENTS = frozenset({"explanation", "risk", "news", "sentiment"})
_OPERATION_MAP = {
    "comparison": "compare",
    "trend": "trend",
    "projection": "project",
    "fact_lookup": "lookup",
    "explanation": "explain",
    "risk": "summarize_risk",
    "news": "summarize_news",
    "sentiment": "summarize_sentiment",
}
# Alias values include the ingestors' real store keys (e.g. yfinance
# fundamentals use ``revenue_ttm``/``gross_margin_ttm``/``eps_ttm``) so an
# obligation named by the user-facing metric can be satisfied by stored rows.
_METRIC_ALIASES = {
    "revenue": frozenset({"total_revenue", "revenues", "sales", "revenue_ttm"}),
    "total_revenue": frozenset({"revenue", "revenues", "sales", "revenue_ttm"}),
    "gross_margin": frozenset({"gross_profit_margin", "gross_margin_pct", "gross_margin_ttm"}),
    "net_income": frozenset({"net_earnings", "profit"}),
    "eps": frozenset({"earnings_per_share", "diluted_eps", "eps_ttm"}),
    "pe_ratio": frozenset({"pe", "price_to_earnings", "pe_ratio_ttm"}),
    "market_cap": frozenset({"market_capitalization", "marketcap"}),
}
_STALE = frozenset({"stale", "expired", "never_fetched"})


def _dedup(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def validated_metric_aliases(metrics: Iterable[str]) -> tuple[str, ...]:
    """Expand metrics with the grader's finite, audited alias catalog."""
    expanded: list[str] = []
    for metric in metrics:
        normalized = str(metric).strip().lower()
        if not normalized:
            continue
        expanded.append(normalized)
        expanded.extend(sorted(_METRIC_ALIASES.get(normalized, ())))
    return _dedup(expanded)


def build_obligations(plan: QueryPlan) -> tuple[EvidenceObligation, ...]:
    """Build evidence obligations directly from an already validated plan."""
    plan.validate()
    rows: list[EvidenceObligation] = []
    for subquery in plan.subqueries:
        intents = subquery.intents or tuple(plan.intents)
        modes = tuple(_MODE_MAP[mode] for mode in subquery.retrieval_modes if mode in _MODE_MAP)
        if not modes:
            inferred: list[str] = []
            if subquery.metrics or plan.metrics:
                inferred.append("fact")
            if set(intents) & _QUALITATIVE_INTENTS:
                inferred.append("document")
            modes = tuple(inferred or ["document"])
        if "documents" in subquery.retrieval_modes:
            if "news" in intents:
                modes = tuple("news" if mode == "document" else mode for mode in modes)
            elif "risk" in intents:
                modes = tuple("filing" if mode == "document" else mode for mode in modes)
        if "projection" in intents:
            modes = tuple("estimate" if mode == "fact" else mode for mode in modes)
        if "macro" in modes:
            modes = tuple(mode for mode in modes if mode != "fact")
        if "calculation" in modes:
            modes = tuple(mode for mode in modes if mode != "fact")
        source_class = "internal"
        if "risk" in intents:
            source_class = "filing"
        elif "news" in intents:
            source_class = "news"
        elif "projection" in intents:
            source_class = "estimate"
        elif "macro" in modes:
            source_class = "macro"
        rows.append(EvidenceObligation(
            subquery_id=subquery.id,
            entities=_dedup(subquery.entity_tickers or tuple(plan.tickers)),
            metrics=_dedup(subquery.metrics or tuple(plan.metrics)),
            periods=_dedup(subquery.periods or tuple(plan.periods)),
            modalities=_dedup(modes),
            operations=_dedup(_OPERATION_MAP.get(intent, intent) for intent in intents),
            freshness_required=bool(set(intents) & {"news", "projection"}),
            minimum_source_class=source_class,
        ))
    return tuple(rows)


def _row_modality(row: dict, *, document: bool) -> str:
    source = str(evidence_field(row, "source_type", evidence_field(row, "source", ""))).lower()
    kind = str(row.get("kind") or "").lower()
    if kind in {"calculation", "tool_result"}:
        return "calculation"
    if source in {"fred", "macro"} or evidence_field(row, "ticker") == "MACRO":
        return "macro"
    if source in {"estimates", "analyst_estimate"}:
        return "estimate"
    if "news" in source or source == "gdelt":
        return "news"
    if source.startswith("sec_") or source in {"filing", "sec"}:
        return "filing" if document else "fact"
    return "document" if document else "fact"


def _source_compatible(row: dict, required: str) -> bool:
    if required == "internal":
        return True
    modality = _row_modality(row, document=bool(document_body(row)))
    return modality == required or (required == "filing" and modality == "fact")


def _metric_match(actual: object, requested: str) -> tuple[bool, bool]:
    value = str(actual or "").strip().lower()
    requested = requested.lower()
    if value == requested:
        return True, False
    aliases = _METRIC_ALIASES.get(requested, frozenset())
    return value in aliases, value in aliases


def _doc_identity(row: dict) -> tuple:
    metadata = row.get("metadata") or {}
    parent = metadata.get("parent_id") or row.get("parent_id")
    chunk = metadata.get("chunk_index", metadata.get("chunk"))
    if parent is not None and chunk is not None:
        return ("parent_chunk", parent, chunk)
    return ("id", evidence_id(row, prefix="document"))


def _usable_rows(retrieval: dict) -> tuple[list[dict], list[dict], list[str]]:
    facts: list[dict] = []
    documents: list[dict] = []
    reasons: list[str] = []
    for row in retrieval.get("facts", []) or []:
        if isinstance(row, dict) and row.get("value") is not None:
            facts.append(row)
        else:
            reasons.append("missing_value")
    seen: set[tuple] = set()
    for row in retrieval.get("documents", []) or []:
        if not isinstance(row, dict) or not document_body(row):
            reasons.append("blank_body")
            continue
        identity = _doc_identity(row)
        if identity in seen:
            reasons.append("duplicate_evidence")
            continue
        seen.add(identity)
        documents.append(row)
    return facts, documents, list(dict.fromkeys(reasons))


def _entity_matches(row: dict, entity: str) -> bool:
    raw = evidence_field(row, "entities")
    entities = raw if isinstance(raw, (list, tuple, set)) else [evidence_field(row, "ticker")]
    return entity in {str(item).upper() for item in entities if item}


def _period_matches(row: dict, periods: tuple[str, ...]) -> bool:
    if not periods:
        return True
    actual = evidence_field(row, "period", evidence_field(row, "as_of"))
    return str(actual) in periods


def _is_stale(row: dict, obligation: EvidenceObligation) -> bool:
    status = str(evidence_field(row, "freshness_status", evidence_field(row, "freshness", ""))).lower()
    required = obligation.freshness_required or bool(row.get("freshness_required"))
    return required and status in _STALE


def _conflicts(facts: list[dict], obligation: EvidenceObligation) -> list[dict]:
    grouped: dict[tuple, dict[str, list[dict]]] = {}
    for row in facts:
        entity = str(evidence_field(row, "ticker", ""))
        metric = str(row.get("metric") or "")
        period = str(evidence_field(row, "period", ""))
        if obligation.entities and entity not in obligation.entities:
            continue
        if obligation.metrics and not any(_metric_match(metric, req)[0] for req in obligation.metrics):
            continue
        if obligation.periods and period not in obligation.periods:
            continue
        unit = str(row.get("unit") or "").upper()
        grouped.setdefault((entity, metric, period), {}).setdefault(unit, []).append(row)
    conflicts: list[dict] = []
    for key, units in grouped.items():
        nonblank = {unit for unit in units if unit}
        if len(nonblank) > 1:
            conflicts.append({"type": "conflicting_units", "slot": key,
                              "units": sorted(nonblank)})
            continue
        values = {repr(row.get("value")) for rows in units.values() for row in rows}
        if len(values) > 1:
            conflicts.append({"type": "conflicting_values", "slot": key,
                              "values": sorted(values)})
            continue
        # Authoritative CompanyFacts folds multiple filed values into one row
        # flagged ``conflict`` (2.2.5.3): disclose it even when a single value
        # survived retrieval, so the grader never hides a filed-value dispute.
        flagged = [row for rows in units.values() for row in rows if row.get("conflict")]
        if flagged:
            reasons = sorted({
                str(row.get("conflict_reason") or "conflicting_filed_values")
                for row in flagged
            })
            conflicts.append({"type": "conflicting_values", "slot": key,
                              "reasons": reasons})
    return conflicts


def _grade_obligation(
    obligation: EvidenceObligation,
    facts: list[dict],
    documents: list[dict],
) -> tuple[CoverageResult, bool]:
    covered: list[str] = []
    missing: list[str] = []
    support: list[str] = []
    reasons: list[str] = []
    alias_seen = False
    entities = obligation.entities or ("*",)

    for modality in obligation.modalities:
        is_document = modality in {"document", "news", "filing"}
        candidates = documents if is_document else facts
        for entity in entities:
            metrics: tuple[str | None, ...] = (
                (None,) if is_document or not obligation.metrics
                else tuple(obligation.metrics)
            )
            periods: tuple[str | None, ...] = (
                tuple(obligation.periods) if obligation.periods else (None,)
            )
            for metric in metrics:
                for period in periods:
                    slot_parts = [modality, entity]
                    if metric is not None:
                        slot_parts.append(metric)
                    if period is not None:
                        slot_parts.append(period)
                    slot = ":".join(slot_parts)
                    compatible: list[dict] = []
                    for row in candidates:
                        if entity != "*" and not _entity_matches(row, entity):
                            continue
                        actual_modality = _row_modality(row, document=is_document)
                        if actual_modality != modality:
                            if modality == "document" and actual_modality in {"filing", "news"}:
                                pass
                            # Filing facts may cover a structured fact obligation.
                            elif not (modality == "fact" and actual_modality == "fact"):
                                continue
                        if not _source_compatible(row, obligation.minimum_source_class):
                            reasons.append("insufficient_source_authority")
                            continue
                        if _is_stale(row, obligation):
                            reasons.append("stale_required_source")
                            continue
                        if period is not None and not _period_matches(row, (period,)):
                            reasons.append("missing_period")
                            continue
                        if metric is not None:
                            metric_match, used_alias = _metric_match(row.get("metric"), metric)
                            if not metric_match:
                                continue
                            alias_seen = alias_seen or used_alias
                        compatible.append(row)
                    if compatible:
                        covered.append(slot)
                        support.extend(
                            evidence_id(row, prefix=modality) for row in compatible)
                    else:
                        missing.append(slot)
                        if candidates and entity != "*" and not any(
                            _entity_matches(row, entity) for row in candidates
                        ):
                            reasons.append("wrong_entity")
                        if is_document:
                            reasons.append("missing_qualitative_evidence")
                        elif metric is not None:
                            reasons.append("missing_metric")
                        else:
                            reasons.append("missing_modality")

    return CoverageResult(
        subquery_id=obligation.subquery_id,
        covered_fields=_dedup(covered),
        missing_fields=_dedup(missing),
        supporting_evidence_ids=_dedup(support),
        reason_codes=_dedup(reasons),
    ), alias_seen


def _choose_action(
    plan: QueryPlan,
    coverage: tuple[CoverageResult, ...],
    reasons: tuple[str, ...],
    alias_seen: bool,
    facts: list[dict],
    documents: list[dict],
) -> CorrectiveAction:
    if "stale_required_source" in reasons or any(
        reason.startswith("conflicting_") for reason in reasons
    ):
        return CorrectiveAction.NONE
    if alias_seen:
        return CorrectiveAction.APPLY_VALIDATED_ALIAS
    if "missing_metric" in reasons and "missing_period" not in reasons and any(
        metric in _METRIC_ALIASES for metric in plan.metrics
    ) and any(
        not plan.tickers or any(_entity_matches(row, ticker) for ticker in plan.tickers)
        for row in facts
    ):
        return CorrectiveAction.APPLY_VALIDATED_ALIAS
    if "wrong_entity" in reasons and any(
        entity.confidence < 0.9 and entity.source != "override" for entity in plan.entities
    ):
        return CorrectiveAction.BROADEN_TICKER_FILTER
    if "missing_qualitative_evidence" in reasons and any(
        evidence_field(row, "parent_id") for row in documents
    ):
        return CorrectiveAction.EXPAND_PARENT_SECTION
    if any(subquery.derived and subquery.id in {
        row.subquery_id for row in coverage if not row.covered
    } for subquery in plan.subqueries):
        # 2.2.4.2 plugs execution into this reserved deterministic seam.
        return CorrectiveAction.RUN_DERIVED_SUBQUERIES
    has_compatible_coverage = any(row.covered_fields for row in coverage)
    if (has_compatible_coverage or "missing_period" in reasons) and any(
        not row.covered for row in coverage
    ):
        return CorrectiveAction.ALTERNATE_INTERNAL_MODALITY
    return CorrectiveAction.NONE


def grade_evidence(plan: QueryPlan, retrieval: dict) -> SufficiencyResult:
    """Grade usable evidence against every obligation in a validated plan."""
    from .stream_events import current_emitter

    emitter = current_emitter()
    started_at = perf_counter()
    if emitter is not None:
        emitter.stage("grade", "started")
    obligations = build_obligations(plan)
    facts, documents, base_reasons = _usable_rows(retrieval or {})
    coverage_rows: list[CoverageResult] = []
    conflicts: list[dict] = []
    alias_seen = False
    for obligation in obligations:
        row, obligation_alias = _grade_obligation(obligation, facts, documents)
        coverage_rows.append(row)
        alias_seen = alias_seen or obligation_alias
        conflicts.extend(_conflicts(facts, obligation))

    for conflict in conflicts:
        base_reasons.append(str(conflict["type"]))
    reasons = _dedup([
        *base_reasons,
        *(reason for row in coverage_rows for reason in row.reason_codes),
    ])
    coverage = tuple(coverage_rows)
    total_fields = sum(len(row.covered_fields) + len(row.missing_fields) for row in coverage)
    covered_fields = sum(len(row.covered_fields) for row in coverage)
    score = covered_fields / total_fields if total_fields else 0.0

    if coverage and all(row.covered for row in coverage) and not conflicts:
        status = SufficiencyStatus.SUFFICIENT
        action = CorrectiveAction.NONE
    else:
        action = _choose_action(plan, coverage, reasons, alias_seen, facts, documents)
        status = (SufficiencyStatus.BORDERLINE
                  if action is not CorrectiveAction.NONE
                  else SufficiencyStatus.MISSING)
    result = SufficiencyResult(
        status=status,
        overall_score=round(score, 4),
        reason_codes=reasons,
        coverage=coverage,
        conflicts=tuple(conflicts),
        allowed_action=action,
    )
    if emitter is not None:
        emitter.stage(
            "grade", "completed",
            elapsed_ms=(perf_counter() - started_at) * 1000,
            reason=result.status.value,
        )
    return result

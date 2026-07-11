"""
tests/test_evidence_grader.py
Offline tests for deterministic evidence sufficiency and corrective actions.
"""

from src.middleware.evidence_grader import (
    CorrectiveAction,
    SufficiencyStatus,
    build_obligations,
    grade_evidence,
)
from src.middleware.config import MiddlewareConfig
from src.middleware.query_plan import (
    QueryEntity,
    QueryPlan,
    QuerySubquery,
    normalize_question,
)


def _plan(
    *,
    entities=("NVDA",),
    metrics=("total_revenue",),
    periods=(),
    intents=("fact_lookup",),
    modes=("facts",),
) -> QueryPlan:
    question = "test question"
    resolved = [
        QueryEntity(ticker=t, resolved_name=None, confidence=1.0,
                    source="known_ticker", mention=t, start=i)
        for i, t in enumerate(entities)
    ]
    subquery = QuerySubquery(
        id="sq0", text=question, entity_tickers=tuple(entities),
        intents=tuple(intents), metrics=tuple(metrics), periods=tuple(periods),
        retrieval_modes=tuple(modes),
    )
    return QueryPlan(
        original_question=question,
        retrieval_query=question,
        normalized_question=normalize_question(question),
        entities=resolved,
        intents=list(intents),
        metrics=list(metrics),
        periods=list(periods),
        subqueries=[subquery],
        primary_intent=intents[0],
    ).validate()


def _fact(**overrides):
    row = {
        "evidence_id": "f1", "ticker": "NVDA", "metric": "total_revenue",
        "value": 0, "period": "FY2025", "unit": "USD",
        "source_type": "sec_10k", "freshness_status": "fresh",
    }
    row.update(overrides)
    return row


def _doc(**overrides):
    row = {
        "id": "d1", "document": "Management cited data-center demand.",
        "metadata": {"ticker": "NVDA", "source_type": "sec_10k",
                     "period": "FY2025", "freshness_status": "fresh"},
    }
    row.update(overrides)
    return row


def test_obligations_are_built_from_validated_subquery_fields():
    plan = _plan(
        entities=("NVDA", "AMD"), metrics=("gross_margin",),
        periods=("FY2025",), intents=("comparison",), modes=("facts",),
    )

    obligations = build_obligations(plan)

    assert [(o.subquery_id, o.entities, o.metrics, o.periods, o.modalities)
            for o in obligations] == [
        ("sq0", ("NVDA", "AMD"), ("gross_margin",), ("FY2025",), ("fact",))
    ]
    assert obligations[0].operations == ("compare",)


def test_zero_valued_exact_fact_is_sufficient():
    result = grade_evidence(
        _plan(periods=("FY2025",)), {"facts": [_fact()], "documents": []}
    )

    assert result.status is SufficiencyStatus.SUFFICIENT
    assert result.coverage[0].supporting_evidence_ids == ("f1",)
    assert result.allowed_action is CorrectiveAction.NONE


def test_count_alone_never_yields_sufficient_for_unrelated_broad_facts():
    facts = [
        _fact(evidence_id=f"f{i}", ticker="AMD", metric=f"other_{i}")
        for i in range(10)
    ]

    result = grade_evidence(_plan(), {"facts": facts, "documents": []})

    assert result.status is not SufficiencyStatus.SUFFICIENT
    assert "wrong_entity" in result.reason_codes
    assert "missing_metric" in result.reason_codes


def test_blank_body_and_wrong_ticker_do_not_cover_document_obligation():
    plan = _plan(metrics=(), intents=("risk",), modes=("documents",))
    result = grade_evidence(plan, {
        "facts": [],
        "documents": [
            _doc(document="   "),
            _doc(id="d2", metadata={"ticker": "AMD", "source_type": "sec_10k"}),
        ],
    })

    assert result.status is SufficiencyStatus.MISSING
    assert "blank_body" in result.reason_codes
    assert "wrong_entity" in result.reason_codes
    assert "missing_qualitative_evidence" in result.reason_codes


def test_missing_period_is_not_covered_and_selects_one_internal_action():
    result = grade_evidence(
        _plan(periods=("FY2025",)),
        {"facts": [_fact(period="FY2024")], "documents": []},
    )

    assert result.status is SufficiencyStatus.BORDERLINE
    assert result.allowed_action is CorrectiveAction.ALTERNATE_INTERNAL_MODALITY
    assert result.reason_codes.count("missing_period") == 1


def test_stale_required_source_is_missing_without_retry():
    result = grade_evidence(
        _plan(periods=("FY2025",)),
        {"facts": [_fact(freshness_status="stale", freshness_required=True)],
         "documents": []},
    )

    assert result.status is SufficiencyStatus.MISSING
    assert "stale_required_source" in result.reason_codes
    assert result.allowed_action is CorrectiveAction.NONE


def test_mixed_fact_and_document_cover_compound_plan():
    plan = _plan(
        periods=("FY2025",), intents=("fact_lookup", "explanation"),
        modes=("facts", "documents"),
    )
    result = grade_evidence(
        plan, {"facts": [_fact()], "documents": [_doc()]}
    )

    assert result.status is SufficiencyStatus.SUFFICIENT
    assert result.covered_subqueries == ("sq0",)
    assert result.missing_subqueries == ()


def test_every_requested_metric_and_period_requires_coverage():
    plan = _plan(metrics=("total_revenue", "gross_margin"),
                 periods=("FY2024", "FY2025"))
    partial = grade_evidence(plan, {"facts": [
        _fact(metric="total_revenue", period="FY2025"),
    ], "documents": []})

    assert partial.status is SufficiencyStatus.BORDERLINE
    assert len(partial.coverage[0].covered_fields) == 1
    assert len(partial.coverage[0].missing_fields) == 3

    complete = grade_evidence(plan, {"facts": [
        _fact(evidence_id="r24", metric="total_revenue", period="FY2024"),
        _fact(evidence_id="r25", metric="total_revenue", period="FY2025"),
        _fact(evidence_id="m24", metric="gross_margin", period="FY2024"),
        _fact(evidence_id="m25", metric="gross_margin", period="FY2025"),
    ], "documents": []})
    assert complete.status is SufficiencyStatus.SUFFICIENT


def test_projection_requires_and_accepts_estimate_evidence():
    plan = _plan(metrics=("estimate_eps_next_y",), intents=("projection",),
                 modes=("facts",))
    result = grade_evidence(plan, {"facts": [{
        "evidence_id": "estimate", "ticker": "NVDA",
        "metric": "estimate_eps_next_y", "value": 4.2,
        "source_type": "estimates", "freshness_status": "fresh",
    }], "documents": []})

    assert build_obligations(plan)[0].modalities == ("estimate",)
    assert result.status is SufficiencyStatus.SUFFICIENT


def test_macro_route_requires_and_accepts_macro_evidence_once():
    plan = _plan(entities=(), metrics=("GDP",), intents=("fact_lookup",),
                 modes=("facts", "macro"))
    result = grade_evidence(plan, {"facts": [{
        "evidence_id": "fred", "ticker": "MACRO", "metric": "GDP",
        "value": 3.1, "source_type": "fred",
    }], "documents": []})

    assert build_obligations(plan)[0].modalities == ("macro",)
    assert result.status is SufficiencyStatus.SUFFICIENT


def test_conflicting_units_make_required_evidence_missing():
    result = grade_evidence(
        _plan(periods=("FY2025",)),
        {"facts": [_fact(evidence_id="usd", unit="USD"),
                   _fact(evidence_id="eur", unit="EUR")], "documents": []},
    )

    assert result.status is SufficiencyStatus.MISSING
    assert "conflicting_units" in result.reason_codes
    assert result.conflicts


def test_conflicting_values_for_same_slot_are_missing():
    result = grade_evidence(
        _plan(periods=("FY2025",)),
        {"facts": [_fact(evidence_id="a", value=10),
                   _fact(evidence_id="b", value=11)], "documents": []},
    )

    assert result.status is SufficiencyStatus.MISSING
    assert "conflicting_values" in result.reason_codes


def test_missing_known_metric_alias_selects_alias_correction():
    result = grade_evidence(
        _plan(metrics=("total_revenue",)),
        {"facts": [_fact(metric="gross_margin")], "documents": []},
    )

    assert result.status is SufficiencyStatus.BORDERLINE
    assert result.allowed_action is CorrectiveAction.APPLY_VALIDATED_ALIAS


def test_low_confidence_inferred_wrong_ticker_selects_broaden_filter():
    plan = _plan()
    entity = plan.entities[0]
    plan.entities[0] = QueryEntity(
        ticker=entity.ticker, resolved_name=entity.resolved_name, confidence=0.6,
        source="fuzzy", mention=entity.mention, start=entity.start,
    )
    result = grade_evidence(
        plan, {"facts": [_fact(ticker="AMD")], "documents": []}
    )

    assert result.status is SufficiencyStatus.BORDERLINE
    assert result.allowed_action is CorrectiveAction.BROADEN_TICKER_FILTER


def test_parent_hit_with_missing_period_selects_section_expansion():
    plan = _plan(metrics=(), periods=("FY2025",), intents=("risk",),
                 modes=("documents",))
    result = grade_evidence(plan, {
        "facts": [],
        "documents": [_doc(metadata={
            "ticker": "NVDA", "source_type": "sec_10k", "period": "FY2024",
            "parent_id": "filing", "chunk_index": 2,
        })],
    })

    assert result.status is SufficiencyStatus.BORDERLINE
    assert result.allowed_action is CorrectiveAction.EXPAND_PARENT_SECTION


def test_missing_derived_subquery_selects_reserved_2_2_4_2_seam():
    plan = _plan()
    plan.subqueries.append(QuerySubquery(
        id="sq1", text="derived", entity_tickers=("NVDA",),
        metrics=("total_revenue",), retrieval_modes=("facts",),
        derived=True, parent_id="sq0",
    ))
    plan.validate()

    result = grade_evidence(plan, {"facts": [], "documents": []})

    assert result.status is SufficiencyStatus.BORDERLINE
    assert result.allowed_action is CorrectiveAction.RUN_DERIVED_SUBQUERIES


def test_duplicate_parent_chunks_do_not_create_extra_coverage():
    plan = _plan(metrics=(), intents=("risk",), modes=("documents",))
    first = _doc(id="p#0", metadata={"ticker": "NVDA", "source_type": "sec_10k",
                                     "parent_id": "p", "chunk_index": 0})
    duplicate = _doc(id="copy", metadata={"ticker": "NVDA", "source_type": "sec_10k",
                                          "parent_id": "p", "chunk_index": 0})

    result = grade_evidence(plan, {"facts": [], "documents": [first, duplicate]})

    assert result.status is SufficiencyStatus.SUFFICIENT
    assert result.coverage[0].supporting_evidence_ids == ("p#0",)
    assert "duplicate_evidence" in result.reason_codes


def test_corrective_config_defaults_off_and_retry_limit_is_hard_clamped(tmp_path):
    missing = tmp_path / "missing.yaml"
    defaults = MiddlewareConfig(config_path=missing)
    assert defaults.enable_evidence_sufficiency is False
    assert defaults.enable_corrective_retry is False
    assert defaults.max_corrective_retries == 1

    config_file = tmp_path / "middleware.yaml"
    config_file.write_text("max_corrective_retries: 9\n", encoding="utf-8")
    assert MiddlewareConfig(config_path=config_file).max_corrective_retries == 1

    config_file.write_text("max_corrective_retries: -3\n", encoding="utf-8")
    assert MiddlewareConfig(config_path=config_file).max_corrective_retries == 0

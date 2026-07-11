"""Offline tests for the query-plan contract and quality gate (2.2.3.1).

Covers QueryPlan.validate() invariants, the legacy adapter projection, and the
checked-in compound-query micro-F1 gate. Everything is in-memory: an empty
symbol catalog keeps the parser offline and deterministic (the fixture uses
companies from IntentParser's local map). No network, Chroma, middleware, or
model is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "compound_queries.json"


def _empty_catalog(tmp_path) -> Path:
    path = tmp_path / "symbol_catalog.json"
    path.write_text(
        json.dumps(
            {"generated_at": "2099-01-01T00:00:00+00:00", "ttl_hours": 168, "entries": []}
        ),
        encoding="utf-8",
    )
    return path


def _parser(tmp_path):
    from src.middleware.intent_parser import IntentParser
    from src.middleware.symbol_resolver import SymbolResolver

    return IntentParser(resolver=SymbolResolver(catalog_path=_empty_catalog(tmp_path)))


def _entity(ticker, start=0, source="known_ticker"):
    from src.middleware.query_plan import QueryEntity

    return QueryEntity(
        ticker=ticker, resolved_name=ticker, confidence=1.0,
        source=source, mention=ticker, start=start,
    )


def _valid_plan(**overrides):
    from src.middleware.query_plan import QueryPlan, QuerySubquery

    base = dict(
        original_question="What is NVDA revenue?",
        retrieval_query="What is NVDA revenue?",
        entities=[_entity("NVDA")],
        intents=["fact_lookup"],
        metrics=["total_revenue"],
        periods=[],
        primary_intent="fact_lookup",
    )
    base.update(overrides)
    sq0 = QuerySubquery(
        id="sq0",
        text=base["retrieval_query"],
        entity_tickers=tuple(e.ticker for e in base["entities"]),
        intents=tuple(base["intents"]),
        metrics=tuple(base["metrics"]),
        periods=tuple(base["periods"]),
        retrieval_modes=("facts",),
    )
    base.setdefault("subqueries", [sq0])
    return QueryPlan(**base)


class TestQueryPlanValidate:
    """QueryPlan.validate() enforces every documented invariant."""

    def test_valid_plan_passes(self):
        plan = _valid_plan()
        assert plan.validate() is plan, "A well-formed plan validates and returns self"

    def test_blank_original_question_rejected(self):
        from src.middleware.query_plan import QueryPlanError

        plan = _valid_plan(original_question="   ")
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "blank_original_question" in exc.value.reason_codes

    def test_blank_retrieval_query_rejected(self):
        from src.middleware.query_plan import QueryPlanError

        plan = _valid_plan(retrieval_query="")
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "blank_retrieval_query" in exc.value.reason_codes

    def test_lowercase_entity_rejected(self):
        from src.middleware.query_plan import QueryPlanError

        plan = _valid_plan(entities=[_entity("nvda")])
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "entity_not_uppercase" in exc.value.reason_codes

    def test_duplicate_entity_rejected(self):
        from src.middleware.query_plan import QueryPlanError

        plan = _valid_plan(entities=[_entity("NVDA", start=0), _entity("NVDA", start=5)])
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "entity_duplicate" in exc.value.reason_codes

    def test_entities_out_of_mention_order_rejected(self):
        from src.middleware.query_plan import QueryPlanError

        plan = _valid_plan(entities=[_entity("AMD", start=10), _entity("NVDA", start=2)])
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "entity_out_of_mention_order" in exc.value.reason_codes

    def test_override_entity_first_is_allowed(self):
        """An override (start=-1) ahead of a located entity is valid ordering."""
        plan = _valid_plan(
            entities=[_entity("NVDA", start=-1, source="override"), _entity("AMD", start=8)]
        )
        assert plan.validate() is plan

    def test_subquery_count_bounds(self):
        from src.middleware.query_plan import QueryPlanError, QuerySubquery

        plan = _valid_plan()
        plan.subqueries = []
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "subquery_count_out_of_bounds" in exc.value.reason_codes

        plan = _valid_plan()
        extra = QuerySubquery(id="sqx", text="x", derived=True, parent_id="sq0")
        plan.subqueries = [plan.subqueries[0], extra, extra, extra]
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "subquery_count_out_of_bounds" in exc.value.reason_codes

    def test_sq0_must_be_retrieval_query(self):
        from src.middleware.query_plan import QueryPlanError

        plan = _valid_plan()
        plan.subqueries[0] = plan.subqueries[0].__class__(
            id="sq0", text="something else", retrieval_modes=("facts",)
        )
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "sq0_not_retrieval_query" in exc.value.reason_codes

    def test_invalid_retrieval_mode_rejected(self):
        from src.middleware.query_plan import QueryPlanError, QuerySubquery

        plan = _valid_plan()
        plan.subqueries[0] = QuerySubquery(
            id="sq0", text=plan.retrieval_query, retrieval_modes=("facts", "web"),
        )
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "invalid_retrieval_mode" in exc.value.reason_codes

    def test_derived_metric_and_period_drift_rejected(self):
        from src.middleware.query_plan import QueryPlanError, QuerySubquery

        plan = _valid_plan(metrics=["total_revenue"], periods=["q1 2026"])
        plan.subqueries.append(
            QuerySubquery(
                id="sq1", text="net income q2", metrics=("net_income",),
                periods=("q2 2026",), derived=True, parent_id="sq0",
            )
        )
        with pytest.raises(QueryPlanError) as exc:
            plan.validate()
        assert "derived_metric_drift" in exc.value.reason_codes
        assert "derived_period_drift" in exc.value.reason_codes

    def test_derived_subquery_narrowing_allowed(self):
        """A derived subquery that only narrows the plan is valid."""
        from src.middleware.query_plan import QuerySubquery

        plan = _valid_plan(
            entities=[_entity("NVDA", start=8), _entity("AMD", start=17)],
            metrics=["total_revenue", "net_income"],
        )
        plan.subqueries.append(
            QuerySubquery(
                id="sq1", text="NVDA revenue", entity_tickers=("NVDA",),
                metrics=("total_revenue",), derived=True, parent_id="sq0",
            )
        )
        assert plan.validate() is plan


class TestLegacyAdapter:
    """to_legacy_intent() projects onto the current parse() dict shape."""

    def test_projects_primary_entity_and_scalars(self, tmp_path):
        parser = _parser(tmp_path)
        plan = parser.parse_plan("What is Meta's PE ratio in Q1 2026?")
        legacy = plan.to_legacy_intent()
        assert legacy["ticker"] == "META"
        assert legacy["question_type"] == "fact_lookup"
        assert legacy["timeframe"] == "q1 2026"
        assert legacy["timeframe_type"] == "quarter"
        assert set(legacy.keys()) == {
            "ticker", "metrics", "question_type", "timeframe", "timeframe_type",
            "original_question", "ticker_confidence", "resolved_name", "ticker_source",
        }

    def test_adapter_does_not_mutate_plan(self, tmp_path):
        parser = _parser(tmp_path)
        plan = parser.parse_plan("Compare NVDA and AMD revenue")
        before = (list(plan.tickers), list(plan.metrics), list(plan.intents))
        plan.to_legacy_intent()
        after = (list(plan.tickers), list(plan.metrics), list(plan.intents))
        assert before == after, "Legacy projection must not mutate the plan"

    def test_no_entity_projects_none(self, tmp_path):
        parser = _parser(tmp_path)
        plan = parser.parse_plan("What is the weather today?")
        legacy = plan.to_legacy_intent()
        assert legacy["ticker"] is None
        assert legacy["ticker_source"] == "none"
        assert legacy["ticker_confidence"] == 0.0


def _micro_f1(pairs):
    """Micro-averaged F1 over (predicted_set, gold_set) label pairs."""
    tp = fp = fn = 0
    for predicted, gold in pairs:
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class TestCompoundQueryQualityGate:
    """Micro-F1 >= 0.90 on the checked-in compound-query fixture, zero mutations."""

    def _run(self, tmp_path):
        parser = _parser(tmp_path)
        data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        cases = data["cases"]
        assert len(cases) >= 12, "Fixture must be a substantive compound-query set"
        rows = []
        for case in cases:
            plan = parser.parse_plan(
                case["question"],
                retrieval_query=case.get("retrieval_query"),
                override_ticker=case.get("override_ticker"),
            )
            rows.append((case, plan))
        return rows

    def test_raw_question_never_mutated(self, tmp_path):
        for case, plan in self._run(tmp_path):
            assert plan.original_question == case["question"], (
                f"{case['id']}: raw question was mutated"
            )

    def test_micro_f1_per_category_and_pooled(self, tmp_path):
        rows = self._run(tmp_path)
        fields = {
            "entities": lambda p: set(p.tickers),
            "intents": lambda p: set(p.intents),
            "metrics": lambda p: set(p.metrics),
            "periods": lambda p: set(p.periods),
        }
        pooled = []
        for name, extract in fields.items():
            pairs = [
                (extract(plan), set(case["expected"][name])) for case, plan in rows
            ]
            score = _micro_f1(pairs)
            assert score >= 0.90, f"{name} micro-F1 {score:.3f} below 0.90 gate"
            # Namespace labels so pooling cannot cross-credit categories.
            pooled.extend(
                ({f"{name}:{v}" for v in pred}, {f"{name}:{v}" for v in gold})
                for pred, gold in pairs
            )
        assert _micro_f1(pooled) >= 0.90, "Pooled micro-F1 below 0.90 gate"


# ══════════════════════════════════════════════════════════════════════
# 2.2.4.2 — Selective query decomposition + derived-query drift validation
# ══════════════════════════════════════════════════════════════════════


def _plan(question, *, entities=(), metrics=(), intents=("general",), periods=()):
    """Build a validated sq0-only plan for decomposition tests."""
    from src.middleware.query_plan import (
        QueryPlan,
        QuerySubquery,
        normalize_question,
    )

    ents = [_entity(t, start=i) for i, t in enumerate(entities)]
    sq0 = QuerySubquery(
        id="sq0", text=question,
        entity_tickers=tuple(e.ticker for e in ents),
        intents=tuple(intents), metrics=tuple(metrics), periods=tuple(periods),
        retrieval_modes=("facts",) if metrics else (),
    )
    return QueryPlan(
        original_question=question, retrieval_query=question,
        normalized_question=normalize_question(question),
        entities=ents, intents=list(intents), metrics=list(metrics),
        periods=list(periods), subqueries=[sq0], primary_intent=intents[0],
    ).validate()


class TestDecomposePlan:
    """decompose_plan only splits genuinely compound / low-coverage plans."""

    def test_simple_fact_lookup_yields_no_derived(self):
        from src.middleware.query_plan import decompose_plan

        plan = _plan("What is NVDA revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"])
        assert decompose_plan(plan) == []

    def test_single_qualitative_topic_yields_no_derived(self):
        from src.middleware.query_plan import decompose_plan

        plan = _plan("Why did NVDA stock drop", entities=["NVDA"],
                     intents=["explanation"])
        assert decompose_plan(plan) == []

    def test_fact_plus_risk_splits_by_modality(self):
        from src.middleware.query_plan import decompose_plan

        plan = _plan("NVDA revenue and risk factors", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup", "risk"])
        derived = decompose_plan(plan)
        assert [s.id for s in derived] == ["sq1", "sq2"]
        modes = {m for s in derived for m in s.retrieval_modes}
        assert "facts" in modes and "documents" in modes
        assert all(s.derived and s.parent_id == "sq0" for s in derived)
        assert all(s.derivation_source == "deterministic" for s in derived)
        # Each derived subquery records the obligations it covers.
        assert all(s.covers_obligations for s in derived)

    def test_comparison_splits_one_lookup_per_entity(self):
        from src.middleware.query_plan import decompose_plan

        plan = _plan("Compare NVDA and AMD revenue",
                     entities=["NVDA", "AMD"], metrics=["total_revenue"],
                     intents=["comparison"])
        derived = decompose_plan(plan)
        assert [s.entity_tickers for s in derived] == [("NVDA",), ("AMD",)]
        assert all(s.reason_code == "decompose_entity" for s in derived)

    def test_two_tickers_split_per_entity(self):
        from src.middleware.query_plan import decompose_plan

        plan = _plan("NVDA revenue FY2024 versus AMD revenue FY2025",
                     entities=["NVDA", "AMD"], metrics=["total_revenue"],
                     intents=["comparison"], periods=["FY2024", "FY2025"])
        derived = decompose_plan(plan)
        assert [s.entity_tickers for s in derived] == [("NVDA",), ("AMD",)]
        # Derived periods only ever narrow the parent plan.
        for s in derived:
            assert set(s.periods) <= {"FY2024", "FY2025"}

    def test_macro_plus_company_splits_macro_and_company(self):
        from src.middleware.query_plan import decompose_plan

        plan = _plan("How does inflation affect NVDA", entities=["NVDA"],
                     intents=["explanation"])
        derived = decompose_plan(plan)
        reasons = {s.reason_code for s in derived}
        assert reasons == {"decompose_macro", "decompose_company"}
        macro = next(s for s in derived if s.reason_code == "decompose_macro")
        assert macro.retrieval_modes == ("macro",) and macro.entity_tickers == ()

    def test_at_most_two_derived_even_with_three_entities(self):
        from src.middleware.query_plan import decompose_plan

        plan = _plan("Compare NVDA AMD INTC revenue",
                     entities=["NVDA", "AMD", "INTC"], metrics=["total_revenue"],
                     intents=["comparison"])
        derived = decompose_plan(plan)
        assert len(derived) == 2

    def test_decompose_does_not_mutate_plan(self):
        from src.middleware.query_plan import decompose_plan

        plan = _plan("NVDA revenue and risk", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup", "risk"])
        before = [s.id for s in plan.subqueries]
        decompose_plan(plan)
        assert [s.id for s in plan.subqueries] == before == ["sq0"]

    def test_decompose_is_idempotent_on_already_derived_plan(self):
        from src.middleware.query_plan import (
            attach_derived_subqueries,
            decompose_plan,
        )

        plan = _plan("NVDA revenue and risk", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup", "risk"])
        extended = attach_derived_subqueries(plan, decompose_plan(plan))
        assert decompose_plan(extended) == []


class TestAttachDerivedSubqueries:
    def test_attach_keeps_plan_capped_and_valid(self):
        from src.middleware.query_plan import (
            attach_derived_subqueries,
            decompose_plan,
        )

        plan = _plan("Compare NVDA AMD INTC revenue",
                     entities=["NVDA", "AMD", "INTC"], metrics=["total_revenue"],
                     intents=["comparison"])
        extended = attach_derived_subqueries(plan, decompose_plan(plan))
        assert len(extended.subqueries) == 3  # sq0 + at most 2 derived
        extended.validate()  # still valid, no drift

    def test_attach_fails_soft_on_invalid_derived(self):
        from src.middleware.query_plan import QuerySubquery, attach_derived_subqueries

        plan = _plan("NVDA revenue", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"])
        # A derived subquery that drifts (new metric) cannot be attached.
        bad = QuerySubquery(
            id="sq1", text="NVDA net income", entity_tickers=("NVDA",),
            metrics=("net_income",), derived=True, parent_id="sq0")
        assert attach_derived_subqueries(plan, [bad]) is plan


class TestDerivedDriftValidation:
    """Every derived field must come from the validated plan or known aliases."""

    def _base(self):
        return _plan("NVDA revenue FY2025", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup"],
                     periods=["FY2025"])

    def _sub(self, **over):
        from src.middleware.query_plan import QuerySubquery

        base = dict(id="sq1", text="NVDA revenue FY2025",
                    entity_tickers=("NVDA",), metrics=("total_revenue",),
                    periods=("FY2025",), retrieval_modes=("facts",),
                    derived=True, parent_id="sq0")
        base.update(over)
        return QuerySubquery(**base)

    def test_valid_narrowing_subquery_passes(self):
        from src.middleware.query_plan import validate_derived_subquery

        ok, reasons = validate_derived_subquery(self._base(), self._sub())
        assert ok and reasons == []

    def test_alias_metric_is_allowed(self):
        from src.middleware.query_plan import validate_derived_subquery

        # "revenues" is a known grader alias of the plan's total_revenue.
        ok, _ = validate_derived_subquery(
            self._base(), self._sub(metrics=("revenues",), text="NVDA revenues FY2025"))
        assert ok

    def test_invented_ticker_rejected(self):
        from src.middleware.query_plan import validate_derived_subquery

        ok, reasons = validate_derived_subquery(
            self._base(), self._sub(entity_tickers=("TSLA",), text="TSLA revenue FY2025"))
        assert not ok and "drift_invented_entity" in reasons

    def test_new_metric_rejected(self):
        from src.middleware.query_plan import validate_derived_subquery

        ok, reasons = validate_derived_subquery(
            self._base(), self._sub(metrics=("gross_margin",)))
        assert not ok and "drift_new_metric" in reasons

    def test_new_period_rejected(self):
        from src.middleware.query_plan import validate_derived_subquery

        ok, reasons = validate_derived_subquery(
            self._base(), self._sub(periods=("FY2099",), text="NVDA revenue FY2099"))
        assert not ok and "drift_new_period" in reasons

    def test_new_number_in_text_rejected(self):
        from src.middleware.query_plan import validate_derived_subquery

        ok, reasons = validate_derived_subquery(
            self._base(), self._sub(text="NVDA revenue above 500"))
        assert not ok and "drift_new_number" in reasons

    def test_empty_or_stopword_query_rejected(self):
        from src.middleware.query_plan import validate_derived_subquery

        ok, reasons = validate_derived_subquery(
            self._base(), self._sub(text="the of a", entity_tickers=(), metrics=(),
                                    periods=(), retrieval_modes=()))
        assert not ok and "drift_empty_query" in reasons

    def test_invalid_mode_rejected(self):
        from src.middleware.query_plan import validate_derived_subquery

        ok, reasons = validate_derived_subquery(
            self._base(), self._sub(retrieval_modes=("web",)))
        assert not ok and "drift_invalid_mode" in reasons

    def test_broader_topic_intent_rejected(self):
        from src.middleware.query_plan import validate_derived_subquery

        ok, reasons = validate_derived_subquery(
            self._base(), self._sub(intents=("sentiment",)))
        assert not ok and "drift_broader_topic" in reasons


class TestSelectDerivedSubqueries:
    """Shared validator/dedup/cap for deterministic + planner-proposed drafts."""

    def _plan(self):
        return _plan("NVDA revenue and risk factors", entities=["NVDA"],
                     metrics=["total_revenue"], intents=["fact_lookup", "risk"])

    def _cand(self, cid, **over):
        from src.middleware.query_plan import QuerySubquery

        base = dict(id=cid, text="NVDA revenue", entity_tickers=("NVDA",),
                    metrics=("total_revenue",), retrieval_modes=("facts",),
                    derived=True, parent_id="sq0")
        base.update(over)
        return QuerySubquery(**base)

    def test_duplicate_obligations_deduped(self):
        from src.middleware.query_plan import select_derived_subqueries

        # Two candidates covering the identical obligation set.
        cands = [self._cand("a"), self._cand("b")]
        accepted, reasons = select_derived_subqueries(
            self._plan(), cands, source="deterministic")
        assert len(accepted) == 1
        assert "drift_duplicate_obligations" in reasons

    def test_caps_at_two_and_reids(self):
        from src.middleware.query_plan import select_derived_subqueries

        cands = [
            self._cand("a", retrieval_modes=("facts",)),
            self._cand("b", retrieval_modes=("documents",), metrics=(),
                       text="NVDA risk"),
            self._cand("c", entity_tickers=("NVDA",), retrieval_modes=("macro",),
                       metrics=(), text="NVDA macro"),
        ]
        accepted, _ = select_derived_subqueries(
            self._plan(), cands, source="planner")
        assert [s.id for s in accepted] == ["sq1", "sq2"]
        assert all(s.derivation_source == "planner" for s in accepted)

    def test_invalid_candidate_reason_recorded(self):
        from src.middleware.query_plan import select_derived_subqueries

        cands = [self._cand("a", entity_tickers=("TSLA",), text="TSLA revenue")]
        accepted, reasons = select_derived_subqueries(
            self._plan(), cands, source="deterministic")
        assert accepted == []
        assert "drift_invented_entity" in reasons

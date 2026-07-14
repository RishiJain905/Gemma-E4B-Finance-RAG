"""
tests/test_answer_validator.py
Deterministic, offline tests for src/middleware/answer_validator.py (2.2.4.3).

Every test uses fixed answer strings and evidence dicts — no network, no model,
no live dependency. Covers the spec matrix: exact/rounded display formats,
percentages, negatives, billions, ratios, zero, dates, different periods,
incompatible currencies, derived calculations, multiple citations, unknown ids,
uncited numbers, and the validator exception path.
"""

from decimal import Decimal

from src.middleware.answer_validator import (
    AnswerValidation,
    _extract_numbers,
    validate_answer,
)
from src.middleware.evidence import assign_evidence_ids, build_evidence_items


def _ledger(facts, documents=None):
    return assign_evidence_ids(build_evidence_items(facts, documents or []))


def _fact(metric, value, unit, *, ticker="NVDA", period="2026-Q1",
          source_type="sec_10q"):
    return {"metric": metric, "value": value, "unit": unit, "ticker": ticker,
            "period": period, "source_type": source_type}


def _claim_map(report):
    return {c.text: (c.status, c.reason) for c in report.claims}


# ── Number extraction ──────────────────────────────────────────────────────

class TestNumberExtraction:
    def test_currency_percent_ratio_scaled_suffix(self):
        kinds = {c.text: c.kind for c in _extract_numbers(
            "$26.0 billion, 42.5%, 15.9x, 26.0 billion, 26B")}
        assert kinds["$26.0 billion"] == "currency"
        assert kinds["42.5%"] == "percent"
        assert kinds["15.9x"] == "ratio"
        assert kinds["26.0 billion"] == "scaled"
        assert kinds["26B"] == "scaled"

    def test_magnitude_and_precision(self):
        (claim,) = _extract_numbers("$26.0 billion")
        assert claim.magnitude == Decimal("26.0") * (Decimal(10) ** 9)
        # One displayed decimal at billion scale -> precision unit of 0.1 billion.
        assert claim.precision == Decimal("1e8")

    def test_dates_years_and_quarters_are_not_numbers(self):
        # Bare years, quarter labels, and hyphenated filing types carry no unit.
        assert _extract_numbers("In 2026 Q1 the 10-K and 8-K filings") == []

    def test_negative_and_zero(self):
        neg = _extract_numbers("-$1.2 billion")[0]
        assert neg.magnitude < 0
        zero = _extract_numbers("$0")[0]
        assert zero.magnitude == Decimal(0)


# ── Numeric-claim validation ───────────────────────────────────────────────

class TestNumericValidation:
    def test_exact_and_rounded_currency_supported(self):
        ledger = _ledger([_fact("total_revenue", 26.04, "billion_usd")])
        # Rounded display (26.0 rounds from 26.04).
        assert validate_answer("Revenue was $26.0 billion [E1].",
                               ledger).numeric_claims_supported == 1
        # Exact full magnitude.
        assert validate_answer("Revenue was $26,040,000,000 [E1].",
                               ledger).numeric_claims_supported == 1

    def test_rounded_out_of_tolerance_is_unsupported(self):
        ledger = _ledger([_fact("total_revenue", 26.04, "billion_usd")])
        report = validate_answer("Revenue was $26.1 billion [E1].", ledger)
        assert report.numeric_claims_unsupported == 1
        assert _claim_map(report)["$26.1 billion"] == ("unsupported", "value_mismatch")

    def test_percentage_and_ratio_bridge(self):
        ledger = _ledger([
            _fact("gross_margin", 0.70, "ratio"),
            _fact("forward_pe", 15.9, "x", ticker="META", period="2026"),
        ])
        assert validate_answer("Gross margin was 70% [E1].",
                               ledger).numeric_claims_supported == 1
        assert validate_answer("It trades at 15.9x [E2].",
                               ledger).numeric_claims_supported == 1

    def test_negative_value_supported(self):
        ledger = _ledger([_fact("net_income", -1.2, "billion_usd")])
        assert validate_answer("Net income was -$1.2 billion [E1].",
                               ledger).numeric_claims_supported == 1

    def test_zero_value_supported(self):
        ledger = _ledger([_fact("cash", 0.0, "usd")])
        assert validate_answer("Cash was $0 [E1].",
                               ledger).numeric_claims_supported == 1

    def test_billions_suffix_supported(self):
        ledger = _ledger([_fact("total_revenue", 5.8e9, "usd", ticker="AMD")])
        report = validate_answer("Revenue was $5.8B [E1].", ledger)
        assert report.numeric_claims_supported == 1

    def test_incompatible_currency_unsupported(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer("Revenue was €26.0 billion [E1].", ledger)
        assert _claim_map(report)["€26.0 billion"] == ("unsupported", "unit_mismatch")

    def test_different_period_unsupported(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd", period="2026-Q1")])
        report = validate_answer("In 2025 revenue was $26.0 billion [E1].", ledger)
        assert _claim_map(report)["$26.0 billion"] == ("unsupported", "period_mismatch")

    def test_multi_period_sentence_no_false_positive(self):
        # A sentence naming several years must not falsely flag a period mismatch
        # when the item's year is among them.
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd", period="2026-Q1")])
        report = validate_answer(
            "Compared with 2025, 2026 revenue was $26.0 billion [E1].", ledger)
        assert report.numeric_claims_supported == 1

    def test_entity_mismatch_unsupported(self):
        # Cross-ticker confusion within the retrieved set: the sentence is about
        # NVDA but cites AMD's item (both are ledger entities).
        ledger = _ledger([
            _fact("total_revenue", 26.0, "billion_usd", ticker="NVDA"),
            _fact("total_revenue", 5.8, "billion_usd", ticker="AMD"),
        ])
        report = validate_answer("NVDA revenue was $5.8 billion [E2].", ledger)
        assert _claim_map(report)["$5.8 billion"] == ("unsupported", "entity_mismatch")

    def test_multiple_citations_in_one_sentence(self):
        ledger = _ledger([
            _fact("forward_pe", 15.9, "x", ticker="META", period="2026"),
            _fact("total_revenue", 5.8, "billion_usd", ticker="AMD"),
        ])
        report = validate_answer(
            "META trades at 15.9x while AMD revenue was $5.8 billion [E1][E2].",
            ledger)
        assert report.numeric_claims_supported == 2
        assert report.numeric_claims_unsupported == 0

    def test_uncited_but_present_is_ambiguous(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer("Revenue was $26.0 billion.", ledger)
        assert report.numeric_claims_ambiguous == 1
        assert _claim_map(report)["$26.0 billion"] == ("ambiguous", "uncited_but_present")

    def test_uncited_and_absent_is_unsupported(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer("Margins were 88.5%.", ledger)
        assert _claim_map(report)["88.5%"] == ("unsupported", "uncited")

    def test_cited_id_absent_from_ledger_is_unsupported(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer("Revenue was $26.0 billion [E9].", ledger)
        assert _claim_map(report)["$26.0 billion"] == ("unsupported", "no_matching_evidence")

    def test_derived_calculation_supports_claim(self):
        ledger = _ledger([_fact("total_revenue", 5.8, "billion_usd", ticker="AMD")])
        calcs = [{"id": "calc1", "result": 41.5, "unit": "billion_usd",
                  "formula": "nvda + amd"}]
        report = validate_answer("Combined revenue is $41.5 billion [E1].",
                                ledger, calculations=calcs)
        claim = report.claims[0]
        assert claim.status == "supported"
        assert claim.reason == "calculation"
        assert claim.matched_evidence_id == "calc1"

    def test_example_sentence_not_flagged(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer(
            "Revenue was $26.0 billion [E1]. For example, a P/E of 20x is high.",
            ledger)
        assert report.numeric_claims_total == 1  # only the real claim
        assert report.numeric_claims_supported == 1

    def test_dict_ledger_is_accepted(self):
        ledger = [{"evidence_id": "E1", "kind": "fact", "metric": "total_revenue",
                   "value": 26.0, "unit": "billion_usd", "ticker": "NVDA",
                   "period": "2026-Q1", "source_type": "sec_10q"}]
        report = validate_answer("Revenue was $26.0 billion [E1].", ledger)
        assert report.numeric_claims_supported == 1


# ── Citation parsing ───────────────────────────────────────────────────────

class TestCitationParsing:
    def test_resolved_evidence_citation_carries_provenance(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd",
                                source_type="sec_10q")])
        report = validate_answer("Revenue was $26.0 billion [E1].", ledger)
        (cit,) = report.citations
        assert cit.evidence_id == "E1"
        assert cit.support_status == "supported"
        assert cit.source_type == "sec_10q"
        assert cit.ticker == "NVDA"
        assert cit.metric == "total_revenue"

    def test_unknown_id_is_missing_not_a_real_citation(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer("Revenue was $26.0 billion [E9].", ledger)
        (cit,) = report.citations
        assert cit.evidence_id == "E9"
        assert cit.support_status == "missing"
        assert cit.source_type is None and cit.ticker is None

    def test_malformed_id_is_flagged(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        for token in ("[E1x]", "[E]"):
            report = validate_answer(f"Revenue was $26.0 billion {token}.", ledger)
            (cit,) = report.citations
            assert cit.support_status == "malformed"
            assert cit.source_type is None

    def test_legacy_source_citation_retained(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd",
                                source_type="sec_10q")])
        report = validate_answer(
            "Revenue was $26.0 billion [Source: sec_10q/NVDA].", ledger)
        (cit,) = report.citations
        assert cit.evidence_id is None
        assert cit.source_type == "sec_10q"
        assert cit.ticker == "NVDA"
        assert cit.support_status == "supported"

    def test_duplicate_citations_collapse_to_one_record(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer(
            "Revenue was $26.0 billion [E1]. Confirmed again [E1].", ledger)
        assert len(report.citations) == 1
        assert report.citations[0].evidence_id == "E1"

    def test_bracketed_prose_is_not_a_citation(self):
        # A non-citation bracket like "[Earnings]" must not become a citation.
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer("[Earnings] were strong.", ledger)
        assert report.citations == ()

    def test_citation_support_rate(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer(
            "A [E1] and B [E9].", ledger)  # one resolved, one missing
        assert report.citation_support_rate == 0.5


# ── Report shape / policy helpers ──────────────────────────────────────────

class TestReportShape:
    def test_metadata_keys(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        meta = validate_answer("Revenue was $26.0 billion [E1].", ledger).to_metadata()
        for key in ("validation_status", "citation_support_rate",
                    "numeric_claims_supported", "numeric_claims_unsupported",
                    "numeric_claims_ambiguous", "numeric_claims_total",
                    "citations_total", "citations_resolved", "citations_missing",
                    "citations_malformed", "mismatch_counts"):
            assert key in meta

    def test_status_supported_when_no_violations(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        assert validate_answer("Revenue was $26.0 billion [E1].",
                              ledger).status == "supported"

    def test_status_unsupported_on_violation(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        assert validate_answer("Revenue was $30.0 billion [E1].",
                              ledger).status == "unsupported"

    def test_status_no_claims_on_empty_answer(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        assert validate_answer("Revenue grew strongly.", ledger).status == "no_claims"

    def test_wholly_unsupported_and_violations(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        report = validate_answer("Revenue was $30.0 billion [E1].", ledger)
        assert report.wholly_unsupported() is True
        assert report.has_violations() is True

    def test_mismatch_counts_bucketed(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd", period="2026-Q1")])
        report = validate_answer("In 2025 revenue was $26.0 billion [E1].", ledger)
        assert report.mismatch_counts["period"] == 1

    def test_unavailable_sentinel(self):
        meta = AnswerValidation.unavailable().to_metadata()
        assert meta["validation_status"] == "report_unavailable"

    def test_empty_ledger_uncited_number_is_unsupported(self):
        report = validate_answer("Revenue was $26.0 billion.", [])
        assert report.numeric_claims_unsupported == 1

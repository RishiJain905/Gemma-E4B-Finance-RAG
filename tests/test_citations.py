"""
tests/test_citations.py
Offline tests for structured citation parsing/resolution (2.2.4.3, Step 2).

Covers the EvidenceCitation response model, resolution of [E#] citations
against the request-local evidence ledger, the legacy [Source: type/ticker]
compatibility window, duplicate collapse, and the rule that unknown/malformed
ids never become a real source citation. Deterministic, no network/model.
"""

from types import SimpleNamespace

from src.middleware import app as middleware_app
from src.middleware.evidence import assign_evidence_ids, build_evidence_items
from src.middleware.models import EvidenceCitation, QueryResponse


def _config(**overrides):
    values = {"answer_validation": "report", "require_evidence_ids": False}
    values.update(overrides)
    return SimpleNamespace(**values)


def _ledger(facts, documents=None):
    return assign_evidence_ids(build_evidence_items(facts, documents or []))


def _fact(metric, value, unit, *, ticker="NVDA", period="2026-Q1",
          source_type="sec_10q", source_url=None):
    row = {"metric": metric, "value": value, "unit": unit, "ticker": ticker,
           "period": period, "source_type": source_type}
    if source_url:
        row["source_url"] = source_url
    return row


def _apply(answer, ledger, *, calculations=None, **cfg):
    """Run app-level validation with a stubbed config; return the 4-tuple."""
    context = {"evidence_ledger": ledger, "calculations": calculations or []}
    original = middleware_app.config
    middleware_app.config = _config(**cfg)
    try:
        return middleware_app._apply_answer_validation(context, answer, "grounded")
    finally:
        middleware_app.config = original


# ── Response model ─────────────────────────────────────────────────────────

class TestEvidenceCitationModel:
    def test_defaults_and_status(self):
        c = EvidenceCitation(evidence_id="E1", source_type="sec_10q", ticker="NVDA")
        assert c.support_status == "supported"
        assert c.metric is None and c.period is None

    def test_response_omits_validation_when_off(self):
        # A plain response (validation off / None) must not surface the fields.
        dumped = QueryResponse(answer="legacy").model_dump()
        assert "answer_validation" not in dumped
        assert "evidence_citations" not in dumped

    def test_response_includes_validation_when_present(self):
        response = QueryResponse(
            answer="ok",
            evidence_citations=[EvidenceCitation(evidence_id="E1")],
            answer_validation={"validation_status": "supported"},
        )
        dumped = response.model_dump()
        assert dumped["answer_validation"]["validation_status"] == "supported"
        assert dumped["evidence_citations"][0]["evidence_id"] == "E1"


# ── App-level resolution ───────────────────────────────────────────────────

class TestCitationResolution:
    def test_off_mode_is_passthrough(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        answer, grounding, meta, cites = _apply(
            "Revenue was $26.0 billion [E1].", ledger, answer_validation="off")
        assert (meta, cites) == (None, None)
        assert grounding == "grounded"

    def test_resolves_to_ledger_with_provenance(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd",
                                source_url="https://sec.gov/x")])
        _a, _g, meta, cites = _apply("Revenue was $26.0 billion [E1].", ledger)
        assert isinstance(cites[0], EvidenceCitation)
        assert cites[0].evidence_id == "E1"
        assert cites[0].support_status == "supported"
        assert cites[0].source_type == "sec_10q"
        assert cites[0].source_url == "https://sec.gov/x"
        assert meta["citation_support_rate"] == 1.0

    def test_unknown_id_is_missing_not_real_citation(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        _a, _g, meta, cites = _apply("Revenue was $26.0 billion [E9].", ledger)
        (cit,) = cites
        assert cit.evidence_id == "E9"
        assert cit.support_status == "missing"
        assert cit.source_type is None and cit.ticker is None
        assert meta["citations_missing"] == 1

    def test_malformed_id_is_flagged_not_real_citation(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        _a, _g, meta, cites = _apply("Revenue was $26.0 billion [E1x].", ledger)
        (cit,) = cites
        assert cit.support_status == "malformed"
        assert cit.source_type is None
        assert meta["citations_malformed"] == 1

    def test_duplicate_citations_collapse(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        _a, _g, _m, cites = _apply(
            "Revenue was $26.0 billion [E1]. Again [E1].", ledger)
        assert len(cites) == 1

    def test_legacy_source_citation_supported_in_window(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        _a, _g, _m, cites = _apply(
            "Revenue was $26.0 billion [Source: sec_10q/NVDA].", ledger)
        (cit,) = cites
        assert cit.evidence_id is None
        assert cit.source_type == "sec_10q" and cit.ticker == "NVDA"
        assert cit.support_status == "supported"

    def test_cited_ids_only_resolve_to_final_ledger(self):
        # Every accepted (supported) citation must carry an id present in the
        # ledger; an id outside it is never accepted.
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        _a, _g, _m, cites = _apply(
            "A [E1] and B [E2] and C [E1].", ledger)
        supported = [c for c in cites if c.support_status == "supported"]
        assert [c.evidence_id for c in supported] == ["E1"]
        assert [c.support_status for c in cites if c.evidence_id == "E2"] == ["missing"]


# ── Legacy extractor unchanged ─────────────────────────────────────────────

class TestLegacyExtractor:
    def test_extract_citations_still_parses_source_labels(self):
        cites = middleware_app._extract_citations(
            "Revenue [Source: sec_10q/NVDA] and margin [Source: yfinance/AMD].")
        assert [(c.source_type, c.ticker) for c in cites] == [
            ("sec_10q", "NVDA"), ("yfinance", "AMD")]

    def test_extract_citations_ignores_evidence_ids(self):
        # The legacy SourceCitation extractor only reads [Source: ...]; [E#]
        # markers are not converted into legacy source citations.
        assert middleware_app._extract_citations("Revenue was $26B [E1].") == []


# ── Enforcement (opt-in) ───────────────────────────────────────────────────

class TestEnforcement:
    def test_report_mode_never_changes_answer_or_grounding(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        answer, grounding, meta, _c = _apply(
            "Revenue was $99.0 billion [E1].", ledger, answer_validation="report")
        assert answer == "Revenue was $99.0 billion [E1]."
        assert grounding == "grounded"
        assert meta["enforcement"] == "none"

    def test_enforce_downgrades_grounded_to_partial_with_warning(self):
        ledger = _ledger([
            _fact("total_revenue", 26.0, "billion_usd"),
            _fact("gross_margin", 0.70, "ratio"),
        ])
        # One supported figure, one fabricated -> downgrade (not wholly unsupported).
        answer, grounding, meta, _c = _apply(
            "Margin was 70% [E2] but revenue was $99.0 billion [E1].",
            ledger, answer_validation="enforce")
        assert grounding == "partial"
        assert meta["enforcement"] == "downgrade"
        assert "Support warning" in answer

    def test_enforce_refuses_wholly_unsupported_answer(self):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])
        answer, grounding, meta, _c = _apply(
            "Revenue was $99.0 billion [E1].", ledger, answer_validation="enforce")
        assert grounding == "refused"
        assert meta["enforcement"] == "refuse"
        assert answer == middleware_app.NO_GENERAL_FALLBACK_MESSAGE

    def test_validator_exception_fails_soft(self, monkeypatch):
        ledger = _ledger([_fact("total_revenue", 26.0, "billion_usd")])

        def boom(*a, **k):
            raise RuntimeError("validator blew up")

        monkeypatch.setattr("src.middleware.answer_validator.validate_answer", boom)
        answer, grounding, meta, cites = _apply(
            "Revenue was $26.0 billion [E1].", ledger, answer_validation="enforce")
        # Query never crashes: answer/grounding untouched, status report_unavailable.
        assert answer == "Revenue was $26.0 billion [E1]."
        assert grounding == "grounded"
        assert meta["validation_status"] == "report_unavailable"
        assert cites is None

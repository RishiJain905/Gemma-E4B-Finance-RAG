"""tests/test_sec_event_classifier.py
Offline tests for deterministic SEC event classification and extraction.
"""

import json
from pathlib import Path

import pytest

from src.sec.event_classifier import RULE_VERSION, SECEventClassifier


FIXTURES = Path(__file__).parent / "fixtures" / "sec" / "events"


@pytest.fixture
def classifier() -> SECEventClassifier:
    return SECEventClassifier()


@pytest.mark.parametrize("case", json.loads((FIXTURES / "classifier_cases.json").read_text()))
def test_classifier_emits_expected_deterministic_labels(
    classifier: SECEventClassifier, case: dict,
) -> None:
    events = classifier.classify(case["filing"], case.get("documents", []))

    assert sorted(event.event_type for event in events) == sorted(case["expected"])
    assert all(event.rule_version == RULE_VERSION for event in events)
    assert all(event.classification_reason for event in events)


def test_financing_fields_are_extracted_only_from_explicit_terms(
    classifier: SECEventClassifier,
) -> None:
    case = json.loads((FIXTURES / "classifier_cases.json").read_text())[0]

    events = classifier.classify(case["filing"])
    debt = next(event for event in events if event.event_type == "debt_raise")

    assert debt.amount == 750_000_000
    assert debt.currency == "USD"
    assert debt.security_type == "convertible senior notes"
    assert debt.maturity == "2032"
    assert debt.rate == 3.25


def test_orcl_like_linked_evidence_does_not_invent_amount(
    classifier: SECEventClassifier,
) -> None:
    payload = json.loads((FIXTURES / "orcl_notes_offering.json").read_text())

    events = classifier.classify(payload["filing"], payload["documents"])
    labels = {event.event_type for event in events}
    debt = next(event for event in events if event.event_type == "debt_raise")

    assert {"debt_raise", "prospectus_update"} <= labels
    assert debt.amount is None
    assert debt.currency is None
    assert debt.source_document_types == ("PRIMARY",)


def test_unrelated_currency_figure_is_not_treated_as_offering_amount(
    classifier: SECEventClassifier,
) -> None:
    events = classifier.classify({
        "accession": "0001-26-000008",
        "form": "424B5",
        "text": (
            "Prospectus supplement for senior notes. "
            "Last year's segment revenue was $750 million. "
            "The principal amount will be determined at pricing."
        ),
    })

    debt = next(event for event in events if event.event_type == "debt_raise")
    assert debt.amount is None
    assert debt.currency is None


def test_unrelated_percentage_is_not_treated_as_interest_rate(
    classifier: SECEventClassifier,
) -> None:
    events = classifier.classify({
        "accession": "0001-26-000013",
        "form": "424B5",
        "text": (
            "Prospectus supplement for senior notes. "
            "The underwriting discount is 2.0%. The interest rate will be set at pricing."
        ),
    })

    debt = next(event for event in events if event.event_type == "debt_raise")
    assert debt.rate is None


def test_every_declared_event_label_has_a_rule(classifier: SECEventClassifier) -> None:
    assert set(classifier.rule_inventory()) == {
        "debt_raise", "equity_raise", "convertible_offering",
        "shelf_registration", "prospectus_update", "acquisition", "divestiture",
        "earnings_release", "guidance_change", "leadership_change", "auditor_change",
        "buyback", "dividend_change", "insider_transaction",
        "beneficial_ownership_change",
    }

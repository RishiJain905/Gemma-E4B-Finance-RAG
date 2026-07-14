"""Contract tests for Phase 2.3 normalized ingestion records."""

from dataclasses import FrozenInstanceError, fields

import pytest

from src.ingestion.normalization import (
    NORMALIZATION_VERSION,
    content_hash,
    normalize_canonical_url,
    normalize_headline,
    syndicated_news_key,
)
from src.ingestion.records import EventRecord, NarrativeRecord, ObservationRecord


PROVENANCE = {
    "source_name": "finnhub",
    "source_category": "news_vendor",
    "provider_record_id": "news-1",
    "original_publisher": "Reuters",
    "source_url": "https://feed.example/items/1",
    "canonical_url": "https://publisher.example/story",
    "published_at": "2026-07-14T12:00:00Z",
    "observed_at": "2026-07-14T12:01:00Z",
    "accessed_at": "2026-07-14T12:02:00Z",
    "ingested_at": "2026-07-14T12:03:00Z",
    "license_label": "provider_entitlement",
    "normalization_version": NORMALIZATION_VERSION,
}


def narrative(**overrides) -> NarrativeRecord:
    values = {
        **PROVENANCE,
        "corpus_item_id": "item-1",
        "item_type": "news",
        "title": "Acme raises guidance",
        "body": "Acme raised its full-year guidance.",
        "summary": "Guidance increased.",
        "language": "en",
        "content_hash": content_hash("Acme raised its full-year guidance."),
        "document_family": "company_news",
        "security_ids": ("sec-acme",),
        "tickers": ("ACME",),
        "index_codes": ("sp500",),
        "sectors": ("Industrials",),
        "metadata": {"provider_revision": "2"},
    }
    values.update(overrides)
    return NarrativeRecord(**values)


def test_record_families_are_frozen_and_have_explicit_contract_fields():
    record = narrative()

    with pytest.raises(FrozenInstanceError):
        record.title = "changed"

    assert {field.name for field in fields(NarrativeRecord)} >= {
        "corpus_item_id", "source_name", "source_category",
        "provider_record_id", "original_publisher", "item_type", "event_type",
        "title", "body", "summary", "language", "published_at", "effective_at",
        "as_of_at", "observed_at", "accessed_at", "ingested_at", "source_url",
        "canonical_url", "security_ids", "tickers", "index_codes", "sectors",
        "content_hash", "metadata", "document_family", "indexing_status",
        "license_label", "normalization_version", "evidence_authority",
    }
    assert {field.name for field in fields(ObservationRecord)} >= {
        "observation_id", "metric_id", "series_id", "value_text",
        "value_numeric", "unit", "frequency", "period_start", "period_end",
        "vintage_at", "as_of_at", "scope", "security_ids", "tickers", "sector",
    }
    assert {field.name for field in fields(EventRecord)} >= {
        "event_id", "event_type", "effective_at", "announced_at", "status",
        "security_ids", "source_corpus_item_ids", "amount", "currency", "rate",
        "ratio", "action_date", "classifier_version", "explanation",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_name": ""}, "source_name"),
        ({"title": ""}, "title"),
        ({"body": ""}, "body"),
        ({"content_hash": "not-a-sha256"}, "content_hash"),
        ({"published_at": "yesterday"}, "published_at"),
        ({"indexing_status": "lost"}, "indexing_status"),
        ({"metadata": {"unbounded_vendor_blob": "x"}}, "metadata"),
    ],
)
def test_narrative_rejects_missing_malformed_or_unknown_fields(overrides, message):
    with pytest.raises(ValueError, match=message):
        narrative(**overrides)


def test_metadata_is_shallowly_immutable_and_values_are_bounded():
    record = narrative(metadata={"provider_revision": "2"})

    with pytest.raises(TypeError):
        record.metadata["provider_revision"] = "3"
    with pytest.raises(ValueError, match="metadata"):
        narrative(metadata={"provider_revision": "x" * 2_001})


def test_vendor_parsed_filing_cannot_claim_direct_sec_authority():
    with pytest.raises(ValueError, match="direct_sec"):
        narrative(
            source_name="massive",
            source_category="data_vendor",
            item_type="filing",
            evidence_authority="direct_sec",
        )

    direct = narrative(
        source_name="sec",
        source_category="regulator",
        item_type="filing",
        evidence_authority="direct_sec",
    )
    assert direct.evidence_authority == "direct_sec"


def test_observation_and_event_validate_required_fields_and_provenance():
    observation = ObservationRecord(
        observation_id="obs-1",
        metric_id="close",
        series_id=None,
        value_text="201.25",
        value_numeric=201.25,
        unit="usd",
        frequency="daily",
        period_start="2026-07-13",
        period_end="2026-07-13",
        vintage_at="2026-07-14T00:00:00Z",
        as_of_at="2026-07-13T20:00:00Z",
        scope="security",
        security_ids=("sec-acme",),
        tickers=("ACME",),
        sector=None,
        metadata={"adjusted": True},
        **PROVENANCE,
    )
    event = EventRecord(
        event_id="event-1",
        event_type="dividend",
        effective_at="2026-08-01T00:00:00Z",
        announced_at="2026-07-14T00:00:00Z",
        status="announced",
        security_ids=("sec-acme", "sec-beta"),
        source_corpus_item_ids=("item-1",),
        amount=0.25,
        currency="USD",
        action_date="2026-08-01",
        metadata={"provider_revision": "1"},
        **PROVENANCE,
    )

    assert observation.value_numeric == 201.25
    assert event.security_ids == ("sec-acme", "sec-beta")
    with pytest.raises(ValueError, match="scope"):
        ObservationRecord(**{**observation.__dict__, "scope": "tickerish"})
    with pytest.raises(ValueError, match="metadata"):
        EventRecord(**{**event.__dict__, "metadata": {"raw_payload": {"x": 1}}})


def test_url_hash_and_syndicated_headline_normalization_are_deterministic():
    assert normalize_canonical_url(
        "HTTPS://Example.COM:443/news//story/?utm_source=feed&b=2&a=1#section"
    ) == "https://example.com/news/story?a=1&b=2"
    assert content_hash("Line one\r\nLine two") == content_hash("Line one\nLine two")
    assert normalize_headline("  ACME\u2019s Q2: Results!  ") == "acme s q2 results"
    assert syndicated_news_key(
        "Acme raises guidance", "2026-07-14T12:05:00Z"
    ) == syndicated_news_key(
        "ACME raises guidance!", "2026-07-14T15:59:00Z"
    )
    assert syndicated_news_key(
        "Acme raises guidance", "2026-07-14T18:01:00Z"
    ) != syndicated_news_key(
        "Acme raises guidance", "2026-07-14T12:05:00Z"
    )

"""Offline integration tests for corpus ledger deduplication and indexing state."""

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

if "chromadb" not in sys.modules:
    chromadb = type(sys)("chromadb")
    chromadb.EmbeddingFunction = object
    chromadb.Documents = list
    chromadb.Embeddings = list
    chromadb.PersistentClient = MagicMock
    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.api"] = MagicMock()

from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash
from src.ingestion.records import EventRecord, NarrativeRecord, ObservationRecord
from src.storage.store import Store


@pytest.fixture
def store(tmp_path: Path):
    with patch("src.storage.store.ChromaStore") as chroma_class:
        chroma = MagicMock()
        chroma_class.return_value = chroma
        instance = Store(db_path=tmp_path / "corpus.db", chroma_path=tmp_path / "chroma")
        yield instance, chroma


def seed_securities(store: Store) -> tuple[str, str]:
    store.upsert_universe_snapshot(
        "ivv",
        "2026-07-14T00:00:00Z",
        [
            {
                "symbol": "ACME", "company_name": "Acme Corporation",
                "source": "ivv", "index_code": "sp500", "exchange": "NYSE",
            },
            {
                "symbol": "BETA", "company_name": "Beta Corporation",
                "source": "ivv", "index_code": "sp500", "exchange": "NASDAQ",
            },
        ],
    )
    return (
        store.resolve_security("ACME")["security_id"],
        store.resolve_security("BETA")["security_id"],
    )


def narrative(item_id: str = "item-1", **overrides) -> NarrativeRecord:
    body = overrides.pop("body", "Acme raised its full-year guidance.")
    values = {
        "corpus_item_id": item_id,
        "source_name": "finnhub",
        "source_category": "news_vendor",
        "provider_record_id": "finnhub-1",
        "original_publisher": "Reuters",
        "item_type": "news",
        "title": "Acme raises full-year guidance",
        "body": body,
        "summary": "Guidance increased.",
        "language": "en",
        "published_at": "2026-07-14T12:05:00Z",
        "observed_at": "2026-07-14T12:06:00Z",
        "accessed_at": "2026-07-14T12:07:00Z",
        "ingested_at": "2026-07-14T12:08:00Z",
        "source_url": "https://finnhub.example/news/1",
        "canonical_url": "https://publisher.example/acme-guidance?utm_source=feed",
        "license_label": "provider_entitlement",
        "normalization_version": NORMALIZATION_VERSION,
        "content_hash": content_hash(body),
        "document_family": "company_news",
        "metadata": {"provider_revision": "1"},
    }
    values.update(overrides)
    return NarrativeRecord(**values)


def test_provider_url_and_hash_replays_create_zero_duplicate_items(store):
    facade, chroma = store
    first = narrative()

    provider_result = facade.upsert_narrative(first)
    facade.upsert_narrative(replace(first, corpus_item_id="provider-replay"))
    facade.upsert_narrative(replace(
        first,
        corpus_item_id="url-replay",
        provider_record_id=None,
        canonical_url="https://PUBLISHER.example:443/acme-guidance#top",
    ))
    facade.upsert_narrative(replace(
        first,
        corpus_item_id="hash-replay",
        source_name="massive",
        provider_record_id="massive-99",
        canonical_url="https://massive.example/story/99",
        source_url="https://massive.example/story/99",
    ))

    assert provider_result["created"] is True
    assert facade.sqlite.count_corpus_items() == 1
    assert chroma.add_document.call_count == 1
    sources = facade.sqlite.list_corpus_item_sources("item-1")
    assert {source["source_name"] for source in sources} == {"finnhub", "massive"}


def test_syndicated_story_merges_origins_but_distinct_story_does_not(store):
    facade, chroma = store
    facade.upsert_narrative(narrative())

    syndicated = narrative(
        "item-massive",
        source_name="massive",
        provider_record_id="massive-1",
        source_url="https://massive.example/1",
        canonical_url="https://another-publisher.example/wire-copy",
        title="ACME raises full year guidance!",
        body="Wire copy with different formatting and a different hash.",
    )
    merged = facade.upsert_narrative(syndicated)
    distinct = narrative(
        "item-distinct",
        source_name="massive",
        provider_record_id="massive-2",
        source_url="https://massive.example/2",
        canonical_url="https://another-publisher.example/interview",
        title="Acme CEO discusses factory expansion",
        body="A separate interview published on the same day.",
    )
    separate = facade.upsert_narrative(distinct)

    assert merged["corpus_item_id"] == "item-1"
    assert merged["deduplication_layer"] == "syndicated_headline"
    assert separate["corpus_item_id"] == "item-distinct"
    assert facade.sqlite.count_corpus_items() == 2
    assert chroma.add_document.call_count == 2


def test_one_item_and_event_link_to_multiple_securities_without_duplication(store):
    facade, _chroma = store
    acme_id, beta_id = seed_securities(facade)
    item = narrative(
        security_ids=(acme_id, beta_id),
        tickers=("ACME", "BETA"),
        index_codes=("sp500",),
        sectors=("Industrials",),
    )
    facade.upsert_narrative(item)
    facade.upsert_narrative(item)
    event = EventRecord(
        event_id="event-1",
        event_type="joint_venture",
        effective_at="2026-08-01T00:00:00Z",
        announced_at="2026-07-14T12:05:00Z",
        status="announced",
        security_ids=(acme_id, beta_id),
        source_corpus_item_ids=("item-1",),
        source_name="sec",
        source_category="regulator",
        provider_record_id="event-provider-1",
        original_publisher=None,
        source_url="https://www.sec.gov/Archives/example",
        canonical_url="https://www.sec.gov/Archives/example",
        published_at="2026-07-14T12:05:00Z",
        observed_at="2026-07-14T12:06:00Z",
        accessed_at="2026-07-14T12:07:00Z",
        ingested_at="2026-07-14T12:08:00Z",
        license_label="public_record",
        evidence_authority="direct_sec",
    )
    facade.upsert_event(event)
    facade.upsert_event(event)

    assert facade.sqlite.count_corpus_items() == 1
    assert {row["security_id"] for row in facade.sqlite.list_corpus_item_securities("item-1")} == {
        acme_id, beta_id,
    }
    stored_event = facade.sqlite.get_event("event-1")
    assert set(stored_event["security_ids"]) == {acme_id, beta_id}
    assert stored_event["source_corpus_item_ids"] == ["item-1"]
    assert facade.sqlite.count_events() == 1


def test_chroma_failure_retains_retryable_sqlite_item_without_refetch(store):
    facade, chroma = store
    record = narrative()
    chroma.add_document.side_effect = RuntimeError("embedding server unavailable")

    result = facade.upsert_narrative(record)

    assert result["indexing_status"] == "error"
    stored = facade.sqlite.get_corpus_item("item-1")
    assert stored["indexing_status"] == "error"
    assert "embedding server unavailable" in stored["index_error"]
    assert [row["corpus_item_id"] for row in facade.sqlite.list_retryable_corpus_items()] == [
        "item-1"
    ]

    chroma.add_document.side_effect = None
    retry = facade.upsert_narrative(record)
    assert retry["indexing_status"] == "indexed"
    assert facade.sqlite.get_corpus_item("item-1")["indexing_status"] == "indexed"
    assert chroma.add_document.call_count == 2


def test_changed_narrative_content_reindexes_only_affected_family(store):
    facade, chroma = store
    first = narrative("item-1")
    second = narrative(
        "item-2",
        provider_record_id="finnhub-2",
        source_url="https://finnhub.example/news/2",
        canonical_url="https://publisher.example/acme-factory",
        title="Acme opens new factory",
        body="Acme opened a new factory.",
    )
    facade.upsert_narrative(first)
    facade.upsert_narrative(second)
    chroma.reset_mock()

    changed = replace(
        first,
        body="Acme raised guidance again after the market close.",
        content_hash=content_hash("Acme raised guidance again after the market close."),
        summary="Guidance increased again.",
        accessed_at="2026-07-14T13:00:00Z",
    )
    result = facade.upsert_narrative(changed)
    facade.upsert_narrative(second)

    assert result["content_changed"] is True
    chroma.delete_document.assert_not_called()
    chroma.delete_filing_section_family.assert_not_called()
    assert chroma.add_document.call_count == 1
    assert chroma.add_document.call_args.kwargs["document_id"] == "item-1"
    assert chroma.add_document.call_args.kwargs["replace_family"] is True


def test_same_source_url_fallback_updates_changed_canonical_content(store):
    facade, chroma = store
    first = narrative(provider_record_id=None)
    facade.upsert_narrative(first)
    chroma.reset_mock()

    changed_body = "Acme raised guidance after publishing an amended release."
    result = facade.upsert_narrative(replace(
        first,
        corpus_item_id="adapter-generated-different-id",
        body=changed_body,
        content_hash=content_hash(changed_body),
        summary="The release was amended.",
        accessed_at="2026-07-14T13:00:00Z",
    ))

    assert result["corpus_item_id"] == "item-1"
    assert result["deduplication_layer"] == "canonical_url"
    assert result["content_changed"] is True
    chroma.delete_document.assert_not_called()
    assert chroma.add_document.call_count == 1
    assert chroma.add_document.call_args.kwargs["replace_family"] is True


def test_headline_window_does_not_merge_two_items_from_same_source(store):
    facade, _chroma = store
    facade.upsert_narrative(narrative(provider_record_id="source-1"))
    second = narrative(
        "item-2",
        provider_record_id="source-2",
        source_url="https://finnhub.example/news/2",
        canonical_url="https://publisher.example/acme-guidance-followup",
        body="A separate follow-up with the same exact headline.",
    )

    result = facade.upsert_narrative(second)

    assert result["created"] is True
    assert facade.sqlite.count_corpus_items() == 2


def test_observation_revisions_are_idempotent_and_never_embedded(store):
    facade, chroma = store
    acme_id, _ = seed_securities(facade)
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
        security_ids=(acme_id,),
        tickers=("ACME",),
        sector=None,
        source_name="massive",
        source_category="market_data",
        provider_record_id="bar-1",
        original_publisher=None,
        source_url="https://massive.example/bars/1",
        canonical_url=None,
        published_at=None,
        observed_at="2026-07-13T20:00:00Z",
        accessed_at="2026-07-14T00:00:00Z",
        ingested_at="2026-07-14T00:01:00Z",
        license_label="provider_entitlement",
        metadata={"adjusted": True},
    )
    start_revision = facade.retrieval_revision()
    first = facade.upsert_observation(observation)
    replay = facade.upsert_observation(observation)

    assert first["created"] is True
    assert replay["changed"] is False
    assert facade.sqlite.count_observations() == 1
    assert facade.retrieval_revision() == start_revision + 1
    chroma.add_document.assert_not_called()

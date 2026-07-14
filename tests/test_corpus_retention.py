"""tests/test_corpus_retention.py: Offline retention and accounting coverage."""

import sys
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
from src.ingestion.records import NarrativeRecord, ObservationRecord
from src.storage.store import Store


@pytest.fixture
def store(tmp_path: Path):
    with patch("src.storage.store.ChromaStore") as chroma_class:
        chroma = MagicMock()
        chroma_class.return_value = chroma
        instance = Store(db_path=tmp_path / "retention.db", chroma_path=tmp_path / "chroma")
        yield instance, chroma


def narrative(item_id: str, *, item_type: str = "news", published_at: str, **overrides):
    body = overrides.pop("body", f"Narrative body for {item_id}.")
    values = {
        "corpus_item_id": item_id,
        "source_name": "finnhub" if item_type == "news" else "treasury",
        "source_category": "news_vendor" if item_type == "news" else "official_government",
        "provider_record_id": item_id,
        "original_publisher": None,
        "item_type": item_type,
        "title": f"Title {item_id}",
        "body": body,
        "summary": body,
        "published_at": published_at,
        "observed_at": published_at,
        "accessed_at": published_at,
        "ingested_at": published_at,
        "source_url": f"https://example.test/{item_id}",
        "canonical_url": f"https://example.test/{item_id}",
        "license_label": "public_record",
        "normalization_version": NORMALIZATION_VERSION,
        "content_hash": content_hash(body),
        "document_family": "company_news" if item_type == "news" else "official_release",
        "evidence_authority": "provider" if item_type == "news" else "official",
    }
    values.update(overrides)
    return NarrativeRecord(**values)


def test_news_retention_expires_only_indexed_news_and_bumps_revision_once(store):
    facade, chroma = store
    facade.upsert_narrative(narrative("old-news", published_at="2024-07-13T00:00:00Z"))
    facade.upsert_narrative(narrative("old-news-2", published_at="2024-01-01T00:00:00Z"))
    facade.upsert_narrative(narrative("recent-news", published_at="2024-07-14T00:00:00Z"))
    facade.upsert_narrative(narrative(
        "official-release", item_type="official_release",
        published_at="2020-01-01T00:00:00Z",
    ))
    chroma.add_document.side_effect = RuntimeError("index unavailable")
    facade.upsert_narrative(narrative("failed-news", published_at="2020-01-01T00:00:00Z"))
    chroma.add_document.side_effect = None
    chroma.reset_mock()
    before = facade.retrieval_revision()

    result = facade.run_retention(as_of="2026-07-14", apply=True)

    assert result == {
        "apply": True,
        "cutoff": "2024-07-14T00:00:00Z",
        "eligible": 2,
        "expired": 2,
        "failed": 0,
        "document_family_ids": ["old-news-2", "old-news"],
    }
    assert [call.args[0] for call in chroma.delete_document_family.call_args_list] == [
        "old-news-2", "old-news",
    ]
    assert facade.retrieval_revision() == before + 1
    expired = facade.sqlite.get_corpus_item("old-news")
    assert expired["is_tombstone"] is True
    assert expired["indexing_status"] == "not_applicable"
    assert expired["narrative_bytes"] == 0
    assert facade.sqlite.get_corpus_item("recent-news")["is_tombstone"] is False
    assert facade.sqlite.get_corpus_item("official-release")["is_tombstone"] is False
    assert facade.sqlite.get_corpus_item("failed-news")["indexing_status"] == "error"


def test_retention_preview_is_read_only(store):
    facade, chroma = store
    facade.upsert_narrative(narrative("old-news", published_at="2020-01-01T00:00:00Z"))
    chroma.reset_mock()
    before = facade.retrieval_revision()

    result = facade.run_retention(as_of="2026-07-14", apply=False)

    assert result["eligible"] == 1
    assert result["expired"] == 0
    assert result["document_family_ids"] == ["old-news"]
    assert facade.retrieval_revision() == before
    assert facade.sqlite.get_corpus_item("old-news")["is_tombstone"] is False
    chroma.delete_document_family.assert_not_called()

    with pytest.raises(ValueError, match="limit"):
        facade.run_retention(as_of="2026-07-14", apply=False, limit=0)


def test_raw_network_payload_metadata_is_not_retained(store):
    facade, chroma = store
    with pytest.raises(ValueError, match="raw_payload"):
        narrative(
            "news-with-raw", published_at="2026-07-14T00:00:00Z",
            metadata={"raw_payload": "secret provider response"},
        )

    assert facade.sqlite.count_corpus_items() == 0
    chroma.add_document.assert_not_called()


def test_accounting_uses_bounded_sqlite_aggregates(store):
    facade, chroma = store
    facade.upsert_narrative(narrative("news-a", published_at="2026-06-01T00:00:00Z"))
    facade.upsert_narrative(narrative("news-b", published_at="2026-06-02T00:00:00Z"))
    facade.upsert_narrative(narrative(
        "release-a", item_type="official_release", published_at="2025-02-01T00:00:00Z",
    ))
    facade.upsert_observation(ObservationRecord(
        observation_id="obs-1",
        metric_id="fed_funds_rate",
        series_id="FEDFUNDS",
        value_text="5.25",
        value_numeric=5.25,
        unit="percent",
        frequency="monthly",
        period_start="2026-06-01",
        period_end="2026-06-30",
        vintage_at="2026-07-01T00:00:00Z",
        as_of_at="2026-06-30T00:00:00Z",
        scope="global",
        security_ids=(),
        tickers=(),
        sector=None,
        source_name="fred",
        source_category="official_macro",
        provider_record_id="FEDFUNDS-2026-06",
        original_publisher="Federal Reserve",
        source_url="https://example.test/fred/fedfunds",
        canonical_url=None,
        published_at="2026-07-01T00:00:00Z",
        observed_at="2026-07-01T00:00:00Z",
        accessed_at="2026-07-01T00:00:00Z",
        ingested_at="2026-07-01T00:00:00Z",
        license_label="public_record",
    ))
    chroma.reset_mock()

    by_source = facade.get_corpus_accounting("source_category", limit=1)
    by_month = facade.get_corpus_accounting("month", item_type="news", limit=10)
    by_type = facade.get_corpus_accounting("item_type", limit=10)
    every_dimension = {
        dimension: facade.get_corpus_accounting(dimension, limit=2)
        for dimension in (
            "source_category", "source", "item_type", "security", "year", "month",
            "indexing_state",
        )
    }

    assert len(by_source) == 1
    assert set(by_source[0]) == {"key", "count", "approximate_bytes"}
    assert by_source[0]["count"] > 0
    assert by_source[0]["approximate_bytes"] > 0
    assert by_month == [{
        "key": "2026-06", "count": 2,
        "approximate_bytes": by_month[0]["approximate_bytes"],
    }]
    assert any(row["key"] == "observation" and row["count"] == 1 for row in by_type)
    assert all(len(rows) <= 2 for rows in every_dimension.values())
    chroma.iter_documents.assert_not_called()
    chroma.collection.get.assert_not_called()


def test_retention_policy_defaults_are_explicit():
    from src.storage.retention import load_retention_policy

    policy = load_retention_policy()

    assert policy.company_news_months == 24
    assert policy.retain_raw_network_payloads is False
    assert {
        "sec_filing", "sec_exhibit", "issuer_release", "official_release",
        "event", "corporate_action", "membership", "observation",
    }.issubset(policy.permanent_item_types)

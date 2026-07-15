"""tests/test_coverage_health.py: Offline SQLite-only coverage health reporting."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

if "chromadb" not in sys.modules:
    chromadb = type(sys)("chromadb")
    chromadb.EmbeddingFunction = object
    chromadb.Documents = list
    chromadb.Embeddings = list
    chromadb.PersistentClient = MagicMock
    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.api"] = MagicMock()


def _seed_health_rows(store):
    with store.sqlite._connect() as conn:
        conn.executemany(
            """INSERT INTO securities (
                security_id, ticker, normalized_ticker, company_name, exchange,
                sector, first_seen_at, last_seen_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("sec-a", "AAA", "AAA", "AAA Corp", "NYSE", "Technology", "2026-01-01", "2026-07-14", "2026-07-14"),
                ("sec-b", "BBB", "BBB", "BBB Corp", "NYSE", "Health Care", "2026-01-01", "2026-07-14", "2026-07-14"),
            ],
        )
        conn.executemany(
            """INSERT INTO security_memberships (
                security_id, index_code, effective_from, active, source, observed_at
            ) VALUES (?, ?, ?, 1, ?, ?)""",
            [
                ("sec-a", "sp500", "2026-01-01", "fixture", "2026-07-14"),
                ("sec-b", "nasdaq100", "2026-01-01", "fixture", "2026-07-14"),
            ],
        )
        conn.execute(
            """INSERT INTO cache_meta (
                ticker, source, last_updated, next_scheduled_update, status
            ) VALUES ('AAA', 'finnhub_news', '2026-07-14 00:00:00', '2026-07-15 00:00:00', 'fresh')"""
        )
        conn.execute(
            """INSERT INTO cache_meta (
                ticker, source, last_updated, next_scheduled_update, status, error_message
            ) VALUES ('BBB', 'finnhub_news', '2026-07-01 00:00:00', '2026-07-01 00:00:00', 'stale', 'provider unavailable')"""
        )
        conn.execute(
            """INSERT INTO cache_meta (
                ticker, source, last_updated, next_scheduled_update, status
            ) VALUES ('SCHEDULER', 'unified:federal_reserve', '2026-07-14 00:00:00', '2099-07-15 00:00:00', 'fresh')"""
        )
        conn.execute(
            """INSERT INTO corpus_items (
                corpus_item_id, source, source_category, provider_record_id,
                item_type, title, normalized_headline, summary, language,
                published_at, accessed_at, ingested_at, source_url, content_hash,
                metadata_json, document_family, indexing_status, license_label,
                normalization_version, evidence_authority, narrative_bytes,
                metadata_bytes
            ) VALUES (
                'news-aaa', 'finnhub', 'news_vendor', 'news-aaa', 'news',
                'AAA headline', 'aaa headline', 'AAA summary', 'en',
                '2026-07-14T00:00:00Z', '2026-07-14T00:00:00Z',
                '2026-07-14T00:00:00Z', 'https://example.test/news-aaa',
                'hash-aaa', '{}', 'company_news', 'indexed', 'provider_summary',
                'test', 'provider', 20, 20
            )"""
        )
        conn.execute(
            """INSERT INTO corpus_items (
                corpus_item_id, source, source_category, provider_record_id,
                item_type, title, normalized_headline, summary, language,
                published_at, accessed_at, ingested_at, source_url, content_hash,
                metadata_json, document_family, indexing_status, index_error,
                license_label, normalization_version, evidence_authority,
                narrative_bytes, metadata_bytes
            ) VALUES (
                'pending-item', 'finnhub', 'news_vendor', 'pending-item', 'news',
                'Pending headline', 'pending headline', 'Pending summary', 'en',
                '2026-07-13T00:00:00Z', '2026-07-13T00:00:00Z',
                '2026-07-13T00:00:00Z', 'https://example.test/pending-item',
                'hash-pending', '{}', 'company_news', 'error', 'index failed',
                'provider_summary', 'test', 'provider', 20, 20
            )"""
        )
        conn.execute(
            """INSERT INTO corpus_item_securities (corpus_item_id, security_id, ticker)
               VALUES ('news-aaa', 'sec-a', 'AAA')"""
        )
        conn.execute(
            """INSERT INTO filings (
                ticker, filing_type, filing_date, accession, status, index_error
            ) VALUES ('AAA', '10-Q', '2026-07-10', 'filing-error', 'index_pending', 'embedding failed')"""
        )
        conn.commit()


def test_status_reports_denominators_and_indexing_backlog_without_network(offline_store, monkeypatch):
    _seed_health_rows(offline_store)
    network = MagicMock(side_effect=AssertionError("coverage status must not call network"))
    monkeypatch.setattr("requests.get", network)

    from src.scheduler import UnifiedScheduler

    coverage = SimpleNamespace(
        revision="coverage-fixture-v1",
        explain=lambda source: {
            "source": source,
            "enabled": True,
            "ticker_count": 2,
            "policy_revision": "coverage-fixture-v1",
        },
        tickers_for=lambda _source: ["AAA", "BBB"],
        is_enabled=lambda _source: True,
    )
    scheduler = UnifiedScheduler(
        store=offline_store,
        coverage_resolver=coverage,
        inter_source_delay=0,
    )
    report = scheduler.status_report()
    health = report["coverage_health"]

    assert health["policy_revision"]
    assert health["active_securities"]["total"] == 2
    assert health["active_securities"]["by_index"]["sp500"]["denominator"] == 2
    assert health["active_securities"]["by_index"]["sp500"]["percentage"] == 50.0
    assert health["indexing"]["error"] >= 1
    assert health["corpus_dates"]["oldest"] == "2026-07-13T00:00:00Z"
    assert health["corpus_dates"]["newest"] == "2026-07-14T00:00:00Z"
    assert health["partitions"]["finnhub_news"]["denominator"] >= 2
    assert health["partitions"]["federal_reserve"]["fresh"] == 1
    assert health["indexing"]["filings"]["error"] == 1
    network.assert_not_called()

"""tests/test_source_cursors.py
Transactional cursor, overlap-window, fairness, and idempotency tests.
"""

from datetime import datetime, timezone

import pytest

from src.ingestion.records import ObservationRecord
from src.scheduler.cursors import CursorManager
from src.storage.sqlite_store import SQLiteStore


@pytest.fixture
def sqlite(tmp_path):
    return SQLiteStore(tmp_path / "cursors.db")


@pytest.fixture
def cursors(sqlite):
    return CursorManager(sqlite)


def test_date_token_and_per_security_cursors_commit_together(cursors, sqlite):
    with cursors.transaction() as transaction:
        transaction.advance(
            "sec_daily", "global", "2026-07-13", kind="date", overlap="2d"
        )
        transaction.advance(
            "agency", "dataset", "next-abc", kind="page_token", overlap=None
        )
        transaction.advance(
            "finnhub_news",
            "AAA",
            "2026-07-14T12:00:00Z",
            kind="timestamp",
            overlap="6h",
        )

    assert sqlite.get_source_cursor("sec_daily", "global") == "2026-07-13"
    assert sqlite.get_source_cursor("agency", "dataset") == "next-abc"
    assert (
        sqlite.get_source_cursor("finnhub_news", "AAA")
        == "2026-07-14T12:00:00Z"
    )


def test_cursor_transaction_rolls_back_every_partition_on_error(cursors, sqlite):
    with pytest.raises(RuntimeError, match="page failed"):
        with cursors.transaction() as transaction:
            transaction.advance("market", "US", "2026-07-13", kind="date")
            transaction.advance("news", "AAA", "token-2", kind="page_token")
            raise RuntimeError("page failed")

    assert sqlite.get_source_cursor("market", "US") is None
    assert sqlite.get_source_cursor("news", "AAA") is None


def test_cursor_advances_only_after_page_callback_finishes(cursors, sqlite):
    def failed_commit():
        raise RuntimeError("record commit failed")

    with pytest.raises(RuntimeError, match="record commit failed"):
        cursors.advance_after_commit(
            "market",
            "US",
            "2026-07-13",
            kind="date",
            commit=failed_commit,
        )

    assert sqlite.get_source_cursor("market", "US") is None

    cursors.advance_after_commit(
        "market",
        "US",
        "2026-07-13",
        kind="date",
        commit=lambda: "committed",
    )
    assert sqlite.get_source_cursor("market", "US") == "2026-07-13"


def test_overlap_windows_refetch_and_idempotent_storage(cursors, sqlite):
    record = ObservationRecord(
        observation_id="market/AAA/2026-07-13/close",
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
        scope="global",
        security_ids=(),
        tickers=("AAA",),
        sector=None,
        source_name="massive",
        source_category="market_data",
        provider_record_id="AAA-2026-07-13",
        original_publisher=None,
        source_url="https://example.test/market/2026-07-13",
        canonical_url=None,
        published_at=None,
        observed_at="2026-07-13T20:00:00Z",
        accessed_at="2026-07-14T00:00:00Z",
        ingested_at="2026-07-14T00:00:00Z",
        license_label="provider_entitlement",
    )

    assert cursors.overlap_start("2026-07-14", kind="date", overlap="2d") == (
        "2026-07-12"
    )
    first = sqlite.upsert_observation_record(record)
    replay = sqlite.upsert_observation_record(record)

    assert first["created"] is True
    assert replay["changed"] is False
    assert sqlite.count_observations() == 1


def test_never_fetched_then_oldest_successful_partitions_are_scheduled_first(
    cursors, sqlite, monkeypatch
):
    timestamps = iter(
        [
            datetime(2026, 7, 10, tzinfo=timezone.utc),
            datetime(2026, 7, 12, tzinfo=timezone.utc),
        ]
    )
    monkeypatch.setattr(cursors, "_now", lambda: next(timestamps))
    cursors.advance(
        "finnhub_news", "AAA", "2026-07-10T00:00:00Z", kind="timestamp"
    )
    cursors.advance(
        "finnhub_news", "BBB", "2026-07-12T00:00:00Z", kind="timestamp"
    )
    sqlite.set_source_status(
        "finnhub_news", "AAA", "error", error_message="temporary failure"
    )

    ordered = cursors.order_partitions(
        "finnhub_news", ["BBB", "CCC", "AAA", "DDD"]
    )

    assert ordered == ["CCC", "DDD", "AAA", "BBB"]


def test_successful_empty_partition_does_not_starve_never_fetched_work(
    cursors, sqlite
):
    sqlite.set_source_status("finnhub_news", "AAA", "success")

    assert cursors.order_partitions("finnhub_news", ["AAA", "BBB"]) == [
        "BBB",
        "AAA",
    ]


def test_cursor_state_is_not_cache_freshness(cursors, sqlite):
    cursors.advance("market", "US", "2026-07-13", kind="date")

    assert sqlite.get_source_cursor("market", "US") == "2026-07-13"
    assert sqlite.get_cache_status("US", "market") is None

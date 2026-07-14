"""tests/test_universe_registry.py
Transactional security identity and membership-history tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.storage.store import Store
from src.universe.models import SnapshotValidationError, UniverseRecord
from src.universe.registry import UniverseRegistry


@pytest.fixture
def store(tmp_path: Path):
    with patch("src.storage.store.ChromaStore") as chroma_cls:
        chroma_cls.return_value = MagicMock()
        yield Store(db_path=tmp_path / "universe.db", chroma_path=tmp_path / "chroma")


def row(
    symbol: str,
    *,
    source: str,
    index_code: str | None = None,
    cik: str | None = None,
    exchange: str | None = None,
    name: str | None = None,
) -> UniverseRecord:
    return UniverseRecord(
        symbol=symbol,
        company_name=name or f"{symbol} Incorporated",
        source=source,
        index_code=index_code,
        cik=cik,
        exchange=exchange,
        source_url=f"https://example.test/{source}",
    )


def test_overlap_does_not_duplicate_and_same_cik_share_classes_stay_distinct(store) -> None:
    registry = UniverseRegistry(store, minimums={"sec": 1, "nasdaq100": 2, "sp500": 2})
    registry.refresh(
        "sec",
        "2026-07-01T00:00:00Z",
        [
            row("GOOG", source="sec", cik="0001652044", exchange="NASDAQ"),
            row("GOOGL", source="sec", cik="0001652044", exchange="NASDAQ"),
            row("BRK-A", source="sec", cik="0001067983", exchange="NYSE"),
            row("BRK-B", source="sec", cik="0001067983", exchange="NYSE"),
        ],
    )

    registry.refresh(
        "nasdaq",
        "2026-07-02T00:00:00Z",
        [
            row("GOOG", source="nasdaq", index_code="nasdaq100", exchange="Nasdaq"),
            row("GOOGL", source="nasdaq", index_code="nasdaq100", exchange="Nasdaq"),
        ],
    )
    registry.refresh(
        "ivv",
        "2026-07-02T00:00:00Z",
        [
            row("BRK.B", source="ivv", index_code="sp500", exchange="NYSE"),
            row("GOOGL", source="ivv", index_code="sp500", exchange="NASDAQ"),
        ],
    )

    securities = store.list_securities(active=True, limit=20, offset=0)
    assert {item["ticker"] for item in securities} == {"BRK-A", "BRK-B", "GOOG", "GOOGL"}
    assert len([item for item in securities if item["cik"] == "0001652044"]) == 2
    assert len([item for item in securities if item["cik"] == "0001067983"]) == 2
    assert len(store.list_memberships(index_code="sp500", active=True)) == 2
    assert len(store.list_memberships(index_code="nasdaq100", active=True)) == 2


def test_two_snapshots_preserve_history_and_identical_replay_is_noop(store) -> None:
    registry = UniverseRegistry(store, minimums={"sp500": 2})
    first = registry.refresh(
        "ivv",
        "2026-07-01T00:00:00Z",
        [
            row("AAA", source="ivv", index_code="sp500"),
            row("BBB", source="ivv", index_code="sp500"),
        ],
    )
    revision_after_first = store.retrieval_revision()
    second_rows = [
        row("BBB", source="ivv", index_code="sp500"),
        row("CCC", source="ivv", index_code="sp500"),
    ]
    second = registry.refresh("ivv", "2026-07-10T00:00:00Z", second_rows)
    revision_after_second = store.retrieval_revision()
    replay = registry.refresh("ivv", "2026-07-11T00:00:00Z", second_rows)

    assert first.changed is True
    assert second.changed is True
    assert second.memberships_opened == 1
    assert second.memberships_closed == 1
    assert replay.changed is False
    assert revision_after_first == 1
    assert revision_after_second == 2
    assert store.retrieval_revision() == revision_after_second
    memberships = store.list_memberships(index_code="sp500", active=None)
    aaa = next(item for item in memberships if item["ticker"] == "AAA")
    bbb = [item for item in memberships if item["ticker"] == "BBB"]
    assert aaa["active"] == 0
    assert aaa["effective_to"] == "2026-07-10"
    assert len(bbb) == 1 and bbb[0]["active"] == 1
    assert [item["ticker"] for item in store.list_securities(
        index="sp500", active=False, limit=10, offset=0,
    )] == ["AAA"]


def test_implausibly_small_snapshot_leaves_active_membership_unchanged(store) -> None:
    registry = UniverseRegistry(store, minimums={"sp500": 2})
    registry.refresh(
        "ivv",
        "2026-07-01T00:00:00Z",
        [
            row("AAA", source="ivv", index_code="sp500"),
            row("BBB", source="ivv", index_code="sp500"),
        ],
    )
    before = store.list_memberships(index_code="sp500", active=True)
    revision = store.retrieval_revision()

    with pytest.raises(SnapshotValidationError, match="at least 2"):
        registry.refresh(
            "ivv",
            "2026-07-02T00:00:00Z",
            [row("AAA", source="ivv", index_code="sp500")],
        )

    assert store.list_memberships(index_code="sp500", active=True) == before
    assert store.retrieval_revision() == revision


def test_cik_ticker_change_creates_former_ticker_alias(store) -> None:
    registry = UniverseRegistry(store, minimums={"sec": 1})
    registry.refresh(
        "sec",
        "2022-01-01T00:00:00Z",
        [row("FB", source="sec", cik="0001326801", exchange="NASDAQ", name="Meta Platforms")],
    )
    registry.refresh(
        "sec",
        "2022-06-09T00:00:00Z",
        [row("META", source="sec", cik="0001326801", exchange="NASDAQ", name="Meta Platforms")],
    )

    canonical = store.resolve_security("META", provider="sec")
    historical = store.resolve_security("FB", provider="sec", as_of="2022-05-01")
    assert canonical["security_id"] == historical["security_id"]
    assert canonical["ticker"] == "META"
    assert store.resolve_security("META", provider="sec", as_of="2022-05-01") is None
    assert store.resolve_security("META", provider="sec", as_of="2022-06-09") == canonical
    assert store.sqlite.list_tickers() == ["META"]


def test_ambiguous_cik_is_recorded_as_reconciliation_error(store) -> None:
    registry = UniverseRegistry(store, minimums={"sec": 1, "sp500": 1})
    registry.refresh(
        "sec",
        "2026-07-01T00:00:00Z",
        [
            row("ONE", source="sec", cik="0000000001", exchange="NYSE"),
            row("TWO", source="sec", cik="0000000001", exchange="NYSE"),
        ],
    )

    result = registry.refresh(
        "ivv",
        "2026-07-02T00:00:00Z",
        [
            row(
                "THREE",
                source="ivv",
                index_code="sp500",
                cik="0000000001",
                exchange="NYSE",
            )
        ],
    )

    errors = store.list_universe_errors(result.run_id)
    assert result.errors == 1
    assert errors[0]["symbol"] == "THREE"
    assert errors[0]["error_code"] == "ambiguous_cik"
    assert store.get_security("THREE") is None


def test_single_row_refresh_does_not_merge_a_different_share_class(store) -> None:
    registry = UniverseRegistry(store, minimums={"sec": 1})
    registry.refresh(
        "sec",
        "2026-07-01T00:00:00Z",
        [
            row(
                "BRK-A",
                source="sec",
                cik="0001067983",
                exchange="NYSE",
                name="Berkshire Hathaway Inc.",
            )
        ],
    )

    registry.refresh(
        "sec",
        "2026-07-02T00:00:00Z",
        [
            row(
                "BRK-B",
                source="sec",
                cik="0001067983",
                exchange="NYSE",
                name="Berkshire Hathaway Inc.",
            )
        ],
    )
    registry.refresh(
        "sec",
        "2026-07-03T00:00:00Z",
        [row("GOOG", source="sec", cik="0001652044", name="Alphabet Inc.")],
    )
    registry.refresh(
        "sec",
        "2026-07-04T00:00:00Z",
        [row("GOOGL", source="sec", cik="0001652044", name="Alphabet Inc.")],
    )

    securities = store.list_securities(active=True, limit=10, offset=0)
    assert {item["ticker"] for item in securities} == {
        "BRK-A", "BRK-B", "GOOG", "GOOGL",
    }


def test_same_symbol_on_different_exchanges_does_not_overwrite_security(store) -> None:
    registry = UniverseRegistry(store, minimums={"sec": 1})
    registry.refresh(
        "sec",
        "2026-07-01T00:00:00Z",
        [row("SAME", source="sec", exchange="NASDAQ")],
    )
    registry.refresh(
        "sec",
        "2026-07-02T00:00:00Z",
        [row("SAME", source="sec", exchange="NYSE")],
    )

    securities = store.list_securities(active=True, limit=10, offset=0)
    assert [(item["ticker"], item["exchange"]) for item in securities] == [
        ("SAME", "NASDAQ"),
        ("SAME", "NYSE"),
    ]

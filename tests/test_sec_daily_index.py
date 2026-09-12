"""tests/test_sec_daily_index.py
Offline tests for index-driven SEC broad-universe filing discovery.
"""

from pathlib import Path
from io import BytesIO
from unittest.mock import MagicMock, patch
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import yaml

from src.sec.daily_index import SECDailyIndexDiscovery, parse_daily_index
from src.storage.store import Store
from src.universe.coverage import CoverageResolver


FIXTURE = Path(__file__).parent / "fixtures" / "sec" / "events" / "master.20260710.idx"


@pytest.fixture
def store(tmp_path: Path):
    with patch("src.storage.store.ChromaStore") as chroma_class:
        chroma_class.return_value = MagicMock()
        yield Store(db_path=tmp_path / "sec.db", chroma_path=tmp_path / "chroma")


def _seed_broad_universe(store: Store) -> None:
    store.upsert_universe_snapshot(
        "ivv",
        "2026-07-10T00:00:00Z",
        [
            {
                "symbol": "ORCL", "company_name": "Oracle Corporation",
                "source": "ivv", "index_code": "sp500", "exchange": "NYSE",
                "cik": "0001341439", "source_url": "https://example.test/ivv",
            },
            {
                "symbol": "MSFT", "company_name": "Microsoft Corporation",
                "source": "ivv", "index_code": "sp500", "exchange": "NASDAQ",
                "cik": "0000789019", "source_url": "https://example.test/ivv",
            },
            *[
                {
                    "symbol": ticker, "company_name": f"{ticker} Corporation",
                    "source": "ivv", "index_code": "sp500", "exchange": "NASDAQ",
                    "source_url": "https://example.test/ivv",
                }
                for ticker in ("AAPL", "AMD", "CRWD", "META", "NVDA")
            ],
        ],
    )


def _config() -> dict:
    return {
        "forms": {
            "periodic_material": ["10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "8-K/A", "6-K"],
            "capital_markets": [
                "S-1", "S-1/A", "S-3", "S-3/A", "S-3ASR", "S-3ASR/A",
                "424B2", "424B3", "424B5", "FWP",
            ],
            "ownership_governance": [
                "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
                "3", "3/A", "4", "4/A", "5", "5/A", "DEF 14A",
            ],
        }
    }


def _bulk_zip() -> bytes:
    payload = BytesIO()
    with ZipFile(payload, "w", ZIP_DEFLATED) as archive:
        for name in ("CIK0001341439.json", "CIK0000789019.json"):
            archive.writestr(name, (FIXTURE.parent / name).read_bytes())
    return payload.getvalue()


def test_parse_and_filter_fixture_against_mixed_cik_universe(store: Store) -> None:
    _seed_broad_universe(store)
    discovery = SECDailyIndexDiscovery(
        store=store,
        coverage_resolver=CoverageResolver(store),
        sec_config=_config(),
        user_agent="Researcher test@example.com",
        request_delay=0,
    )

    rows = discovery.filter_entries(parse_daily_index(FIXTURE.read_text()))

    assert {(row.cik, row.form) for row in rows} == {
        ("0001341439", "8-K"),
        ("0001341439", "424B5"),
        ("0001341439", "S-3ASR"),
        ("0000789019", "10-Q"),
        ("0000789019", "SC 13G/A"),
    }
    assert all(row.accession for row in rows)


def test_every_configured_capability_form_is_discoverable(store: Store) -> None:
    _seed_broad_universe(store)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs" / "sec.yaml").read_text())["sec"]
    discovery = SECDailyIndexDiscovery(
        store=store,
        coverage_resolver=CoverageResolver(store),
        sec_config=config,
        user_agent="Researcher test@example.com",
        request_delay=0,
    )
    configured = {
        str(form) for forms in config["forms"].values() for form in forms
    }
    payload = "\n".join(
        f"1341439|Oracle Corporation|{form}|2026-07-10|"
        f"edgar/data/1341439/0001193125-26-{index:06d}.txt"
        for index, form in enumerate(sorted(configured), start=1)
    )

    discovered = discovery.filter_entries(parse_daily_index(payload))

    assert {entry.form for entry in discovered} == configured


def test_process_index_registers_each_accession_and_replay_is_idempotent(store: Store) -> None:
    _seed_broad_universe(store)
    discovery = SECDailyIndexDiscovery(
        store=store,
        coverage_resolver=CoverageResolver(store),
        sec_config=_config(),
        user_agent="Researcher test@example.com",
        request_delay=0,
    )

    first = discovery.process_index("2026-07-10", FIXTURE.read_text(), "https://sec.test/master.idx")
    replay = discovery.process_index("2026-07-10", FIXTURE.read_text(), "https://sec.test/master.idx")

    assert first == {"discovered": 5, "registered": 5, "replayed": False}
    assert replay == {"discovered": 0, "registered": 0, "replayed": True}
    assert store.sqlite.count_filings() == 5
    assert store.get_sec_daily_index_cursor() == "2026-07-10"
    with store.sqlite._connect() as conn:
        scopes = dict(conn.execute(
            "SELECT accession, discovery_scope FROM filings"
        ).fetchall())
    assert scopes["0000950170-26-099001"] == "deep"
    assert scopes["0001193125-26-188001"] == "broad"


def test_cursor_is_not_advanced_when_registration_transaction_fails(store: Store) -> None:
    _seed_broad_universe(store)
    discovery = SECDailyIndexDiscovery(
        store=store,
        coverage_resolver=CoverageResolver(store),
        sec_config=_config(),
        user_agent="Researcher test@example.com",
        request_delay=0,
    )

    with patch.object(store, "register_sec_daily_index", side_effect=RuntimeError("commit failed")):
        with pytest.raises(RuntimeError, match="commit failed"):
            discovery.process_index("2026-07-10", FIXTURE.read_text(), "https://sec.test/master.idx")

    assert store.get_sec_daily_index_cursor() is None


def test_discovery_downloads_one_index_once_not_one_request_per_ticker(store: Store) -> None:
    _seed_broad_universe(store)
    response = MagicMock(text=FIXTURE.read_text())
    response.raise_for_status.return_value = None
    get = MagicMock(return_value=response)
    discovery = SECDailyIndexDiscovery(
        store=store,
        coverage_resolver=CoverageResolver(store),
        sec_config=_config(),
        user_agent="Researcher test@example.com",
        request_delay=0,
        http_get=get,
    )

    result = discovery.discover_dates(["2026-07-10"])

    assert result["registered"] == 5
    assert get.call_count == 1
    discovery.discover_dates(["2026-07-10"])
    assert get.call_count == 1


def test_bulk_submissions_bootstrap_downloads_once_and_filters_locally(store: Store) -> None:
    _seed_broad_universe(store)
    response = MagicMock(content=_bulk_zip())
    response.raise_for_status.return_value = None
    get = MagicMock(return_value=response)
    discovery = SECDailyIndexDiscovery(
        store=store,
        coverage_resolver=CoverageResolver(store),
        sec_config=_config(),
        user_agent="Researcher test@example.com",
        request_delay=0,
        http_get=get,
    )

    first = discovery.bootstrap_from_submissions_bulk()
    replay = discovery.bootstrap_from_submissions_bulk()

    assert first == {"discovered": 3, "registered": 3, "replayed": False}
    assert replay == {"discovered": 0, "registered": 0, "replayed": True}
    assert get.call_count == 1
    with store.sqlite._connect() as conn:
        rows = conn.execute(
            "SELECT accession, discovery_scope FROM filings ORDER BY accession"
        ).fetchall()
    assert dict(rows) == {
        "0000950170-26-099101": "deep",
        "0001193125-26-188101": "broad",
        "0001193125-26-188102": "broad",
    }


def _discovery_with(store: Store, get) -> SECDailyIndexDiscovery:
    return SECDailyIndexDiscovery(
        store=store,
        coverage_resolver=CoverageResolver(store),
        sec_config=_config(),
        user_agent="Researcher test@example.com",
        request_delay=0,
        http_get=get,
    )


def _http_403() -> MagicMock:
    """EDGAR's S3 answers 403 AccessDenied for daily indexes it never published."""
    response = MagicMock(status_code=403, text="<Error><Code>AccessDenied</Code></Error>")
    error = Exception("403 Client Error: Forbidden")
    error.response = response
    response.raise_for_status.side_effect = error
    return response


def test_holiday_403_is_marked_absent_when_a_later_date_succeeds(store: Store) -> None:
    """A 403 date older than a same-pass success is a holiday, not a rate limit
    (found live 2026-07-17: Juneteenth 2026-06-19 pinned the cursor for days)."""
    _seed_broad_universe(store)
    ok = MagicMock(text=FIXTURE.read_text())
    ok.raise_for_status.return_value = None
    get = MagicMock(side_effect=[_http_403(), ok])
    discovery = _discovery_with(store, get)

    result = discovery.discover_dates(["2026-06-19", "2026-07-10"])

    assert result["absent"] == ["2026-06-19"]
    assert result["failed"] == 0
    assert result.get("errors") == []
    assert "error_class" not in result
    assert store.get_sec_daily_index_status("2026-06-19") == "absent"
    from src.scheduler import UnifiedScheduler

    status, reason = UnifiedScheduler._classify_detail(
        {
            "mode": "daily_index",
            "checked": 1,
            "new_filings": int(result.get("registered") or 0),
            "failed": result["failed"],
            "details": result,
        }
    )
    assert status == "success"
    assert reason is None
    # The absent date is never re-fetched.
    discovery.discover_dates(["2026-06-19"])
    assert get.call_count == 2


def test_provider_wide_403_block_is_not_marked_absent(store: Store) -> None:
    """When every date 403s (a real EDGAR block) nothing is absent-marked and
    the normal retry/backoff classification stays in charge."""
    _seed_broad_universe(store)
    get = MagicMock(side_effect=[_http_403(), _http_403()])
    discovery = _discovery_with(store, get)

    result = discovery.discover_dates(["2026-06-19", "2026-07-10"])

    assert "absent" not in result
    assert result["failed"] == 2
    assert store.get_sec_daily_index_status("2026-06-19") is None
    assert store.get_sec_daily_index_status("2026-07-10") is None


def test_403_newer_than_every_success_is_retried_not_absent(store: Store) -> None:
    """A 403 on the newest date may be publication lag; only dates older than
    a same-pass success are provably unpublished."""
    _seed_broad_universe(store)
    ok = MagicMock(text=FIXTURE.read_text())
    ok.raise_for_status.return_value = None
    get = MagicMock(side_effect=[ok, _http_403()])
    discovery = _discovery_with(store, get)

    result = discovery.discover_dates(["2026-07-10", "2026-07-13"])

    assert "absent" not in result
    assert result["failed"] == 1
    assert store.get_sec_daily_index_status("2026-07-13") is None

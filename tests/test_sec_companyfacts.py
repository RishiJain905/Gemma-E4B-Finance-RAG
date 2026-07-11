"""
tests/test_sec_companyfacts.py
Offline contract tests for SEC CompanyFacts ingestion and selection.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.scheduler import UnifiedScheduler
from src.sec.companyfacts import SECCompanyFactsIngestor
from src.storage.sqlite_store import SQLiteStore
from src.storage.store import select_companyfacts
from src.storage.store import Store


FIXTURE_PATH = Path(__file__).parent / "fixtures/sec/companyfacts_sample.json"


def _config(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "user_agent": "TraceAlchemy Tests tests@example.com",
        "allowed_taxonomies": ["us-gaap"],
        "allowed_forms": ["10-K", "10-K/A", "10-Q", "10-Q/A"],
        "timeout_seconds": 7,
        "retries": 2,
        "backoff_factor": 0.1,
        "request_delay_seconds": 0,
        "metrics": {
            "total_revenue": {
                "concepts": [
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "Revenues",
                ],
                "units": ["USD"],
                "period_kinds": ["quarterly", "ytd", "annual"],
            },
            "total_assets": {
                "concepts": ["Assets"],
                "units": ["USD"],
                "period_kinds": ["instant"],
            },
            "diluted_eps": {
                "concepts": ["EarningsPerShareDiluted"],
                "units": ["USD/shares"],
                "period_kinds": ["quarterly", "ytd", "annual"],
            },
        },
    }


def _session(payload: str | None = None) -> MagicMock:
    session = MagicMock()
    response = MagicMock()
    response.text = payload if payload is not None else FIXTURE_PATH.read_text(encoding="utf-8")
    response.raise_for_status.return_value = None
    session.get.return_value = response
    return session


def _ingestor(tmp_path: Path, *, clock=None, enabled: bool = True):
    sqlite = SQLiteStore(tmp_path / "companyfacts.db")
    session = _session()
    resolver = MagicMock()
    resolver.resolve_cik.return_value = "1045810"
    ingestor = SECCompanyFactsIngestor(
        store=sqlite,
        config=_config(enabled),
        session=session,
        cik_resolver=resolver,
        clock=clock or (lambda: datetime(2026, 7, 10, 12, tzinfo=timezone.utc)),
        sleep=lambda _seconds: None,
    )
    return ingestor, sqlite, session, resolver


def _rows(tmp_path: Path) -> tuple[SQLiteStore, list[dict]]:
    ingestor, sqlite, _session_mock, _resolver = _ingestor(tmp_path)
    summary = ingestor.fetch_for_ticker("nvda")
    assert summary["errors"] == []
    return sqlite, sqlite.query_sec_companyfacts(
        "NVDA",
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "Assets",
            "EarningsPerShareDiluted",
        ],
        as_of="2026-07-10",
    )


def test_request_uses_padded_cik_and_required_user_agent(tmp_path):
    ingestor, _sqlite, session, resolver = _ingestor(tmp_path)

    ingestor.fetch_for_ticker("nvda")

    resolver.resolve_cik.assert_called_once_with("NVDA")
    session.get.assert_called_once_with(
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
        headers={"User-Agent": "TraceAlchemy Tests tests@example.com"},
        timeout=7.0,
    )


def test_normalizes_duration_instant_and_ytd_facts(tmp_path):
    _sqlite, rows = _rows(tmp_path)
    kinds = {(row["concept"], row["period_end"]): row["period_kind"] for row in rows}

    assert kinds[("Assets", "2025-01-26")] == "instant"
    assert kinds[("RevenueFromContractWithCustomerExcludingAssessedTax", "2025-04-27")] == "quarterly"
    assert kinds[("RevenueFromContractWithCustomerExcludingAssessedTax", "2025-10-26")] == "ytd"
    assert kinds[("RevenueFromContractWithCustomerExcludingAssessedTax", "2025-01-26")] == "annual"


def test_decimal_text_and_unit_are_preserved(tmp_path):
    sqlite, _rows_result = _rows(tmp_path)
    eps = sqlite.query_sec_companyfacts(
        "NVDA", ["EarningsPerShareDiluted"], as_of="2026-07-10"
    )[0]

    assert eps["value_text"] == "1.2300"
    assert eps["value_numeric"] == pytest.approx(1.23)
    assert eps["unit"] == "USD/shares"


def test_reingestion_is_idempotent(tmp_path):
    times = iter(
        [
            datetime(2026, 7, 10, 12, tzinfo=timezone.utc),
            datetime(2026, 7, 11, 12, tzinfo=timezone.utc),
        ]
    )
    ingestor, sqlite, _session_mock, _resolver = _ingestor(tmp_path, clock=lambda: next(times))

    first = ingestor.fetch_for_ticker("NVDA")
    count = sqlite.count_sec_companyfacts("NVDA")
    second = ingestor.fetch_for_ticker("NVDA")
    rows = sqlite.query_sec_companyfacts(
        "NVDA", ["EarningsPerShareDiluted"], as_of="2026-07-11"
    )

    assert first["facts_written"] == second["facts_written"] == count
    assert sqlite.count_sec_companyfacts("NVDA") == count
    assert rows[0]["source_accessed_at"] == "2026-07-11T12:00:00Z"


def test_as_of_excludes_later_filing(tmp_path):
    sqlite, _rows_result = _rows(tmp_path)

    rows = sqlite.query_sec_companyfacts(
        "NVDA",
        ["RevenueFromContractWithCustomerExcludingAssessedTax"],
        period_end="2025-01-26",
        as_of="2025-03-01",
    )

    assert {row["accession"] for row in rows} == {"0001045810-25-000023"}
    with pytest.raises(ValueError):
        sqlite.query_sec_companyfacts("NVDA", ["Assets"], as_of="not-a-date")


def test_amendment_precedence_preserves_original(tmp_path):
    sqlite, _rows_result = _rows(tmp_path)
    rules = _config()["metrics"]
    historical = select_companyfacts(
        sqlite.query_sec_companyfacts("NVDA", rules["total_revenue"]["concepts"], as_of="2025-03-01"),
        rules,
        ["total_revenue"],
        periods=["2025-01-26"],
        as_of="2025-03-01",
    )
    current = select_companyfacts(
        sqlite.query_sec_companyfacts("NVDA", rules["total_revenue"]["concepts"], as_of="2026-07-10"),
        rules,
        ["total_revenue"],
        periods=["2025-01-26"],
        as_of="2026-07-10",
    )

    assert sqlite.count_sec_companyfacts("NVDA") > len(current)
    assert historical[0]["accession"] == "0001045810-25-000023"
    assert current[0]["accession"] == "0001045810-25-000099"


def test_conflicting_concepts_are_reported_not_averaged(tmp_path):
    _sqlite, rows = _rows(tmp_path)
    selected = select_companyfacts(
        rows,
        _config()["metrics"],
        ["total_revenue"],
        periods=["2025-01-26"],
        as_of="2026-07-10",
    )[0]

    assert selected["value"] == 61000000000.0
    assert selected["conflict"] is True
    assert {alternative["value_text"] for alternative in selected["alternatives"]} == {
        "60922000000",
        "60000000000",
    }


def test_metric_alias_priority_is_deterministic(tmp_path):
    _sqlite, rows = _rows(tmp_path)
    rules = _config()["metrics"]

    first = select_companyfacts(rows, rules, ["total_revenue"], periods=["2025-01-26"])
    second = select_companyfacts(list(reversed(rows)), rules, ["total_revenue"], periods=["2025-01-26"])

    assert first == second
    assert first[0]["concept"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_store_get_companyfacts_returns_retrieval_contract(tmp_path):
    sqlite, _rows_result = _rows(tmp_path)
    config_path = tmp_path / "companyfacts.yaml"
    config_path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    store = Store.__new__(Store)
    store.sqlite = sqlite
    store.COMPANYFACTS_CONFIG_PATH = config_path

    result = store.get_companyfacts(
        "NVDA", ["total_assets"], periods=["2025-01-26"], as_of="2026-07-10"
    )

    assert result == [
        {
            "ticker": "NVDA",
            "metric": "total_assets",
            "value": 111601000000.0,
            "value_text": "111601000000",
            "unit": "USD",
            "period": "2025-01-26",
            "period_start": "",
            "period_type": "instant",
            "source_type": "sec_companyfacts",
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
            "source_accessed_at": "2026-07-10T12:00:00Z",
            "taxonomy": "us-gaap",
            "concept": "Assets",
            "accession": "0001045810-25-000023",
            "form": "10-K",
            "filed_at": "2025-02-26",
            "as_of": "2026-07-10",
            "conflict": False,
            "alternatives": [],
        }
    ]


def test_bad_fact_is_counted_and_siblings_continue(tmp_path):
    ingestor, sqlite, _session_mock, _resolver = _ingestor(tmp_path)

    summary = ingestor.fetch_for_ticker("NVDA")

    assert summary["facts_seen"] == 8
    assert summary["facts_written"] == 7
    assert summary["facts_skipped"] == 1
    assert sqlite.count_sec_companyfacts("NVDA") == 7
    assert summary["errors"] == []


def test_scheduler_isolates_ticker_failure(tmp_path):
    sqlite = SQLiteStore(tmp_path / "scheduler.db")
    store = MagicMock()
    store.sqlite = sqlite
    scheduler = UnifiedScheduler(store=store, inter_source_delay=0)
    ingestor = MagicMock()
    ingestor.enabled = True
    ingestor.fetch_for_ticker.side_effect = [RuntimeError("CIK unavailable"), {
        "ticker": "AMD", "cik": "0000002488", "facts_seen": 2,
        "facts_written": 2, "facts_skipped": 0, "errors": [],
        "source_accessed_at": "2026-07-10T12:00:00Z",
    }]

    with patch("src.sec.SECCompanyFactsIngestor", return_value=ingestor), patch.object(
        scheduler, "_load_core_tickers", return_value=["NVDA", "AMD"]
    ):
        result = scheduler._run_source("sec_companyfacts")

    assert result["tickers_processed"] == 2
    assert result["tickers_failed"] == 1
    store.mark_source_stale.assert_called_once_with("NVDA", "sec_companyfacts", "CIK unavailable")
    store.mark_source_fresh.assert_called_once_with("AMD", "sec_companyfacts", 24)


def test_disabled_source_does_not_fetch(tmp_path):
    store = MagicMock()
    scheduler = UnifiedScheduler(store=store, inter_source_delay=0)
    ingestor = MagicMock()
    ingestor.enabled = False

    with patch("src.sec.SECCompanyFactsIngestor", return_value=ingestor), patch.object(
        scheduler, "_load_core_tickers", return_value=["NVDA"]
    ):
        result = scheduler._run_source("sec_companyfacts")

    assert result == {"enabled": False, "tickers_processed": 0}
    ingestor.fetch_for_ticker.assert_not_called()
    store.mark_source_fresh.assert_not_called()
    store.mark_source_stale.assert_not_called()

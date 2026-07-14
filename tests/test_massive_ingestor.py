"""Offline tests for the Massive grouped-market and corporate-action adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.storage.store import Store


FIXTURES = Path(__file__).parent / "fixtures" / "providers" / "massive"


class FakeResponse:
    """Small requests-compatible response used by the offline provider tests."""

    def __init__(self, payload: object, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def json(self) -> object:
        return self._payload


def _load(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _store(tmp_path: Path) -> tuple[Store, MagicMock]:
    with patch("src.storage.store.ChromaStore") as chroma_class:
        chroma = MagicMock()
        chroma_class.return_value = chroma
        store = Store(db_path=tmp_path / "massive.db", chroma_path=tmp_path / "chroma")
    store.upsert_universe_snapshot(
        "ivv",
        "2026-07-14T00:00:00Z",
        [
            {"symbol": "AAA", "company_name": "Alpha Corp", "index_code": "sp500"},
            {"symbol": "BBB", "company_name": "Beta Corp", "index_code": "sp500"},
            {"symbol": "CCC", "company_name": "Gamma Corp", "index_code": "sp500"},
        ],
    )
    return store, chroma


def _coverage() -> MagicMock:
    coverage = MagicMock()
    coverage.tickers_for.return_value = ["AAA", "BBB"]
    return coverage


def _http_router(payloads: dict[str, object]):
    calls: list[str] = []

    def http_get(url: str, **_kwargs):
        calls.append(url)
        for key, payload in payloads.items():
            if key in url:
                return FakeResponse(payload)
        raise AssertionError(f"unexpected provider URL: {url}")

    return http_get, calls


def test_grouped_market_summary_filters_locally_and_corrected_bars_upsert(
    tmp_path: Path,
) -> None:
    """One grouped request stores only active symbols and revisions replace the day."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, chroma = _store(tmp_path)
    first = _load("grouped-2026-07-13.json")
    corrected = _load("grouped-2026-07-13-corrected.json")
    responses = iter([first, corrected])
    calls: list[str] = []

    def http_get(url: str, **_kwargs):
        calls.append(url)
        return FakeResponse(next(responses))

    ingestor = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        now_fn=lambda: "2026-07-14T20:00:00Z",
        sleep_fn=lambda _seconds: None,
    )

    first_result = ingestor.ingest_market_data(
        start_date="2026-07-13", end_date="2026-07-13"
    )
    second_result = ingestor.ingest_market_data(
        start_date="2026-07-13", end_date="2026-07-13"
    )

    assert first_result["status"] == "ok"
    assert second_result["updated"] >= 1
    assert len(calls) == 2
    assert all("grouped" in url for url in calls)
    assert all("AAA" not in url and "BBB" not in url for url in calls)
    assert store.sqlite.count_observations() == 10  # 2 securities x 5 OHLCV metrics
    assert store.get_source_cursor("massive_market", "US") == "2026-07-13"
    assert chroma.add_document.call_count == 0

    with store.sqlite._connect() as conn:
        close = conn.execute(
            "SELECT value_numeric, metadata_json FROM corpus_observations "
            "WHERE source_name=? AND metric_id=? AND period_end=? AND tickers_json=?",
            ("massive", "close", "2026-07-13", '["AAA"]'),
        ).fetchone()
    assert close["value_numeric"] == 101.25
    metadata = json.loads(close["metadata_json"])
    assert metadata["adjusted"] is True
    assert metadata["market_date"] == "2026-07-13"
    assert metadata["provider_revision"] == "rev-2"


def test_massive_splits_and_dividends_are_structured_idempotent_events(
    tmp_path: Path,
) -> None:
    """Action dates survive in EventRecord metadata and replay creates no duplicates."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, _chroma = _store(tmp_path)
    http_get, calls = _http_router({
        "/splits": _load("splits.json"),
        "/dividends": _load("dividends.json"),
    })
    ingestor = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        now_fn=lambda: "2026-07-14T20:00:00Z",
        sleep_fn=lambda _seconds: None,
    )

    first = ingestor.ingest_corporate_actions(
        start_date="2026-07-01", end_date="2026-07-14"
    )
    second = ingestor.ingest_corporate_actions(
        start_date="2026-07-01", end_date="2026-07-14"
    )

    assert first["status"] == "ok"
    assert first["stored"] == 2
    assert second["duplicates"] == 2
    assert store.sqlite.count_events() == 2
    assert len([url for url in calls if "/grouped/" in url]) == 0
    dividend = store.sqlite.get_event("event/massive/div-1")
    assert dividend["event_type"] == "dividend"
    assert dividend["amount"] == 0.5
    assert dividend["metadata"]["ex_date"] == "2026-07-15"
    assert dividend["metadata"]["payable_date"] == "2026-07-31"
    assert dividend["metadata"]["record_date"] == "2026-07-16"


def test_massive_entitlement_disables_market_only_and_other_capability_runs(
    tmp_path: Path,
) -> None:
    """A denied grouped endpoint does not prevent entitled corporate actions."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, _chroma = _store(tmp_path)
    calls: list[str] = []

    def http_get(url: str, **_kwargs):
        calls.append(url)
        if "/grouped/" in url:
            return FakeResponse({"error": "plan does not include grouped market data"}, 403)
        if "/splits" in url:
            return FakeResponse(_load("splits.json"))
        return FakeResponse(_load("dividends.json"))

    ingestor = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        sleep_fn=lambda _seconds: None,
    )

    market = ingestor.ingest_market_data(start_date="2026-07-13", end_date="2026-07-13")
    actions = ingestor.ingest_corporate_actions(
        start_date="2026-07-01", end_date="2026-07-14"
    )

    assert market["status"] == "disabled_entitlement"
    assert actions["status"] == "ok"
    assert store.sqlite.count_events() == 2
    assert len(calls) == 3


def test_massive_optional_news_and_vendor_filings_use_secondary_provenance(
    tmp_path: Path,
) -> None:
    """Optional narrative inputs use the shared Store and never claim SEC authority."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, chroma = _store(tmp_path)
    http_get, _calls = _http_router({
        "/news": _load("news.json"),
        "/filings": _load("filings.json"),
    })
    ingestor = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        now_fn=lambda: "2026-07-14T20:00:00Z",
        sleep_fn=lambda _seconds: None,
    )

    news = ingestor.ingest_news(start_date="2026-07-14", end_date="2026-07-14")
    filings = ingestor.ingest_vendor_filings(
        start_date="2026-07-14", end_date="2026-07-14"
    )

    assert news["stored"] == 1
    assert filings["stored"] == 1
    assert chroma.add_document.call_count == 2
    filing = store.sqlite.get_corpus_item("filing/massive/AAA/filing-1")
    assert filing["source_category"] == "vendor_filing_metadata"
    assert filing["evidence_authority"] == "provider"
    assert filing["metadata"]["accession"] == "0000000000-26-000001"


def test_massive_429_exposes_retry_metadata_without_advancing_date_cursor(
    tmp_path: Path,
) -> None:
    """Exhausted grouped-data rate limiting is bounded and replayable."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, _chroma = _store(tmp_path)
    http_get = MagicMock(
        side_effect=[
            FakeResponse({"error": "rate limit"}, 429, {"Retry-After": "11"}),
            FakeResponse({"error": "rate limit"}, 429, {"Retry-After": "11"}),
            FakeResponse({"error": "rate limit"}, 429, {"Retry-After": "11"}),
        ]
    )
    sleeps: list[float] = []
    result = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        sleep_fn=sleeps.append,
        max_attempts=3,
    ).ingest_market_data(start_date="2026-07-13", end_date="2026-07-13")

    assert result["status"] == "rate_limited"
    assert result["retry_after"] == 11.0
    assert http_get.call_count == 3
    assert sleeps == [11.0, 11.0]
    assert store.get_source_cursor("massive_market", "US") is None


@pytest.mark.parametrize(
    ("status_code", "payload", "expected_status"),
    [
        (401, {"error": "invalid api key"}, "disabled_authentication"),
        (403, {"error": "entitlement required for corporate actions"}, "disabled_entitlement"),
    ],
)
def test_massive_auth_and_entitlement_states_are_capability_local(
    tmp_path: Path,
    status_code: int,
    payload: dict,
    expected_status: str,
) -> None:
    """Authentication and plan errors are not treated as transient retries."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, _chroma = _store(tmp_path)
    http_get = MagicMock(return_value=FakeResponse(payload, status_code=status_code))
    result = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        sleep_fn=lambda _seconds: None,
    ).ingest_market_data(start_date="2026-07-13", end_date="2026-07-13")

    assert result["status"] == expected_status
    assert http_get.call_count == 1

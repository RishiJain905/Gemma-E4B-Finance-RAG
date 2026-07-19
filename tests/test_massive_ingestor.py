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
    assert all("/locale/us/market/stocks/" in url for url in calls)
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


def test_grouped_market_prefers_unique_cik_identity_for_ambiguous_ticker(
    tmp_path: Path,
) -> None:
    """Massive US bars use the SEC-backed identity without relaxing defaults."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    with patch("src.storage.store.ChromaStore") as chroma_class:
        chroma_class.return_value = MagicMock()
        store = Store(
            db_path=tmp_path / "massive-cboe.db",
            chroma_path=tmp_path / "chroma",
        )
    store.upsert_universe_snapshot(
        "sec",
        "2026-07-01T00:00:00Z",
        [
            {
                "symbol": "CBOE",
                "company_name": "Cboe Global Markets, Inc.",
                "exchange": "CBOE",
                "cik": "0001374310",
            }
        ],
    )
    store.upsert_universe_snapshot(
        "ivv",
        "2026-07-02T00:00:00Z",
        [
            {
                "symbol": "CBOE",
                "company_name": "CBOE GLOBAL MARKETS INC",
                "exchange": "CBOE BZX",
                "index_code": "sp500",
            }
        ],
    )
    coverage = MagicMock()
    coverage.tickers_for.return_value = ["CBOE"]
    payload = {
        "status": "OK",
        "results": [
            {"T": "CBOE", "o": 239.0, "h": 241.0, "l": 238.0, "c": 240.0, "v": 12345}
        ],
    }

    result = MassiveIngestor(
        store=store,
        coverage_resolver=coverage,
        api_key="test-massive-key",
        http_get=lambda _url, **_kwargs: FakeResponse(payload),
        now_fn=lambda: "2026-07-14T20:00:00Z",
    ).ingest_market_data(start_date="2026-07-13", end_date="2026-07-13")

    assert result["status"] == "ok"
    assert result["malformed"] == 0
    assert store.sqlite.count_observations() == 5


def test_grouped_market_zero_result_envelope_is_an_empty_market_day() -> None:
    from src.ingestion.massive_ingestor import MassiveIngestor

    assert MassiveIngestor._rows_from_payload(
        {"status": "OK", "queryCount": 0, "resultsCount": 0},
    ) == []


def test_grouped_market_does_not_request_incomplete_current_day(tmp_path: Path) -> None:
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, _chroma = _store(tmp_path)
    calls: list[str] = []

    def http_get(url: str, **_kwargs) -> FakeResponse:
        calls.append(url)
        return FakeResponse(_load("grouped-2026-07-13.json"))

    result = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        now_fn=lambda: "2026-07-14T12:00:00Z",
    ).ingest_market_data(start_date="2026-07-13", end_date="2026-07-14")

    assert result["status"] == "ok"
    assert len(calls) == 1
    assert calls[0].endswith("/2026-07-13")
    assert result["cursor_after"] == "2026-07-13"


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


def test_massive_news_uses_timestamp_cursor_with_overlap_and_oldest_first(
    tmp_path: Path,
) -> None:
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, _chroma = _store(tmp_path)
    store.set_source_cursor(
        "massive_news",
        "US",
        "2026-07-14T11:30:00Z",
        cursor_type="timestamp",
        status="success",
    )
    payload = _load("news.json")
    payload["results"][0]["published_utc"] = "2026-07-14T12:45:00Z"
    params_seen: list[dict] = []

    def http_get(_url: str, **kwargs) -> FakeResponse:
        params_seen.append(kwargs["params"])
        return FakeResponse(payload)

    result = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        news_overlap_hours=2,
        now_fn=lambda: "2026-07-14T20:00:00Z",
        sleep_fn=lambda _seconds: None,
    ).ingest_news()

    assert params_seen[0]["published_utc.gte"] == "2026-07-14T09:30:00Z"
    assert params_seen[0]["published_utc.lte"] == "2026-07-14T20:00:00Z"
    assert params_seen[0]["sort"] == "published_utc"
    assert params_seen[0]["order"] == "asc"
    assert result["cursor_before"] == "2026-07-14T11:30:00Z"
    assert result["cursor_after"] == "2026-07-14T12:45:00Z"
    assert store.get_source_cursor("massive_news", "US") == "2026-07-14T12:45:00Z"


def test_massive_news_does_not_advance_cursor_when_indexing_fails(
    tmp_path: Path,
) -> None:
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, chroma = _store(tmp_path)
    store.set_source_cursor(
        "massive_news",
        "US",
        "2026-07-14T11:30:00Z",
        cursor_type="timestamp",
        status="success",
    )
    payload = _load("news.json")
    payload["results"][0]["published_utc"] = "2026-07-14T12:45:00Z"
    chroma.add_document.side_effect = RuntimeError("embedding unavailable")

    result = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=lambda *_args, **_kwargs: FakeResponse(payload),
        news_overlap_hours=2,
        now_fn=lambda: "2026-07-14T20:00:00Z",
        sleep_fn=lambda _seconds: None,
    ).ingest_news()

    assert result["status"] == "partial"
    assert result["cursor_after"] == "2026-07-14T11:30:00Z"
    assert store.get_source_cursor("massive_news", "US") == "2026-07-14T11:30:00Z"


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


def test_massive_provider_429_skips_remaining_capabilities(tmp_path: Path) -> None:
    """One exhausted provider circuit prevents corporate-action requests."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, _chroma = _store(tmp_path)
    response = FakeResponse(
        {"error": "rate limit"}, 429, {"Retry-After": "0"}
    )
    http_get = MagicMock(return_value=response)
    result = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        sleep_fn=lambda _seconds: None,
        max_attempts=3,
    ).ingest_all(start_date="2026-07-13", end_date="2026-07-13")

    assert http_get.call_count == 3
    assert result["market"]["error_class"] == "rate_limited"
    assert result["corporate_actions"]["status"] == "skipped_provider_circuit"
    assert result["corporate_actions"]["remaining_work_skipped"] is True


def test_massive_permanent_market_404_continues_next_date(tmp_path: Path) -> None:
    """A missing daily resource is isolated without opening the provider circuit."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, _chroma = _store(tmp_path)
    requested: list[str] = []

    def http_get(url: str, **_kwargs) -> FakeResponse:
        requested.append(url)
        if url.endswith("2026-07-13"):
            return FakeResponse({"error": "not found"}, 404)
        return FakeResponse(_load("grouped-2026-07-13.json"))

    result = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        sleep_fn=lambda _seconds: None,
    ).ingest_market_data(start_date="2026-07-13", end_date="2026-07-14")

    assert len(requested) == 2
    assert result["status"] == "partial"
    assert result["stored"] > 0
    assert result["rejected_items"] == 1
    assert store.get_source_cursor("massive_market", "US") == "2026-07-14"


def test_massive_permanent_action_404_continues_next_endpoint(
    tmp_path: Path,
) -> None:
    """One unavailable action resource does not suppress another entitled resource."""
    from src.ingestion.massive_ingestor import MassiveIngestor

    store, _chroma = _store(tmp_path)
    requested: list[str] = []

    def http_get(url: str, **_kwargs) -> FakeResponse:
        requested.append(url)
        if "/splits" in url:
            return FakeResponse({"error": "not found"}, 404)
        return FakeResponse(_load("dividends.json"))

    result = MassiveIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-massive-key",
        http_get=http_get,
        sleep_fn=lambda _seconds: None,
    ).ingest_corporate_actions(
        start_date="2026-07-01", end_date="2026-07-14"
    )

    assert len(requested) == 2
    assert result["status"] == "partial"
    assert result["stored"] == 1
    assert result["rejected_items"] == 1


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

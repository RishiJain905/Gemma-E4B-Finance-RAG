"""Offline tests for the Finnhub company-news adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.storage.store import Store


FIXTURES = Path(__file__).parent / "fixtures" / "providers" / "finnhub"


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
        store = Store(db_path=tmp_path / "finnhub.db", chroma_path=tmp_path / "chroma")
    store.upsert_universe_snapshot(
        "ivv",
        "2026-07-14T00:00:00Z",
        [
            {"symbol": "AAA", "company_name": "Alpha Corp", "index_code": "sp500"},
            {"symbol": "BBB", "company_name": "Beta Corp", "index_code": "sp500"},
        ],
    )
    return store, chroma


def _coverage() -> MagicMock:
    coverage = MagicMock()
    coverage.tickers_for.return_value = ["AAA", "BBB"]
    return coverage


def test_finnhub_paginates_overlap_and_advances_cursor_after_storage(tmp_path: Path) -> None:
    """Valid pages are stored through Store and the ticker cursor moves to the newest item."""
    from src.ingestion.finnhub_ingestor import FinnhubIngestor

    store, chroma = _store(tmp_path)
    store.set_source_cursor(
        "finnhub_news", "AAA", "2026-07-14T10:00:00Z", cursor_type="timestamp"
    )
    calls: list[dict] = []

    def http_get(_url: str, **kwargs):
        calls.append(kwargs)
        return FakeResponse(_load("page-1.json" if kwargs["params"].get("page", 1) == 1 else "page-2.json"))

    ingestor = FinnhubIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-finnhub-key",
        http_get=http_get,
        now_fn=lambda: "2026-07-14T13:00:00Z",
        sleep_fn=lambda _seconds: None,
        overlap_hours=6,
    )

    result = ingestor.ingest_ticker_news("AAA")

    assert result["status"] == "partial"
    assert result["stored"] == 2
    assert result["malformed"] == 1
    assert len(calls) == 2
    assert calls[0]["params"]["from"] == "2026-07-14"
    assert store.get_source_cursor("finnhub_news", "AAA") == "2026-07-14T12:00:00Z"
    assert store.sqlite.count_corpus_items() == 2
    assert chroma.add_document.call_count == 2

    stored = store.sqlite.get_corpus_item("news/finnhub/AAA/aaa-1")
    assert stored["title"] == "Alpha announces a new product"
    assert stored["summary"] == "Alpha announced a new product today."
    assert stored["original_publisher"] == "Reuters"
    assert stored["source_url"] == "https://news.example/alpha-1"


def test_finnhub_cursor_does_not_advance_when_storage_fails(tmp_path: Path) -> None:
    """A failed Store write leaves the last committed cursor intact for replay."""
    from src.ingestion.finnhub_ingestor import FinnhubIngestor

    store, _chroma = _store(tmp_path)
    before = "2026-07-14T10:00:00Z"
    store.set_source_cursor("finnhub_news", "AAA", before, cursor_type="timestamp")
    store.upsert_narrative = MagicMock(side_effect=RuntimeError("sqlite unavailable"))
    ingestor = FinnhubIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-finnhub-key",
        http_get=lambda _url, **_kwargs: FakeResponse(_load("page-1.json")),
        now_fn=lambda: "2026-07-14T13:00:00Z",
        sleep_fn=lambda _seconds: None,
    )

    result = ingestor.ingest_ticker_news("AAA")

    assert result["status"] == "error"
    assert store.get_source_cursor("finnhub_news", "AAA") == before
    assert result["cursor_after"] == before


def test_finnhub_missing_key_disables_only_news_without_http(tmp_path: Path) -> None:
    """No credential is a capability state, not a provider-wide exception."""
    from src.ingestion.finnhub_ingestor import FinnhubIngestor

    store, _chroma = _store(tmp_path)
    http_get = MagicMock()
    result = FinnhubIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="",
        http_get=http_get,
    ).ingest_news()

    assert result["status"] == "disabled_missing_key"
    assert result["error_class"] == "authentication"
    http_get.assert_not_called()


@pytest.mark.parametrize(
    ("status_code", "payload", "expected_status"),
    [
        (401, {"error": "invalid api key"}, "disabled_authentication"),
        (403, {"error": "plan does not include company news"}, "disabled_entitlement"),
    ],
)
def test_finnhub_auth_and_entitlement_denials_are_not_retried(
    tmp_path: Path,
    status_code: int,
    payload: dict,
    expected_status: str,
) -> None:
    """401/403 states stop the partition immediately and never consume retries."""
    from src.ingestion.finnhub_ingestor import FinnhubIngestor

    store, _chroma = _store(tmp_path)
    http_get = MagicMock(return_value=FakeResponse(payload, status_code=status_code))
    result = FinnhubIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-finnhub-key",
        http_get=http_get,
        sleep_fn=lambda _seconds: None,
    ).ingest_ticker_news("AAA")

    assert result["status"] == expected_status
    assert http_get.call_count == 1


def test_finnhub_429_honors_retry_metadata_and_keeps_cursor(tmp_path: Path) -> None:
    """A bounded 429 retry exposes Retry-After and does not checkpoint on exhaustion."""
    from src.ingestion.finnhub_ingestor import FinnhubIngestor

    store, _chroma = _store(tmp_path)
    before = "2026-07-14T10:00:00Z"
    store.set_source_cursor("finnhub_news", "AAA", before, cursor_type="timestamp")
    http_get = MagicMock(
        side_effect=[
            FakeResponse({"error": "rate limit"}, status_code=429, headers={"Retry-After": "7"}),
            FakeResponse({"error": "rate limit"}, status_code=429, headers={"Retry-After": "7"}),
            FakeResponse({"error": "rate limit"}, status_code=429, headers={"Retry-After": "7"}),
        ]
    )
    sleeps: list[float] = []
    result = FinnhubIngestor(
        store=store,
        coverage_resolver=_coverage(),
        api_key="test-finnhub-key",
        http_get=http_get,
        sleep_fn=sleeps.append,
        max_attempts=3,
    ).ingest_ticker_news("AAA")

    assert result["status"] == "rate_limited"
    assert result["error_class"] == "rate_limited"
    assert result["retry_after"] == 7.0
    assert http_get.call_count == 3
    assert sleeps == [7.0, 7.0]
    assert store.get_source_cursor("finnhub_news", "AAA") == before

"""Offline tests for free/freemium finance vendor adapters."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.storage.store import Store


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload) if not isinstance(payload, (str, bytes)) else str(payload)
        self.content = self.text.encode("utf-8")

    def json(self) -> object:
        if isinstance(self._payload, (str, bytes)):
            return json.loads(self.text)
        return self._payload


def _store(tmp_path: Path) -> tuple[Store, MagicMock]:
    with patch("src.storage.store.ChromaStore") as chroma_class:
        chroma = MagicMock()
        chroma_class.return_value = chroma
        store = Store(db_path=tmp_path / "vendors.db", chroma_path=tmp_path / "chroma")
    store.upsert_universe_snapshot(
        "ivv",
        "2026-07-14T00:00:00Z",
        [
            {"symbol": "AAA", "company_name": "Alpha Corp", "index_code": "sp500"},
            {"symbol": "BBB", "company_name": "Beta Corp", "index_code": "sp500"},
        ],
    )
    return store, chroma


def _coverage(tickers: list[str] | None = None) -> MagicMock:
    coverage = MagicMock()
    coverage.tickers_for.return_value = tickers or ["AAA"]
    return coverage


def test_alpha_vantage_overview_income_and_news(tmp_path: Path) -> None:
    from src.ingestion.alpha_vantage_ingestor import AlphaVantageIngestor

    store, chroma = _store(tmp_path)
    calls: list[dict] = []

    def http_get(_url: str, **kwargs):
        params = kwargs["params"]
        calls.append(params)
        fn = params.get("function")
        if fn == "OVERVIEW":
            return FakeResponse(
                {
                    "Symbol": "AAA",
                    "MarketCapitalization": "1000000",
                    "PERatio": "20.5",
                    "RevenueTTM": "500000",
                    "LatestQuarter": "2026-03-31",
                }
            )
        if fn == "INCOME_STATEMENT":
            return FakeResponse(
                {
                    "annualReports": [
                        {
                            "fiscalDateEnding": "2025-12-31",
                            "totalRevenue": "480000",
                            "netIncome": "90000",
                        }
                    ]
                }
            )
        if fn == "NEWS_SENTIMENT":
            return FakeResponse(
                {
                    "feed": [
                        {
                            "title": "AAA wins a contract",
                            "url": "https://news.example/aaa-1",
                            "time_published": "20260714T120000",
                            "summary": "Alpha announced a contract win.",
                            "source": "Reuters",
                            "overall_sentiment_score": "0.2",
                            "ticker_sentiment": [
                                {"ticker": "AAA", "ticker_sentiment_score": "0.35"}
                            ],
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected function {fn}")

    result = AlphaVantageIngestor(
        store=store,
        coverage_resolver=_coverage(["AAA"]),
        api_key="test-av-key",
        http_get=http_get,
        now_fn=lambda: "2026-07-14T13:00:00Z",
        sleep_fn=lambda _s: None,
    ).ingest()

    assert result["status"] == "ok"
    assert result["facts_stored"] >= 3
    assert result["narratives_stored"] == 1
    assert any(c.get("function") == "OVERVIEW" for c in calls)
    assert store.get_fundamental("AAA", "market_cap")["value"] == 1_000_000.0
    assert store.sqlite.count_corpus_items() == 1
    assert chroma.add_document.call_count == 1
    items = store.sqlite.list_corpus_items(limit=5)
    assert items
    assert items[0]["title"] == "AAA wins a contract"
    detail = store.sqlite.get_corpus_item(items[0]["corpus_item_id"])
    assert "Sentiment 0.35" in (detail.get("summary") or "")


def test_alpha_vantage_missing_key_disables(tmp_path: Path) -> None:
    from src.ingestion.alpha_vantage_ingestor import AlphaVantageIngestor

    store, _ = _store(tmp_path)
    result = AlphaVantageIngestor(
        store=store,
        coverage_resolver=_coverage(["AAA"]),
        api_key="",
        http_get=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no http")),
    ).ingest()
    assert result["status"] == "disabled_missing_key"


def test_fmp_income_and_ratios(tmp_path: Path) -> None:
    from src.ingestion.fmp_ingestor import FMPIngestor

    store, _ = _store(tmp_path)

    def http_get(url: str, **kwargs):
        params = kwargs.get("params") or {}
        assert "apikey" in params
        if url.rstrip("/").endswith("/income-statement"):
            assert params.get("symbol") == "AAA"
            assert params.get("period") == "annual"
            return FakeResponse(
                [
                    {
                        "date": "2025-12-31",
                        "revenue": 1_200_000,
                        "netIncome": 300_000,
                        "eps": 2.5,
                    }
                ]
            )
        if url.rstrip("/").endswith("/ratios-ttm"):
            assert params.get("symbol") == "AAA"
            return FakeResponse(
                [
                    {
                        "peRatioTTM": 18.2,
                        "returnOnEquityTTM": 0.22,
                        "grossProfitMarginTTM": 0.55,
                    }
                ]
            )
        raise AssertionError(url)

    result = FMPIngestor(
        store=store,
        coverage_resolver=_coverage(["AAA"]),
        api_key="test-fmp",
        http_get=http_get,
        now_fn=lambda: "2026-07-14T13:00:00Z",
        sleep_fn=lambda _s: None,
    ).ingest()

    assert FMPIngestor.BASE_URL.endswith("/stable")
    assert result["status"] == "ok"
    assert result["facts_stored"] >= 4
    assert store.get_fundamental("AAA", "total_revenue")["value"] == 1_200_000.0
    assert store.get_fundamental("AAA", "pe_ratio_ttm")["value"] == 18.2


def test_fmp_symbol_entitlement_skips_ticker_continues_batch(tmp_path: Path) -> None:
    """HTTP 402 on one free-tier symbol must not abort the rest of the ingest."""
    from src.ingestion.fmp_ingestor import FMPIngestor

    store, _ = _store(tmp_path)
    seen: list[str] = []

    def http_get(url: str, **kwargs):
        params = kwargs.get("params") or {}
        symbol = str(params.get("symbol") or "")
        seen.append(symbol)
        if symbol == "CRWD":
            return FakeResponse(
                {
                    "Error Message": (
                        "Premium Query Parameter: "
                        "symbol is not available under your current subscription"
                    )
                },
                status_code=402,
            )
        if url.rstrip("/").endswith("/income-statement"):
            return FakeResponse(
                [
                    {
                        "date": "2025-12-31",
                        "revenue": 2_000_000,
                        "netIncome": 500_000,
                        "eps": 3.1,
                    }
                ]
            )
        if url.rstrip("/").endswith("/ratios-ttm"):
            return FakeResponse([{"peRatioTTM": 25.0, "returnOnEquityTTM": 0.3}])
        raise AssertionError(url)

    result = FMPIngestor(
        store=store,
        coverage_resolver=_coverage(["CRWD", "NVDA"]),
        api_key="test-fmp",
        http_get=http_get,
        now_fn=lambda: "2026-07-14T13:00:00Z",
        sleep_fn=lambda _s: None,
    ).ingest()

    assert result["status"] in {"ok", "partial"}
    assert result.get("remaining_work_skipped") is not True
    assert result.get("error_class") not in {"entitlement", "authentication"}
    assert result["status"] != "disabled_entitlement"
    assert "NVDA" in seen
    assert store.get_fundamental("NVDA", "total_revenue")["value"] == 2_000_000.0
    assert store.get_fundamental("CRWD", "total_revenue") is None
    assert any("CRWD" in err and "entitlement" in err.lower() for err in result["errors"])


def test_fmp_missing_key_disables(tmp_path: Path) -> None:
    from src.ingestion.fmp_ingestor import FMPIngestor

    store, _ = _store(tmp_path)
    result = FMPIngestor(
        store=store,
        coverage_resolver=_coverage(["AAA"]),
        api_key="",
        http_get=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no http")),
    ).ingest()
    assert result["status"] == "disabled_missing_key"


def test_marketaux_news_snippet_only(tmp_path: Path) -> None:
    from src.ingestion.marketaux_ingestor import MarketauxIngestor

    store, chroma = _store(tmp_path)

    def http_get(_url: str, **kwargs):
        assert kwargs["params"]["api_token"] == "test-marketaux"
        assert kwargs["params"]["limit"] == 3
        return FakeResponse(
            {
                "data": [
                    {
                        "uuid": "m-1",
                        "title": "AAA product launch",
                        "snippet": "Alpha launched a product.",
                        "url": "https://news.example/marketaux-1",
                        "published_at": "2026-07-14T11:00:00Z",
                        "source": {"name": "Bloomberg"},
                        "entities": [{"symbol": "AAA", "sentiment_score": 0.4}],
                    }
                ]
            }
        )

    result = MarketauxIngestor(
        store=store,
        coverage_resolver=_coverage(["AAA"]),
        api_key="test-marketaux",
        http_get=http_get,
        now_fn=lambda: "2026-07-14T13:00:00Z",
        sleep_fn=lambda _s: None,
    ).ingest_ticker_news("AAA")

    assert result["status"] == "ok"
    assert result["stored"] == 1
    assert store.get_source_cursor("marketaux_news", "AAA") == "2026-07-14T11:00:00Z"
    assert chroma.add_document.call_count == 1
    item = store.sqlite.get_corpus_item("news/marketaux/AAA/m-1")
    assert item["title"] == "AAA product launch"
    assert "Alpha launched a product." in (item.get("summary") or "")
    assert "Sentiment 0.4" in (item.get("summary") or "")


def test_openfigi_maps_and_registers_alias(tmp_path: Path) -> None:
    from src.ingestion.openfigi import OpenFIGIIngestor

    store, _ = _store(tmp_path)
    security = store.resolve_security("AAA")
    assert security is not None

    def http_post(_url: str, **kwargs):
        return FakeResponse(
            [
                {
                    "data": [
                        {
                            "figi": "BBG000B9XRY4",
                            "name": "Alpha Corp",
                            "ticker": "AAA",
                            "exchCode": "US",
                            "compositeFIGI": "BBG000B9XRY4",
                        }
                    ]
                }
            ]
        )

    result = OpenFIGIIngestor(
        store=store,
        coverage_resolver=_coverage(["AAA"]),
        api_key="test-figi",
        http_post=http_post,
        sleep_fn=lambda _s: None,
    ).ingest()

    assert result["status"] == "ok"
    assert result["mapped"] == 1
    resolved = store.resolve_security("BBG000B9XRY4", provider="openfigi")
    assert resolved is not None
    assert resolved["security_id"] == security["security_id"]


def test_registry_free_adapters_and_gdelt_daily() -> None:
    from src.scheduler.source_registry import SourceRegistry

    registry = SourceRegistry.load(
        environ={
            "MASSIVE_API_KEY": "x",
            "ALPHA_VANTAGE_API_KEY": "x",
            "FMP_API_KEY": "x",
            "MARKETAUX_API_KEY": "x",
        }
    )
    daily = {spec.name for spec in registry.select("daily")}
    hourly = [spec.name for spec in registry.select("hourly")]
    assert {"alpha_vantage", "fmp", "marketaux", "openfigi", "gdelt"} <= daily
    assert hourly == ["massive_news"]
    assert registry.get("gdelt").status == "enabled"
    assert "daily" in registry.get("gdelt").run_modes
    assert "hourly" not in registry.get("gdelt").run_modes

    missing = SourceRegistry.load(environ={"MASSIVE_API_KEY": "x"})
    assert missing.get("alpha_vantage").status == "disabled_missing_key"
    assert missing.get("fmp").status == "disabled_missing_key"
    assert missing.get("marketaux").status == "disabled_missing_key"
    assert missing.get("openfigi").status == "enabled"

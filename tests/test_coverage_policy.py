"""tests/test_coverage_policy.py
Offline tests for source-aware broad, deep, sector, and global coverage policy.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.storage.store import Store
from src.universe.models import UniverseRecord
from src.universe.registry import UniverseRegistry


@pytest.fixture
def store(tmp_path: Path):
    with patch("src.storage.store.ChromaStore") as chroma_cls:
        chroma_cls.return_value = MagicMock()
        yield Store(db_path=tmp_path / "coverage.db", chroma_path=tmp_path / "chroma")


def _row(
    symbol: str,
    *,
    source: str,
    index_code: str | None = None,
    sector: str | None = None,
) -> UniverseRecord:
    return UniverseRecord(
        symbol=symbol,
        company_name=f"{symbol} Incorporated",
        source=source,
        index_code=index_code,
        sector=sector,
        source_url=f"https://example.test/{source}",
    )


def _seed_registry(store: Store) -> None:
    registry = UniverseRegistry(store, minimums={"sp500": 2, "nasdaq100": 2, "sec": 1})
    registry.refresh(
        "sec",
        "2026-07-01T00:00:00Z",
        [
            _row("AAA", source="sec", sector="Information Technology"),
            _row("BBB", source="sec", sector="Health Care"),
            _row("CCC", source="sec", sector="Industrials"),
            _row("OFF", source="sec", sector="Health Care"),
        ],
    )
    registry.refresh(
        "ivv",
        "2026-07-02T00:00:00Z",
        [
            _row("AAA", source="ivv", index_code="sp500", sector="Information Technology"),
            _row("BBB", source="ivv", index_code="sp500", sector="Health Care"),
        ],
    )
    registry.refresh(
        "nasdaq",
        "2026-07-02T00:00:00Z",
        [
            _row("AAA", source="nasdaq", index_code="nasdaq100", sector="Information Technology"),
            _row("CCC", source="nasdaq", index_code="nasdaq100", sector="Industrials"),
        ],
    )


def _write_policy(path: Path, **overrides) -> Path:
    policy = {
        "revision": "test-r1",
        "fallback_when_registry_empty": "deep",
        "deep": {"tickers": ["AAA"], "allow_outside_indexes": []},
        "broad": {"additions": []},
        "sector": {"rules": {"health": ["Health Care"]}},
        "sources": {
            "broad_source": {
                "enabled": True,
                "scopes": ["broad"],
                "capabilities": ["market_summary"],
            },
            "deep_source": {
                "enabled": True,
                "scopes": ["deep"],
                "capabilities": ["full_text"],
            },
            "overlap_source": {
                "enabled": True,
                "scopes": ["broad", "deep"],
                "capabilities": ["company_news"],
            },
            "sector_source": {
                "enabled": True,
                "scopes": ["sector"],
                "sector_rules": ["health"],
                "capabilities": ["health_events"],
            },
            "global_source": {
                "enabled": True,
                "scopes": ["global"],
                "capabilities": ["official_macro"],
            },
            "universe_source": {
                "enabled": True,
                "scopes": ["universe"],
                "capabilities": ["identity_maintenance"],
            },
            "disabled_source": {
                "enabled": False,
                "scopes": ["deep"],
                "capabilities": ["optional_news"],
            },
        },
    }
    for key, value in overrides.items():
        policy[key] = value
    path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    return path


def test_broad_is_unique_active_index_union_and_overlap_is_deduplicated(
    store: Store, tmp_path: Path,
) -> None:
    from src.universe.coverage import CoverageResolver

    _seed_registry(store)
    resolver = CoverageResolver(store, config_path=_write_policy(tmp_path / "coverage.yaml"))

    assert resolver.tickers_for("broad_source") == ["AAA", "BBB", "CCC"]
    assert resolver.tickers_for("overlap_source") == ["AAA", "BBB", "CCC"]


def test_deep_and_sector_sources_remain_bounded(store: Store, tmp_path: Path) -> None:
    from src.universe.coverage import CoverageResolver

    _seed_registry(store)
    policy = _write_policy(
        tmp_path / "coverage.yaml",
        deep={"tickers": ["AAA", "OFF"], "allow_outside_indexes": ["OFF"]},
    )
    resolver = CoverageResolver(store, config_path=policy)

    assert resolver.tickers_for("deep_source") == ["AAA", "OFF"]
    assert resolver.tickers_for("sector_source") == ["BBB", "OFF"]
    assert set(resolver.tickers_for("deep_source")) < set(
        resolver.tickers_for("broad_source") + ["OFF"]
    )


def test_global_universe_disabled_and_unknown_source_behavior(
    store: Store, tmp_path: Path,
) -> None:
    from src.universe.coverage import CoverageResolver

    _seed_registry(store)
    resolver = CoverageResolver(store, config_path=_write_policy(tmp_path / "coverage.yaml"))

    assert resolver.tickers_for("global_source") == []
    assert resolver.tickers_for("universe_source") == []
    assert resolver.tickers_for("disabled_source") == []
    assert resolver.is_enabled("global_source") is True
    assert resolver.is_enabled("disabled_source") is False
    with pytest.raises(ValueError, match="unknown coverage source"):
        resolver.tickers_for("missing")


def test_scopes_and_explanation_are_deterministic_and_redacted(
    store: Store, tmp_path: Path,
) -> None:
    from src.universe.coverage import CoverageResolver

    _seed_registry(store)
    resolver = CoverageResolver(store, config_path=_write_policy(tmp_path / "coverage.yaml"))
    security = store.get_security("AAA")

    assert resolver.scopes_for(security["security_id"]) == ["universe", "broad", "deep"]
    explanation = resolver.explain("deep_source", security["security_id"])
    assert explanation == {
        "source": "deep_source",
        "enabled": True,
        "included": True,
        "security_id": security["security_id"],
        "ticker": "AAA",
        "coverage_scopes": ["deep"],
        "inclusion_reasons": ["explicit deep ticker"],
        "enabled_capabilities": ["full_text"],
        "policy_revision": "test-r1",
    }
    serialized = repr(explanation).lower()
    assert "coverage.yaml" not in serialized
    assert "api_key" not in serialized


def test_unknown_deep_ticker_requires_explicit_outside_index_opt_in(
    store: Store, tmp_path: Path,
) -> None:
    from src.universe.coverage import CoverageResolver

    _seed_registry(store)
    invalid = _write_policy(
        tmp_path / "invalid.yaml",
        deep={"tickers": ["AAA", "TYPO"], "allow_outside_indexes": []},
    )
    with pytest.raises(ValueError, match="TYPO"):
        CoverageResolver(store, config_path=invalid)

    valid = _write_policy(
        tmp_path / "valid.yaml",
        deep={"tickers": ["AAA", "TYPO"], "allow_outside_indexes": ["TYPO"]},
    )
    resolver = CoverageResolver(store, config_path=valid)
    assert resolver.tickers_for("deep_source") == ["AAA", "TYPO"]


def test_invalid_scope_is_rejected(store: Store, tmp_path: Path) -> None:
    from src.universe.coverage import CoverageResolver

    policy_path = _write_policy(tmp_path / "coverage.yaml")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    policy["sources"]["broad_source"]["scopes"] = ["everything"]
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid coverage scope"):
        CoverageResolver(store, config_path=policy_path)


def test_legacy_watchlist_warns_and_preserves_core_as_deep(
    store: Store, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    from src.universe.coverage import CoverageResolver

    legacy_path = tmp_path / "watchlist.yaml"
    legacy_path.write_text(
        yaml.safe_dump(
            {
                "core": ["NVDA", "AMD", "NVDA"],
                "extended": ["AAPL", "AMD"],
                "macro_tickers": ["SPY"],
                "schedule": {"fundamentals": 24},
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        resolver = CoverageResolver(store, config_path=legacy_path)

    assert resolver.tickers_for("sec_companyfacts") == ["AMD", "NVDA"]
    assert resolver.tickers_for("yfinance_fundamentals") == ["AMD", "NVDA"]
    assert "legacy watchlist" in caplog.text.lower()


def test_empty_registry_falls_back_to_deep_without_crashing(
    store: Store, tmp_path: Path,
) -> None:
    from src.universe.coverage import CoverageResolver

    resolver = CoverageResolver(store, config_path=_write_policy(tmp_path / "coverage.yaml"))

    assert resolver.tickers_for("broad_source") == ["AAA"]
    assert resolver.tickers_for("deep_source") == ["AAA"]


def _fake_coverage() -> MagicMock:
    coverage = MagicMock()
    ticker_map = {
        "yfinance": ["AAA", "BBB", "CCC"],
        "yfinance_fundamentals": ["AAA", "BBB", "CCC"],
        "yfinance_news": ["AAA", "BBB", "CCC"],
        "sec_filings": ["AAA", "BBB", "CCC"],
        "sec_filing_text": ["AAA"],
        "sec_companyfacts": ["AAA"],
        "earnings_transcripts": ["NVDA"],
        "ir_pages": ["NVDA"],
        "estimates": ["AAA"],
        "gdelt": ["AAA"],
    }
    coverage.tickers_for.side_effect = lambda source, as_of=None: list(ticker_map[source])
    coverage.is_enabled.return_value = True
    coverage.explain.side_effect = lambda source: {
        "source": source,
        "enabled": True,
        "coverage_scopes": ["deep"],
        "ticker_count": len(ticker_map.get(source, [])),
        "enabled_capabilities": [source],
        "policy_revision": "test-r1",
    }
    return coverage


def test_yfinance_batch_methods_use_broad_policy(store: Store) -> None:
    from src.ingestion.yfinance_ingestor import YFinanceIngestor

    coverage = _fake_coverage()
    ingestor = YFinanceIngestor(store=store, coverage_resolver=coverage)
    ingestor._fundamentals_fresh = MagicMock(return_value=False)
    ingestor._news_fresh = MagicMock(return_value=True)
    ingestor._fetch_ticker = MagicMock(return_value=None)

    ingestor.ingest_fundamentals()

    assert [call.args[0] for call in ingestor._fetch_ticker.call_args_list] == [
        "AAA", "BBB", "CCC"
    ]
    coverage.tickers_for.assert_any_call("yfinance_fundamentals")


def test_deep_batch_ingestors_do_not_receive_broad_universe(store: Store) -> None:
    from src.macros.earnings_transcripts import EarningsTranscriptIngestor
    from src.macros.gdelt_ingestor import GDELTIngestor
    from src.macros.ir_ingestor import IRIngestor

    coverage = _fake_coverage()

    transcripts = EarningsTranscriptIngestor(
        store=store, request_delay=0, coverage_resolver=coverage
    )
    transcripts.fetch_and_process = MagicMock(return_value={"status": "success"})
    assert list(transcripts.fetch_all_core()) == ["NVDA"]

    gdelt = GDELTIngestor(store=store, coverage_resolver=coverage)
    gdelt.fetch_and_store_for_ticker = MagicMock(return_value=1)
    assert gdelt.fetch_and_store_all() == {"AAA": 1}

    ir = IRIngestor(store=store, coverage_resolver=coverage)
    ir.request_delay = 0
    ir.fetch_for_ticker = MagicMock(return_value={"status": "success", "items_stored": 1})
    assert list(ir.fetch_all_core()) == ["NVDA"]


def test_estimates_batch_uses_deep_policy_without_loading_environment(store: Store) -> None:
    from src.macros.estimates_ingestor import EstimatesIngestor

    coverage = _fake_coverage()
    with patch("src.macros.estimates_ingestor.load_env"):
        ingestor = EstimatesIngestor(store=store, coverage_resolver=coverage)
    ingestor.request_delay = 0
    ingestor.fetch_for_ticker = MagicMock(return_value={"status": "success"})

    assert ingestor.fetch_all_core() == {"AAA": {"status": "success"}}
    coverage.tickers_for.assert_any_call("estimates")


def test_sec_discovery_is_broad_but_full_text_processing_is_deep(store: Store) -> None:
    from src.sec.scheduler import FilingScheduler

    coverage = _fake_coverage()
    processor = MagicMock()
    processor.discover_new_filings.return_value = 0
    processor.process_ticker.return_value = {"processed": 1, "failed": 0, "errors": []}
    scheduler = FilingScheduler(
        store=store,
        processor=processor,
        coverage_resolver=coverage,
    )

    report = scheduler.run_full_pipeline(force=True)

    assert [call.args[0] for call in processor.discover_new_filings.call_args_list] == [
        "AAA", "BBB", "CCC"
    ]
    processor.process_ticker.assert_called_once_with("AAA", limit=50)
    processor.process_pending_filings.assert_not_called()
    assert report["processing"]["processed"] == 1


def test_scheduler_skips_disabled_policy_and_exposes_coverage_status(store: Store) -> None:
    from src.scheduler import UnifiedScheduler

    coverage = _fake_coverage()
    coverage.is_enabled.side_effect = lambda source: source != "gdelt"
    scheduler = UnifiedScheduler(
        store=store,
        inter_source_delay=0,
        coverage_resolver=coverage,
    )
    scheduler._run_source = MagicMock(return_value={"ok": True})

    result = scheduler.run_all_stale(force=True)
    report = scheduler.status_report()

    assert result["gdelt"]["status"] == "skipped"
    assert result["gdelt"]["reason"] == "policy_disabled"
    assert result["gdelt"]["terminal_status"] == "skipped"
    assert result["gdelt"]["requests"] == 0
    assert "gdelt" not in [call.args[0] for call in scheduler._run_source.call_args_list]
    assert report["sources"]["yfinance"]["coverage"]["policy_revision"] == "test-r1"

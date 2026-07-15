"""tests/test_scheduler_cli_phase2_3.py: Offline Phase 2.3.4.3 scheduler controls."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

if "chromadb" not in sys.modules:
    chromadb = type(sys)("chromadb")
    chromadb.EmbeddingFunction = object
    chromadb.Documents = list
    chromadb.Embeddings = list
    chromadb.PersistentClient = MagicMock
    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.api"] = MagicMock()

from src.ingestion.errors import ErrorClass, ProviderError
from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash
from src.ingestion.records import NarrativeRecord
from src.scheduler import UnifiedScheduler
from src.storage.store import Store


@pytest.fixture
def store(tmp_path: Path):
    with patch("src.storage.store.ChromaStore") as chroma_class:
        chroma = MagicMock()
        chroma.heartbeat.return_value = True
        chroma.count.return_value = 0
        chroma_class.return_value = chroma
        yield Store(db_path=tmp_path / "phase2343.db", chroma_path=tmp_path / "chroma"), chroma


@pytest.fixture
def scheduler(store, monkeypatch):
    instance, _chroma = store
    for name in (
        "FINNHUB_API_KEY",
        "MASSIVE_API_KEY",
        "BLS_API_KEY",
        "BEA_API_KEY",
        "EIA_API_KEY",
        "OPENFDA_API_KEY",
    ):
        monkeypatch.setenv(name, "test-key")
    return UnifiedScheduler(store=instance, inter_source_delay=0)


def _news(item_id: str, *, published_at: str = "2024-01-01T00:00:00Z") -> NarrativeRecord:
    body = f"Summary for {item_id}."
    return NarrativeRecord(
        corpus_item_id=item_id,
        source_name="finnhub",
        source_category="news_vendor",
        provider_record_id=item_id,
        original_publisher="Example Publisher",
        item_type="news",
        title=f"Headline {item_id}",
        body=body,
        summary=body,
        published_at=published_at,
        observed_at=published_at,
        accessed_at=published_at,
        ingested_at=published_at,
        source_url=f"https://example.test/{item_id}",
        canonical_url=f"https://example.test/{item_id}",
        license_label="provider_summary",
        normalization_version=NORMALIZATION_VERSION,
        content_hash=content_hash(body),
        document_family="company_news",
        evidence_authority="provider",
    )


def test_bootstrap_resume_skips_completed_partitions_without_duplicate_writes(scheduler):
    calls: list[str] = []
    failed_once = {"B"}

    scheduler._bootstrap_partitions = MagicMock(return_value=["A", "B"])

    def run_partition(source, partition, *, since, run_id):
        calls.append(partition)
        if partition in failed_once:
            failed_once.remove(partition)
            raise RuntimeError("interrupted partition")
        return {"status": "success", "items": 1, "new": 1}

    scheduler._run_bootstrap_partition = run_partition

    first = scheduler.run_bootstrap(source="finnhub", since="2026-01-01")
    resumed = scheduler.run_bootstrap(
        source="finnhub", since="2026-01-01", resume=True,
    )

    assert first["status"] == "error"
    assert resumed["status"] == "success"
    assert first["run_id"] == resumed["run_id"]
    assert calls == ["A", "B", "B"]


def test_daily_sec_refresh_does_not_enable_bootstrap_history(scheduler, monkeypatch):
    filing_scheduler = MagicMock()
    filing_scheduler.run_discovery.return_value = {"status": "success"}
    fake_sec = ModuleType("src.sec")
    fake_sec.FilingScheduler = MagicMock(return_value=filing_scheduler)
    monkeypatch.setitem(sys.modules, "src.sec", fake_sec)

    scheduler._run_source("sec_filings", force=True)

    filing_scheduler.run_discovery.assert_called_once_with(
        force=True, allow_bootstrap=False,
    )


def test_repair_reindexes_stored_content_without_a_provider_request(scheduler, store, monkeypatch):
    instance, chroma = store
    chroma.add_document.side_effect = RuntimeError("embedding service unavailable")
    instance.upsert_narrative(_news("repair-me"))
    chroma.add_document.side_effect = None
    chroma.reset_mock()
    provider_request = MagicMock(side_effect=AssertionError("repair must not download"))
    monkeypatch.setattr("requests.get", provider_request)

    result = scheduler.run_repair(source="finnhub", limit=10)

    assert result["completed"] == 1
    assert result["failed"] == 0
    assert chroma.add_document.call_count == 1
    provider_request.assert_not_called()
    assert instance.sqlite.get_corpus_item("repair-me")["indexing_status"] == "indexed"


def test_retention_preview_is_read_only_and_apply_uses_displayed_ids(scheduler, store):
    instance, chroma = store
    instance.upsert_narrative(_news("old-news"))
    chroma.reset_mock()

    preview = scheduler.run_retention(as_of="2026-07-14", preview=True)
    assert preview["eligible"] == 1
    assert preview["expired"] == 0
    chroma.delete_document_family.assert_not_called()

    applied = scheduler.run_retention(
        as_of="2026-07-14",
        apply=True,
        eligible_ids=preview["document_family_ids"],
    )
    assert applied["expired"] == 1
    assert [call.args[0] for call in chroma.delete_document_family.call_args_list] == [
        "old-news",
    ]
    assert scheduler.status_report()["runs"][0]["mode"] == "retention"

    # An empty displayed set is still an explicit empty set, never "all".
    no_match = scheduler.run_retention(
        as_of="2026-07-14", apply=True, eligible_ids=[],
    )
    assert no_match["expired"] == 0


def test_run_status_records_success_rate_limit_and_missing_key_without_freshness_inference(
    store, monkeypatch,
):
    instance, _chroma = store
    registry = __import__("src.scheduler.source_registry", fromlist=["SourceRegistry"]).SourceRegistry.load(
        environ={}
    )
    scheduler = UnifiedScheduler(
        store=instance, registry=registry, inter_source_delay=0,
    )

    def run_source(name, deep=False, force=False):
        if name == "fred":
            raise ProviderError(
                "rate limit token=hidden",
                error_class=ErrorClass.RATE_LIMITED,
                attempts=3,
                retry_after=60,
            )
        return {"status": "ok", "requests": 1, "items": 1}

    scheduler._run_source = run_source
    result = scheduler.run_daily(force=True)
    report = scheduler.status_report()

    assert result["fred"]["status"] == "skipped"
    assert result["fred"]["error_class"] == "rate_limited"
    assert result["finnhub"]["reason"] == "disabled_missing_key"
    assert result["yfinance"]["status"] == "success"
    assert report["runs"][0]["mode"] == "daily"
    source_status = {row["source"]: row for row in report["runs"][0]["sources"]}
    assert source_status["fred"]["error_class"] == "rate_limited"
    assert source_status["finnhub"]["error_class"] == "authentication"
    assert source_status["yfinance"]["status"] == "success"
    assert instance.get_cache_status("SCHEDULER", "unified:finnhub")["status"] != "fresh"


def test_cli_dispatches_new_modes_and_json_flag(monkeypatch, capsys):
    from src import scheduler as scheduler_module

    instance = MagicMock()
    instance.run_bootstrap.return_value = {"mode": "bootstrap"}
    instance.run_daily.return_value = {"mode": "daily"}
    instance.run_repair.return_value = {"mode": "repair"}
    instance.run_retention.return_value = {"mode": "retention"}
    instance.status_report.return_value = {"mode": "status"}
    monkeypatch.setattr(scheduler_module, "UnifiedScheduler", MagicMock(return_value=instance))

    commands = [
        (["bootstrap", "--source", "finnhub", "--since", "2026-01-01", "--resume"], "run_bootstrap"),
        (["daily", "--source", "fred", "--scope", "global", "--force"], "run_daily"),
        (["repair", "--source", "finnhub", "--limit", "2"], "run_repair"),
        (["retention", "--preview"], "run_retention"),
        (["status", "--source", "fred", "--json"], "status_report"),
    ]
    for argv, method in commands:
        monkeypatch.setattr(sys, "argv", ["scheduler", *argv])
        scheduler_module.main()
        assert getattr(instance, method).called

    output = capsys.readouterr().out
    assert '"mode": "status"' in output


def test_refresh_tool_rejects_unbounded_requests_without_provider_work(store):
    instance, _chroma = store
    from src.middleware.tools.data_tools import refresh_data_handler

    result = refresh_data_handler(instance, "AAA", sources=["all"])

    assert "unbounded" in result["error"]
    assert result["ticker"] == "AAA"


def test_adapter_failure_counts_never_classify_as_success():
    assert UnifiedScheduler._classify_detail({"failed": 1})[0] == "error"
    assert UnifiedScheduler._classify_detail({"failed": 1, "registered": 2})[0] == "partial"


def test_bootstrap_resume_uses_durable_manifest_and_cumulative_counts(scheduler):
    scheduler._bootstrap_partitions = MagicMock(return_value=["A", "B"])
    failed_once = {"B"}

    def run_partition(source, partition, *, since, run_id):
        if partition in failed_once:
            failed_once.remove(partition)
            raise RuntimeError("interrupt")
        return {"status": "success", "items": 1, "new": 1}

    scheduler._run_bootstrap_partition = run_partition
    first = scheduler.run_bootstrap(source="finnhub", since="2026-01-01")
    scheduler._bootstrap_partitions = MagicMock(return_value=["C"])

    resumed = scheduler.run_bootstrap(resume=True)

    assert resumed["run_id"] == first["run_id"]
    assert resumed["status"] == "success"
    assert resumed["sources"]["finnhub"]["partitions"] == 2
    assert resumed["sources"]["finnhub"]["items"] == 2
    scheduler._bootstrap_partitions.assert_not_called()


def test_repair_skips_non_news_when_full_body_was_not_retained(scheduler, store):
    instance, chroma = store
    transcript = replace(
        _news("transcript-item"),
        item_type="transcript",
        document_family="earnings_transcript",
        body="Full transcript body that SQLite intentionally does not retain.",
    )
    chroma.add_document.side_effect = RuntimeError("index unavailable")
    instance.upsert_narrative(transcript)
    chroma.add_document.side_effect = None
    chroma.reset_mock()

    result = scheduler.run_repair(source="finnhub", limit=10)

    assert result["completed"] == 0
    assert result["skipped"] == 1
    chroma.add_document.assert_not_called()


def test_retention_delete_failure_is_not_recorded_as_success(scheduler, store):
    instance, chroma = store
    instance.upsert_narrative(_news("retention-failure"))
    preview = scheduler.run_retention(as_of="2026-07-14", preview=True)
    chroma.delete_document_family.side_effect = RuntimeError("delete failed")

    result = scheduler.run_retention(
        as_of="2026-07-14",
        apply=True,
        eligible_ids=preview["document_family_ids"],
    )
    run = scheduler.status_report()["runs"][0]

    assert result["failed"] == 1
    assert run["status"] == "error"
    assert run["sources"][0]["status"] == "error"


def test_history_pruning_preserves_resumable_bootstrap(store):
    instance, _chroma = store
    active = instance.start_scheduler_run(
        "bootstrap",
        policy_revision="p",
        config_revision="c",
        requested_sources=["finnhub"],
        started_at="2020-01-01T00:00:00Z",
    )
    for index in range(3):
        run_id = instance.start_scheduler_run(
            "daily",
            policy_revision="p",
            config_revision="c",
            requested_sources=["fred"],
            started_at=f"2026-07-1{index + 1}T00:00:00Z",
        )
        instance.complete_scheduler_run(run_id, status="success")

    instance.prune_scheduler_history(1)

    assert instance.get_resumable_bootstrap_run()["run_id"] == active


def test_bounded_refresh_cannot_bypass_durable_request_quota(
    scheduler, store, monkeypatch,
):
    instance, _chroma = store
    spec = scheduler.registry.get("finnhub")
    now = datetime.now(timezone.utc)
    instance.record_source_budget_usage(
        "finnhub",
        day_start=now.date().isoformat(),
        minute_start=now.strftime("%Y-%m-%dT%H:%MZ"),
        attempted_requests=spec.requests_per_day,
        successful_requests=0,
    )
    provider = MagicMock(side_effect=AssertionError("quota must reject before HTTP"))
    monkeypatch.setattr("requests.get", provider)

    with pytest.raises(Exception, match="bounded refresh rejected: requests_per_"):
        scheduler.run_bounded_security_refresh("AAA", "finnhub_news")

    provider.assert_not_called()


def test_repair_item_exception_is_isolated_and_run_terminalizes(scheduler, store):
    instance, chroma = store
    chroma.add_document.side_effect = RuntimeError("index unavailable")
    instance.upsert_narrative(_news("repair-exception"))
    chroma.add_document.side_effect = None
    instance.repair_corpus_item = MagicMock(side_effect=RuntimeError("unexpected"))

    result = scheduler.run_repair(source="finnhub", limit=10)
    run = scheduler.status_report()["runs"][0]

    assert result["failed"] == 1
    assert run["status"] == "error"


def test_source_scoped_resume_keeps_other_bootstrap_partitions_resumable(
    scheduler, store,
):
    instance, _chroma = store
    run_id = instance.start_scheduler_run(
        "bootstrap",
        policy_revision=scheduler.coverage.revision,
        config_revision=scheduler.registry.version,
        requested_sources=["finnhub", "massive"],
        bootstrap_manifest={"finnhub": ["AAA"], "massive": ["__global__"]},
    )
    scheduler._run_bootstrap_partition = MagicMock(
        return_value={"status": "success", "items": 1, "new": 1},
    )

    result = scheduler.run_bootstrap(source="finnhub", resume=True)

    assert result["run_id"] == run_id
    assert result["status"] == "partial"
    assert instance.get_resumable_bootstrap_run()["run_id"] == run_id


def test_refresh_batch_is_charged_per_source_against_query_limit(store, monkeypatch):
    instance, _chroma = store
    from src.middleware import app as middleware_app
    from src.middleware.tools import ToolContext, dispatch_tool

    provider_work = MagicMock()
    monkeypatch.setattr(middleware_app, "_refresh_ticker_sources", provider_work)
    result = dispatch_tool(
        {
            "function": {
                "name": "refresh_data",
                "arguments": '{"ticker":"AAA","sources":["news","gdelt","finnhub"]}',
            }
        },
        instance,
        ToolContext(allow_write=True, max_refreshes=2),
    )

    assert result == {"error": "max refreshes exceeded", "tool": "refresh_data"}
    provider_work.assert_not_called()


def test_sec_daily_index_preserves_provider_cooldown_metadata(store):
    instance, _chroma = store
    from src.sec.daily_index import SECDailyIndexDiscovery

    reset_at = "2026-07-14T12:05:00Z"
    request = MagicMock(side_effect=ProviderError(
        "rate limited",
        error_class=ErrorClass.RATE_LIMITED,
        retry_after=300,
        reset_at=reset_at,
        provider_wide=True,
        circuit_open=True,
    ))
    discovery = SECDailyIndexDiscovery(
        store=instance,
        coverage_resolver=MagicMock(),
        sec_config={"forms": {}},
        user_agent="test@example.com",
        request_delay=0,
        http_get=request,
    )

    result = discovery.discover_dates(["2026-07-14"])

    assert result["error_class"] == "rate_limited"
    assert result["retry_after"] == 300
    assert result["reset_at"] == reset_at
    assert result["provider_wide"] is True


def test_sec_bootstrap_persists_provider_wide_cooldown(scheduler):
    reset_at = "2099-07-14T12:05:00Z"
    scheduler._bootstrap_partitions = MagicMock(return_value=["2026-07-14"])
    scheduler._run_bootstrap_partition = MagicMock(return_value={
        "failed": 1,
        "errors": ["rate limited"],
        "error_class": "rate_limited",
        "retry_after": 300,
        "reset_at": reset_at,
        "provider_wide": True,
        "circuit_open": True,
    })

    scheduler.run_bootstrap(source="sec_filings")
    provider = scheduler.store.get_source_cursor_state(
        "sec_filings", "__provider__",
    )

    assert provider["status"] == "circuit_open"
    assert provider["error_class"] == "rate_limited"
    assert provider["cursor_value"] == reset_at

"""tests/test_source_registry.py
Offline tests for validated scheduler source registry configuration.
"""

from pathlib import Path

import pytest
import yaml

from src.scheduler.source_registry import SourceRegistry


def _write_registry(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _source_config(**overrides) -> dict:
    source = {
        "capability_group": "news",
        "enabled": True,
        "scope": "broad",
        "cadence": "daily",
        "run_modes": ["daily", "all"],
        "priority": 1,
        "dependencies": [],
        "cursor_kind": "timestamp",
        "overlap": "6h",
        "requests_per_minute": 10,
        "requests_per_day": 100,
        "requests_per_run": 10,
        "batch_size": 1,
        "max_work_items_per_run": 5,
        "retry_policy": "default",
        "ttl_key": "news",
        "ttl_hours": 6,
    }
    source.update(overrides)
    return {"sources": {"candidate": source}}


def test_registry_loads_valid_entries_in_priority_order(tmp_path):
    config_path = _write_registry(
        tmp_path / "sources.yaml",
        """
version: "1"
sources:
  later:
    capability_group: news
    enabled: true
    scope: broad
    cadence: daily
    run_modes: [daily, all]
    priority: 20
    dependencies: []
    cursor_kind: timestamp
    overlap: 6h
    requests_per_minute: 30
    requests_per_day: 500
    requests_per_run: 20
    batch_size: 1
    max_work_items_per_run: 20
    retry_policy: vendor
    ttl_key: news
    ttl_hours: 6
  first:
    capability_group: filings
    enabled: true
    scope: broad
    cadence: daily
    run_modes: [daily, weekly, all]
    priority: 10
    dependencies: []
    cursor_kind: date
    overlap: 2d
    requests_per_minute: 10
    requests_per_day: 100
    requests_per_run: 10
    batch_size: 1
    max_work_items_per_run: 10
    retry_policy: sec
    ttl_key: filings
    ttl_hours: 12
""",
    )

    registry = SourceRegistry.load(config_path, environ={})

    assert [spec.name for spec in registry.select("daily")] == ["first", "later"]
    assert registry.get("first").cursor_kind == "date"
    assert registry.get("later").scope == "broad"
    assert registry.get("later").is_available is True


def test_invalid_entry_is_disabled_without_hiding_valid_source(tmp_path):
    config_path = _write_registry(
        tmp_path / "sources.yaml",
        """
sources:
  valid:
    capability_group: macro
    enabled: true
    scope: global
    cadence: daily
    run_modes: [daily, all]
    priority: 1
    dependencies: []
    cursor_kind: date
    overlap: 1d
    requests_per_minute: 10
    requests_per_day: 100
    requests_per_run: 10
    batch_size: 1
    max_work_items_per_run: 5
    retry_policy: default
    ttl_key: macro
    ttl_hours: 24
  broken:
    capability_group: news
    enabled: true
    scope: everywhere
    cadence: hourly
    run_modes: [hourly, all]
    priority: 2
    dependencies: []
    cursor_kind: timestamp
    requests_per_minute: 10
    requests_per_day: 100
    requests_per_run: 10
    batch_size: 1
    max_work_items_per_run: 5
    retry_policy: default
    ttl_key: news
    ttl_hours: 6
""",
    )

    registry = SourceRegistry.load(config_path, environ={})

    assert registry.get("valid").is_available is True
    assert registry.get("broken").is_available is False
    assert registry.get("broken").status == "invalid_configuration"
    assert "scope" in registry.get("broken").disabled_reason
    assert [spec.name for spec in registry.select("daily")] == ["valid"]


def test_missing_required_key_disables_only_that_source(tmp_path):
    config_path = _write_registry(
        tmp_path / "sources.yaml",
        """
sources:
  public:
    capability_group: macro
    enabled: true
    scope: global
    cadence: daily
    run_modes: [daily, all]
    priority: 1
    dependencies: []
    cursor_kind: date
    overlap: 1d
    requests_per_minute: 10
    requests_per_day: 100
    requests_per_run: 10
    batch_size: 1
    max_work_items_per_run: 5
    retry_policy: default
    ttl_key: macro
    ttl_hours: 24
  keyed:
    capability_group: news
    enabled: true
    required_env: VENDOR_KEY
    scope: broad
    cadence: daily
    run_modes: [daily, all]
    priority: 2
    dependencies: []
    cursor_kind: timestamp
    overlap: 6h
    requests_per_minute: 10
    requests_per_day: 100
    requests_per_run: 10
    batch_size: 1
    max_work_items_per_run: 5
    retry_policy: default
    ttl_key: news
    ttl_hours: 6
""",
    )

    registry = SourceRegistry.load(config_path, environ={})

    assert registry.get("public").is_available is True
    assert registry.get("keyed").status == "disabled_missing_key"
    assert registry.get("keyed").disabled_reason == "missing VENDOR_KEY"
    assert [spec.name for spec in registry.select("all")] == ["public"]
    assert registry.status()["keyed"]["status"] == "disabled_missing_key"


def test_registry_filters_by_source_and_scope(tmp_path):
    config_path = _write_registry(
        tmp_path / "sources.yaml",
        """
sources:
  broad_news:
    capability_group: news
    enabled: true
    scope: broad
    cadence: daily
    run_modes: [daily, all]
    priority: 1
    dependencies: []
    cursor_kind: timestamp
    overlap: 6h
    requests_per_minute: 10
    requests_per_day: 100
    requests_per_run: 10
    batch_size: 1
    max_work_items_per_run: 5
    retry_policy: default
    ttl_key: news
    ttl_hours: 6
  deep_docs:
    capability_group: documents
    enabled: true
    scope: deep
    cadence: weekly
    run_modes: [weekly, all]
    priority: 2
    dependencies: []
    cursor_kind: none
    overlap: null
    requests_per_minute: 10
    requests_per_day: 100
    requests_per_run: 10
    batch_size: 1
    max_work_items_per_run: 5
    retry_policy: default
    ttl_key: documents
    ttl_hours: 168
""",
    )
    registry = SourceRegistry.load(config_path, environ={})

    assert [item.name for item in registry.select("all", scope="deep")] == [
        "deep_docs"
    ]
    assert [item.name for item in registry.select("daily", source="broad_news")] == [
        "broad_news"
    ]
    assert registry.select("daily", source="deep_docs") == []


def test_default_hourly_registry_uses_massive_news_and_disables_gdelt():
    registry = SourceRegistry.load(environ={"MASSIVE_API_KEY": "test-key"})

    assert [spec.name for spec in registry.select("hourly")] == ["massive_news"]
    news = registry.get("massive_news")
    assert news.capability_group == "company_news"
    assert news.cursor_kind == "timestamp"
    assert news.overlap == "2h"
    assert news.requests_per_run == 1
    assert news.ttl_hours == 1

    gdelt = registry.get("gdelt")
    assert gdelt.status == "configured_disabled"
    assert gdelt.disabled_reason == "disabled by configuration"


@pytest.mark.parametrize(
    "override, expected_error",
    [
        ({"cadence": "fortnightly"}, "cadence"),
        ({"overlap": "yesterday"}, "overlap"),
        ({"batch_size": 11}, "batch_size"),
        ({"cursor_kind": "page_token", "overlap": "2d"}, "overlap"),
    ],
)
def test_registry_disables_incoherent_cadence_overlap_and_limits(
    tmp_path, override, expected_error
):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        yaml.safe_dump(_source_config(**override)), encoding="utf-8"
    )

    registry = SourceRegistry.load(config_path, environ={})

    assert registry.get("candidate").status == "invalid_configuration"
    assert expected_error in registry.get("candidate").disabled_reason

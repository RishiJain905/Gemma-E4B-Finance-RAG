"""Offline coverage inventory and deterministic coverage-answer tests."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.middleware import deterministic_router as dr
from src.middleware.query_plan import QueryEntity, QueryPlan, QuerySubquery
from src.middleware.tools import REGISTRY, ToolContext, dispatch_named_tool
from src.middleware.tools.coverage_tools import describe_coverage_handler
from src.storage.sqlite_store import SQLiteStore
from src.storage.store import Store


FIXTURE = Path(__file__).parent / "fixtures/evaluation/phase2_3_coverage_queries.json"


def _coverage_store(tmp_path: Path) -> Store:
    store = object.__new__(Store)
    store.sqlite = SQLiteStore(tmp_path / "coverage.db")
    store.chroma = MagicMock()
    _seed_canonical_inventory(store.sqlite)
    return store


def _seed_canonical_inventory(sqlite: SQLiteStore) -> None:
    securities = [
        ("sec-nvda", "NVDA", "NVIDIA Corporation", "Information Technology", "Semiconductors", 1),
        ("sec-amd", "AMD", "Advanced Micro Devices", "Information Technology", "Semiconductors", 1),
        ("sec-aapl", "AAPL", "Apple Inc.", "Information Technology", "Consumer Electronics", 1),
        ("sec-hum", "HUM", "Humana Inc.", "Health Care", "Health Care Providers", 1),
        ("sec-old", "OLD", "Inactive Example", "Information Technology", "Software", 0),
    ]
    with sqlite._connect() as conn:
        for security_id, ticker, name, sector, industry, active in securities:
            conn.execute(
                """INSERT INTO securities (
                    security_id, ticker, normalized_ticker, company_name, exchange,
                    sector, industry, active, first_seen_at, last_seen_at, updated_at
                ) VALUES (?, ?, ?, ?, 'NASDAQ', ?, ?, ?, '2026-01-01', '2026-07-01', '2026-07-01')""",
                (security_id, ticker, ticker, name, sector, industry, active),
            )
        conn.executemany(
            """INSERT INTO security_memberships (
                security_id, index_code, effective_from, active, source, observed_at
            ) VALUES (?, ?, '2026-01-01', ?, 'seed', '2026-07-01T00:00:00Z')""",
            [
                ("sec-nvda", "sp500", 1),
                ("sec-amd", "nasdaq100", 1),
                ("sec-aapl", "sp500", 1),
                ("sec-hum", "sp500", 1),
                ("sec-old", "sp500", 0),
            ],
        )
        _insert_corpus_item(
            conn, "item-nvda-filing", "sec_filings", "sec_filing",
            "NVDA filing", "2026-07-02", "sec-nvda",
        )
        _insert_corpus_item(
            conn, "item-nvda-market", "yfinance", "market_bar",
            "NVDA market bar", "2026-07-03", "sec-nvda",
        )
        _insert_corpus_item(
            conn, "item-amd-filing", "sec_filings", "sec_filing",
            "AMD filing", "2026-07-04", "sec-amd",
        )
        _insert_corpus_item(
            conn, "item-hum-event", "sec_filings", "regulatory_event",
            "HUM event", "2026-07-05", "sec-hum",
        )
        conn.execute(
            """INSERT INTO corpus_observations (
                observation_id, metric_id, value_text, value_numeric, unit,
                frequency, period_end, scope, tickers_json, source_name,
                source_category, source_url, accessed_at, ingested_at,
                license_label, normalization_version, evidence_authority
            ) VALUES (
                'obs-nvda-revenue', 'revenue', '100', 100, 'usd', 'quarterly',
                '2026-06-30', 'security', '[\"NVDA\"]', 'sec_companyfacts',
                'structured_facts', 'https://example.test/revenue',
                '2026-07-06', '2026-07-06', 'seed', 'test', 'direct_sec'
            )"""
        )
        conn.execute(
            "INSERT INTO observation_securities (observation_id, security_id) VALUES ('obs-nvda-revenue', 'sec-nvda')"
        )
        conn.execute(
            """INSERT INTO corpus_observations (
                observation_id, metric_id, value_text, value_numeric, unit,
                frequency, period_end, scope, tickers_json, source_name,
                source_category, source_url, accessed_at, ingested_at,
                license_label, normalization_version, evidence_authority
            ) VALUES (
                'obs-amd-eps', 'eps_diluted', '2.5', 2.5, 'usd', 'quarterly',
                '2026-06-30', 'security', '[\"AMD\"]', 'sec_companyfacts',
                'structured_facts', 'https://example.test/eps',
                '2026-07-06', '2026-07-06', 'seed', 'test', 'direct_sec'
            )"""
        )
        conn.execute(
            "INSERT INTO observation_securities (observation_id, security_id) VALUES ('obs-amd-eps', 'sec-amd')"
        )
        conn.executemany(
            """INSERT INTO source_cursors (
                source, partition_key, status, last_successful_at, updated_at
            ) VALUES (?, '__provider__', ?, ?, ?)""",
            [
                ("sec_filings", "success", "2026-07-06T00:00:00Z", "2026-07-06T00:00:00Z"),
                ("yfinance", "partial", "2026-07-06T00:00:00Z", "2026-07-07T00:00:00Z"),
            ],
        )
        conn.execute("UPDATE store_revision SET revision = 42, updated_at = '2026-07-07'")
        conn.commit()


def _insert_corpus_item(
    conn: sqlite3.Connection,
    item_id: str,
    source: str,
    item_type: str,
    title: str,
    published_at: str,
    security_id: str,
) -> None:
    conn.execute(
        """INSERT INTO corpus_items (
            corpus_item_id, source, source_category, item_type, title,
            normalized_headline, language, published_at, accessed_at, ingested_at,
            source_url, tickers_json, content_hash, metadata_json, document_family,
            document_family_id, indexing_status, license_label,
            normalization_version, evidence_authority
        ) VALUES (?, ?, ?, ?, ?, ?, 'en', ?, ?, ?, ?, '[]', ?, '{}', ?, ?,
                  'indexed', 'seed', 'test', 'direct')""",
        (
            item_id, source, item_type if item_type == "sec_filing" else source,
            item_type, title, title.lower(), published_at, published_at,
            published_at, f"https://example.test/{item_id}", item_id, source,
            item_id,
        ),
    )
    conn.execute(
        """INSERT INTO corpus_item_securities (corpus_item_id, security_id, ticker)
        SELECT ?, security_id, ticker FROM securities WHERE security_id = ?""",
        (item_id, security_id),
    )


@pytest.fixture
def coverage_store(tmp_path: Path) -> Store:
    return _coverage_store(tmp_path)


def test_all_tickers_are_complete_sorted_and_exactly_counted(coverage_store: Store):
    result = coverage_store.describe_coverage(
        operation="list_securities", ticker_only=True,
    )

    assert [row["ticker"] for row in result["securities"]] == ["AAPL", "AMD", "HUM", "NVDA"]
    assert result["result_count"] == result["total_matching"] == 4
    assert result["active_securities"] == 4
    assert result["complete"] is True
    assert result["next_cursor"] is None
    assert result["data_revision"] == 42
    assert result["coverage_basis"] == "canonical"


def test_detail_pages_are_bounded_and_truthful(coverage_store: Store):
    first = coverage_store.describe_coverage(
        operation="list_securities", limit=1,
    )
    assert first["result_count"] == 1
    assert first["total_matching"] == 4
    assert first["complete"] is False
    assert first["next_cursor"]

    second = coverage_store.describe_coverage(
        operation="list_securities", limit=2, cursor=first["next_cursor"],
    )
    assert second["securities"][0]["ticker"] == "AMD"
    assert second["complete"] is False
    assert second["next_cursor"]


def test_membership_evidence_capability_and_freshness_are_distinct(coverage_store: Store):
    result = coverage_store.describe_coverage(
        operation="security_sources", ticker="NVDA",
    )
    sec = next(row for row in result["sources"] if row["source"] == "sec_filings")
    assert result["covered"] is True
    assert sec["evidence_count"] == 1
    assert sec["has_evidence"] is True
    assert sec["capability"]["configured"] is True
    assert sec["capability"]["last_terminal_status"] == "success"
    assert "fresh" not in sec["capability"] or sec["capability"]["fresh"] is not True

    no_evidence = coverage_store.describe_coverage(
        operation="security_sources", ticker="HUM",
    )
    yfinance = next(row for row in no_evidence["sources"] if row["source"] == "yfinance")
    assert yfinance["has_evidence"] is False
    assert yfinance["capability"]["configured"] is True


def test_filters_cover_index_sector_tier_item_type_source_and_date(coverage_store: Store):
    by_index = coverage_store.describe_coverage(
        operation="list_securities", filters={"index": "sp500"},
    )
    assert [row["ticker"] for row in by_index["securities"]] == ["AAPL", "HUM", "NVDA"]

    by_sector = coverage_store.describe_coverage(
        operation="list_securities", filters={"sector": "Health Care"},
    )
    assert [row["ticker"] for row in by_sector["securities"]] == ["HUM"]

    by_tier = coverage_store.describe_coverage(
        operation="list_securities", filters={"coverage_tier": "deep"},
    )
    assert {row["ticker"] for row in by_tier["securities"]} >= {"AAPL", "NVDA"}

    by_item = coverage_store.describe_coverage(
        operation="list_securities", filters={"item_type": "sec_filing"},
    )
    assert [row["ticker"] for row in by_item["securities"]] == ["AMD", "NVDA"]

    by_source = coverage_store.describe_coverage(
        operation="list_securities", filters={"source_category": "sec_filing"},
    )
    assert [row["ticker"] for row in by_source["securities"]] == ["AMD", "NVDA"]

    by_date = coverage_store.describe_coverage(
        operation="list_item_types", filters={"date_from": "2026-07-03"},
    )
    assert set(by_date["item_types"]) == {
        "market_bar", "sec_filing", "regulatory_event",
    }


def test_coverage_operations_expose_stable_inventory_envelope(coverage_store: Store):
    for operation, kwargs in (
        ("summary", {}),
        ("list_sources", {}),
        ("list_item_types", {}),
        ("list_metrics", {"ticker": "NVDA"}),
    ):
        result = coverage_store.describe_coverage(operation=operation, **kwargs)
        assert {"filters_applied", "result_count", "total_matching", "complete",
                "next_cursor", "data_revision", "universe_snapshot_at"} <= result.keys()
        assert result["complete"] is True


def test_contains_security_returns_suggestions_for_unknown_ticker(coverage_store: Store):
    result = coverage_store.describe_coverage(
        operation="contains_security", ticker="NVD",
    )
    assert result["covered"] is False
    assert "suggestions" in result
    assert "NVDA" in result["suggestions"]


def test_missing_phase23_tables_use_legacy_partial_projection(coverage_store: Store):
    with coverage_store.sqlite._connect() as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TABLE corpus_item_securities")
        conn.execute("DROP TABLE corpus_items")
        conn.execute("DROP TABLE security_memberships")
        conn.execute("DROP TABLE security_aliases")
        conn.execute("DROP TABLE securities")
        conn.execute(
            "INSERT INTO fundamentals (ticker, metric, value, unit, period, period_type, source_type) "
            "VALUES ('LEG', 'revenue', 1, 'usd', '2026-Q1', 'quarterly', 'legacy_source')"
        )
        conn.commit()

    result = coverage_store.describe_coverage(
        operation="list_securities", ticker_only=True,
    )
    assert result["coverage_basis"] == "legacy_partial"
    assert result["complete"] is True
    assert [row["ticker"] for row in result["securities"]] == ["LEG"]


def test_store_failure_is_structured_unavailable_and_never_guesses():
    broken = MagicMock()
    broken.describe_coverage.side_effect = RuntimeError("inventory offline")
    result = describe_coverage_handler(broken, operation="summary")
    assert result["status"] == "unavailable"
    assert result["coverage_basis"] == "unavailable"
    assert result["complete"] is False
    assert result["answer_origin"] == "deterministic_coverage"
    assert "inventory offline" not in result.get("message", "")


def test_describe_coverage_is_registered_read_only_and_cannot_refresh(coverage_store: Store):
    assert "describe_coverage" in REGISTRY
    assert REGISTRY["describe_coverage"].write is False
    refresh = MagicMock()
    coverage_store.refresh_data = refresh
    result, name, _args = dispatch_named_tool(
        "describe_coverage", {"operation": "summary"}, coverage_store,
        ToolContext(allow_write=False, max_refreshes=0),
    )
    assert name == "describe_coverage"
    assert result["status"] == "ok"
    refresh.assert_not_called()


def test_legacy_list_metrics_delegates_inventory_to_store_contract(coverage_store: Store, monkeypatch):
    expected = {"metrics": ["revenue"], "tickers": ["NVDA"]}
    spy = MagicMock(return_value={
        "metrics": ["revenue"],
        "securities": [{"ticker": "NVDA"}],
        "complete": True,
    })
    monkeypatch.setattr(coverage_store, "describe_coverage", spy)

    from src.middleware.tools.data_tools import list_metrics_handler

    assert list_metrics_handler(coverage_store, "NVDA") == expected
    spy.assert_called_once_with(operation="list_metrics", ticker="NVDA", ticker_only=True)


def _plan(question: str, tickers: list[str] | None = None) -> QueryPlan:
    tickers = tickers or []
    entities = [
        QueryEntity(ticker=ticker, resolved_name=None, confidence=1.0,
                    source="resolved", mention=ticker, start=index)
        for index, ticker in enumerate(tickers)
    ]
    return QueryPlan(
        original_question=question,
        retrieval_query=question,
        entities=entities,
        intents=["general"],
        subqueries=[QuerySubquery(id="sq0", text=question)],
    )


def test_coverage_questions_route_to_read_only_tool_and_render_deterministically(coverage_store: Store):
    decision = dr.route(_plan("What tickers do you know about?"), [])
    assert decision.matched is True
    assert decision.complete is True
    assert decision.tool_invocations[0].name == "describe_coverage"
    assert decision.tool_invocations[0].arguments["operation"] == "list_securities"

    execution = dr.execute_route(decision, coverage_store)
    assert execution.error is False
    assert execution.answer_origin == "deterministic"
    assert execution.answer_metadata["complete"] is True
    assert "AAPL" in execution.answer
    assert "4" in execution.answer


def test_coverage_failure_renders_unavailable_without_model_fallback():
    broken = MagicMock()
    broken.describe_coverage.side_effect = RuntimeError("db unavailable")
    decision = dr.route(_plan("What tickers do you know about?"), [])
    execution = dr.execute_route(decision, broken)
    assert execution.error is False
    assert execution.answer_origin == "deterministic"
    assert "unavailable" in execution.answer.lower()
    assert "db unavailable" not in execution.answer


@pytest.mark.asyncio
async def test_query_response_preserves_coverage_origin_without_model_probe(monkeypatch):
    from src.middleware import app as middleware_app
    from src.middleware.adaptive_orchestrator import Lane, OrchestrationResult
    from src.middleware.config import MiddlewareConfig
    from src.middleware.evidence import assign_evidence_ids, build_evidence_items

    async def fail_model_probe():
        raise AssertionError("coverage answer must not probe the model")

    monkeypatch.setattr(middleware_app, "_check_model_health", fail_model_probe)
    config = MiddlewareConfig(config_path=None)
    config.enable_deterministic_answers = True
    monkeypatch.setattr(middleware_app, "config", config)
    plan = _plan("What tickers do you know about?")
    execution = dr.ExecutionResult(
        invocations=[dr.ExecutedInvocation(
            name="describe_coverage",
            arguments={"operation": "list_securities", "ticker_only": True},
            subquery_id="sq0",
            reason_code=dr.REASON_COVERAGE,
            result={
                "status": "ok",
                "complete": True,
                "next_cursor": None,
                "securities": [
                    {"ticker": ticker} for ticker in ("AAPL", "AMD", "HUM", "NVDA")
                ],
                "result_count": 4,
                "total_matching": 4,
            },
        )],
        complete=True,
    )
    orchestration_result = OrchestrationResult(
        lane=Lane.CATALOG,
        plan=plan,
        tool_execution=execution,
        set_complete=True,
        result_set_size=4,
    )
    facts = [{
        "ticker": "CATALOG",
        "metric": "coverage_list_securities",
        "value": "AAPL, AMD, HUM, NVDA",
        "source_type": "catalog",
    }]
    ledger = assign_evidence_ids(build_evidence_items(facts, []))
    context = {
        "start": time.time(),
        "timings": {},
        "intent": {"ticker": None, "question_type": "general"},
        "freshness": {"overall": "unknown", "stale_sources_used": []},
        "retrieval": {
            "facts": facts, "documents": [], "retrieval_strategy": "fast",
        },
        "grounding_level": "grounded",
        "augmented_prompt": "",
        "include_evidence_trace": False,
        "conversation": None,
        "compiled": None,
        "retrieval_query": None,
        "orchestration": {
            "lane": "catalog", "deterministic_tools": ["describe_coverage"],
            "model_calls": 0,
        },
        "answer_origin": None,
        "coverage_metadata": {
            "data_revision": 42, "complete": True, "total_matching": 4,
        },
        "evidence_ledger": ledger,
        "graph_evidence_ids": ["E1"],
        "calculations": [],
        "evidence_sufficiency": None,
        "_orchestration_result": orchestration_result,
    }

    response = await middleware_app._answer_query_context(
        middleware_app.QueryRequest(question="What tickers do you know about?"),
        context,
    )
    assert response.answer_origin == "deterministic"
    assert response.generation_skipped is True
    assert response.coverage_metadata["data_revision"] == 42
    assert "AAPL" in response.answer


def test_evaluation_fixture_covers_all_declared_operations():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert {row["operation"] for row in fixture["queries"]} == {
        "summary", "list_securities", "contains_security", "security_sources",
        "list_sources", "list_item_types", "list_metrics",
    }

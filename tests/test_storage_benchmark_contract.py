"""Offline contract tests for the Phase 2.3.7.6 storage benchmark."""

from __future__ import annotations

import re
import os
from pathlib import Path

import httpx

from scripts import benchmark_rag_storage as benchmark


def test_generator_is_seeded_and_uses_stable_ledger_ids() -> None:
    first = benchmark.generate_corpus(chunks=128, seed=2376)
    second = benchmark.generate_corpus(chunks=128, seed=2376)
    other = benchmark.generate_corpus(chunks=128, seed=2377)

    assert first.digest == second.digest
    assert first.digest != other.digest
    assert [row.chunk_id for row in first.chunks] == [row.chunk_id for row in second.chunks]
    assert all(re.fullmatch(r"[^#]+(?:#\d+)?", row.chunk_id) for row in first.chunks)
    assert {len(row.embedding) for row in first.chunks} == {benchmark.EMBEDDING_DIMENSION}


def test_fixture_covers_required_labeled_query_categories() -> None:
    fixture = benchmark.load_query_fixture()
    categories = {str(query["category"]) for query in fixture["queries"]}

    assert {
        "exact-symbol",
        "phrase",
        "finance-terminology",
        "filtered",
    } <= categories


def test_small_benchmark_executes_the_full_contract_matrix(tmp_path: Path) -> None:
    result = benchmark.run_benchmark(
        chunks=64,
        seed=2376,
        workdir=tmp_path / "storage-benchmark",
        repeats=1,
        batch_size=32,
    )

    expected_workloads = {
        "dense_top_k",
        "lexical_top_k",
        "hybrid_top_k",
        "filtered_ticker",
        "filtered_source",
        "filtered_item_type",
        "filtered_date",
        "filtered_compound",
        "inventory_count",
        "inventory_source_counts",
        "inventory_ticker_counts",
        "incremental_mutations",
        "restart_first_query",
        "reconciliation",
        "concurrency",
    }
    workloads = result["metrics"]["workloads"]

    assert result["schema_version"] == "phase2.3.7.6"
    assert result["corpus"]["chunks"] == 64
    assert result["corpus"]["digest"]
    assert expected_workloads <= set(workloads)
    assert result["environment"]["embedding"]["network_calls"] == 0
    assert result["environment"]["embedding"]["model_calls"] == 0
    assert result["gates"]["evaluations"]
    assert all(
        evaluation["status"] in {"PASS", "FAIL", "N/A"}
        for evaluation in result["gates"]["evaluations"].values()
    )
    assert all(
        run["latency_ms"] >= 0
        for workload in workloads.values()
        for run in workload.get("runs", [])
    )


def test_gate_evaluation_has_the_spec_thresholds() -> None:
    result = benchmark.evaluate_gates(
        {
            "metrics": {
                "workloads": {
                    "lexical_top_k": {"summary": {"p95_ms": 10}},
                    "dense_top_k": {"summary": {"p95_ms": 10}},
                    "hybrid_top_k": {"summary": {"p95_ms": 10}},
                    "filtered_compound": {"summary": {"p95_ms": 10}},
                    "inventory_count": {"summary": {"p95_ms": 10}},
                    "restart_first_query": {"summary": {"p95_ms": 10}},
                },
                "quality": {"aggregate": {"baseline_ndcg_at_10": 1.0, "hybrid_ndcg_at_10": 1.0}},
                "throughput": {"incremental": {"elapsed_s": 1.0}},
                "identity": {"unrepaired": 0},
            }
        },
        chunks=100_000,
        refresh_window_seconds=60,
    )

    assert result["hard_gate_applicable"] is True
    assert result["thresholds"]["lexical_p95_ms"] == 100
    assert result["thresholds"]["dense_p95_ms"] == 200
    assert result["thresholds"]["hybrid_p95_ms"] == 250
    assert result["thresholds"]["filtered_hybrid_p95_ms"] == 300
    assert result["thresholds"]["inventory_p95_ms"] == 50
    assert result["thresholds"]["restart_readiness_ms"] == 10_000
    assert result["thresholds"]["quality_ndcg_gap"] == 0.02
    assert all(item["status"] == "PASS" for item in result["evaluations"].values())


def test_benchmark_never_calls_llama_server(tmp_path: Path, monkeypatch) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("benchmark attempted a model/network call")

    monkeypatch.setattr(httpx.Client, "post", fail_network)

    previous_telemetry_setting = os.environ.get("ANONYMIZED_TELEMETRY")
    result = benchmark.run_benchmark(
        chunks=24,
        seed=2376,
        workdir=tmp_path / "offline-only",
        repeats=1,
        batch_size=12,
    )

    assert result["environment"]["embedding"]["network_calls"] == 0
    assert os.environ.get("ANONYMIZED_TELEMETRY") == previous_telemetry_setting

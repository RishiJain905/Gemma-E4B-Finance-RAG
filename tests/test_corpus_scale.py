"""tests/test_corpus_scale.py
Offline scale + performance gates for the read-only corpus explorer (2.3.5.3).

Builds the deterministic Phase 2.3 corpus scale fixture (600 securities, 100k
corpus metadata items, mixed states) and measures that bounded aggregate reads
stay within the documented warm p95 targets, that filtered hot paths keep an
indexed query plan, and that every response respects the element/byte caps.

The heavy full-scale cases are marked ``slow`` but still run in the default
offline gate; a smaller always-on variant guards the same invariants cheaply.
"""

from __future__ import annotations

import time

import pytest

from src.middleware.corpus_graph import CorpusGraph
from tests.fixtures.graph.phase2_3_corpus_scale import build_corpus_scale


def _p95(operation, *, samples: int = 25) -> float:
    """Return the warm p95 latency of ``operation`` in milliseconds."""
    for _ in range(3):
        operation()
    timings = []
    for _ in range(samples):
        started = time.perf_counter()
        operation()
        timings.append((time.perf_counter() - started) * 1000)
    return sorted(timings)[max(0, int(0.95 * samples) - 1)]


def _assert_consistent(store, summary):
    for dimension, tally in (
        ("source_category", summary.by_source_category),
        ("item_type", summary.by_item_type),
        ("index", summary.by_index),
        ("sector", summary.by_sector),
        ("indexing_state", summary.by_indexing_state),
        ("event_type", summary.by_event_type),
        ("year", summary.by_year),
    ):
        rows = store.get_corpus_accounting(dimension, limit=200)
        counts = {row["key"]: row["count"] for row in rows}
        assert counts == dict(tally), f"dimension {dimension} drifted from seed"


def test_small_scale_aggregate_counts_match_seed(offline_store):
    summary = build_corpus_scale(
        offline_store, securities=120, items=3_000, observations=150,
        events=150, seed=3,
    )
    _assert_consistent(offline_store, summary)


def test_small_scale_point_lookups_use_indexes(offline_store):
    build_corpus_scale(
        offline_store, securities=80, items=2_000, observations=100,
        events=100, seed=4,
    )
    with offline_store.sqlite._connect() as conn:
        for query, params in (
            ("SELECT * FROM corpus_items WHERE corpus_item_id=?", ("CI-0000001",)),
            ("SELECT * FROM corpus_item_securities WHERE corpus_item_id=?",
             ("CI-0000001",)),
            ("SELECT * FROM corpus_item_sources WHERE corpus_item_id=?",
             ("CI-0000001",)),
        ):
            plan = " ".join(
                row["detail"]
                for row in conn.execute("EXPLAIN QUERY PLAN " + query, params)
            )
            assert "SCAN" not in plan, f"unexpected full scan: {plan}"


@pytest.mark.slow
def test_full_scale_counts_and_warm_p95_targets(offline_store):
    summary = build_corpus_scale(offline_store, securities=600, items=100_000)
    assert summary.corpus_items == 100_000
    assert summary.securities == 600
    _assert_consistent(offline_store, summary)

    graph = CorpusGraph(offline_store)

    overview_p95 = _p95(lambda: graph.groups("source_category", limit=50))
    facet_p95 = _p95(lambda: graph.facets(filters={"index": "sp500"}))
    filtered_p95 = _p95(
        lambda: graph.search(
            kinds=["corpus_item"], filters={"source_category": "company_news"},
            limit=50))

    # Warm targets from the spec (Step 5). Recorded here so a regression is loud.
    assert overview_p95 < 250, f"overview/facet p95 {overview_p95:.1f}ms >= 250ms"
    assert facet_p95 < 250, f"facet p95 {facet_p95:.1f}ms >= 250ms"
    assert filtered_p95 < 300, f"filtered first page p95 {filtered_p95:.1f}ms >= 300ms"


@pytest.mark.slow
def test_full_scale_responses_respect_element_and_byte_caps(offline_store):
    build_corpus_scale(offline_store, securities=600, items=100_000)
    graph = CorpusGraph(offline_store, element_limit=500, excerpt_bytes=300)

    groups = graph.groups("source_category", limit=50)
    node = next(n for n in groups["nodes"]
                if n["metadata"]["bucket_value"] == "company_news")
    page = graph.neighbors(node["id"], limit=100)
    assert len(page["nodes"]) + len(page["edges"]) <= 500
    assert page["next_cursor"]  # oversized branch stays paged behind a cursor

    detail = graph.item_detail(page["nodes"][0]["id"])
    body = detail["nodes"][0]
    assert len(body.get("excerpt", "")) <= 300
    # No unbounded arrays leak into a single response.
    assert len(body["metadata"].get("provenance", [])) <= 20

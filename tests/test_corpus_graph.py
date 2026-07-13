"""tests/test_corpus_graph.py
Offline tests for the read-only corpus projection and explorer limits.
"""

from __future__ import annotations

import re
import time

import pytest

from src.middleware.config import MiddlewareConfig
from src.middleware.corpus_graph import CorpusGraph


def _seed_corpus(store):
    store.sqlite.upsert_fundamental(
        "NVDA", "revenue", 26.0, "usd", "2026-Q1", source_type="sec",
        source_url="https://sec.example/nvda",
    )
    store.sqlite.upsert_fundamental(
        "NVDA", "gross_margin", 0.74, "percent", "2026-Q1",
        source_type="yfinance", source_url="https://finance.example/nvda",
    )
    store.sqlite.register_filing(
        "NVDA", "10-Q", "2026-05-15", "2026-Q1", "ACC-NVDA-1",
        "https://sec.example/filing/ACC-NVDA-1",
    )
    store.sqlite.mark_filing_parsed(
        "ACC-NVDA-1", embedding_id="sec:ACC-NVDA-1:item_1",
        file_path=r"C:\private\parsed\ACC-NVDA-1.txt",
        section_count=2, chunk_count=3,
    )
    store.sqlite.mark_cache_fresh("NVDA", "yfinance_fundamentals", ttl_hours=24)
    store.sqlite.upsert_cache_stale(
        "NVDA", "sec_filings", "token=canary C:\\private\\error.txt",
    )
    store.chroma.records = [
        {
            "id": "sec:ACC-NVDA-1:item_1#0",
            "document": "Item 1. Business. " + "body " * 400,
            "metadata": {
                "source": "sec_filing", "ticker": "NVDA",
                "accession": "ACC-NVDA-1", "form": "10-Q",
                "filing_date": "2026-05-15", "section_key": "item_1",
                "section_heading": "Item 1. Business", "section_index": 0,
                "parent_id": "sec:ACC-NVDA-1:item_1", "chunk_index": 0,
                "chunk_count": 2, "source_url": "https://sec.example/filing",
                "parsed_path": r"C:\private\parsed\ACC-NVDA-1.txt",
            },
        },
        {
            "id": "sec:ACC-NVDA-1:item_1#1",
            "document": "Item 1 continuation",
            "metadata": {
                "source": "sec_filing", "ticker": "NVDA",
                "accession": "ACC-NVDA-1", "form": "10-Q",
                "filing_date": "2026-05-15", "section_key": "item_1",
                "section_heading": "Item 1. Business", "section_index": 0,
                "parent_id": "sec:ACC-NVDA-1:item_1", "chunk_index": 1,
                "chunk_count": 2, "source_url": "https://sec.example/filing",
                "parsed_path": r"C:\private\parsed\ACC-NVDA-1.txt",
            },
        },
        {
            "id": "sec:ACC-NVDA-1:item_2#0",
            "document": "Item 2. Risk factors",
            "metadata": {
                "source": "sec_filing", "ticker": "NVDA",
                "accession": "ACC-NVDA-1", "form": "10-Q",
                "filing_date": "2026-05-15", "section_key": "item_2",
                "section_heading": "Item 2. Risk Factors", "section_index": 1,
                "parent_id": "sec:ACC-NVDA-1:item_2", "chunk_index": 0,
                "chunk_count": 1, "source_url": "https://sec.example/filing",
                "parsed_path": r"C:\private\parsed\ACC-NVDA-1.txt",
            },
        },
        {
            "id": "ir:nvda:release-1",
            "document": "NVIDIA release",
            "metadata": {
                "source": "ir", "ticker": "NVDA", "date": "2026-05-16",
                "parent_id": r"C:\\private\\ir\\release-1",
                "chunk_count": 1, "source_url": "https://ir.example/release",
            },
        },
    ]


def test_overview_is_metadata_only_revision_aware_and_redacts_paths(offline_store):
    _seed_corpus(offline_store)
    graph = CorpusGraph(offline_store, overview_ttl_s=60)

    result = graph.overview()
    assert result["corpus_revision"] == offline_store.retrieval_revision()
    assert {node["kind"] for node in result["nodes"]} >= {
        "source", "ticker", "freshness", "scheduler_source",
    }
    payload = str(result)
    assert "parsed_path" not in payload
    assert "C:\\private" not in payload
    assert "token=canary" not in payload
    assert not any("document" in node for node in result["nodes"])
    assert len(result["nodes"]) + len(result["edges"]) < 500


def test_search_filters_metrics_and_document_families_without_embedding(offline_store):
    _seed_corpus(offline_store)
    graph = CorpusGraph(offline_store)

    metrics = graph.search(q="revenue", kinds=["metric"], ticker="NVDA", limit=10)
    assert metrics["nodes"]
    assert all(node["kind"] == "metric" for node in metrics["nodes"])
    assert all("revenue" in node["label"].lower() for node in metrics["nodes"])

    docs = graph.search(
        q="release", kinds=["document_family"], sources=["ir"],
        ticker="NVDA", limit=10,
    )
    assert [node["kind"] for node in docs["nodes"]] == ["document_family"]
    assert "ir:nvda" not in docs["nodes"][0]["id"]


def test_node_ids_are_opaque_and_details_are_bounded(offline_store):
    _seed_corpus(offline_store)
    graph = CorpusGraph(offline_store, id_ttl_s=60)
    filing = next(node for node in graph.search(kinds=["filing"], limit=10)["nodes"])

    assert re.fullmatch(r"cg1_[A-Za-z0-9_-]+", filing["id"])
    assert "ACC-NVDA-1" not in filing["id"]

    detail = graph.detail(filing["id"])
    assert detail is not None
    assert detail["nodes"][0]["metadata"]["accession"] == "ACC-NVDA-1"
    assert "file_path" not in str(detail)

    section = next(node for node in graph.search(kinds=["section"], limit=10)["nodes"])
    section_detail = graph.detail(section["id"])
    assert len(section_detail["nodes"][0].get("excerpt", "")) <= 1000
    assert "C:\\private" not in str(section_detail)


def test_section_neighbors_are_paginated_and_cap_elements(offline_store):
    _seed_corpus(offline_store)
    graph = CorpusGraph(offline_store, element_limit=4, page_limit=2)
    section = next(node for node in graph.search(kinds=["section"], limit=2)["nodes"])

    result = graph.neighbors(section["id"], limit=2)
    assert result["nodes"]
    assert result["corpus_revision"] == offline_store.retrieval_revision()
    assert len(result["nodes"]) + len(result["edges"]) <= 4


def test_expired_node_ids_and_old_cursors_are_rejected(offline_store):
    _seed_corpus(offline_store)
    now = [100.0]
    graph = CorpusGraph(offline_store, id_ttl_s=1, clock=lambda: now[0])
    result = graph.search(kinds=["filing"], limit=1)
    node_id = result["nodes"][0]["id"]
    now[0] = 102.0
    with pytest.raises(ValueError, match="expired"):
        graph.detail(node_id)

    page = graph = CorpusGraph(offline_store, clock=lambda: now[0])
    first = page.search(kinds=["metric"], limit=1)
    if first["next_cursor"]:
        offline_store.bump_retrieval_revision("test")
        with pytest.raises(ValueError, match="revision"):
            page.search(kinds=["metric"], limit=1, cursor=first["next_cursor"])


def test_explorer_never_calls_embedding_model_network_or_mutation_methods(offline_store, monkeypatch):
    _seed_corpus(offline_store)
    graph = CorpusGraph(offline_store)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden explorer side effect")

    for name in ("search", "add_document", "delete_document", "heartbeat"):
        monkeypatch.setattr(offline_store.chroma, name, forbidden, raising=False)
    monkeypatch.setattr(offline_store.sqlite, "bump_store_revision", forbidden)

    graph.overview()
    graph.search(q="NVDA", kinds=["metric", "fact"], limit=10)


def test_config_clamps_corpus_limits_and_reads_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_PAGE_LIMIT", "9999")
    monkeypatch.setenv("CORPUS_ELEMENT_LIMIT", "9999")
    monkeypatch.setenv("CORPUS_VISIBLE_NODE_TARGET", "9999")
    monkeypatch.setenv("CORPUS_OVERVIEW_CACHE_TTL_S", "9999")
    monkeypatch.setenv("CORPUS_OPAQUE_ID_TTL_S", "9999")

    config = MiddlewareConfig(config_path=tmp_path / "missing.yaml")

    assert config.corpus_page_limit == 200
    assert config.corpus_element_limit == 2000
    assert config.corpus_visible_node_target == 499
    assert config.corpus_overview_cache_ttl_s == 60.0
    assert config.corpus_opaque_id_ttl_s == 3600.0


def test_offline_latency_and_element_caps_stay_within_gate(offline_store):
    _seed_corpus(offline_store)
    graph = CorpusGraph(offline_store, element_limit=6, page_limit=10)
    section = next(node for node in graph.search(kinds=["section"], limit=10)["nodes"])

    for _ in range(3):
        graph.overview()
        graph.search(q="NVDA", kinds=["metric", "fact"], limit=10)
        graph.neighbors(section["id"], limit=10)

    def p95(operation):
        samples = []
        for _ in range(20):
            started = time.perf_counter()
            result = operation()
            samples.append((time.perf_counter() - started) * 1000)
            assert len(result["nodes"]) + len(result["edges"]) <= 6
        return sorted(samples)[18]

    assert p95(graph.overview) < 100
    assert p95(lambda: graph.search(q="NVDA", kinds=["metric"], limit=10)) < 200
    assert p95(lambda: graph.neighbors(section["id"], limit=10)) < 200

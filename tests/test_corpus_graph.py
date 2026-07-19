"""tests/test_corpus_graph.py
Offline tests for the read-only corpus projection and explorer limits.
"""

from __future__ import annotations

import re
import time

import pytest

from src.middleware.config import MiddlewareConfig
from src.middleware.corpus_graph import CorpusGraph, CorpusRevisionChanged
from tests.fixtures.graph.phase2_3_corpus_scale import build_corpus_scale


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


# ── 2.3.5.3: source-independent aggregate nodes, facets, and leaf drill ───────


@pytest.fixture
def scale_store(offline_store):
    """A small deterministic slice of the Phase 2.3 corpus scale fixture."""
    summary = build_corpus_scale(
        offline_store, securities=120, items=2_400, observations=120,
        events=120, seed=7,
    )
    return offline_store, summary


def test_groups_project_registry_dimensions_with_authoritative_counts(scale_store):
    store, summary = scale_store
    graph = CorpusGraph(store)

    result = graph.groups("index", limit=10)
    counts = {n["metadata"]["bucket_value"]: n["metadata"]["count"]
              for n in result["nodes"]}
    assert counts == dict(summary.by_index)
    assert all(n["kind"] == "index" for n in result["nodes"])
    assert result["applied_filters"] == {}
    assert result["total_count"] == len(summary.by_index)
    # Aggregate nodes carry counts + filter payload, never thousands of child ids.
    assert all("filters" in n["metadata"] for n in result["nodes"])
    assert not result["edges"]

    sectors = graph.groups("sector", limit=50)
    sector_counts = {n["metadata"]["bucket_value"]: n["metadata"]["count"]
                     for n in sectors["nodes"]}
    assert sector_counts == dict(summary.by_sector)


def test_facets_are_cached_by_revision_and_reflect_filters(scale_store):
    store, summary = scale_store
    calls = {"n": 0}
    real = store.get_corpus_accounting

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    store.get_corpus_accounting = counting
    graph = CorpusGraph(store)

    first = graph.facets()
    assert set(first["facets"]) >= {"source_category", "item_type", "index"}
    after_first = calls["n"]
    graph.facets()  # identical filters -> served from the revision cache
    assert calls["n"] == after_first

    filtered = graph.facets(filters={"index": "sp500"})
    total = sum(b["count"] for b in filtered["facets"]["source_category"]["buckets"])
    assert total <= summary.corpus_items + summary.events
    assert filtered["applied_filters"] == {"index": "sp500"}

    store.bump_retrieval_revision("test")
    graph.facets()  # revision changed -> recompute
    assert calls["n"] > after_first


def test_aggregate_drill_is_bounded_and_pages_leaf_items(scale_store):
    store, _ = scale_store
    graph = CorpusGraph(store, page_limit=25)
    groups = graph.groups("source_category", limit=20)
    node = next(n for n in groups["nodes"]
                if n["metadata"]["bucket_value"] == "company_news")

    page = graph.neighbors(node["id"], limit=10)
    assert page["nodes"] and all(n["kind"] == "corpus_item" for n in page["nodes"])
    assert len(page["nodes"]) == 10
    assert page["next_cursor"]  # oversized branch requires a next page
    assert all(e["relation"] == "links_item" for e in page["edges"])
    assert page["applied_filters"] == {"source_category": "company_news"}

    page2 = graph.neighbors(node["id"], cursor=page["next_cursor"], limit=10)
    first_ids = {n["id"] for n in page["nodes"]}
    assert first_ids.isdisjoint({n["id"] for n in page2["nodes"]})


def test_search_applies_server_side_facets_and_leaf_kind(scale_store):
    store, _ = scale_store
    graph = CorpusGraph(store)

    result = graph.search(kinds=["corpus_item"], filters={"item_type": "news"}, limit=15)
    assert result["nodes"]
    assert all(n["metadata"]["item_type"] == "news" for n in result["nodes"])
    assert result["applied_filters"] == {"item_type": "news"}


def test_item_detail_is_bounded_and_carries_provenance(scale_store):
    store, _ = scale_store
    graph = CorpusGraph(store, excerpt_bytes=200)
    result = graph.search(kinds=["corpus_item"], limit=1)
    node = result["nodes"][0]

    detail = graph.item_detail(node["id"])
    assert detail is not None
    body = detail["nodes"][0]
    assert body["kind"] == "corpus_item"
    assert "provenance" in body["metadata"] and "securities" in body["metadata"]
    assert len(body.get("excerpt", "")) <= 200


def test_groups_reject_bad_dimension_cursor_and_limit(scale_store):
    store, _ = scale_store
    graph = CorpusGraph(store)
    with pytest.raises(ValueError, match="group_by"):
        graph.groups("not_a_dimension")
    with pytest.raises(ValueError, match="limit"):
        graph.groups("index", limit=99_999)

    first = graph.groups("source_category", limit=1)
    if first["next_cursor"]:
        store.bump_retrieval_revision("test")
        with pytest.raises(CorpusRevisionChanged):
            graph.groups("source_category", limit=1, cursor=first["next_cursor"])


def test_explorer_aggregation_never_enumerates_chroma(scale_store, monkeypatch):
    store, _ = scale_store
    graph = CorpusGraph(store)

    def forbidden(*_a, **_k):
        raise AssertionError("aggregation must not scan Chroma bodies")

    for name in ("get_metadata", "search", "add_document", "count"):
        monkeypatch.setattr(store.chroma, name, forbidden, raising=False)

    graph.groups("index", limit=10)
    graph.facets(filters={"sector": "Technology"})
    graph.search(kinds=["corpus_item"], filters={"item_type": "news"}, limit=10)


def _insert_canary_item(store):
    """Insert one corpus item whose fields carry a secret + local path."""
    now = "2026-06-01T00:00:00Z"
    with store.sqlite._connect() as conn:
        conn.execute(
            "INSERT INTO corpus_items (corpus_item_id, source, source_category, "
            "item_type, title, normalized_headline, language, published_at, "
            "accessed_at, ingested_at, source_url, content_hash, document_family, "
            "document_family_id, indexing_status, license_label, "
            "normalization_version, evidence_authority, narrative_bytes, "
            "metadata_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("CI-CANARY", "finnhub", "company_news", "news",
             r"Secret C:\private\leak.txt", "secret", "en", now, now, now,
             "https://example.test/x?api_key=CANARYSECRET", "c" * 64, "famc",
             "CI-CANARY", "indexed", "public", "v1", "provider", 100, 20),
        )
        conn.execute(
            "INSERT INTO corpus_item_sources (corpus_item_id, source_key, "
            "source_name, source_category, source_url, accessed_at, ingested_at, "
            "license_label, evidence_authority) VALUES (?,?,?,?,?,?,?,?,?)",
            ("CI-CANARY", "k1", "finnhub", "company_news",
             "https://example.test/prov?token=CANARYTOKEN", now, now, "public",
             "provider"),
        )
        conn.commit()


def test_new_responses_scrub_secrets_local_paths_and_bound_text(offline_store):
    _insert_canary_item(offline_store)
    graph = CorpusGraph(offline_store)

    listing = graph.search(kinds=["corpus_item"], filters={"item_type": "news"}, limit=5)
    node = next(n for n in listing["nodes"] if "Secret" in n["label"])
    detail = graph.item_detail(node["id"])
    payload = str(detail) + str(listing) + str(graph.facets())
    assert "CANARYSECRET" not in payload
    assert "CANARYTOKEN" not in payload
    assert "C:\\private" not in payload
    assert "api_key" not in payload

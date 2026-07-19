"""Offline tests for Phase 2.3 stable evidence taxonomy, filters, and ranking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "fixtures/evaluation/phase2_3_sources.json"


@pytest.fixture
def documents() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["documents"]


def test_vocabulary_is_versioned_and_source_independent() -> None:
    from src.middleware.evidence_taxonomy import (
        EVENT_TYPES,
        ITEM_TYPES,
        SOURCE_CATEGORIES,
        TAXONOMY_VERSION,
    )

    assert TAXONOMY_VERSION == "finance-evidence-taxonomy-v1"
    assert SOURCE_CATEGORIES == (
        "sec", "issuer", "market_data", "company_news", "central_bank",
        "treasury", "economic_agency", "regulator", "sector_agency",
        "transcript", "estimates", "global_news",
    )
    assert ITEM_TYPES == (
        "filing", "filing_exhibit", "press_release", "news", "speech",
        "policy_release", "economic_release", "market_observation",
        "corporate_action", "regulatory_event", "contract_award", "recall",
        "transcript", "estimate",
    )
    assert {
        "debt_raise", "equity_raise", "beneficial_ownership_change",
        "economic_release", "monetary_policy_decision", "recall",
        "contract_award",
    } <= set(EVENT_TYPES)
    assert not any(provider in SOURCE_CATEGORIES for provider in ("finnhub", "massive", "gdelt"))


def test_policy_config_defaults_on_and_clamps_authority_bound(tmp_path: Path) -> None:
    from src.middleware.config import MiddlewareConfig

    config_path = tmp_path / "middleware.yaml"
    config_path.write_text(
        "authority_max_boost: 9\nmax_secondary_per_event: 99\n",
        encoding="utf-8",
    )
    config = MiddlewareConfig(config_path=config_path)
    assert config.enable_evidence_taxonomy is True
    assert config.enable_authority_ranking is True
    assert config.enable_duplicate_coverage_packing is True
    assert config.authority_max_boost == 0.025
    assert config.max_secondary_per_event == 5


@pytest.mark.parametrize(
    ("document_id", "expected"),
    [
        ("sec-orcl-notes-2026", ("sec", "filing", 1)),
        ("finnhub-orcl-notes-2026", ("company_news", "news", 3)),
        ("bls-cpi-2026-06", ("economic_agency", "economic_release", 1)),
        ("nhtsa-recall-2026", ("sector_agency", "recall", 1)),
    ],
)
def test_normalization_maps_provider_metadata_to_stable_facets(
    documents: list[dict], document_id: str, expected: tuple[str, str, int]
) -> None:
    from src.middleware.evidence_taxonomy import normalize_evidence

    row = next(row for row in documents if row["id"] == document_id)
    normalized = normalize_evidence(row)
    metadata = normalized["metadata"]
    assert (metadata["source_category"], metadata["item_type"], metadata["authority_rank"]) == expected
    assert metadata["taxonomy_version"] == "finance-evidence-taxonomy-v1"
    assert metadata["source"] == row["metadata"]["source"]


@pytest.mark.parametrize(
    ("filters", "expected_id"),
    [
        ({"security": "ORCL"}, "sec-orcl-notes-2026"),
        ({"index_membership": "sp500"}, "sec-orcl-notes-2026"),
        ({"sector": "Information Technology"}, "sec-orcl-notes-2026"),
        ({"industry": "Software—Infrastructure"}, "sec-orcl-notes-2026"),
        ({"source_category": "company_news"}, "finnhub-orcl-notes-2026"),
        ({"source": "bls"}, "bls-cpi-2026-06"),
        ({"item_type": "recall"}, "nhtsa-recall-2026"),
        ({"event_type": "economic_release"}, "bls-cpi-2026-06"),
        ({"form": "10-Q"}, "meta-historical-2021"),
        ({"item": "2.03"}, "sec-orcl-notes-2026"),
        ({"exhibit": "EX-10.1"}, "sec-orcl-notes-2026"),
        ({"published_from": "2026-07-12", "published_to": "2026-07-12T23:59:59Z"}, "bls-cpi-2026-06"),
        ({"effective_from": "2026-07-11", "effective_to": "2026-07-11T23:59:59Z"}, "nhtsa-recall-2026"),
        ({"as_of_from": "2021-09-01", "as_of_to": "2021-09-30T23:59:59Z"}, "meta-historical-2021"),
        ({"freshness_status": "fresh"}, "sec-orcl-notes-2026"),
        ({"indexing_status": "indexed"}, "sec-orcl-notes-2026"),
        ({"authority_tier": "direct_sec"}, "sec-orcl-notes-2026"),
    ],
)
def test_every_stable_filter_facet_matches_seeded_corpus(
    documents: list[dict], filters: dict, expected_id: str
) -> None:
    from src.middleware.evidence_taxonomy import evidence_matches_filters

    matches = [row["id"] for row in documents if evidence_matches_filters(row, filters)]
    assert expected_id in matches


def test_filters_compose_on_mixed_seeded_corpus(documents: list[dict]) -> None:
    from src.middleware.evidence_taxonomy import filter_evidence

    filters = {
        "security": "ORCL",
        "index_membership": "sp500",
        "sector": "Information Technology",
        "industry": "Software—Infrastructure",
        "source_category": "sec",
        "source": "sec",
        "item_type": "filing",
        "event_type": "debt_raise",
        "form": "424B5",
        "item": "2.03",
        "exhibit": "EX-10.1",
        "published_from": "2026-07-10",
        "published_to": "2026-07-10T23:59:59Z",
        "effective_from": "2026-07-10",
        "effective_to": "2026-07-10T23:59:59Z",
        "as_of_from": "2026-07-10",
        "as_of_to": "2026-07-10T23:59:59Z",
        "freshness_status": "fresh",
        "indexing_status": "indexed",
        "authority_tier": "direct_sec",
    }
    assert [row["id"] for row in filter_evidence(documents, filters)] == [
        "sec-orcl-notes-2026"
    ]


def test_plural_filter_values_are_or_within_and_across_facets(documents: list[dict]) -> None:
    from src.middleware.evidence_taxonomy import filter_evidence

    matches = filter_evidence(documents, {
        "source_categories": ["sec", "issuer"],
        "industries": ["Software—Infrastructure"],
        "authority_tiers": ["direct_sec", "issuer"],
    })
    assert [row["id"] for row in matches] == [
        "sec-orcl-notes-2026", "issuer-orcl-notes-2026"
    ]


def test_domain_timestamp_never_uses_ingestion_time(documents: list[dict]) -> None:
    from src.middleware.evidence_taxonomy import normalize_evidence

    historical = next(row for row in documents if row["id"] == "meta-historical-2021")
    metadata = normalize_evidence(historical)["metadata"]
    assert metadata["domain_timestamp"] == "2021-10-25T20:00:00Z"
    assert metadata["domain_timestamp_kind"] == "published_at"
    assert metadata["ingestion_time"] == "2026-07-14T20:00:00Z"


def test_structured_fact_receives_self_describing_taxonomy() -> None:
    from src.middleware.evidence import EvidenceItem

    item = EvidenceItem.from_row({
        "id": "fact-1", "ticker": "ORCL", "metric": "total_debt",
        "value": 42, "period": "2025-12-31", "as_of": "2026-02-01",
        "source_type": "sec_companyfacts",
    }, kind="fact")
    assert item.item_type == "filing"
    assert item.authority_tier == "primary"
    assert item.source == "sec_companyfacts"
    assert item.canonical_security == "ORCL"
    assert item.date_semantics["source_vintage"] == "2026-02-01"


def test_provider_filing_summary_is_not_promoted_to_direct_observation() -> None:
    from src.middleware.evidence_taxonomy import normalize_evidence

    metadata = normalize_evidence({
        "id": "vendor-filing", "document": "Provider filing summary",
        "metadata": {
            "source": "massive", "source_category": "vendor_filing_metadata",
            "item_type": "filing", "authority_tier": "provider",
        },
    })["metadata"]
    assert metadata["source_category"] == "market_data"
    assert metadata["authority_rank"] == 3


def test_provider_news_is_categorized_by_content_not_provider() -> None:
    from src.middleware.evidence_taxonomy import normalize_evidence

    metadata = normalize_evidence({
        "id": "massive-news", "document": "Licensed company story",
        "metadata": {
            "source": "massive", "source_category": "news_vendor",
            "item_type": "news", "authority_tier": "licensed",
        },
    })["metadata"]
    assert metadata["source_category"] == "company_news"
    assert metadata["authority_rank"] == 3


def test_store_search_resolves_former_ticker_as_of_requested_period(
    documents: list[dict],
) -> None:
    from src.storage.store import Store

    class FakeChroma:
        last_search_timings = {}

        def search(self, query, n_results=5, filter_dict=None):
            return [dict(row) for row in documents[:n_results]]

    class FakeSQLite:
        def resolve_security(self, symbol, provider=None, as_of=None):
            assert (symbol, provider, as_of) == ("FB", None, "2021-10-01")
            return {"security_id": "sec-meta", "ticker": "META"}

        def search_facts(self, ticker, limit):
            return []

    store = Store.__new__(Store)
    store.chroma = FakeChroma()
    store.sqlite = FakeSQLite()
    result = store.search(
        "FB historical filing", n_results=5,
        filters={
            "security": "FB", "as_of": "2021-10-01",
            "item_type": "filing", "published_to": "2021-12-31T23:59:59Z",
        },
    )
    assert result["ticker"] == "META"
    assert [row["id"] for row in result["documents"]] == ["meta-historical-2021"]
    assert result["documents"][0]["metadata"]["canonical_security"] == "META"


def test_store_pushes_composable_exact_and_range_filters_to_chroma() -> None:
    from src.storage.store import Store

    class FakeChroma:
        last_search_timings = {}
        filter_dict = None

        def search(self, query, n_results=5, filter_dict=None):
            self.filter_dict = filter_dict
            return []

    class FakeSQLite:
        def search_facts(self, ticker, limit):
            return []

    store = Store.__new__(Store)
    store.chroma = FakeChroma()
    store.sqlite = FakeSQLite()
    store.search(
        "Oracle financing", n_results=5, ticker="ORCL",
        filters={
            "source_category": "sec", "item_type": "filing",
            "published_from": "2026-01-01", "published_to": "2026-12-31",
        },
    )
    assert store.chroma.filter_dict == {"$and": [
        {"ticker": "ORCL"},
        {"source_category": "sec"},
        {"item_type": "filing"},
        {"published_at": {"$gte": "2026-01-01", "$lte": "2026-12-31"}},
    ]}


def test_authority_is_bounded_after_relevance_and_retains_secondary(documents: list[dict]) -> None:
    from src.middleware.evidence_taxonomy import rank_evidence

    event_docs = [
        row for row in documents
        if row["id"] in {"sec-orcl-notes-2026", "finnhub-orcl-notes-2026"}
    ]
    ranked = rank_evidence(
        event_docs,
        query="Oracle debt financing filing",
        filters={"security": "ORCL", "event_type": "debt_raise"},
        authority_max_boost=0.025,
        recency_max_boost=0.0,
    )
    assert [row["id"] for row in ranked] == [
        "sec-orcl-notes-2026", "finnhub-orcl-notes-2026"
    ]
    assert ranked[0]["authority_boost"] == pytest.approx(0.025)
    assert 0.0 <= ranked[1]["authority_boost"] <= 0.025

    highly_relevant_secondary = dict(event_docs[1], rerank_score=0.95)
    weak_primary = dict(event_docs[0], rerank_score=0.70)
    relevance_wins = rank_evidence(
        [weak_primary, highly_relevant_secondary],
        query="investor reaction",
        authority_max_boost=0.025,
        recency_max_boost=0.0,
    )
    assert relevance_wins[0]["id"] == "finnhub-orcl-notes-2026"

    rrf_relevance_wins = rank_evidence(
        [dict(event_docs[0], fusion_score=0.030),
         dict(event_docs[1], fusion_score=0.032)],
        query="investor reaction", authority_max_boost=0.025,
        recency_max_boost=0.0,
    )
    assert rrf_relevance_wins[0]["id"] == "finnhub-orcl-notes-2026"


def test_recency_is_strong_for_latest_news_and_weak_for_historical_filings(
    documents: list[dict],
) -> None:
    from src.middleware.evidence_taxonomy import rank_evidence

    older = dict(next(row for row in documents if row["id"] == "finnhub-orcl-notes-2026"))
    newer = dict(older)
    newer["id"] = "newer-news"
    newer["metadata"] = {**older["metadata"], "published_at": "2026-07-14T14:00:00Z"}
    latest = rank_evidence([older, newer], query="latest Oracle news")
    assert latest[0]["id"] == "newer-news"
    assert latest[0]["recency_boost"] - latest[1]["recency_boost"] > 0.01

    filing_2020 = dict(next(row for row in documents if row["id"] == "sec-orcl-notes-2026"))
    filing_2020["id"] = "filing-2020"
    filing_2020["metadata"] = {**filing_2020["metadata"], "published_at": "2020-03-01T00:00:00Z"}
    filing_2024 = dict(filing_2020)
    filing_2024["id"] = "filing-2024"
    filing_2024["metadata"] = {**filing_2020["metadata"], "published_at": "2024-03-01T00:00:00Z"}
    historical = rank_evidence([filing_2024, filing_2020], query="Oracle 2020 historical filing")
    by_id = {row["id"]: row for row in historical}
    assert by_id["filing-2020"]["date_match_boost"] > by_id["filing-2024"]["date_match_boost"]
    assert max(row["recency_boost"] for row in historical) <= 0.005


def test_duplicate_coverage_packing_keeps_primary_and_material_secondary(
    documents: list[dict],
) -> None:
    from src.middleware.evidence_taxonomy import pack_event_coverage, rank_evidence

    ranked = rank_evidence(documents, query="latest Oracle financing news")
    packed = pack_event_coverage(ranked, limit=5, max_secondary_per_event=2)
    ids = [row["id"] for row in packed]
    assert "sec-orcl-notes-2026" in ids
    assert "finnhub-orcl-notes-2026" in ids
    assert not ({"syndicated-orcl-notes-a", "syndicated-orcl-notes-b"} <= set(ids))
    assert len(ids) == 5


def test_evidence_ledger_and_graph_share_taxonomy_metadata(documents: list[dict]) -> None:
    from src.middleware.evidence import EvidenceItem, assign_evidence_ids
    from src.middleware.graph_observer import event_graph_deltas
    from src.middleware.stream_events import QueryEventEmitter
    from src.middleware.evidence_taxonomy import normalize_evidence

    row = normalize_evidence(next(row for row in documents if row["id"] == "sec-orcl-notes-2026"))
    item = assign_evidence_ids([EvidenceItem.from_row(row, kind="document")])[0]
    emitter = QueryEventEmitter(query_id="q-taxonomy")
    event = emitter.graph_evidence(
        evidence_id=item.evidence_id,
        kind=item.kind,
        excerpt=item.document,
        metadata=item.taxonomy_metadata(),
        source_type=item.source_type,
        rank=1,
    )
    node = next(
        delta.node for delta in event_graph_deltas(
            event, excerpt_chars=1000, question_preview_chars=200
        ) if delta.node and delta.node.kind == "evidence"
    )
    ledger = item.to_dict()
    for key in (
        "item_type", "event_type", "authority_tier", "source",
        "date_semantics", "canonical_security", "coverage_tier",
    ):
        assert node.metadata[key] == ledger[key]
    assert ledger["store_id"] == "sec-orcl-notes-2026"


def test_resolved_citation_and_graph_node_share_taxonomy_metadata(
    documents: list[dict], monkeypatch,
) -> None:
    from src.middleware import app as middleware_app
    from src.middleware.answer_validator import validate_answer
    from src.middleware.config import MiddlewareConfig
    from src.middleware.evidence import EvidenceItem, assign_evidence_ids
    from src.middleware.graph_observer import TraceHub
    from src.middleware.models import QueryRequest

    row = next(row for row in documents if row["id"] == "sec-orcl-notes-2026")
    item = assign_evidence_ids([EvidenceItem.from_row(row, kind="document")])[0]
    citation_record = validate_answer("Oracle raised debt [E1].", [item]).citations[0]
    citation = middleware_app._evidence_citation_model(citation_record)

    config = MiddlewareConfig()
    config.enable_graph_observer = True
    hub = TraceHub()
    monkeypatch.setattr(middleware_app, "config", config)
    monkeypatch.setattr(middleware_app, "graph_hub", hub)
    emitter = middleware_app._install_query_emitter(
        QueryRequest(question="How did Oracle finance debt?"), chat_events=False,
    )
    middleware_app._emit_graph_evidence(
        {"facts": [], "documents": [row]}, [item],
    )
    middleware_app._emit_graph_terminal(
        {
            "retrieval": {"facts": [], "documents": [row]},
            "grounding_level": "grounded", "graph_evidence_ids": ["E1"],
        },
        model_available=True, evidence_citations=[citation], validation={},
    )
    citation_node = next(
        node for node in hub.snapshot(emitter.query_id)["nodes"]
        if node["kind"] == "citation"
    )
    expected = item.taxonomy_metadata()
    for key, value in expected.items():
        assert getattr(citation, key) == value
        assert citation_node["metadata"][key] == value
    assert citation.evidence_id == item.evidence_id == "E1"

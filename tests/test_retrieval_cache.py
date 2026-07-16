"""
tests/test_retrieval_cache.py
Offline tests for the versioned retrieval cache (Phase 2.2.6.2).

Everything runs without the middleware, model, ChromaDB process, or network:
clocks, stores, and evidence are all fakes, and llama-server is never contacted.
The suite proves EXACT invalidation (fact insert/update, document add / replace /
delete including a same-count replacement, refresh, config/model fingerprint
change, TTL expiry, explicit bypass), immutable cached copies, LRU eviction,
concurrent lookup, and the orchestrator wiring (hit reuses evidence without
re-retrieving; miss stores; refresh + revision bumps invalidate).
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Mock chromadb before importing storage/middleware modules (mirrors sibling
# test suites) so collection never touches a real ChromaDB install.
if "chromadb" not in sys.modules:  # pragma: no cover - import shim
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from src.middleware.adaptive_orchestrator import Lane, orchestrate
from src.middleware.config import MiddlewareConfig
from src.middleware.query_plan import QueryEntity, QueryPlan, QuerySubquery, normalize_question
from src.middleware.retrieval_cache import (
    RetrievalCache,
    build_cache_key,
    config_fingerprint,
)


# ── Builders ──────────────────────────────────────────────


class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def make_config(**overrides) -> MiddlewareConfig:
    cfg = MiddlewareConfig()
    cfg.enable_adaptive_rag = True
    cfg.enable_retrieval_cache = True
    cfg.enable_reranker = False
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_plan(question="tell me about the company", *, entities=(), metrics=(),
              intents=("general",), periods=()) -> QueryPlan:
    plan_entities = [
        QueryEntity(ticker=t.upper(), resolved_name=None, confidence=1.0,
                    source="resolved", mention=t, start=i)
        for i, t in enumerate(entities)
    ]
    intents = list(intents) or ["general"]
    sq0 = QuerySubquery(
        id="sq0", text=question,
        entity_tickers=tuple(e.ticker for e in plan_entities),
        intents=tuple(intents), metrics=tuple(metrics), periods=tuple(periods))
    return QueryPlan(
        original_question=question, retrieval_query=question,
        normalized_question=normalize_question(question),
        entities=plan_entities, intents=intents, metrics=list(metrics),
        periods=list(periods), subqueries=[sq0], primary_intent=intents[0])


class FakeRetriever:
    """Counts retrieve() calls so a cache hit is observable as 'not retrieved'."""

    def __init__(self):
        self.calls = 0

    def retrieve(self, query, intent, top_k_documents=5, top_k_facts=10):
        self.calls += 1
        return {
            "facts": [{"metric": "m", "value": self.calls, "ticker": "NVDA",
                       "period": None, "source_type": "yfinance"}],
            "documents": [{"id": f"doc-{self.calls}", "document": "context body",
                           "metadata": {"ticker": "NVDA"}}],
            "retrieval_strategy": "vector",
        }


def fake_store(revision=0):
    holder = {"rev": revision}
    store = SimpleNamespace(
        retrieval_revision=lambda: holder["rev"],
        sqlite=SimpleNamespace(list_metrics=lambda: []),
    )
    return store, holder


# ── RetrievalCache unit ───────────────────────────────────


def test_set_get_roundtrip():
    cache = RetrievalCache(max_entries=4, ttl_s=100, clock=_Clock())
    assert cache.get("k") is None
    assert cache.set("k", {"documents": [{"id": "d"}]}) is True
    assert cache.get("k") == {"documents": [{"id": "d"}]}


def test_cached_values_are_immutable_copies():
    """Mutating a returned value must never poison a later hit (frozen copies)."""
    cache = RetrievalCache(clock=_Clock())
    cache.set("k", {"documents": [{"id": "d", "tags": ["a"]}]})

    first = cache.get("k")
    first["documents"][0]["tags"].append("MUTATED")
    first["documents"].append({"id": "injected"})

    second = cache.get("k")
    assert second == {"documents": [{"id": "d", "tags": ["a"]}]}


def test_stored_value_isolated_from_caller_mutation():
    """A caller mutating the source dict after set() must not change the entry."""
    cache = RetrievalCache(clock=_Clock())
    source = {"documents": [{"id": "d"}]}
    cache.set("k", source)
    source["documents"][0]["id"] = "CHANGED"
    assert cache.get("k") == {"documents": [{"id": "d"}]}


def test_ttl_expiry_with_injected_clock():
    clock = _Clock(0.0)
    cache = RetrievalCache(ttl_s=100, clock=clock)
    cache.set("k", {"v": 1})

    clock.t = 99.9
    assert cache.get("k") == {"v": 1}      # still fresh
    clock.t = 100.0
    assert cache.get("k") is None          # expired exactly at the bound
    assert cache.stats["expired"] == 1


def test_lru_eviction_by_entry_count():
    cache = RetrievalCache(max_entries=2, ttl_s=1000, clock=_Clock())
    cache.set("a", {"v": 1})
    cache.set("b", {"v": 2})
    cache.get("a")                 # touch a -> b is now least-recent
    cache.set("c", {"v": 3})       # evicts b

    assert cache.get("a") == {"v": 1}
    assert cache.get("c") == {"v": 3}
    assert cache.get("b") is None
    assert cache.stats["evictions"] == 1


def test_oversized_value_refused():
    cache = RetrievalCache(max_value_chars=50, clock=_Clock())
    assert cache.set("k", {"documents": [{"body": "x" * 500}]}) is False
    assert cache.get("k") is None
    assert cache.stats["too_large"] == 1


def test_fingerprint_change_clears_cache():
    cache = RetrievalCache(fingerprint="fp1", clock=_Clock())
    cache.set("k", {"v": 1})
    assert cache.check_fingerprint("fp1") is False
    assert cache.get("k") == {"v": 1}

    assert cache.check_fingerprint("fp2") is True   # config/model changed
    assert cache.get("k") is None
    assert len(cache) == 0


def test_concurrent_lookup_is_thread_safe():
    cache = RetrievalCache(max_entries=64, ttl_s=1000, clock=_Clock())
    errors: list = []

    def worker(n):
        try:
            for i in range(200):
                key = f"k{(n + i) % 32}"
                cache.set(key, {"n": n, "i": i})
                cache.get(key)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(cache) <= 64        # bound never exceeded under concurrency


# ── Cache key: exact invalidation signals ─────────────────


def test_key_changes_with_store_revision():
    cfg = make_config()
    plan = make_plan(entities=("NVDA",))
    k1 = build_cache_key(plan=plan, config=cfg, lane=Lane.STANDARD, revision=1)
    k2 = build_cache_key(plan=plan, config=cfg, lane=Lane.STANDARD, revision=2)
    assert k1 != k2


def test_key_changes_with_corpus_revision():
    cfg = make_config()
    plan = make_plan(entities=("NVDA",))
    first = build_cache_key(
        plan=plan, config=cfg, lane=Lane.STANDARD, revision=7, corpus_revision=3
    )
    second = build_cache_key(
        plan=plan, config=cfg, lane=Lane.STANDARD, revision=7, corpus_revision=4
    )
    assert first != second


def test_key_changes_with_config_fingerprint():
    plan = make_plan(entities=("NVDA",))
    cfg_a = make_config(top_k_documents=5)
    cfg_b = make_config(top_k_documents=8)
    k_a = build_cache_key(plan=plan, config=cfg_a, lane=Lane.STANDARD, revision=1)
    k_b = build_cache_key(plan=plan, config=cfg_b, lane=Lane.STANDARD, revision=1)
    assert k_a != k_b
    assert config_fingerprint(cfg_a) != config_fingerprint(cfg_b)


def test_key_includes_lexical_backend_profile_filters_and_candidate_limits():
    plan = make_plan(entities=("NVDA",))
    plan.evidence_filters = {"item_type": "sec_filing", "security": "nvda"}
    base = make_config(
        lexical_backend="fts5", profile="recommended", rerank_candidates=30,
    )
    rollback = make_config(
        lexical_backend="memory", profile="legacy", rerank_candidates=30,
    )
    wider = make_config(
        lexical_backend="fts5", profile="recommended", rerank_candidates=60,
    )

    key = build_cache_key(plan=plan, config=base, lane=Lane.STANDARD, revision=4)

    assert '"lexical_backend": "fts5"' in key
    assert '"retrieval_profile": "recommended"' in key
    assert '"item_type": "sec_filing"' in key
    assert key != build_cache_key(
        plan=plan, config=rollback, lane=Lane.STANDARD, revision=4
    )
    assert key != build_cache_key(
        plan=plan, config=wider, lane=Lane.STANDARD, revision=4
    )


def test_key_stable_for_identical_request():
    cfg = make_config()
    plan = make_plan(entities=("NVDA",), metrics=("total_revenue",))
    k1 = build_cache_key(plan=plan, config=cfg, lane=Lane.STANDARD, revision=3,
                         available_metrics=("total_revenue",))
    k2 = build_cache_key(plan=plan, config=cfg, lane=Lane.STANDARD, revision=3,
                         available_metrics=("total_revenue",))
    assert k1 == k2


def test_key_changes_with_lane_and_plan_fields():
    cfg = make_config()
    base = make_plan(entities=("NVDA",))
    assert (
        build_cache_key(plan=base, config=cfg, lane=Lane.STANDARD, revision=1)
        != build_cache_key(plan=base, config=cfg, lane=Lane.COMPLEX, revision=1)
    )
    other = make_plan(entities=("AMD",))
    assert (
        build_cache_key(plan=base, config=cfg, lane=Lane.STANDARD, revision=1)
        != build_cache_key(plan=other, config=cfg, lane=Lane.STANDARD, revision=1)
    )


# ── Orchestrator wiring ───────────────────────────────────


def test_orchestrator_hit_reuses_evidence_without_retrieving():
    cfg = make_config()
    cache = RetrievalCache(fingerprint=config_fingerprint(cfg), clock=_Clock())
    store, _ = fake_store(revision=1)
    retr = FakeRetriever()

    first = orchestrate(make_plan(), store, cfg, retriever=retr,
                        retrieval_cache=cache)
    assert retr.calls == 1
    assert "retrieval_cache_miss" in first.reason_codes

    second = orchestrate(make_plan(), store, cfg, retriever=retr,
                         retrieval_cache=cache)
    assert retr.calls == 1  # served from cache — no second retrieval
    assert "retrieval_cache_hit" in second.reason_codes
    assert second.merged_documents == first.merged_documents


def test_orchestrator_revision_bump_invalidates():
    cfg = make_config()
    cache = RetrievalCache(fingerprint=config_fingerprint(cfg), clock=_Clock())
    store, holder = fake_store(revision=1)
    retr = FakeRetriever()

    orchestrate(make_plan(), store, cfg, retriever=retr, retrieval_cache=cache)
    assert retr.calls == 1

    holder["rev"] = 2  # an ingestion write advanced the store revision
    orchestrate(make_plan(), store, cfg, retriever=retr, retrieval_cache=cache)
    assert retr.calls == 2  # stale entry never served


def test_orchestrator_refresh_bypasses_lookup():
    cfg = make_config()
    cache = RetrievalCache(fingerprint=config_fingerprint(cfg), clock=_Clock())
    store, _ = fake_store(revision=1)
    retr = FakeRetriever()

    orchestrate(make_plan(), store, cfg, retriever=retr, retrieval_cache=cache)
    assert retr.calls == 1
    # Explicit refresh must re-retrieve even though the key is unchanged.
    result = orchestrate(make_plan(), store, cfg, retriever=retr,
                         retrieval_cache=cache, refresh=True)
    assert retr.calls == 2
    assert "retrieval_cache_hit" not in result.reason_codes


def test_orchestrator_disabled_flag_never_caches():
    cfg = make_config(enable_retrieval_cache=False)
    cache = RetrievalCache(fingerprint=config_fingerprint(cfg), clock=_Clock())
    store, _ = fake_store(revision=1)
    retr = FakeRetriever()

    orchestrate(make_plan(), store, cfg, retriever=retr, retrieval_cache=cache)
    orchestrate(make_plan(), store, cfg, retriever=retr, retrieval_cache=cache)
    assert retr.calls == 2  # flag off -> no caching
    assert len(cache) == 0


def test_orchestrator_store_without_revision_bypasses():
    cfg = make_config()
    cache = RetrievalCache(fingerprint=config_fingerprint(cfg), clock=_Clock())
    store = SimpleNamespace(sqlite=SimpleNamespace(list_metrics=lambda: []))
    retr = FakeRetriever()

    orchestrate(make_plan(), store, cfg, retriever=retr, retrieval_cache=cache)
    orchestrate(make_plan(), store, cfg, retriever=retr, retrieval_cache=cache)
    assert retr.calls == 2  # no revision -> cannot invalidate safely -> bypass


def test_orchestrator_cache_failure_is_a_miss_not_an_error():
    """A cache whose get()/set() raise must degrade to a plain retrieval."""
    cfg = make_config()
    store, _ = fake_store(revision=1)
    retr = FakeRetriever()

    boom = MagicMock()
    boom.check_fingerprint.side_effect = RuntimeError("boom")
    result = orchestrate(make_plan(), store, cfg, retriever=retr,
                         retrieval_cache=boom)
    assert retr.calls == 1
    assert result.merged_documents  # query still answered from fresh retrieval


# ── Store facade revision bumps (exact invalidation source) ───


@pytest.fixture
def revision_store(tmp_path: Path):
    from src.storage.store import Store
    with patch("src.storage.store.ChromaStore") as mock_cls:
        chroma = MagicMock()
        chroma.heartbeat.return_value = True
        chroma.count.return_value = 0
        chroma.count_filing_section_chunks.side_effect = None
        mock_cls.return_value = chroma
        store = Store(db_path=tmp_path / "rev.db", chroma_path=tmp_path / "chroma")
        yield store, chroma


def test_fact_insert_and_update_bump_revision(revision_store):
    store, _chroma = revision_store
    r0 = store.retrieval_revision()
    store.save_fundamental("NVDA", "revenue", 26.0, period="2026-Q1")
    r1 = store.retrieval_revision()
    assert r1 > r0
    store.save_fundamental("NVDA", "revenue", 27.0, period="2026-Q1")  # update
    assert store.retrieval_revision() > r1


def test_document_add_bumps_revision(revision_store):
    store, _chroma = revision_store
    r0 = store.retrieval_revision()
    store.save_document("sec/NVDA/10-K", "body", ticker="NVDA")
    assert store.retrieval_revision() > r0


def test_document_delete_bumps_revision(revision_store):
    store, _chroma = revision_store
    r0 = store.retrieval_revision()
    store.delete_filing_section_family("sec:ACC-1:item_2")
    assert store.retrieval_revision() > r0


def test_same_count_document_replacement_bumps_revision(revision_store):
    """A same-count section replacement changes the docs even though Chroma's
    count is unchanged, so it must still bump the revision (count is never used
    for invalidation)."""
    store, chroma = revision_store
    from src.sec.filing_sections import FilingSection

    section = FilingSection(
        accession="ACC-1", ticker="NVDA", form="10-Q",
        filing_date="2026-05-15", report_period="2026-03-31",
        section_key="item_2", section_heading="Item 2. MD&A", section_index=0,
        text="Item 2. MD&A\nUpdated revenue.", source_url="https://sec.example",
        parsed_path="parsed/ACC-1.txt")
    chroma.count_filing_section_chunks.side_effect = [1, 1]  # same count before/after

    r0 = store.retrieval_revision()
    store.add_filing_sections([section])
    assert store.retrieval_revision() > r0


def test_revision_survives_bump_when_sqlite_errors(revision_store, monkeypatch):
    """A revision-bump failure must never crash a Store mutation (fail-soft)."""
    store, _chroma = revision_store
    monkeypatch.setattr(
        store.sqlite, "bump_store_revision",
        MagicMock(side_effect=RuntimeError("locked")))
    # Must not raise even though the bump fails.
    store.save_document("sec/NVDA/8-K", "body", ticker="NVDA")

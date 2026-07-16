"""Offline contracts for the persistent SQLite FTS5 lexical channel."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.middleware.lexical_index import LexicalIndex, compile_fts_query
from src.middleware.config import MiddlewareConfig
from src.middleware.retriever import Retriever
from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash
from src.ingestion.records import NarrativeRecord
from src.storage.sqlite_store import SQLiteStore
from src.storage.store import Store


def _chunk(
    chunk_id: str,
    body: str,
    *,
    family_id: str | None = None,
    ticker: str = "ACME",
    source_category: str = "regulatory_filing",
    item_type: str = "sec_filing",
    published_at: str = "2026-07-01",
) -> dict:
    return {
        "id": chunk_id,
        "document": body,
        "metadata": {
            "document_family_id": family_id or chunk_id.split("#", 1)[0],
            "title": "Acme quarterly filing",
            "ticker": ticker,
            "tickers": ticker,
            "source_category": source_category,
            "source_name": "sec",
            "item_type": item_type,
            "published_at": published_at,
            "form": "10-Q",
            "authority_tier": "direct_sec",
        },
    }


class _HydratingChroma:
    def __init__(self, rows: list[dict], revision: int):
        self.rows = {row["id"]: row for row in rows}
        self.revision = revision
        self.requested_ids: list[str] = []
        self.iterated = False
        self.last_search_timings = {"embedding": 0.0, "chroma": 0.0}

    def get_documents(self, ids: list[str]) -> list[dict]:
        self.requested_ids.extend(ids)
        return [dict(self.rows[value]) for value in ids if value in self.rows]

    def corpus_revision(self) -> int:
        return self.revision

    def search(self, **kwargs) -> list[dict]:
        limit = int(kwargs.get("n_results", 5))
        return [dict(row) for row in list(self.rows.values())[:limit]]

    def iter_documents(self, *args, **kwargs):
        self.iterated = True
        raise AssertionError("FTS5 query path must not materialize the Chroma corpus")


def _lexical(store: SQLiteStore, chroma: _HydratingChroma) -> LexicalIndex:
    facade = SimpleNamespace(
        sqlite=store,
        chroma=chroma,
        retrieval_revision=store.get_store_revision,
    )
    return LexicalIndex(facade, backend="fts5")


def _narrative(body: str) -> NarrativeRecord:
    return NarrativeRecord(
        corpus_item_id="news-1",
        source_name="gdelt",
        source_category="global_news",
        provider_record_id="story-1",
        original_publisher=None,
        item_type="news",
        title="Acme launches a new product",
        body=body,
        summary=body,
        language="en",
        published_at="2026-07-01T12:00:00Z",
        observed_at="2026-07-01T12:01:00Z",
        accessed_at="2026-07-01T12:02:00Z",
        ingested_at="2026-07-01T12:03:00Z",
        source_url="https://example.test/story-1",
        canonical_url="https://example.test/story-1",
        content_hash=content_hash(body),
        license_label="test",
        normalization_version=NORMALIZATION_VERSION,
        document_family="company_news",
        tickers=("ACME",),
    )


class _MutationChroma:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.revision = 0

    def prepare_document(self, document_id: str, text: str, **kwargs) -> list[dict]:
        metadata = dict(kwargs.get("metadata") or {})
        metadata.setdefault("document_family_id", document_id)
        return [{"id": f"{document_id}#0", "document": text, "metadata": metadata}]

    def add_document(self, document_id: str, text: str, **kwargs) -> None:
        for row in self.prepare_document(document_id, text, **kwargs):
            self.rows[row["id"]] = row

    def mark_corpus_revision(self, revision: int) -> None:
        self.revision = revision

    def corpus_revision(self) -> int:
        return self.revision


def test_narrative_ledger_and_lexical_delta_commit_at_one_revision(tmp_path: Path) -> None:
    chroma = _MutationChroma()
    with patch("src.storage.store.ChromaStore", return_value=chroma):
        facade = Store(db_path=tmp_path / "corpus.db", chroma_path=tmp_path / "chroma")
    if not facade.sqlite.fts5_available():
        pytest.skip("test runtime has no SQLite FTS5")

    first = facade.upsert_narrative(_narrative("legacy product"))
    second_record = replace(
        _narrative("MI300X accelerator"),
        ingested_at="2026-07-01T12:04:00Z",
        accessed_at="2026-07-01T12:04:00Z",
    )
    second = facade.upsert_narrative(second_record)

    assert second["revision"] == first["revision"] + 1
    assert facade.sqlite.get_lexical_index_state()["indexed_revision"] == second["revision"]
    assert facade.sqlite.search_lexical(
        compile_fts_query("MI300X"), limit=5, revision=second["revision"]
    )[0]["chunk_id"] == "news-1#0"
    assert facade.sqlite.search_lexical(
        compile_fts_query("legacy"), limit=5, revision=second["revision"]
    ) == []
    assert chroma.corpus_revision() == second["revision"]


def test_migration_creates_content_bearing_fts_and_state(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "lexical.db")
    if not store.fts5_available():
        pytest.skip("test runtime has no SQLite FTS5")

    with store._connect() as conn:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='corpus_fts'"
        ).fetchone()[0]
        columns = {row[1] for row in conn.execute("PRAGMA table_info(corpus_fts)")}
        state_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(lexical_index_state)")
        }

    assert "content=''" not in sql.lower()
    assert {"title", "body", "ticker", "source_category", "item_type", "chunk_id"} <= columns
    assert {"schema_version", "indexed_revision", "indexed_at", "row_count"} <= state_columns


def test_incremental_family_delta_is_revision_based_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "lexical.db"
    store = SQLiteStore(path)
    if not store.fts5_available():
        pytest.skip("test runtime has no SQLite FTS5")

    first = [_chunk("family-1#0", "legacy margin pressure")]
    revision_1 = store.replace_lexical_family("family-1", first)
    state_1 = store.get_lexical_index_state()

    replacement = [_chunk("family-1#0", "MI300X revenue acceleration")]
    revision_2 = store.replace_lexical_family("family-1", replacement)
    reopened = SQLiteStore(path)
    state_2 = reopened.get_lexical_index_state()

    assert revision_2 == revision_1 + 1
    assert state_1["row_count"] == state_2["row_count"] == 1
    assert state_2["indexed_revision"] == revision_2
    assert reopened.search_lexical("MI300X", limit=5, revision=revision_2)[0]["chunk_id"] == "family-1#0"
    assert reopened.search_lexical("legacy", limit=5, revision=revision_2) == []

    revision_3 = reopened.delete_lexical_families(["family-1"])
    assert revision_3 == revision_2 + 1
    assert reopened.get_lexical_index_state()["row_count"] == 0


def test_fts_query_compiler_never_passes_raw_syntax() -> None:
    compiled = compile_fts_query('MI300X OR "unterminated" NEAR(10-Q)')

    assert compiled
    assert "NEAR(" not in compiled
    assert '"mi300x"' in compiled
    assert len(compiled) <= 2048
    assert compile_fts_query("*** :::") is None


def test_filters_apply_before_candidate_limit_and_hydration_is_bounded(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "lexical.db")
    if not store.fts5_available():
        pytest.skip("test runtime has no SQLite FTS5")
    rows = [
        _chunk("a#0", "revenue revenue revenue", ticker="OTHER"),
        _chunk("b#0", "revenue", ticker="ACME"),
    ]
    revision = store.replace_lexical_families({"a": [rows[0]], "b": [rows[1]]})
    chroma = _HydratingChroma(rows, revision)

    hits = _lexical(store, chroma).search(
        "revenue", k=1, where={"ticker": "ACME"}, filters={"item_type": "sec_filing"},
        corpus_revision=revision,
    )

    assert [row["id"] for row in hits] == ["b#0"]
    assert chroma.requested_ids == ["b#0"]
    assert chroma.iterated is False


def test_revision_mismatch_and_corrupt_index_fail_soft_without_hydration(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "lexical.db")
    if not store.fts5_available():
        pytest.skip("test runtime has no SQLite FTS5")
    row = _chunk("family#0", "cash flow")
    revision = store.replace_lexical_family("family", [row])
    chroma = _HydratingChroma([row], revision - 1)
    lexical = _lexical(store, chroma)

    assert lexical.search("cash flow", k=5, corpus_revision=revision) == []
    assert lexical.last_status["reason"] == "revision_mismatch"
    assert chroma.requested_ids == []

    chroma.revision = revision
    with store._connect() as conn:
        conn.execute("DROP TABLE corpus_fts")
        conn.commit()
    assert lexical.search("cash flow", k=5, corpus_revision=revision) == []
    assert lexical.last_status["mode"] == "degraded"
    assert lexical.last_status["reason"] == "fts_query_error"


def test_unavailable_fts5_reports_degraded_without_memory_build(tmp_path: Path, monkeypatch) -> None:
    store = SQLiteStore(tmp_path / "lexical.db")
    chroma = _HydratingChroma([], 0)
    monkeypatch.setattr(store, "fts5_available", lambda: False)
    lexical = _lexical(store, chroma)

    lexical.warm()

    assert lexical.mode == "degraded"
    assert lexical.search("anything", k=5, corpus_revision=0) == []
    assert chroma.iterated is False


def test_retriever_uses_fts_backend_captures_revision_and_traces_degradation(tmp_path: Path) -> None:
    sqlite_store = SQLiteStore(tmp_path / "lexical.db")
    if not sqlite_store.fts5_available():
        pytest.skip("test runtime has no SQLite FTS5")
    row = _chunk("family#0", "MI300X revenue")
    revision = sqlite_store.replace_lexical_family("family", [row])
    chroma = _HydratingChroma([row], revision)
    facade = SimpleNamespace(
        sqlite=sqlite_store,
        chroma=chroma,
        retrieval_revision=sqlite_store.get_store_revision,
    )
    config = MiddlewareConfig(config_path=Path(__file__).parent / "missing.yaml")
    config.enable_lexical = True
    config.lexical_backend = "fts5"
    config.enable_reranker = False
    retriever = Retriever(facade, config=config)

    retriever.warm_lexical_index()
    result = retriever.retrieve_candidates(
        "MI300X revenue",
        {"ticker": "ACME", "question_type": "news", "metrics": []},
        top_k_documents=2,
    )

    assert result["lexical_ids"] == ["family#0"]
    assert result["retrieval_trace"]["corpus_revision"] == revision
    assert result["retrieval_trace"]["lexical"]["mode"] == "fts5"
    assert chroma.iterated is False

    chroma.revision = revision - 1
    degraded = retriever.retrieve_candidates(
        "MI300X revenue",
        {"ticker": "ACME", "question_type": "news", "metrics": []},
        top_k_documents=2,
    )
    assert degraded["documents"]
    assert degraded["lexical_ids"] == []
    assert degraded["retrieval_trace"]["lexical"]["reason"] == "revision_mismatch"


def test_lexical_backend_defaults_to_fts5_and_has_profile_and_env_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    assert MiddlewareConfig(config_path=root / "missing.yaml").lexical_backend == "fts5"
    for name in ("legacy", "recommended", "evaluation"):
        config = MiddlewareConfig(config_path=root / "configs" / "profiles" / f"{name}.yaml")
        assert config.lexical_backend in {"fts5", "memory"}
    monkeypatch.setenv("LEXICAL_BACKEND", "memory")
    assert MiddlewareConfig(config_path=root / "missing.yaml").lexical_backend == "memory"

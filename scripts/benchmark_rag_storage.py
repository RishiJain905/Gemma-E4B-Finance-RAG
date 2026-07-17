"""scripts/benchmark_rag_storage.py
Offline storage benchmark for Phase 2.3.7.6.

The benchmark builds an isolated SQLite + Chroma corpus through the Store
facade, injects a deterministic Chroma embedding function, exercises the
persistent FTS5 and RRF retrieval paths, and evaluates the documented
keep/replace gates. It never reads or writes the repository's ``data/`` tree.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import MethodType
from typing import Any, Callable, ClassVar, Mapping, Sequence

from chromadb import Documents, EmbeddingFunction, Embeddings


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
QUERY_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "benchmarks"
    / "phase2_3_storage_queries.json"
)

DEFAULT_CHUNKS = 100_000
DEFAULT_SEED = 2_376
DEFAULT_REPEATS = 3
DEFAULT_BATCH_SIZE = 500
DEFAULT_REFRESH_WINDOW_SECONDS = 86_400.0
EMBEDDING_DIMENSION = 48
SCHEMA_VERSION = "phase2.3.7.6"

REQUIRED_WORKLOADS = (
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
    "inventory_family_search",
    "incremental_mutations",
    "restart_first_query",
    "reconciliation",
    "concurrency",
)

GATE_THRESHOLDS = {
    "lexical_p95_ms": 100.0,
    "dense_p95_ms": 200.0,
    "hybrid_p95_ms": 250.0,
    "filtered_hybrid_p95_ms": 300.0,
    "inventory_p95_ms": 50.0,
    "restart_readiness_ms": 10_000.0,
    "identity_drift": 0.0,
    "quality_ndcg_gap": 0.02,
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class SyntheticChunk:
    """One deterministic narrative chunk and its stable storage metadata."""

    item_id: str
    chunk_id: str
    family_id: str
    text: str
    metadata: dict[str, Any]
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class SyntheticCorpus:
    """Generated corpus plus digest and the precomputed embedding lookup."""

    chunks: tuple[SyntheticChunk, ...]
    digest: str
    seed: int
    embedding_dimension: int
    embeddings_by_text: dict[str, tuple[float, ...]]

    @property
    def families(self) -> dict[str, tuple[SyntheticChunk, ...]]:
        families: dict[str, list[SyntheticChunk]] = {}
        for chunk in self.chunks:
            families.setdefault(chunk.family_id, []).append(chunk)
        return {key: tuple(value) for key, value in families.items()}


def _stable_embedding(text: str, *, seed: int, dimension: int) -> tuple[float, ...]:
    """Create a normalized, seeded bag-of-token vector without model calls."""
    values = [0.0] * dimension
    for token in _TOKEN_RE.findall(text.lower()):
        digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
        for slot in range(4):
            offset = int.from_bytes(digest[slot * 4 : slot * 4 + 4], "big") % dimension
            sign = 1.0 if digest[16 + slot] & 1 else -1.0
            values[offset] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        values[0] = 1.0
        norm = 1.0
    return tuple(round(value / norm, 8) for value in values)


def _topic_for_index(index: int) -> str:
    """Distribute a small set of labeled anchor documents through the corpus."""
    if index % 19 == 0:
        return "exact-symbol"
    if index % 23 == 1:
        return "phrase"
    if index % 29 == 2:
        return "finance-terminology"
    if index % 31 == 3:
        return "filtered"
    return "general"


def _metadata_for_chunk(
    *,
    item_id: str,
    chunk_id: str,
    ticker: str,
    source_category: str,
    source: str,
    item_type: str,
    published_at: str,
    topic: str,
    title: str,
) -> dict[str, Any]:
    """Build only primitive Chroma metadata values in the ledger shape."""
    return {
        "corpus_item_id": item_id,
        "document_family_id": chunk_id,
        "parent_id": chunk_id,
        "chunk_index": 0,
        "chunk_count": 1,
        "child_chunk_id": chunk_id,
        "chunk_ordinal": 0,
        "ticker": ticker,
        "tickers": ticker,
        "source": source,
        "source_name": source,
        "source_category": source_category,
        "item_type": item_type,
        "published_at": published_at,
        "published_at_ordinal": date.fromisoformat(published_at).toordinal(),
        "effective_at": published_at,
        "as_of_at": published_at,
        "date": published_at,
        "title": title,
        "benchmark_topic": topic,
        "indexing_status": "indexed",
        "authority_tier": "direct_sec" if source == "sec" else "provider",
        "evidence_authority": "direct_sec" if source == "sec" else "provider",
    }


def generate_corpus(
    chunks: int = DEFAULT_CHUNKS,
    seed: int = DEFAULT_SEED,
    embedding_dimension: int = EMBEDDING_DIMENSION,
) -> SyntheticCorpus:
    """Generate a deterministic finance-flavored corpus of ``chunks`` rows."""
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 1:
        raise ValueError("chunks must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(embedding_dimension, int) or embedding_dimension < 8:
        raise ValueError("embedding_dimension must be at least 8")

    rng = random.Random(seed)
    tickers = (
        "NVDA",
        "AMD",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "JPM",
        "XOM",
        "AVGO",
        "ORCL",
        "CRM",
        "NFLX",
        "COST",
        "ADBE",
    )
    source_rows = (
        ("regulatory_filing", "sec"),
        ("market_news", "gdelt"),
        ("company_ir", "ir"),
        ("earnings_transcript", "transcript"),
        ("macro_release", "fred"),
        ("analyst_research", "finnhub"),
    )
    item_types = ("10-K", "10-Q", "news", "earnings", "transcript", "guidance", "risk")
    date_start = date(2023, 1, 1)
    digest = hashlib.sha256()
    rows: list[SyntheticChunk] = []
    embeddings_by_text: dict[str, tuple[float, ...]] = {}

    for index in range(chunks):
        item_id = f"bench/item-{index:08d}"
        chunk_id = item_id if index % 7 else f"{item_id}#0"
        topic = _topic_for_index(index)
        ticker = tickers[rng.randrange(len(tickers))]
        source_category, source = source_rows[rng.randrange(len(source_rows))]
        item_type = item_types[rng.randrange(len(item_types))]
        published = date_start + timedelta(days=rng.randrange(1_095))

        if topic == "exact-symbol":
            ticker = "NVDA"
        elif topic == "filtered":
            ticker = "AMD"
            source_category = "regulatory_filing"
            source = "sec"
            item_type = "earnings"
            published = date(2025, 6, 15)

        published_at = published.isoformat()
        title = f"{ticker} {item_type} update {published_at}"
        text = (
            f"{title}. The filing reports revenue growth, operating margin, cash flow, "
            f"capital allocation, and forward guidance for {ticker}. "
            f"Source category {source_category} was observed on {published_at}."
        )
        if topic == "exact-symbol":
            text += " NVDA benchmark revenue guidance is the labeled exact-symbol anchor."
        elif topic == "phrase":
            text += " Free cash flow conversion and operating leverage are the labeled phrase anchor."
        elif topic == "finance-terminology":
            text += " Net interest margin, liquidity coverage ratio, and credit quality are the labeled finance terminology anchor."
        elif topic == "filtered":
            text += " AMD benchmark guidance disclosure is the labeled filtered anchor."

        metadata = _metadata_for_chunk(
            item_id=item_id,
            chunk_id=chunk_id,
            ticker=ticker,
            source_category=source_category,
            source=source,
            item_type=item_type,
            published_at=published_at,
            topic=topic,
            title=title,
        )
        embedding = _stable_embedding(
            text,
            seed=seed,
            dimension=embedding_dimension,
        )
        embeddings_by_text[text] = embedding
        rows.append(
            SyntheticChunk(
                item_id=item_id,
                chunk_id=chunk_id,
                family_id=chunk_id,
                text=text,
                metadata=metadata,
                embedding=embedding,
            )
        )
        digest.update(
            json.dumps(
                {
                    "id": chunk_id,
                    "item_id": item_id,
                    "text": text,
                    "metadata": metadata,
                    "embedding": embedding,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")

    return SyntheticCorpus(
        chunks=tuple(rows),
        digest=digest.hexdigest(),
        seed=seed,
        embedding_dimension=embedding_dimension,
        embeddings_by_text=embeddings_by_text,
    )


class DeterministicEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedder backed only by seeded local vectors."""

    _seed: ClassVar[int] = DEFAULT_SEED
    _dimension: ClassVar[int] = EMBEDDING_DIMENSION
    _precomputed: ClassVar[Mapping[str, Sequence[float]]] = {}

    @classmethod
    def configure(
        cls,
        *,
        seed: int,
        dimension: int,
        precomputed: Mapping[str, Sequence[float]],
    ) -> None:
        """Set the corpus vector registry used by the next Store open."""
        cls._seed = int(seed)
        cls._dimension = int(dimension)
        cls._precomputed = precomputed

    def __init__(
        self,
        endpoint: str = "",
        model: str = "deterministic",
        batch_size: int = 10,
        timeout: int = 60,
        embedding_cache_size: int = 256,
    ) -> None:
        del endpoint, model, batch_size, timeout, embedding_cache_size
        self.dimension = self._dimension
        self.seed = self._seed
        self.precomputed = self._precomputed
        self.last_call_timing_ms = 0.0
        self.network_calls = 0
        self.model_calls = 0
        self.embedding_calls = 0

    def __call__(self, input: Documents) -> Embeddings:
        """Return precomputed vectors or the same deterministic local formula."""
        start = time.perf_counter()
        self.embedding_calls += 1
        output: list[list[float]] = []
        for text in input:
            vector = self.precomputed.get(text)
            if vector is None:
                vector = _stable_embedding(
                    text,
                    seed=self.seed,
                    dimension=self.dimension,
                )
            output.append([float(value) for value in vector])
        self.last_call_timing_ms = round((time.perf_counter() - start) * 1000, 3)
        return output


def load_query_fixture(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the size-independent labeled query set."""
    fixture_path = path or QUERY_FIXTURE
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("queries"), list) or not payload["queries"]:
        raise ValueError("storage benchmark query fixture must contain queries")
    required = {"id", "category", "query", "k", "filters", "relevance"}
    for query in payload["queries"]:
        if not required <= set(query):
            raise ValueError(f"invalid storage benchmark query: {query!r}")
    return payload


def _validate_isolated_path(path: Path) -> Path:
    """Reject repository data paths before creating any benchmark state."""
    resolved = path.expanduser().resolve()
    data_root = (REPO_ROOT / "data").resolve()
    if resolved == data_root or data_root in resolved.parents:
        raise ValueError(f"benchmark path may not be inside {data_root}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _open_store(
    workdir: Path,
    corpus: SyntheticCorpus,
    *,
    collection_name: str = "phase2_3_7_6_benchmark",
):
    """Open Store with the benchmark-only deterministic Chroma embedder."""
    from src.storage import chroma_store as chroma_module
    from src.storage.store import Store

    DeterministicEmbeddingFunction.configure(
        seed=corpus.seed,
        dimension=corpus.embedding_dimension,
        precomputed=corpus.embeddings_by_text,
    )
    original = chroma_module.TraceAlchemyEmbeddingFunction
    chroma_module.TraceAlchemyEmbeddingFunction = DeterministicEmbeddingFunction
    try:
        store = Store(
            db_path=workdir / "finance.db",
            chroma_path=workdir / "chroma",
            collection_name=collection_name,
            embedding_endpoint="http://benchmark.invalid/v1/embeddings",
        )
        # Chroma 1.5 rejects re-submitting the immutable hnsw distance metadata
        # during collection.modify(). The repository adapter predates that
        # validation; keep this compatibility seam local to the benchmark.
        def mark_revision(chroma_store: Any, revision: int) -> None:
            chroma_store.collection.modify(metadata={"corpus_revision": int(revision)})

        store.chroma.mark_corpus_revision = MethodType(mark_revision, store.chroma)
        return store
    finally:
        chroma_module.TraceAlchemyEmbeddingFunction = original


def _build_corpus(store: Any, corpus: SyntheticCorpus, batch_size: int) -> dict[str, Any]:
    """Batch-load deterministic rows through Store and measure build throughput."""
    started = time.perf_counter()
    for offset in range(0, len(corpus.chunks), batch_size):
        batch = corpus.chunks[offset : offset + batch_size]
        store.save_documents_batch(
            [row.chunk_id for row in batch],
            [row.text for row in batch],
            [row.metadata for row in batch],
        )
    # Merge FTS5 segments after the bulk load, matching what the production
    # rebuild path (scripts/rebuild_lexical_index.py) does so ranked lexical
    # scans over common terms are measured against a compacted index.
    optimize = getattr(store, "optimize_lexical_index", None)
    if callable(optimize):
        optimize()
    elapsed_s = max(time.perf_counter() - started, 1e-9)
    return {
        "chunks": len(corpus.chunks),
        "elapsed_s": round(elapsed_s, 6),
        "throughput_chunks_per_s": round(len(corpus.chunks) / elapsed_s, 3),
        "batch_size": batch_size,
        "embedding_dimension": corpus.embedding_dimension,
    }


def _validate_collection_embedding_dimension(store: Any, corpus: SyntheticCorpus) -> int:
    """Confirm that Chroma persisted the configured vector dimension."""
    result = store.chroma.collection.get(
        ids=[corpus.chunks[0].chunk_id],
        include=["embeddings"],
    )
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        raise RuntimeError("Chroma returned no persisted benchmark embedding")
    dimension = len(embeddings[0])
    if dimension != corpus.embedding_dimension:
        raise RuntimeError(
            f"benchmark embedding dimension mismatch: expected {corpus.embedding_dimension}, "
            f"got {dimension}"
        )
    return dimension


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50_ms": round(_percentile(values, 0.50), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "worst_ms": round(max(values) if values else 0.0, 3),
        "mean_ms": round(statistics.fmean(values) if values else 0.0, 3),
    }


def _measure_workload(
    name: str,
    operation: Callable[[], Any],
    repeats: int,
) -> dict[str, Any]:
    """Run every latency operation repeatedly and retain every run."""
    runs: list[dict[str, Any]] = []
    for index in range(max(1, repeats)):
        started = time.perf_counter()
        operation()
        elapsed_ms = (time.perf_counter() - started) * 1000
        runs.append(
            {
                "run": index + 1,
                "temperature": "cold" if index == 0 else "warm",
                "latency_ms": round(elapsed_ms, 3),
            }
        )
    return {"name": name, "runs": runs, "summary": _latency_summary([r["latency_ms"] for r in runs])}


def _new_retriever(store: Any):
    from src.middleware.config import MiddlewareConfig
    from src.middleware.retriever import Retriever

    config = MiddlewareConfig(config_path=REPO_ROOT / "configs" / "missing-benchmark.yaml")
    config.enable_lexical = True
    config.lexical_backend = "fts5"
    config.enable_reranker = False
    config.rerank_candidates = 30
    config.rrf_k = 60
    return Retriever(store, config=config)


def _query_intent(query: Mapping[str, Any]) -> dict[str, Any]:
    filters = dict(query.get("filters") or {})
    return {
        "ticker": filters.get("ticker"),
        "question_type": "news" if query.get("category") == "filtered" else "general",
        "metrics": [],
        "evidence_filters": filters,
    }


def _run_channel(
    store: Any,
    retriever: Any,
    query: Mapping[str, Any],
    channel: str,
) -> list[dict[str, Any]]:
    text = str(query["query"])
    k = int(query.get("k", 10))
    filters = dict(query.get("filters") or {})
    ticker = filters.get("ticker")
    if channel == "dense":
        # Chroma 1.5 requires one operator per range expression. The Store
        # facade's committed helper combines both bounds for older runtimes;
        # use the equivalent backend filter only for this compatibility edge.
        if filters.get("published_from") or filters.get("published_to"):
            return list(
                store.chroma.search(
                    query=text,
                    n_results=k,
                    filter_dict=_benchmark_chroma_filter(filters, ticker),
                )
            )
        return list(
            store.search(
                query=text,
                n_results=k,
                ticker=ticker,
                filters=filters,
            ).get("documents", [])
        )
    if channel == "lexical":
        where = {"ticker": ticker} if ticker else None
        return list(
            retriever.lexical.search(
                query=text,
                k=k,
                where=where,
                filters=filters,
                corpus_revision=store.corpus_revision(),
            )
        )
    if channel == "hybrid":
        result = retriever.retrieve_candidates(
            text,
            _query_intent(query),
            top_k_documents=k,
            top_k_facts=0,
        )
        return list(result.get("documents", []))[:k]
    raise ValueError(f"unknown benchmark channel: {channel}")


def _benchmark_chroma_filter(
    filters: Mapping[str, Any], ticker: str | None,
) -> dict[str, Any] | None:
    """Translate benchmark facets to Chroma 1.5-compatible predicates."""
    clauses: list[dict[str, Any]] = []
    if ticker:
        clauses.append({"ticker": ticker})
    exact_fields = {
        "source_category": "source_category",
        "source": "source_name",
        "item_type": "item_type",
    }
    for key, metadata_key in exact_fields.items():
        value = filters.get(key)
        if value not in (None, ""):
            clauses.append({metadata_key: value})
    if filters.get("published_from"):
        clauses.append(
            {
                "published_at_ordinal": {
                    "$gte": date.fromisoformat(str(filters["published_from"])).toordinal()
                }
            }
        )
    if filters.get("published_to"):
        clauses.append(
            {
                "published_at_ordinal": {
                    "$lte": date.fromisoformat(str(filters["published_to"])).toordinal()
                }
            }
        )
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _metadata_matches(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(str(metadata.get(key)) == str(value) for key, value in expected.items())


def derive_relevance_labels(
    corpus: SyntheticCorpus,
    query: Mapping[str, Any],
) -> dict[str, int]:
    """Resolve fixture selectors against generated metadata, independent of N."""
    labels: dict[str, int] = {}
    for row in corpus.chunks:
        for rule in query.get("relevance", []):
            if _metadata_matches(row.metadata, rule.get("metadata") or {}):
                labels[row.chunk_id] = max(labels.get(row.chunk_id, 0), int(rule.get("grade", 1)))
    return labels


def _rank_quality(ids: Sequence[str], labels: Mapping[str, int], k: int) -> dict[str, float]:
    top = list(ids[:k])
    relevant = {key for key, grade in labels.items() if grade > 0}
    denominator = min(k, len(relevant))
    recall = len(set(top) & relevant) / denominator if denominator else 1.0

    def gain(grade: int) -> float:
        return float((2**grade) - 1)

    dcg = sum(
        gain(labels.get(doc_id, 0)) / math.log2(index + 2)
        for index, doc_id in enumerate(top[:10])
    )
    ideal_grades = sorted(labels.values(), reverse=True)[:10]
    ideal = sum(gain(grade) / math.log2(index + 2) for index, grade in enumerate(ideal_grades))
    return {
        "recall_at_k": round(recall, 6),
        "ndcg_at_10": round(dcg / ideal if ideal else 1.0, 6),
        "relevant_count": len(relevant),
        "returned_count": len(top),
    }


def _quality_metrics(
    store: Any,
    retriever: Any,
    corpus: SyntheticCorpus,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    per_query: dict[str, Any] = {}
    channel_totals: dict[str, list[dict[str, float]]] = {name: [] for name in ("dense", "lexical", "hybrid")}
    for query in fixture["queries"]:
        labels = derive_relevance_labels(corpus, query)
        query_results: dict[str, Any] = {
            "category": query["category"],
            "relevant_count": len(labels),
            "channels": {},
        }
        for channel in channel_totals:
            hits = _run_channel(store, retriever, query, channel)
            quality = _rank_quality(
                [str(hit.get("id")) for hit in hits if hit.get("id")],
                labels,
                int(query.get("k", 10)),
            )
            query_results["channels"][channel] = quality
            channel_totals[channel].append(quality)
        per_query[str(query["id"])] = query_results

    aggregate: dict[str, Any] = {
        "baseline": "labeled-oracle",
        "baseline_ndcg_at_10": 1.0,
        "baseline_recall_at_k": 1.0,
    }
    for channel, rows in channel_totals.items():
        aggregate[f"{channel}_recall_at_k"] = round(
            statistics.fmean(row["recall_at_k"] for row in rows), 6
        )
        aggregate[f"{channel}_ndcg_at_10"] = round(
            statistics.fmean(row["ndcg_at_10"] for row in rows), 6
        )
    aggregate["quality_ndcg_gap"] = round(
        abs(aggregate["baseline_ndcg_at_10"] - aggregate["hybrid_ndcg_at_10"]),
        6,
    )
    return {"queries": per_query, "aggregate": aggregate}


def _replace_family(store: Any, family_id: str, rows: Sequence[Mapping[str, Any]]) -> int:
    """Apply one benchmark family delta using the existing Store backends."""
    normalized = [
        {
            "id": str(row["id"]),
            "document": str(row.get("document") or row.get("text") or ""),
            "metadata": dict(row.get("metadata") or {}),
        }
        for row in rows
    ]
    revision = store.sqlite.replace_lexical_family(family_id, normalized)
    store.chroma.collection.upsert(
        ids=[row["id"] for row in normalized],
        documents=[row["document"] for row in normalized],
        metadatas=[row["metadata"] for row in normalized],
    )
    existing = store.chroma.collection.get(
        where={"document_family_id": family_id},
        include=["metadatas"],
    )
    expected_ids = {row["id"] for row in normalized}
    stale_ids = [
        str(document_id)
        for document_id, metadata in zip(
            existing.get("ids") or [], existing.get("metadatas") or []
        )
        if str(document_id) not in expected_ids
        and str((metadata or {}).get("document_family_id")) == family_id
    ]
    if stale_ids:
        store.chroma.collection.delete(ids=stale_ids)
    store.chroma.mark_corpus_revision(revision)
    return int(revision)


def _delete_family(store: Any, family_id: str, ids: Sequence[str]) -> None:
    revision = store.sqlite.delete_lexical_families([family_id])
    if ids:
        store.chroma.collection.delete(ids=list(ids))
    else:
        store.chroma.delete_document_family(family_id)
    store.chroma.mark_corpus_revision(revision)


def _mutation_metrics(store: Any, corpus: SyntheticCorpus, seed: int) -> dict[str, Any]:
    """Measure insert/update/delete and a same-count family replacement."""
    sample = corpus.chunks[-1]
    operations: dict[str, dict[str, Any]] = {}
    temporary_id = f"bench/mutation/insert-{seed}"
    temporary_metadata = dict(sample.metadata)
    temporary_metadata.update(
        {
            "corpus_item_id": temporary_id,
            "document_family_id": temporary_id,
            "parent_id": temporary_id,
            "benchmark_topic": "mutation",
        }
    )

    started = time.perf_counter()
    store.save_documents_batch(
        [temporary_id],
        ["Temporary incremental insert for the storage benchmark."],
        [temporary_metadata],
    )
    operations["insert"] = {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}

    updated_text = sample.text + " Incremental update marker for the storage benchmark."
    started = time.perf_counter()
    _replace_family(
        store,
        sample.family_id,
        [{"id": sample.chunk_id, "document": updated_text, "metadata": sample.metadata}],
    )
    operations["update"] = {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
    _replace_family(
        store,
        sample.family_id,
        [{"id": sample.chunk_id, "document": sample.text, "metadata": sample.metadata}],
    )

    started = time.perf_counter()
    _delete_family(store, temporary_id, [temporary_id])
    operations["delete"] = {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}

    family_id = f"bench/mutation/family-{seed}"
    family_rows = []
    for index in range(2):
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "corpus_item_id": family_id,
                "document_family_id": family_id,
                "parent_id": family_id,
                "chunk_index": index,
                "chunk_count": 2,
                "child_chunk_id": f"{family_id}#{index}",
                "benchmark_topic": "mutation-family",
            }
        )
        family_rows.append(
            {
                "id": f"{family_id}#{index}",
                "document": f"Family replacement before state {index}.",
                "metadata": metadata,
            }
        )
    started = time.perf_counter()
    _replace_family(store, family_id, family_rows)
    first_replacement_ms = (time.perf_counter() - started) * 1000
    replacement_rows = [
        {
            **row,
            "document": f"Family replacement after state {index}.",
        }
        for index, row in enumerate(family_rows)
    ]
    started = time.perf_counter()
    _replace_family(store, family_id, replacement_rows)
    second_replacement_ms = (time.perf_counter() - started) * 1000
    operations["same_count_family_replacement"] = {
        "elapsed_ms": round(first_replacement_ms + second_replacement_ms, 3),
        "before_count": len(family_rows),
        "after_count": len(replacement_rows),
        "stable_ids": [row["id"] for row in replacement_rows],
    }
    _delete_family(store, family_id, [row["id"] for row in replacement_rows])

    elapsed_s = sum(
        float(value["elapsed_ms"])
        for value in operations.values()
    ) / 1000.0
    return {
        "operations": operations,
        "elapsed_s": round(elapsed_s, 6),
        "throughput_operations_per_s": round(len(operations) / max(elapsed_s, 1e-9), 3),
    }


def _restart_metrics(
    workdir: Path,
    corpus: SyntheticCorpus,
    query: Mapping[str, Any],
    repeats: int,
    collection_name: str,
) -> tuple[dict[str, Any], Any, Any]:
    runs: list[dict[str, Any]] = []
    current_store = None
    current_retriever = None
    for index in range(max(1, repeats)):
        opened = time.perf_counter()
        current_store = _open_store(workdir, corpus, collection_name=collection_name)
        open_ms = (time.perf_counter() - opened) * 1000
        ready_started = time.perf_counter()
        heartbeat = current_store.heartbeat()
        current_retriever = _new_retriever(current_store)
        current_retriever.warm_lexical_index()
        readiness_ms = (time.perf_counter() - ready_started) * 1000
        first_started = time.perf_counter()
        _run_channel(current_store, current_retriever, query, "hybrid")
        first_query_ms = (time.perf_counter() - first_started) * 1000
        total_ms = open_ms + readiness_ms + first_query_ms
        runs.append(
            {
                "run": index + 1,
                "temperature": "cold" if index == 0 else "warm",
                "latency_ms": round(total_ms, 3),
                "open_ms": round(open_ms, 3),
                "readiness_ms": round(readiness_ms, 3),
                "first_query_ms": round(first_query_ms, 3),
                "heartbeat": heartbeat,
            }
        )
        gc.collect()
    return (
        {
            "name": "restart_first_query",
            "runs": runs,
            "summary": _latency_summary([run["latency_ms"] for run in runs]),
        },
        current_store,
        current_retriever,
    )


def _reconciliation_metrics(store: Any, corpus: SyntheticCorpus) -> dict[str, Any]:
    """Leave FTS one-sided, measure drift, then repair it against Chroma."""
    sample = corpus.chunks[-1]
    interrupted_text = sample.text + " interrupted cross-store mutation marker"
    original_add = store.chroma.add_document

    def fail_after_lexical(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("simulated Chroma interruption")

    store.chroma.add_document = fail_after_lexical
    try:
        try:
            store.save_document(
                document_id=sample.chunk_id,
                text=interrupted_text,
                ticker=str(sample.metadata["ticker"]),
                source=str(sample.metadata["source"]),
                date=str(sample.metadata["published_at"]),
                metadata=sample.metadata,
            )
        except RuntimeError:
            pass
    finally:
        store.chroma.add_document = original_add

    drift_started = time.perf_counter()
    before = store.reconcile_lexical_index(repair=False, batch_size=200)
    drift_ms = (time.perf_counter() - drift_started) * 1000
    repair_started = time.perf_counter()
    repaired = store.reconcile_lexical_index(repair=True, batch_size=200)
    repair_ms = (time.perf_counter() - repair_started) * 1000
    verification_started = time.perf_counter()
    after = store.reconcile_lexical_index(repair=False, batch_size=200)
    verification_ms = (time.perf_counter() - verification_started) * 1000
    after_counts = after.get("counts") or {}
    unrepaired = sum(int(value or 0) for value in after_counts.values())
    before_counts = before.get("counts") or {}
    return {
        "interruption": "lexical_committed_before_chroma",
        "before": before,
        "repair": repaired,
        "after": after,
        "identity": {
            "missing": int(before_counts.get("missing", 0)),
            "duplicate": int(before_counts.get("duplicate", 0)),
            "stale": int(before_counts.get("stale", 0)),
            "orphan": int(before_counts.get("orphan", 0)),
            "unrepaired": unrepaired,
        },
        "drift_detection_ms": round(drift_ms, 3),
        "repair_time_ms": round(repair_ms, 3),
        "verification_time_ms": round(verification_ms, 3),
    }


def _concurrency_metrics(
    store: Any,
    corpus: SyntheticCorpus,
    fixture_query: Mapping[str, Any],
    repeats: int,
) -> dict[str, Any]:
    """Run bounded readers alongside one writer against the isolated stores."""
    reader_count = 3
    runs: list[dict[str, Any]] = []
    for round_index in range(max(1, repeats)):
        errors: list[str] = []
        temporary_ids = [f"bench/concurrency/{round_index}-{index}" for index in range(2)]

        def writer() -> int:
            for temporary_id in temporary_ids:
                sample = corpus.chunks[round_index % len(corpus.chunks)]
                metadata = dict(sample.metadata)
                metadata.update(
                    {
                        "corpus_item_id": temporary_id,
                        "document_family_id": temporary_id,
                        "parent_id": temporary_id,
                        "benchmark_topic": "concurrency",
                    }
                )
                store.save_documents_batch(
                    [temporary_id],
                    [sample.text + " concurrent writer update"],
                    [metadata],
                )
            return len(temporary_ids)

        def reader() -> int:
            reader_retriever = _new_retriever(store)
            count = 0
            for _ in range(2):
                try:
                    _run_channel(store, reader_retriever, fixture_query, "hybrid")
                    count += 1
                except Exception as exc:  # noqa: BLE001 - report all reader failures
                    errors.append(str(exc))
            return count

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=reader_count + 1) as executor:
            futures = [executor.submit(reader) for _ in range(reader_count)]
            writer_future = executor.submit(writer)
            reader_operations = sum(future.result() for future in futures)
            writer_operations = writer_future.result()
        elapsed_ms = (time.perf_counter() - started) * 1000
        for temporary_id in temporary_ids:
            _delete_family(store, temporary_id, [temporary_id])
        runs.append(
            {
                "run": round_index + 1,
                "temperature": "cold" if round_index == 0 else "warm",
                "latency_ms": round(elapsed_ms, 3),
                "readers": reader_count,
                "writers": 1,
                "reader_operations": reader_operations,
                "writer_operations": writer_operations,
                "errors": errors,
            }
        )
    return {
        "name": "concurrency",
        "readers": reader_count,
        "writers": 1,
        "runs": runs,
        "summary": _latency_summary([run["latency_ms"] for run in runs]),
        "errors": [error for run in runs for error in run["errors"]],
    }


class _MemorySampler:
    """Low-dependency process RSS sampler for peak and steady measurements."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak = 0

    def start(self) -> None:
        self.peak = _rss_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.peak = max(self.peak, _rss_bytes())

    def stop(self) -> dict[str, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        steady = _rss_bytes()
        self.peak = max(self.peak, steady)
        return {"peak_rss_bytes": int(self.peak), "steady_rss_bytes": int(steady)}


def _rss_bytes() -> int:
    if os.name == "nt":
        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(MemoryCounters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_process = kernel.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(MemoryCounters),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        if get_info(get_process(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return 0
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage * (1024 if platform.system() != "Darwin" else 1))
    except (ImportError, AttributeError):
        return 0


def _path_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _fts_bytes(db_path: Path) -> int:
    """Measure FTS pages without reading the benchmark corpus into Python."""
    try:
        with sqlite3.connect(db_path) as connection:
            try:
                row = connection.execute(
                    "SELECT COALESCE(SUM(pgsize), 0) FROM dbstat "
                    "WHERE name LIKE 'corpus_fts%'"
                ).fetchone()
                if row and int(row[0] or 0):
                    return int(row[0])
            except sqlite3.Error:
                pass
            row = connection.execute(
                "SELECT COALESCE(SUM(length(COALESCE(title, '')) + "
                "length(COALESCE(body, '')) + length(COALESCE(ticker, '')) + "
                "length(COALESCE(source_category, '')) + length(COALESCE(item_type, '')) + "
                "length(COALESCE(chunk_id, '')) + length(COALESCE(family_id, '')) + "
                "length(COALESCE(source, '')) + length(COALESCE(event_type, '')) + "
                "length(COALESCE(form, '')) + length(COALESCE(item, '')) + "
                "length(COALESCE(authority_tier, '')) + length(COALESCE(indexing_status, '')) + "
                "length(COALESCE(published_at, '')) + length(COALESCE(effective_at, '')) + "
                "length(COALESCE(as_of_at, ''))), 0) FROM corpus_fts"
            ).fetchone()
        return int(row[0] or 0)
    except sqlite3.Error:
        return 0


def _disk_metrics(workdir: Path) -> dict[str, int]:
    db_path = workdir / "finance.db"
    return {
        "sqlite_bytes": _path_bytes(db_path)
        + _path_bytes(workdir / "finance.db-wal")
        + _path_bytes(workdir / "finance.db-shm"),
        "fts_bytes": _fts_bytes(db_path),
        "chroma_bytes": _path_bytes(workdir / "chroma"),
    }


def _hardware() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def _versions() -> dict[str, str]:
    try:
        chromadb_version = importlib.metadata.version("chromadb")
    except importlib.metadata.PackageNotFoundError:
        chromadb_version = "unknown"
    return {
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "chromadb": chromadb_version,
    }


def _distribution(corpus: SyntheticCorpus) -> dict[str, dict[str, int]]:
    fields = ("ticker", "source_category", "source", "item_type", "benchmark_topic")
    result: dict[str, dict[str, int]] = {}
    for field in fields:
        result[field] = dict(sorted(Counter(str(row.metadata[field]) for row in corpus.chunks).items()))
    return result


def _metric_p95(workloads: Mapping[str, Any], name: str) -> float | None:
    workload = workloads.get(name)
    if not workload:
        return None
    summary = workload.get("summary") or {}
    value = summary.get("p95_ms")
    return None if value is None else float(value)


def evaluate_gates(
    result_or_metrics: Mapping[str, Any],
    *,
    chunks: int | None = None,
    refresh_window_seconds: float = DEFAULT_REFRESH_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Apply the Step 3 hard gates to measured values without inventing data."""
    metrics = result_or_metrics.get("metrics", result_or_metrics)
    effective_chunks = int(chunks if chunks is not None else result_or_metrics.get("corpus", {}).get("chunks", 0))
    applicable = effective_chunks >= DEFAULT_CHUNKS
    workloads = metrics.get("workloads") or {}
    quality = metrics.get("quality", {}).get("aggregate", {})
    identity = metrics.get("identity") or metrics.get("reconciliation", {}).get("identity") or {}
    throughput = metrics.get("throughput", {}).get("incremental", {})
    startup = metrics.get("startup") or {}
    values: dict[str, float | None] = {
        "lexical_p95_ms": _metric_p95(workloads, "lexical_top_k"),
        "dense_p95_ms": _metric_p95(workloads, "dense_top_k"),
        "hybrid_p95_ms": _metric_p95(workloads, "hybrid_top_k"),
        "filtered_hybrid_p95_ms": _metric_p95(workloads, "filtered_compound"),
        "inventory_p95_ms": max(
            value
            for value in (
                _metric_p95(workloads, "inventory_count"),
                _metric_p95(workloads, "inventory_source_counts"),
                _metric_p95(workloads, "inventory_ticker_counts"),
                _metric_p95(workloads, "inventory_family_search"),
            )
            if value is not None
        )
        if any(
            value is not None
            for value in (
                _metric_p95(workloads, "inventory_count"),
                _metric_p95(workloads, "inventory_source_counts"),
                _metric_p95(workloads, "inventory_ticker_counts"),
                _metric_p95(workloads, "inventory_family_search"),
            )
        )
        else None,
        "restart_readiness_ms": startup.get("readiness_p95_ms")
        if startup.get("readiness_p95_ms") is not None
        else _metric_p95(workloads, "restart_first_query"),
        "identity_drift": identity.get("unrepaired"),
        "quality_ndcg_gap": quality.get("quality_ndcg_gap")
        if quality.get("quality_ndcg_gap") is not None
        else (
            abs(
                float(quality["baseline_ndcg_at_10"])
                - float(quality["hybrid_ndcg_at_10"])
            )
            if quality.get("baseline_ndcg_at_10") is not None
            and quality.get("hybrid_ndcg_at_10") is not None
            else None
        ),
        "daily_incremental_elapsed_s": throughput.get("elapsed_s"),
    }
    thresholds = {
        **GATE_THRESHOLDS,
        "daily_incremental_elapsed_s": float(refresh_window_seconds),
    }
    evaluations: dict[str, dict[str, Any]] = {}
    for key, threshold in thresholds.items():
        value = values.get(key)
        if not applicable:
            status = "N/A"
        elif value is None:
            status = "FAIL"
        elif key in {"identity_drift", "quality_ndcg_gap"}:
            status = "PASS" if float(value) <= threshold else "FAIL"
        else:
            status = "PASS" if float(value) < threshold else "FAIL"
        evaluations[key] = {
            "status": status,
            "value": None if value is None else round(float(value), 6),
            "threshold": round(float(threshold), 6),
            "operator": "<=" if key in {"identity_drift", "quality_ndcg_gap"} else "<",
        }
    passed = applicable and all(item["status"] == "PASS" for item in evaluations.values())
    return {
        "hard_gate_applicable": applicable,
        "all_passed": passed,
        "decision": "keep" if passed else "replace_investigation" if applicable else "pending_100k",
        "thresholds": {key: int(value) if float(value).is_integer() else value for key, value in thresholds.items()},
        "evaluations": evaluations,
    }


def _print_report(result: Mapping[str, Any]) -> None:
    print(
        f"Phase 2.3.7.6 storage benchmark | chunks={result['corpus']['chunks']} "
        f"seed={result['corpus']['seed']} digest={result['corpus']['digest'][:16]}"
    )
    print("Workload                         p50 ms    p95 ms   worst ms   runs")
    print("-------------------------------  --------  --------  --------  ----")
    for name, workload in result["metrics"]["workloads"].items():
        summary = workload.get("summary") or {}
        print(
            f"{name:31}  {summary.get('p50_ms', 0):8.3f}  "
            f"{summary.get('p95_ms', 0):8.3f}  {summary.get('worst_ms', 0):8.3f}  "
            f"{len(workload.get('runs', [])):4d}"
        )
    print("\nHard gates")
    for name, evaluation in result["gates"]["evaluations"].items():
        value = evaluation["value"]
        value_text = "n/a" if value is None else str(value)
        print(
            f"{evaluation['status']:4} {name:32} value={value_text} "
            f"threshold={evaluation['threshold']}"
        )
    print(f"Decision: {result['gates']['decision']}")


def _run_benchmark(
    *,
    chunks: int = DEFAULT_CHUNKS,
    seed: int = DEFAULT_SEED,
    workdir: Path | None = None,
    repeats: int = DEFAULT_REPEATS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    refresh_window_seconds: float = DEFAULT_REFRESH_WINDOW_SECONDS,
    collection_name: str = "phase2_3_7_6_benchmark",
) -> dict[str, Any]:
    """Execute the full Phase 2.3.7.6 workload matrix in an isolated path."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    target_workdir = _validate_isolated_path(
        workdir if workdir is not None else Path(tempfile.mkdtemp(prefix="phase2_3_7_6-"))
    )
    corpus = generate_corpus(chunks=chunks, seed=seed)
    fixture = load_query_fixture()
    sampler = _MemorySampler()
    sampler.start()
    build_started = time.perf_counter()
    store = _open_store(target_workdir, corpus, collection_name=collection_name)
    build = _build_corpus(store, corpus, batch_size=batch_size)
    collection_dimension = _validate_collection_embedding_dimension(store, corpus)
    build["open_elapsed_s"] = round(time.perf_counter() - build_started - build["elapsed_s"], 6)
    retriever = _new_retriever(store)
    retriever.warm_lexical_index()

    query_by_category = {str(query["category"]): query for query in fixture["queries"]}
    exact_query = query_by_category["exact-symbol"]
    filtered_query = query_by_category["filtered"]
    phrase_query = query_by_category["phrase"]
    terminology_query = query_by_category["finance-terminology"]
    workloads: dict[str, Any] = {}
    workloads["dense_top_k"] = _measure_workload(
        "dense_top_k", lambda: _run_channel(store, retriever, exact_query, "dense"), repeats
    )
    workloads["lexical_top_k"] = _measure_workload(
        "lexical_top_k", lambda: _run_channel(store, retriever, exact_query, "lexical"), repeats
    )
    workloads["hybrid_top_k"] = _measure_workload(
        "hybrid_top_k", lambda: _run_channel(store, retriever, exact_query, "hybrid"), repeats
    )
    ticker_query = dict(exact_query)
    ticker_query["filters"] = {"ticker": "NVDA"}
    workloads["filtered_ticker"] = _measure_workload(
        "filtered_ticker", lambda: _run_channel(store, retriever, ticker_query, "hybrid"), repeats
    )
    source_query = dict(phrase_query)
    source_query["filters"] = {"source": "sec"}
    workloads["filtered_source"] = _measure_workload(
        "filtered_source", lambda: _run_channel(store, retriever, source_query, "hybrid"), repeats
    )
    type_query = dict(terminology_query)
    type_query["filters"] = {"item_type": "earnings"}
    workloads["filtered_item_type"] = _measure_workload(
        "filtered_item_type", lambda: _run_channel(store, retriever, type_query, "hybrid"), repeats
    )
    date_query = dict(phrase_query)
    date_query["filters"] = {"published_from": "2025-01-01", "published_to": "2025-12-31"}
    workloads["filtered_date"] = _measure_workload(
        "filtered_date", lambda: _run_channel(store, retriever, date_query, "hybrid"), repeats
    )
    workloads["filtered_compound"] = _measure_workload(
        "filtered_compound", lambda: _run_channel(store, retriever, filtered_query, "hybrid"), repeats
    )
    workloads["inventory_count"] = _measure_workload(
        "inventory_count", lambda: store.heartbeat(), repeats
    )
    workloads["inventory_source_counts"] = _measure_workload(
        "inventory_source_counts", lambda: store.get_source_counts(limit=100), repeats
    )
    workloads["inventory_ticker_counts"] = _measure_workload(
        "inventory_ticker_counts", lambda: store.get_ticker_counts(limit=100), repeats
    )
    workloads["inventory_family_search"] = _measure_workload(
        "inventory_family_search",
        lambda: store.search_document_families(limit=100),
        repeats,
    )

    quality = _quality_metrics(store, retriever, corpus, fixture)
    incremental = _mutation_metrics(store, corpus, seed)
    workloads["incremental_mutations"] = {
        "name": "incremental_mutations",
        "runs": [
            {
                "run": 1,
                "temperature": "warm",
                "latency_ms": round(incremental["elapsed_s"] * 1000, 3),
            }
        ],
        "summary": _latency_summary([incremental["elapsed_s"] * 1000]),
        **incremental,
    }

    restart_workload, store, retriever = _restart_metrics(
        target_workdir,
        corpus,
        exact_query,
        repeats,
        collection_name,
    )
    workloads["restart_first_query"] = restart_workload
    startup_runs = restart_workload["runs"]
    startup = {
        "open_p50_ms": round(_percentile([run["open_ms"] for run in startup_runs], 0.50), 3),
        "open_p95_ms": round(_percentile([run["open_ms"] for run in startup_runs], 0.95), 3),
        "readiness_p50_ms": round(_percentile([run["readiness_ms"] for run in startup_runs], 0.50), 3),
        "readiness_p95_ms": round(_percentile([run["readiness_ms"] for run in startup_runs], 0.95), 3),
        "first_query_p50_ms": round(_percentile([run["first_query_ms"] for run in startup_runs], 0.50), 3),
        "first_query_p95_ms": round(_percentile([run["first_query_ms"] for run in startup_runs], 0.95), 3),
    }
    reconciliation = _reconciliation_metrics(store, corpus)
    workloads["reconciliation"] = {
        "name": "reconciliation",
        "runs": [
            {
                "run": 1,
                "temperature": "warm",
                "latency_ms": reconciliation["repair_time_ms"],
            }
        ],
        "summary": _latency_summary([reconciliation["repair_time_ms"]]),
    }
    workloads["concurrency"] = _concurrency_metrics(store, corpus, filtered_query, repeats)
    memory = sampler.stop()
    disk = _disk_metrics(target_workdir)
    embedding = store.chroma.embedding_fn
    identity = dict(reconciliation["identity"])
    metrics = {
        "workloads": workloads,
        "quality": quality,
        "throughput": {"build": build, "incremental": incremental},
        "startup": startup,
        "memory": memory,
        "disk": disk,
        "identity": identity,
        "reconciliation": reconciliation,
        "config": {
            "cold_run": "first run after open",
            "warm_runs": "remaining repeated runs",
            "repeats": repeats,
            "batch_size": batch_size,
            "refresh_window_seconds": refresh_window_seconds,
            "lexical_backend": "fts5",
            "reranker": False,
        },
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "phase": "2.3.7.6",
            "objective": "storage benchmark and database decision gate",
            "cold_warm_recorded": True,
        },
        "corpus": {
            "chunks": len(corpus.chunks),
            "families": len(corpus.families),
            "seed": corpus.seed,
            "digest": corpus.digest,
            "embedding_dimension": corpus.embedding_dimension,
            "id_format": "{item_id} or {item_id}#{i}",
            "distribution": _distribution(corpus),
        },
        "environment": {
            "workdir": str(target_workdir),
            "hardware": _hardware(),
            "versions": _versions(),
            "schema": {
                "sqlite_store": "SQLiteStore + migrations",
                "lexical_index": "corpus_fts FTS5",
                "vector_store": "Chroma PersistentClient",
                "metadata_fields": sorted(corpus.chunks[0].metadata),
            },
            "config": metrics["config"],
            "embedding": {
                "implementation": "DeterministicEmbeddingFunction",
                "dimension": corpus.embedding_dimension,
                "collection_dimension": collection_dimension,
                "seed": corpus.seed,
                "precomputed_corpus_vectors": len(corpus.embeddings_by_text),
                "network_calls": int(getattr(embedding, "network_calls", 0)),
                "model_calls": int(getattr(embedding, "model_calls", 0)),
            },
        },
        "metrics": metrics,
    }
    result["gates"] = evaluate_gates(
        result,
        chunks=len(corpus.chunks),
        refresh_window_seconds=refresh_window_seconds,
    )
    return result


def run_benchmark(
    *,
    chunks: int = DEFAULT_CHUNKS,
    seed: int = DEFAULT_SEED,
    workdir: Path | None = None,
    repeats: int = DEFAULT_REPEATS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    refresh_window_seconds: float = DEFAULT_REFRESH_WINDOW_SECONDS,
    collection_name: str = "phase2_3_7_6_benchmark",
) -> dict[str, Any]:
    """Run the benchmark while restoring process-wide Chroma settings."""
    previous_telemetry_setting = os.environ.get("ANONYMIZED_TELEMETRY")
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    try:
        return _run_benchmark(
            chunks=chunks,
            seed=seed,
            workdir=workdir,
            repeats=repeats,
            batch_size=batch_size,
            refresh_window_seconds=refresh_window_seconds,
            collection_name=collection_name,
        )
    finally:
        if previous_telemetry_setting is None:
            os.environ.pop("ANONYMIZED_TELEMETRY", None)
        else:
            os.environ["ANONYMIZED_TELEMETRY"] = previous_telemetry_setting


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=_positive_int, default=DEFAULT_CHUNKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write raw JSON results here")
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--refresh-window-seconds",
        type=float,
        default=DEFAULT_REFRESH_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--collection-name",
        default="phase2_3_7_6_benchmark",
        help="isolated Chroma collection name",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(
        chunks=args.chunks,
        seed=args.seed,
        workdir=args.workdir,
        repeats=args.repeats,
        batch_size=args.batch_size,
        refresh_window_seconds=args.refresh_window_seconds,
        collection_name=args.collection_name,
    )
    if args.out is not None:
        out_path = _validate_isolated_path(args.out.parent) / args.out.name
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON results: {out_path}")
    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

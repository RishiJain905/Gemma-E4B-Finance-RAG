# Phase 2.3.7.6 — Storage Benchmark Results and Decision

Executed: **2026-07-16** on the reference machine (AMD64 Family 25 Model 97,
16 logical CPUs, Windows 11, Python 3.13.2, SQLite 3.45.3, chromadb 1.5.9).
Harness: `scripts/benchmark_rag_storage.py` (committed `3407d4b`), seeded
deterministic corpus + precomputed 48-dim embeddings
(`DeterministicEmbeddingFunction`; `model_calls: 0`, `network_calls: 0` —
storage and retrieval are measured, never model or network latency).

## Decision: **KEEP SQLite + Chroma**

All nine hard gates pass at the 100,000-chunk Phase 2.3 scale after one
measured, bounded optimization pass. No replacement pilot is triggered.

## Runs

| Run | Corpus | Digest | Outcome |
|---|---:|---|---|
| 100k initial | 100,000 chunks, seed 2376 | `cdaf6035d1d1…` | 6/9 gates PASS, 3 FAIL (below) |
| 100k optimized | 100,000 chunks, seed 2376 (same digest) | `cdaf6035d1d1…` | **9/9 gates PASS** |

An earlier 100k attempt crashed with Chroma `too many SQL variables` — an
unbounded `collection.get(include=["metadatas"])` full scan in
`ChromaStore._all_metadata()`; fixed by bounded 5k-page scanning
(commit `b80b316`). The crash was a genuine store-boundary defect that only
manifests past ~30k chunks — the benchmark gate found it before production
growth did.

## Hard-gate table (100k, cold+warm, 3 repeats, worst-run reported)

| Gate | Threshold | Initial | Optimized | Verdict |
|---|---:|---:|---:|---|
| lexical p95 | < 100 ms | 139.3 ms | **92.3 ms** | PASS |
| dense p95 | < 200 ms | 60.0 ms | 60.6 ms | PASS |
| hybrid p95 (pre-generation) | < 250 ms | 537.0 ms | **94.9 ms** | PASS |
| filtered hybrid p95 | < 300 ms | 247.1 ms | 220.9 ms | PASS |
| authoritative inventory/count p95 | < 50 ms | 6,237.3 ms | **15.3 ms** | PASS |
| restart/open + readiness | < 10 s | 11.6 ms | 20.7 ms | PASS |
| unrepaired identity drift after interruption | 0 | 0 | 0 | PASS |
| nDCG@10 vs labeled baseline | ≤ 0.02 gap | 0.0 | 0.0 | PASS |
| daily incremental mutation volume | within refresh window | 0.7 s | 0.7 s | PASS |

Supporting measurements (optimized run): first query p95 94.5 ms; disk
436 MB Chroma + 76 MB SQLite + 36 MB FTS; build throughput ≈ 45 chunks/s
(one-time cost); bounded 3-reader + 1-writer concurrency clean; interrupted
cross-store mutation reconciled to zero drift.

## The bounded optimization pass (what changed)

1. **Inventory/count queries** — `Store.get_source_counts`/ticker counts
   previously materialized every Chroma metadata row per call (O(corpus));
   now served from the SQLite corpus ledger via indexed `GROUP BY`
   (additive migration `007_lexical_meta.sql`), with the Chroma scan kept
   only as the legacy fallback when ledger tables are absent. 6,237 → 15 ms.
2. **Hybrid hydration** — fused candidates now hydrate through one bounded,
   deduplicated id batch instead of per-stage fetches with dense/lexical
   overlap fetched twice. 537 → 95 ms.
3. **FTS5 maintenance** — `corpus_fts` `optimize` after bulk build, `rank`
   auxiliary instead of per-row `bm25()` recomputation, 200-candidate cap
   enforced before scoring joins. 139 → 92 ms.

No schema rewrites beyond additive indexes, no API shape changes, fail-soft
and revision-consistency semantics from 2.3.7.5 preserved. Offline gate:
`VERIFY: PASS` (1,831 passed).

## Rejected alternatives

- **LanceDB / Qdrant pilots** — not triggered: every hard gate passes and no
  approved requirement (multi-host service, higher writer concurrency)
  exists. Per spec, adapters are not built speculatively.
- **500k forecast run** — not executed: the measured source growth model
  (~24k items ≈ 30k chunks after ~2 weeks of Phase 2.3 ingestion) does not
  make 500k plausible before the next architecture review.

## Known limits

- lexical p95 (92.3 ms) passes with ~8% headroom; it is the first gate to
  re-check as the corpus grows.
- Single-writer local deployment remains the supported concurrency model.
- Benchmark corpora rebuild deterministically from seed 2376 (~37 min at
  100k); the harness currently regenerates rather than snapshotting a built
  corpus — a `--reuse-corpus` snapshot/restore flag is a noted follow-up.

## Next review trigger

Re-run the 100k gate (and add the 500k forecast run) when any of: live corpus
exceeds ~60k chunks; lexical p95 exceeds its gate in the 2.3.7.7 live
measurements; a multi-host or multi-writer requirement is approved; or a
major SQLite/Chroma version migration lands.

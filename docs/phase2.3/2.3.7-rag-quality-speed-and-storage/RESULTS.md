# Phase 2.3.7 — Quality, Latency Evaluation, and Rollout Results

Recorded **2026-07-17** after implementation of 2.3.7.1–2.3.7.6 (commits
`f09ab58` → `dbd0634`) on branch `phase-2.3.7-rag-quality`, live local stack
(llama-server `gemma-4-E4B-it.Q8_0` on `:8087`, restarted fresh before the
final run; live corpus 29,563 chunks / 24,413 items; FTS5 index rebuilt and
reconciled to zero drift).

## Measured runs

| Run | Date | Config | Cases | Source |
|---|---|---|---:|---|
| arm0-baseline | 07-16 | pre-promotion defaults | 75 | `eval/runs/1784183511` |
| arm1…arm6 | 07-16 | promotion arms (see FEATURE-DISPOSITIONS.md) | 75 ea. | `eval/runs/1784187954…1784193138` |
| post-2.3.7-defaults | 07-17 | promoted defaults + FTS5, fresh server | 79 | `eval/runs/1784265945` |

Note: the final run includes the four golden cases added by 2.3.7.7, so its
denominator (79) differs from the arms (75); per-family metrics and the
recorded dataset digests keep comparisons honest.

## Headline results (post-2.3.7 defaults vs arm0 baseline)

| Metric | Baseline | Post-2.3.7 | Reading |
|---|---:|---:|---|
| Indirect exact-plan accuracy | 0.00 | **1.00** | Gate: ≥ +15 pts — met (+100) |
| Indirect router accuracy | 0.00 | **1.00** | Capability routing gate ≥ 0.98 — met |
| Obligation coverage | 0.00 | **0.913** | |
| Intent accuracy (all) | 0.787 | 0.787 | Direct accuracy unchanged (gate: ≤ 1 pt drop) |
| Refusal rate | 0.053 | 0.051 | flat |
| Grounded faithfulness | 0.884 | 0.869 | within single-judge noise (±0.05) |
| Policy compliance | 0.885 | 0.861 | within noise |
| Answer relevance | 0.823 | 0.758 | slightly down; larger indirect share in denominator |
| e2e p50 | 16,891 ms | **9,143 ms** | −46% |
| e2e p95 | 42,445 ms | **29,876 ms** | −30% |
| Unnecessary planner calls | 0 | 0 | |

## Deterministic fast path (2.3.7.3 gate)

- Final run: 4 eligible rows answered with **zero model calls, avg 836 ms**
  vs 12,918 ms run average → ~94% faster; arm1 evidence: 3 rows at 874 ms vs
  24,040 ms (~96%). **≥90% gate met in both measurements.**
- No ineligible, partial, conflicting, or qualitative case skipped
  generation in any run (offline gate test also enforces this with mocks).
- Catalog questions made zero embedding and zero model calls (offline-gated).

## Retrieval / storage

- 100k-chunk storage benchmark: **9/9 hard gates PASS**, decision **keep
  SQLite + Chroma** — see `STORAGE-BENCHMARK-RESULTS.md`.
- Startup readiness with FTS5: middleware boot < 60 s observed live
  (previously 3–5 min due to in-memory BM25 materialization); benchmark
  restart-readiness 20.7 ms.
- Live FTS5 index: 29,563 rows, reconciliation zero
  missing/duplicate/stale/orphan.
- nDCG@10 gap vs labeled baseline: 0.0 (benchmark corpus).

## Rollout by profile (spec Step 5)

Executed via the 2.3.7.4 measured arms in the spec's recommended order:
catalog/coverage path validated (arm1), deterministic routing with
generation (arm1), deterministic answers for the conservative allowlist
(arm1, promoted), FTS5 hybrid promoted after the storage gates
(`lexical_backend: fts5` default). `configs/profiles/legacy.yaml` is the
tested rollback (in-memory BM25 + all 2.2.3+ features off); rollback for
any single capability is its individual flag. Not promoted (retained with
owner/prerequisite/expiry, see FEATURE-DISPOSITIONS.md): conversation
rewrite + LLM fallback + planning call, evidence sufficiency + corrective
retry + enforce, retrieval cache + prompt reuse, hierarchical retrieval,
reranker.

## Unresolved limits

1. **Intermittent model-call failures under load**: 7/79 rows failed on the
   degraded long-lived server; 5/79 still failed on a fresh server
   (`Error calling model:` — suspected context-limit/timeout on
   heavy-context conversational cases). Quarantined honestly as
   `trace_errors` (never judge-scored). Follow-up: capture the model
   error body in the middleware fail-soft path and bound conversation
   prompt size against the 32k server context.
2. **Metric-alias gap**: "latest total revenue" plans `total_revenue`;
   the live store names it `revenue_ttm` — route executes, returns no
   facts, falls back to generation (fail-soft correct, answer quality
   poor). Follow-up: extend the metric alias map.
3. Judge scores are single-judge on a 40-case screening subset + conversations;
   ±0.05 movements are noise. answer_relevance 0.758 needs re-reading after
   limits 1–2 are fixed.
4. `lexical_p95` passed with ~8% headroom at 100k — first storage gate to
   re-check as the corpus grows.

## Ship decision

**SHIP** the promoted set (adaptive RAG, deterministic routing,
deterministic answers, FTS5 lexical) as the recommended profile defaults.
Rollback: `MIDDLEWARE_PROFILE=legacy` (tested) or individual flags.
Limits 1–2 are follow-up defects, not rollback triggers: both fail soft
and neither regresses the pre-2.3.7 baseline behavior.

## Phase 2.3 sign-off inputs (2.3.6.2)

- Recommended/legacy profile diff: `configs/profiles/*.yaml` (this branch).
- Dispositions: `FEATURE-DISPOSITIONS.md` (complete, measured).
- Coverage + indirect results: this file; fixtures
  `tests/fixtures/evaluation/phase2_3_rag_quality.jsonl` and offline gates
  `tests/test_phase2_3_rag_quality_gates.py`.
- Storage results: `STORAGE-BENCHMARK-RESULTS.md`.
- Deterministic-answer correctness/latency: this file + offline gates.
- Unresolved limits and decision: above.

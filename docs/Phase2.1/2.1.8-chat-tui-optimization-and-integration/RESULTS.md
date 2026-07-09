# 2.1.8 — Chat/TUI Optimization & Integration: Results

## 2.1.8.1 — Latency measurement & middleware optimizations

Setup: live middleware `:8000` + TraceAlchemy `:8087`; 12-case golden subset
via `eval/run_eval.py --limit 12` before/after, plus manual `/query` samples
reading the new `timings` field.

### Where the time actually goes

Sample `/query` breakdowns (after instrumentation):

| stage | facts query | docs query (cold) | docs query (repeat) |
|---|---|---|---|
| intent_parse | 0.2 ms | 0.1 ms | 0.1 ms |
| freshness_check | 6.8 ms | 2.2 ms | 2.2 ms |
| retrieval (total) | 0.6 ms | 76.2 ms | **6.2 ms** |
| — embedding | 0.0 | 69.5 ms | **0.0 (LRU hit)** |
| — chroma | 0.0 | 4.8 ms | 4.5 ms |
| — sqlite | 0.6 ms | 0.0 | 0.0 |
| prompt_build | 0.0 ms | 0.0 ms | 0.0 ms |
| model_call | 21,952 ms | 54,445 ms | 30,136 ms |

**>99.9% of `/query` latency is model token generation.** Total non-generation
overhead is ~8 ms (facts path) to ~80 ms (cold docs path). The 12-case avg
total latency (before 17,289 ms → after 19,495 ms) differs only by generation
variance between runs — answer length dominates, middleware overhead is noise
at this scale.

### Optimizations landed

- Per-stage `timings` in `QueryResponse` (behind `return_timings`, default on),
  with retrieval sub-timings (embedding/chroma/sqlite) from the retriever.
- Model-health check cached (10s monotonic TTL); any successful model POST
  refreshes it — a query burst no longer pays one `GET :8087/health` each.
- `/health` builds `UnifiedScheduler` once (lazy module-level) and caches the
  scheduler-status + 6-ticker freshness summary for 3s (the chat client hits
  `/health` on startup).
- Query-embedding LRU cache (normalized text key, bounded by
  `embedding_cache_size`, default 256) in `TraceAlchemyEmbeddingFunction`:
  repeat-query embedding round-trip drops 69.5 ms → 0.
- Embedding HTTP client confirmed persistent (one `httpx.Client` per store).

Gate: `VERIFY: PASS` (772 passed; includes `tests/test_latency_optimizations.py`).

## 2.1.8.2 — Chat client performance & streaming

_Pending._

## 2.1.8.3 — TUI integration

_Pending._

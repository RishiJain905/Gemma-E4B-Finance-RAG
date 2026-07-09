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

- `scripts/chat.py` now uses one persistent `httpx.Client` (`ChatSession`) for
  `/query`, `/refresh`, `/health`, `/tools`; `--timeout` / `--no-stream` flags
  and a `/verbose` toggle (prints the server `timings` block).
- New `/query/stream` SSE endpoint (behind `enable_streaming`, default on):
  shares `_build_query_context` with `/query` (refactored so the two paths
  cannot drift), streams llama-server token deltas as `token` events, then one
  terminal `metadata` event (grounding, citations, timings, counts, strategy).
  Server-side it falls back to the non-streaming answer if the stream errors
  mid-flight; it returns 404 when streaming is disabled, the model is down, or
  **tools are enabled** (the 2.1.4 tool loop is multi-turn and cannot stream) —
  the client detects 404 once and uses the non-streaming path with an
  elapsed-time spinner (TTY-only).
- Note: this deployment's `configs/middleware.yaml` has `enable_tools: true`,
  so streaming only activates with tools off (e.g. `ENABLE_TOOLS=false`).
  2.1.8.3 surfaces active capabilities at TUI startup so this is visible.

Live smoke test (tools off, model on :8087): HTTP 200, 163 token events, then
metadata (`grounding=grounded`, 4 citations, full timings). First token at
35.5s of a 43.5s generation — first-token latency is dominated by llama-server
prompt prefill (model-side), but tokens render ~8s before the full answer and
the TUI no longer looks hung.

Gate: `VERIFY: PASS` (777 passed; includes `tests/test_chat_client.py`).

## 2.1.8.3 — TUI integration

- One defensive renderer (`_render_metadata`) shared by the streaming and
  non-streaming paths: colored grounding tag (TTY-only), `used: <tools>`,
  retrieval strategy, fetched-on-miss notice, resolved-ticker confirmation,
  staleness warnings, `/verbose` timings. A minimal old-server response
  renders without KeyErrors; the dead pre-ChatSession `do_query()` was removed.
- Middleware additions (all optional/additive): `QueryResponse.tools_used`
  (per-request ContextVar filled by the tool loop — `_call_model` signature
  untouched), `QueryResponse.resolved_ticker` (only for non-exact
  resolutions), `HealthResponse.capabilities` (tools/streaming/answer_policy),
  and `QueryRequest.answer_policy` per-request override backing `/grounding`.
- New commands: `/grounding strict|graded|clear`, `/eval [N]`, hardened
  `/tools`; `/help` updated; startup prints active capabilities from `/health`.
- Docs: README chat section refreshed; `scripts/CHAT.md` created.

Manual end-to-end check (live stack, tools on → streaming falls back):
- Analytical: "Which tracked stock has the lowest forward P/E?" → tool-driven
  answer (`used: query_facts`), `[GROUNDED]` tag, strategy shown.
- Projection: "consensus outlook for NVDA next quarter" → grounded analyst
  consensus (revenue + EPS estimates, 40 analysts) with caveat and staleness
  warning rendered.

Gate: `VERIFY: PASS` (786 passed; +9 tests in `tests/test_chat_client.py`).

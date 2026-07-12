# 2.2.6 — Chat Runtime & Phase Integration: Results

**Date:** 2026-07-11
**Branch:** `phase-2.2.6` (one commit per task: 2.2.6.1 `7245d9a`, 2.2.6.2 `65fdffc`, 2.2.6.3 doc/sign-off commit follows)

## Setup

- Offline gate: `scripts\verify.ps1` (ruff + `pytest tests/ -m "not live"`).
- Live smoke: llama-server on `:8087`, middleware on `:8000` with `ENABLE_TOOL_FINAL_STREAMING=1 ENABLE_STREAM_PROGRESS_EVENTS=1 ENABLE_MODEL_TOOLS=1` (committed defaults remain **off**).
- Implementation provenance: both tasks by opus-xhigh (Codex credits exhausted for the whole phase).

## Offline regression gate

| Point | Result |
|---|---|
| Phase start (post-2.2.5 merge) | 1211 passed — VERIFY: PASS |
| After 2.2.6.1 | 1237 passed (+26) — VERIFY: PASS |
| After 2.2.6.2 | 1271 passed (+34) — VERIFY: PASS |

## Live smoke (2.2.6.1, flags on)

`POST /query/stream` with tools enabled returned a well-formed SSE sequence: `query_started` → `stage` events (`compile` 7.3 ms, `retrieve` 6.2 ms, `pack` 0.8 ms, `generate` started) → legacy-shaped `token` deltas streaming the final answer including its `[E#]` citation. Every event carried `schema_version: 1`, the request's `query_id`, and a monotonic `sequence`. `/health.capabilities` reported `streaming: true, streaming_tool_final: true`. No prompt text, document bodies, or tool arguments appeared in any progress event.

## Not run here (deliberate)

- **Warm/cold TTFT and llama-server prompt-reuse measurements** (2.2.6.2 promotion gates: warm TTFT −30%, cold p95 +≤5%) require the scheduled GPU evaluation — the spec forbids silently running it as part of sign-off. Status: **PENDING — GPU evaluation not yet scheduled**.
- Retrieval-cache hit-rate telemetry is implemented and unit-proven (exact invalidation incl. same-count replacement, injected clocks, zero stale hits in tests) but has no live traffic sample yet.

## Caveats

- The stream smoke exercised the standard lane; a tools-enabled request that actually triggers planning rounds (phase 1 → final stream) is covered by offline tests with mocked model/tool boundaries, not by a live run.
- `store_revision` bumps rely on ingestors using the Store facade; direct SQLite writers must call the documented bump helper (enforced by convention + tests, not by the database).

## Regression-gate outcome

`VERIFY: PASS` at both task boundaries. Default-config behavior byte-identical (all new flags off; legacy streaming/token/metadata/error consumers unchanged, verified by the pre-existing suite).

## Ship/rollback decision

**Ship dark (merged, flags off).** Promote `enable_stream_progress_events` + `enable_tool_final_streaming` after the chat-client TTFT check in the scheduled live run; promote `enable_retrieval_cache` and `llama_cache_prompt` only after the live cache/TTFT gates clear. Rollback for each = flip its flag; no storage migration involved (`store_revision` is additive and harmless when the cache is off).

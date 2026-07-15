# Feature-flag tracker

Live inventory of every optional capability flag across `configs/*.yaml`: what
is enabled on this deployment, what is deliberately off, and why. Values in
YAML are authoritative; the listed env vars are optional *overrides* (a flag
does **not** need an `.env` entry to take effect). The middleware reads flags
at startup — restart it after edits.

Last updated: **2026-07-15** (Phase 2.3 rollout enablement, commit `d8b783e`).

## Enabled

| Flag | File | Since | Why |
|---|---|---|---|
| `feature_flags.universe_refresh` | `configs/universe.yaml` | 2026-07-15 | Expands coverage from the 6 deep tickers to the active S&P 500 ∪ Nasdaq-100 union (~550 securities). |
| `feature_flags.sec_broad_events` | `configs/sources.yaml` | 2026-07-15 | SEC event discovery (registrations, 424B, 8-K items, exhibits) across the broad universe. |
| `feature_flags.company_news` | `configs/sources.yaml` | 2026-07-15 | Finnhub company news, broad scope (key verified). |
| `feature_flags.grouped_market_data` | `configs/sources.yaml` | 2026-07-15 | Massive grouped daily market summary + corporate actions (key verified). |
| `feature_flags.official_feeds` | `configs/sources.yaml` | 2026-07-15 | Fed, Treasury, BLS, BEA, EIA, NY Fed, CFTC official feeds (keys verified). |
| `feature_flags.sector_feeds` | `configs/sources.yaml` | 2026-07-15 | openFDA / NHTSA / USAspending, bounded to their configured sectors. |
| `enable_phase2_3_retrieval` | `configs/middleware.yaml` | 2026-07-15 | Phase 2.3 retrieval facets/ranking at query time. Env override: `ENABLE_PHASE2_3_RETRIEVAL`. |
| `enable_phase2_3_corpus_projection` | `configs/middleware.yaml` | 2026-07-15 | Aggregation-first Corpus Explorer projection. Env override: `ENABLE_PHASE2_3_CORPUS_PROJECTION`. |
| `enable_graph_observer` | `configs/middleware.yaml` | 2026-07-15 | Live Trace + Corpus Explorer UI at `/graph` (loopback-only, read-only, redacted). Env: `ENABLE_GRAPH_OBSERVER`. |
| `enable_tool_final_streaming` | `configs/middleware.yaml` | 2026-07-15 | `/query/stream` works with tools enabled (tool rounds non-streaming, final answer streams). |
| `enable_stream_progress_events` | `configs/middleware.yaml` | 2026-07-15 | Redacted stage/tool progress events on the SSE stream (chat client renders a live progress line). |
| `enabled` (CompanyFacts) | `configs/sec_companyfacts.yaml` | 2026-07-15 | Authoritative filed GAAP observations for deep tickers; SQLite-only, preferred automatically at retrieval. |
| `deep.allow_outside_indexes` | `configs/coverage.yaml` | 2026-07-15 | Transitional: registry is populated (backfill) but the first universe snapshot hadn't landed, so the six deep tickers needed the documented off-index opt-in. Inert once memberships exist. |
| `enable_lexical`, `enable_evidence_taxonomy`, `enable_authority_ranking`, `enable_duplicate_coverage_packing`, `enable_tools`, `enable_fetch_on_miss`, `enable_streaming`, `enable_citations` | `configs/middleware.yaml` | pre-2.3.4 | Long-standing defaults, evaluated in their own phases. |
| `answer_validation: report` | `configs/middleware.yaml` | 2.2.4.3 | Metadata-only citation validation; changes no answer. |

## Disabled — 2.3.7 promotion candidates

Each of these shipped complete but **behavior-changing**, so Phase 2.3.7.4's
feature-disposition gate promotes, retains-with-expiry, or removes them after
*measured* quality/latency comparison — not by blind enablement. Expect these
to flip (or be removed) during 2.3.7.

| Flag | File | What it would do | Gated by |
|---|---|---|---|
| `enable_adaptive_rag` (+ `adaptive_enable_planning_call`) | `configs/middleware.yaml` | Fast/standard/complex lane orchestration with budgets. | 2.3.7 quality/latency comparison. |
| `enable_deterministic_tool_routing` | `configs/middleware.yaml` | Route safe analytical asks to read-only tools before any model call. | 2.3.7.2 indirect-query gate. |
| `enable_deterministic_answers` | `configs/middleware.yaml` | Fully-covered deterministic answers skip model generation (≥90% latency win target). | 2.3.7.3 deterministic-answer gate. |
| `enable_evidence_sufficiency` / `enable_corrective_retry` / `enable_query_decomposition` | `configs/middleware.yaml` | Route-aware sufficiency grading + one bounded corrective retrieval. | 2.3.7 quality gates. |
| `enable_hierarchical_retrieval` | `configs/middleware.yaml` | Filing → section → child expansion of precise SEC hits. | Long-document eval gates; also needs `sec.index_filing_text`. |
| `sec.index_filing_text` | `configs/sec.yaml` | Persist/index full SEC filing sections (embedding cost on deep tickers). | Same long-document gates as above. |
| `enable_reranker` | `configs/middleware.yaml` | Cross-encoder/LLM re-rank of fused candidates (adds latency; cross-encoder downloads a HF model). | 2.3.7.5 retrieval latency/quality gates. |
| `enable_conversation_rewrite` / `enable_llm_rewrite_fallback` | `configs/middleware.yaml` | Compile follow-ups into standalone retrieval queries. | Conversational carryover gates. |
| `enable_retrieval_cache` / `llama_cache_prompt` | `configs/middleware.yaml` | Revision-keyed pre-prompt evidence reuse; llama-server prompt cache. | 2.3.7.5 speed gates. |
| `answer_validation: enforce` / `require_evidence_ids` | `configs/middleware.yaml` | Downgrade/refuse unsupported answers instead of reporting. | 2.3.7 quality gates. |
| `twelve_data` source | `configs/coverage.yaml` / `configs/sources.yaml` | Optional market-data fallback. | Source policy: stays off unless Massive coverage proves insufficient. |

## How to flip a flag

1. Edit the YAML value (`false` → `true`).
2. Restart the middleware (or let `scripts/chat.py` start a fresh one).
3. `python -m src.scheduler status` / `GET /health` `capabilities` to confirm.
4. Update this file (move the row, stamp the date).

Rollback for every flag above is the reverse edit — all are non-destructive;
stored evidence and schema stay intact (see `docs/CONFIGURATION.md`).

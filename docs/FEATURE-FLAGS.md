# Feature-flag tracker

Live inventory of every optional capability flag across `configs/*.yaml`: what
is enabled on this deployment, what is deliberately off, and why. Values in
YAML are authoritative; the listed env vars are optional *overrides* (a flag
does **not** need an `.env` entry to take effect). The middleware reads flags
at startup — restart it after edits.

Last updated: **2026-07-17** (Phase 2.3.7 measured promotion; see
`phase2.3/2.3.7-rag-quality-speed-and-storage/RESULTS.md` and
`FEATURE-DISPOSITIONS.md` for the measurements behind every row below).
Since 2.3.7.4, flags belong to documented profiles
(`configs/profiles/legacy|recommended|evaluation.yaml`, selected by the
`profile:` key or `MIDDLEWARE_PROFILE`); `legacy` is the tested rollback.

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
| `enable_adaptive_rag` | `configs/middleware.yaml` | 2026-07-17 | **Promoted by measured 2.3.7.4 arm1**: fast/standard/complex/catalog lane orchestration. Env: `ENABLE_ADAPTIVE_RAG`. |
| `enable_deterministic_tool_routing` | `configs/middleware.yaml` | 2026-07-17 | **Promoted (arm1)**: safe analytical asks routed to read-only tools; refusals 5.3%→0%, policy compliance 0.885→0.958. Env: `ENABLE_DETERMINISTIC_TOOL_ROUTING`. |
| `enable_deterministic_answers` | `configs/middleware.yaml` | 2026-07-17 | **Promoted (arm1 + 2.3.7.7)**: fully-covered typed answers skip generation — ~836 ms vs ~13 s (≥90% gate met). Env: `ENABLE_DETERMINISTIC_ANSWERS`. |
| `lexical_backend: fts5` | `configs/middleware.yaml` | 2026-07-17 | **Promoted (2.3.7.5/2.3.7.6)**: persistent SQLite FTS5 lexical index; startup no longer materializes the corpus (boot minutes→seconds); 9/9 storage gates. `memory` = legacy rollback. Env: `LEXICAL_BACKEND`. |
| `profile: recommended` | `configs/middleware.yaml` | 2.3.7.4 | Profile selection key; `MIDDLEWARE_PROFILE=legacy` is the tested one-switch rollback. |

## Disabled — retained temporarily after measured 2.3.7 arms

The 2.3.7.4 promotion arms **measured** each of these and declined to promote
as configured. Every row is retained with an owner (repo owner), a named
prerequisite, and expiry at Phase 2.3 sign-off — details and the arm numbers
in `phase2.3/2.3.7-rag-quality-speed-and-storage/FEATURE-DISPOSITIONS.md`.

| Flag | File | Measured outcome (2026-07-16 arms) | Prerequisite to re-arm |
|---|---|---|---|
| `enable_conversation_rewrite` / `enable_llm_rewrite_fallback` | `configs/middleware.yaml` | arm2: entity carryover 0.833→0.167, intent accuracy 0.787→0.680. | Rewrite entity-preservation fix. |
| `adaptive_enable_planning_call` | `configs/middleware.yaml` | Only measured bundled in arm2 (net negative). | Isolated arm. |
| `enable_evidence_sufficiency` / `enable_corrective_retry` | `configs/middleware.yaml` | arm3: refusal rate 0.053→0.613; keyword coverage 0.889→0.296. | Grader threshold calibration. |
| `answer_validation: enforce` / `require_evidence_ids` | `configs/middleware.yaml` | Bundled in arm3 (not promoted); stays `report`. | Same calibration as above. |
| `enable_retrieval_cache` / `llama_cache_prompt` | `configs/middleware.yaml` | arm6: warm pass showed no hit/latency change (eval workload defeats the key). | Repeat-traffic hit-rate measurement. |
| `enable_hierarchical_retrieval` | `configs/middleware.yaml` | Not armed — prerequisite missing. | `sec.index_filing_text` + filing corpus backfill. |
| `sec.index_filing_text` | `configs/sec.yaml` | Not armed (embedding cost decision). | Same long-document gates. |
| `enable_reranker` | `configs/middleware.yaml` | Not armed — measured only after the FTS5 baseline landed. | Rerank arm on the persistent lexical baseline. |
| `enable_query_decomposition` | `configs/middleware.yaml` | Not armed in 2.3.7. | Own quality gate. |
| `twelve_data` source | `configs/coverage.yaml` / `configs/sources.yaml` | Source policy, not a 2.3.7 candidate. | Massive coverage proving insufficient. |

## How to flip a flag

1. Edit the YAML value (`false` → `true`).
2. Restart the middleware (or let `scripts/chat.py` start a fresh one).
3. `python -m src.scheduler status` / `GET /health` `capabilities` to confirm.
4. Update this file (move the row, stamp the date).

Rollback for every flag above is the reverse edit — all are non-destructive;
stored evidence and schema stay intact (see `docs/CONFIGURATION.md`).

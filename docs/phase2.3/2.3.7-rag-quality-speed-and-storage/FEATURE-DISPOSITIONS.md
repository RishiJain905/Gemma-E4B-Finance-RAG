# Phase 2.3.7.4 Feature Dispositions

Audited: **2026-07-16** on branch `phase-2.3.7-rag-quality`, after the
2.3.7.1–2.3.7.3 commits (`f09ab58`, `e10a296`, `fe00081`). The measured
promotion arms ran **2026-07-16** (see "Measured promotion arms" below);
the Disposition column records their outcome.

## Measured promotion arms (2026-07-16)

Method: cumulative arms in the spec's recommended order, one capability
group changed per arm, each against a fresh middleware on the committed
code (`cc30d5c`) and the live local stack (llama-server `:8087`,
`gemma-4-E4B-it.Q8_0`). Fixed workload per arm: the first 40
`eval/golden/finance_qa.jsonl` cases plus all conversation scenarios
(75 scored rows, dataset digest `9e819623…`), scored with
`eval/score.py --policy graded` including the LLM judge. Runs/summaries:
`eval/runs/1784183511…1784193138`. Caveats: n=75 per arm, single-judge
scores carry ~±0.05 noise; the definitive full-set evaluation is 2.3.7.7.

| Arm | Flags changed | faith | policy | relev | refusal | kw-cov | ent-carry | met-carry | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arm0-baseline | (current defaults) | 0.884 | 0.885 | 0.823 | 0.053 | 0.889 | 0.833 | 0.111 | 16891 | 42445 |
| arm1-deterministic | + adaptive_rag, deterministic_tool_routing, deterministic_answers | 0.856 | **0.958** | 0.788 | **0.000** | 0.889 | 0.833 | 0.000 | 18774 | 60325 |
| arm2-conversation | + conversation_rewrite, llm_rewrite_fallback, planning_call | 0.878 | 0.892 | 0.743 | 0.013 | 0.741 | **0.167** | 0.444 | 17874 | 60321 |
| arm3-sufficiency | + evidence_sufficiency, corrective_retry, answer_validation=enforce | 0.817 | 0.795 | 0.691 | **0.613** | **0.296** | 0.167 | 0.444 | 9138 | 41658 |
| arm6-cache-cold | + retrieval_cache, llama_cache_prompt | 0.753 | 0.697 | 0.701 | 0.640 | 0.296 | 0.167 | 0.444 | 9723 | 33385 |
| arm6-cache-warm | (same process, repeat pass) | 0.881 | 0.750 | 0.700 | 0.640 | 0.296 | 0.167 | 0.444 | 10151 | 49827 |

Arm-1 fast-path evidence (from `eval/runs/1784187954.jsonl` orchestration
metadata): 72 rows generated normally (avg 24,040 ms); **3 eligible rows
answered with zero model calls at avg 874 ms (~96% latency reduction,
meets the ≥90% 2.3.7.3 gate)**; 4 catalog-lane routes; 40 rows used
deterministic read-only tools; eligibility stayed conservative (no
ineligible row skipped generation).

Key readings:

- **arm1**: refusals eliminated, policy compliance up, ticker accuracy
  0.547→0.653, faithfulness within judge noise → the group promotes.
  (`retrieval_hit_rate` drops to 0.707 because deterministic/catalog rows
  answer without document retrieval — an artifact of the metric, not a
  recall regression.)
- **arm2**: single-turn intent accuracy 0.787→0.680 and entity carryover
  0.833→0.167; the rewrite is compiling entity references away on this
  workload. Metric/timeframe carryover improves (0.111→0.444). Net
  negative as configured → not promoted.
- **arm3**: the sufficiency grader + `enforce` refuses 61% of the
  workload (baseline 5.3%) and keyword coverage collapses → thresholds
  are miscalibrated for current corpus coverage → not promoted.
- **arm6**: warm pass shows no latency or hit-rate improvement over cold
  (p50 9.7s→10.2s) — the eval workload's per-request conversation ids
  defeat the cache key, so the arm demonstrates nothing either way →
  not promoted pending a realistic repeat-traffic measurement.

## Audit rules

- “Current default” means the committed value in `configs/middleware.yaml`.
  Where the source file intentionally promotes an already-shipped capability,
  the code fallback is called out separately.
- Owning modules, config keys, docs, and tests below were traced from the
  current imports/references and the offline test suite; a listed test is not a
  live-arm result.
- `pending measurement (2.3.7.4 arms)` is deliberately retained until the
  orchestrator runs the isolated comparison arms.
- Temporary retention is used only for the two prerequisites named by the
  spec. Both expire at **Phase 2.3 sign-off** and are owned by the **repo owner**.

## Inventory manifest

| Capability | Current default | Owning code modules | Config keys / env overrides | Documentation sections | Offline tests | Exclusive dependency | Disposition status |
|---|---|---|---|---|---|---|---|
| Conversation rewrite | `off` | `src/middleware/conversation.py`, `src/middleware/query_rewriter.py`, `src/middleware/app.py`, `src/middleware/config.py` | `enable_conversation_rewrite`, `conversation_rewrite_timeout_s`; `ENABLE_CONVERSATION_REWRITE`, `CONVERSATION_REWRITE_TIMEOUT_S` | `docs/CONFIGURATION.md` — Conversation memory & follow-up rewriting (2.2.2); `docs/ARCHITECTURE.md` — Conversational query understanding | `tests/test_conversation.py`, `tests/test_query_rewriter.py`, `tests/test_middleware.py` | None beyond the existing middleware/model endpoint; no feature-only package | **retain temporarily — arm2 regressed entity carryover 0.833→0.167 and intent accuracy; prerequisite: rewrite entity-preservation fix + re-arm; owner: repo owner; expiry: Phase 2.3 sign-off** |
| LLM rewrite fallback | `off` | `src/middleware/query_rewriter.py`, `src/middleware/app.py`, `src/middleware/config.py` | `enable_llm_rewrite_fallback`, `conversation_rewrite_timeout_s`; `ENABLE_LLM_REWRITE_FALLBACK`, `CONVERSATION_REWRITE_TIMEOUT_S` | `docs/CONFIGURATION.md` — Conversation memory & follow-up rewriting; `docs/FEATURE-FLAGS.md` — Disabled promotion candidates | `tests/test_query_rewriter.py`, `tests/test_feature_profiles.py` | Existing local llama-server/httpx call only; no exclusive package | **retain temporarily — bundled in arm2 (not promoted); follows conversation rewrite; owner: repo owner; expiry: Phase 2.3 sign-off** |
| Adaptive RAG | `off` | `src/middleware/adaptive_orchestrator.py`, `src/middleware/app.py`, `src/middleware/query_plan.py`, `src/middleware/config.py` | `enable_adaptive_rag`, `adaptive_max_subqueries`, `adaptive_max_retrieval_rounds`, `adaptive_max_planning_calls`, `adaptive_max_context_chars`, `adaptive_conditional_rerank`; `ENABLE_ADAPTIVE_RAG`, `ADAPTIVE_MAX_SUBQUERIES`, `ADAPTIVE_MAX_RETRIEVAL_ROUNDS`, `ADAPTIVE_MAX_PLANNING_CALLS`, `ADAPTIVE_MAX_CONTEXT_CHARS`, `ADAPTIVE_CONDITIONAL_RERANK` | `docs/CONFIGURATION.md` — Adaptive orchestration & deterministic tool routing (2.2.3); `docs/ARCHITECTURE.md` — Adaptive orchestration; `eval/README.md` — Adaptive orchestration comparison | `tests/test_adaptive_orchestrator.py`, `tests/test_query_plan.py`, `tests/test_capability_intent.py`, `tests/test_middleware.py` | Existing retriever/router and optional local planning call; no exclusive package | **PROMOTE (arm1, 2026-07-16) — enabled by default** |
| Planning call | `off` | `src/middleware/adaptive_orchestrator.py`, `src/middleware/query_plan.py`, `src/middleware/app.py`, `src/middleware/config.py` | `adaptive_enable_planning_call`, `adaptive_max_planning_calls`; `ADAPTIVE_ENABLE_PLANNING_CALL`, `ADAPTIVE_MAX_PLANNING_CALLS`; prerequisite `enable_adaptive_rag` | `docs/CONFIGURATION.md` — Adaptive orchestration & deterministic tool routing; `docs/ARCHITECTURE.md` — Adaptive orchestration | `tests/test_adaptive_orchestrator.py`, `tests/test_query_plan.py`, `tests/test_feature_profiles.py` | Existing local llama-server/httpx call only; no exclusive package | **retain temporarily — only measured bundled inside arm2 (net negative); prerequisite: isolated arm; owner: repo owner; expiry: Phase 2.3 sign-off** |
| Deterministic routing | `off` | `src/middleware/deterministic_router.py`, `src/middleware/adaptive_orchestrator.py`, `src/middleware/app.py`, `src/middleware/config.py` | `enable_deterministic_tool_routing`, `max_deterministic_tools_per_query`; `ENABLE_DETERMINISTIC_TOOL_ROUTING`, `MAX_DETERMINISTIC_TOOLS_PER_QUERY` | `docs/CONFIGURATION.md` — Adaptive orchestration & deterministic tool routing; `docs/ARCHITECTURE.md` — Adaptive orchestration; `docs/FEATURE-FLAGS.md` | `tests/test_deterministic_router.py`, `tests/test_capability_intent.py`, `tests/test_middleware.py` | Existing read-only finance tools and Store; no exclusive package | **PROMOTE (arm1, 2026-07-16) — enabled by default** |
| Deterministic answers | `off` | `src/middleware/deterministic_answers.py`, `src/middleware/deterministic_router.py`, `src/middleware/adaptive_orchestrator.py`, `src/middleware/app.py`, `src/middleware/answer_validator.py`, `src/middleware/evidence_trace.py`, `src/middleware/stream_events.py`, `src/middleware/config.py` | `enable_deterministic_answers`; `ENABLE_DETERMINISTIC_ANSWERS`; prerequisites `enable_deterministic_tool_routing` + `enable_adaptive_rag` | `docs/CONFIGURATION.md` — Adaptive orchestration & deterministic tool routing; `docs/ARCHITECTURE.md` — Query/answer flow and deterministic fast path; `docs/FEATURE-FLAGS.md` | `tests/test_deterministic_answer_fast_path.py`, `tests/test_deterministic_router.py`, `tests/test_coverage_tools.py` | Existing typed evidence/provenance validators; no exclusive package | **PROMOTE (arm1, 2026-07-16) — enabled by default; ~96% latency cut on eligible rows** |
| Evidence sufficiency | `off` | `src/middleware/evidence_grader.py`, `src/middleware/evidence.py`, `src/middleware/adaptive_orchestrator.py`, `src/middleware/app.py`, `src/middleware/config.py` | `enable_evidence_sufficiency`; `ENABLE_EVIDENCE_SUFFICIENCY` | `docs/CONFIGURATION.md` — Evidence sufficiency & corrective retrieval (2.2.4.1–2.2.4.2); `docs/ARCHITECTURE.md` — Corrective retrieval & provenance | `tests/test_evidence_grader.py`, `tests/test_answer_policy.py`, `tests/test_adaptive_orchestrator.py` | Existing deterministic grader and evidence models; no exclusive package | **retain temporarily — arm3 refusal rate 0.613 vs 0.053 baseline; prerequisite: grader threshold calibration + re-arm; owner: repo owner; expiry: Phase 2.3 sign-off** |
| Corrective retry | `off` | `src/middleware/evidence_grader.py`, `src/middleware/adaptive_orchestrator.py`, `src/middleware/app.py`, `src/middleware/retrieval_cache.py`, `src/middleware/config.py` | `enable_corrective_retry`, `max_corrective_retries`; `ENABLE_CORRECTIVE_RETRY`, `MAX_CORRECTIVE_RETRIES`; prerequisite `enable_evidence_sufficiency` | `docs/CONFIGURATION.md` — Evidence sufficiency & corrective retrieval; `docs/ARCHITECTURE.md` — Corrective retrieval & provenance | `tests/test_evidence_grader.py`, `tests/test_answer_policy.py`, `tests/test_adaptive_orchestrator.py` | `enable_evidence_sufficiency` and existing Retriever; no exclusive package | **retain temporarily — follows evidence sufficiency (arm3 not promoted); owner: repo owner; expiry: Phase 2.3 sign-off** |
| Hierarchical retrieval | `off` | `src/middleware/hierarchical_retrieval.py`, `src/middleware/retriever.py`, `src/middleware/adaptive_orchestrator.py`, `src/middleware/retrieval_cache.py`, `src/sec/filing_processor.py`, `scripts/index_sec_filing_text.py`, `src/middleware/config.py` | `enable_hierarchical_retrieval`, `hierarchy_max_siblings`, `hierarchy_max_adjacent_sections`, `hierarchy_max_expanded_items`; `ENABLE_HIERARCHICAL_RETRIEVAL`, `HIERARCHY_MAX_SIBLINGS`, `HIERARCHY_MAX_ADJACENT_SECTIONS`, `HIERARCHY_MAX_EXPANDED_ITEMS`; SEC prerequisite `sec.index_filing_text` (`SEC_INDEX_FILING_TEXT` is validation-only) | `docs/CONFIGURATION.md` — Hierarchical retrieval (2.2.5.3); `docs/ARCHITECTURE.md` — Hierarchical retrieval & authoritative facts; `docs/FEATURE-FLAGS.md` | `tests/test_hierarchical_retrieval.py`, `tests/test_filing_sections.py`, `tests/test_index_sec_filing_text.py`, `tests/test_feature_profiles.py` | Indexed SEC filing-section corpus and backfill; fair evaluation requires `sec.index_filing_text: true` and populated sections | **retain temporarily — prerequisite: SEC filing-section indexing plus corpus backfill; owner: repo owner; expiry: Phase 2.3 sign-off** |
| Cross-encoder/LLM reranker | `off` | `src/middleware/reranker.py`, `src/middleware/retriever.py`, `src/middleware/adaptive_orchestrator.py`, `src/middleware/retrieval_cache.py`, `src/middleware/config.py` | `enable_reranker`, `reranker_backend`, `reranker_model`, `rerank_candidates`, `rerank_top_n`; `ENABLE_RERANKER`, `RERANKER_BACKEND`, `RERANKER_MODEL`, `RERANK_CANDIDATES`, `RERANK_TOP_N` | `docs/CONFIGURATION.md` — Retrieval & re-ranking (2.1.2); `docs/ARCHITECTURE.md` — Retrieval Pipeline; `docs/FEATURE-FLAGS.md` | `tests/test_reranker.py`, `tests/test_retrieval_integration.py`, `tests/test_lexical_and_fusion.py`, `tests/test_retriever.py` | Cross-encoder backend uses `sentence-transformers` / torch and downloads `cross-encoder/ms-marco-MiniLM-L-6-v2`; LLM backend reuses llama-server | **retain temporarily — prerequisite: 2.3.7.5 persistent lexical baseline; owner: repo owner; expiry: Phase 2.3 sign-off** |
| Retrieval cache | `off` | `src/middleware/retrieval_cache.py`, `src/middleware/app.py`, `src/middleware/adaptive_orchestrator.py`, `src/middleware/config.py` | `enable_retrieval_cache`, `retrieval_cache_max_entries`, `retrieval_cache_ttl_s`, `retrieval_cache_max_value_chars`; `ENABLE_RETRIEVAL_CACHE`, `RETRIEVAL_CACHE_MAX_ENTRIES`, `RETRIEVAL_CACHE_TTL_S`, `RETRIEVAL_CACHE_MAX_VALUE_CHARS` | `docs/CONFIGURATION.md` — Versioned retrieval cache & prompt efficiency (2.2.6.2); `docs/ARCHITECTURE.md` — Runtime: Streaming & Caching | `tests/test_retrieval_cache.py`, `tests/test_latency_optimizations.py` | Existing Store revision and in-process LRU; no exclusive package | **retain temporarily — arm6 warm pass showed no hit/latency change (cache key defeated by eval workload); prerequisite: repeat-traffic hit-rate measurement; owner: repo owner; expiry: Phase 2.3 sign-off** |
| Llama prompt reuse | `off` | `src/middleware/app.py`, `src/middleware/prompt_policy.py`, `src/middleware/config.py` | `llama_cache_prompt`; `LLAMA_CACHE_PROMPT` | `docs/CONFIGURATION.md` — Versioned retrieval cache & prompt efficiency; `docs/ARCHITECTURE.md` — Runtime: Streaming & Caching | `tests/test_latency_optimizations.py`, `tests/test_feature_profiles.py` | llama-server `cache_prompt` compatibility and reused-token timings; no package beyond existing model server | **retain temporarily — bundled in arm6 (no demonstrated win); owner: repo owner; expiry: Phase 2.3 sign-off** |
| Tool-final streaming | `on` (2026-07-15; code fallback `off`) | `src/middleware/app.py`, `src/middleware/stream_events.py`, `src/middleware/config.py` | `enable_tool_final_streaming`, `enable_streaming`, `enable_tools`; `ENABLE_TOOL_FINAL_STREAMING`, `ENABLE_STREAMING`, `ENABLE_TOOLS` | `docs/CONFIGURATION.md` — Tool-aware final streaming & progress events (2.2.6.1); `docs/ARCHITECTURE.md` — Runtime: Streaming & Caching; `docs/FEATURE-FLAGS.md` | `tests/test_streaming.py`, `tests/test_chat_client.py`, `tests/test_deterministic_answer_fast_path.py` | Existing SSE stream and bounded tools; no exclusive package | **PROMOTE (confirmed) — on since 2026-07-15 and active during every 2.3.7.4 arm with no adverse signal** |
| Progress events | `on` (2026-07-15; code fallback `off`) | `src/middleware/stream_events.py`, `src/middleware/app.py`, `src/middleware/graph_observer.py`, `src/middleware/config.py` | `enable_stream_progress_events`, `stream_progress_include_counts`; `ENABLE_STREAM_PROGRESS_EVENTS`, `STREAM_PROGRESS_INCLUDE_COUNTS` | `docs/CONFIGURATION.md` — Tool-aware final streaming & progress events; `docs/ARCHITECTURE.md` — Runtime: Streaming & Caching; `docs/FEATURE-FLAGS.md` | `tests/test_streaming.py`, `tests/test_chat_client.py`, `tests/test_graph_observer.py` | Existing SSE emitter; no exclusive package | **PROMOTE (confirmed) — on since 2026-07-15 and active during every 2.3.7.4 arm with no adverse signal** |
| Graph observer | `on` (2026-07-15; code fallback `off`) | `src/middleware/graph_observer.py`, `src/middleware/corpus_graph.py`, `src/middleware/graph_api.py`, `src/middleware/graph_models.py`, `src/middleware/stream_events.py`, `src/middleware/app.py`, `src/middleware/static/graph/`, `src/middleware/config.py` | `enable_graph_observer`, graph/corpus bounded budgets, and `enable_phase2_3_corpus_projection`; `ENABLE_GRAPH_OBSERVER`, `GRAPH_*`, `CORPUS_*`, `ENABLE_PHASE2_3_CORPUS_PROJECTION` | `docs/CONFIGURATION.md` — Live retrieval-graph observer (2.2.7); `docs/ARCHITECTURE.md` — Live retrieval graph; `docs/FEATURE-FLAGS.md` | `tests/test_graph_observer.py`, `tests/test_graph_api.py`, `tests/test_graph_ui_contract.py`, `tests/test_corpus_graph.py`, `tests/test_corpus_scale.py`, `tests/test_chat_client.py` | Loopback-only local observer and vendored Cytoscape/font assets; no model/provider dependency | **PROMOTE (confirmed) — on since 2026-07-15 and active during every 2.3.7.4 arm with no adverse signal** |

## Profile assignment

All inventory capabilities are explicitly present in `configs/profiles/legacy.yaml`,
`configs/profiles/recommended.yaml`, and `configs/profiles/evaluation.yaml`.
`legacy` turns the post-2.2.2 optional surfaces off; `recommended` mirrors the
current committed enabled set; `evaluation` mirrors `recommended` and adds
trace/progress metadata. No profile row is a measured promotion claim.

## Dependency validation and rollback

`MiddlewareConfig` validates these relationships after profile/YAML/env loading:

| Dependent capability | Required prerequisite | Invalid explicit/env override | Invalid reviewed profile |
|---|---|---|---|
| `enable_deterministic_answers` | `enable_deterministic_tool_routing` and `enable_adaptive_rag` | Clamp dependent flag off and log a warning. | Hard error. |
| `enable_corrective_retry` | `enable_evidence_sufficiency` | Clamp dependent flag off and log a warning. | Hard error. |
| `enable_hierarchical_retrieval` | `sec.index_filing_text` | Clamp dependent flag off and log a warning. | Hard error. |
| `adaptive_enable_planning_call` | `enable_adaptive_rag` | Clamp dependent flag off and log a warning. | Hard error. |
| `enable_llm_rewrite_fallback` | `enable_conversation_rewrite` | Clamp dependent flag off and log a warning. | Hard error. |

The tested rollback is selecting `MIDDLEWARE_PROFILE=legacy`; it does not edit
or change the current defaults in `configs/middleware.yaml`.

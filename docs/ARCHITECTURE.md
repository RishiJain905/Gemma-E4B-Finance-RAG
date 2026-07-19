# Architecture

Gemma-E4B-Finance-RAG is a hybrid RAG system. Structured facts and free-text
documents are ingested from seven sources into two storage backends, then served
to a fine-tuned Gemma 4 E4B model through a FastAPI middleware that performs
intent parsing, hybrid retrieval, and prompt augmentation.

```
                          7 data sources
   SEC EDGAR · Yahoo Finance · FRED · GDELT · Earnings transcripts · IR pages · Estimates
                               │
                       Ingestion pipeline
                  (UnifiedScheduler, resilience layer)
                               │
                ┌──────────────┴──────────────┐
                ▼                              ▼
          SQLite                          ChromaDB
   fundamentals · filings ·        document embeddings
   cache_meta · ingestion_log ·    (cosine similarity,
   dead_letter                      TraceAlchemy embeddings)
                └──────────────┬──────────────┘
                               ▼
                   FastAPI middleware (:8000)
   intent_parser → retriever (hybrid) → prompt_augmenter → model call
                               │
                               ▼
              llama-server (:8087) — TraceAlchemy
              Gemma 4 E4B (chat + embeddings)
```

---

## Components

| Component | Module | Responsibility |
|-----------|--------|----------------|
| Storage facade | `src/storage/store.py` (`Store`) | Unifies SQLite + ChromaDB; hybrid `search()`, freshness reporting, filing pipeline support |
| Structured store | `src/storage/sqlite_store.py` (`SQLiteStore`) | Fundamentals, filing index, cache freshness, ingestion log |
| Vector store | `src/storage/chroma_store.py` (`ChromaStore`) | Document embeddings + custom `TraceAlchemyEmbeddingFunction`, chunking |
| Middleware app | `src/middleware/app.py` | FastAPI endpoints + query pipeline |
| Intent parser | `src/middleware/intent_parser.py` (`IntentParser`) | Extracts ticker, metrics, question type, timeframe |
| Retriever | `src/middleware/retriever.py` (`Retriever`) | Selects a retrieval strategy and queries both stores |
| Prompt augmenter | `src/middleware/prompt_augmenter.py` (`PromptAugmenter`) | Builds the grounded prompt from retrieval results |
| Unified scheduler | `src/scheduler/__init__.py` (`UnifiedScheduler`) | Runs explicit bootstrap, incremental refresh, repair, and retention controls with budgets/cursors/TTL tracking |
| Scheduler status | `src/scheduler/status.py` | SQLite-only run summaries and denominator-explicit coverage health |
| Ingestors | `src/ingestion/`, `src/macros/`, `src/sec/` | Per-source data fetching |
| Resilience utils | `src/utils/resilience.py` | Retry/backoff, circuit breaker, dead-letter queue |
| Lexical index | `src/middleware/lexical_index.py` (`LexicalIndex`) | BM25 keyword index over the ChromaDB corpus (Phase 2.1.2.1) |
| Re-ranker | `src/middleware/reranker.py` (`Reranker`) | Cross-encoder / LLM re-ranker of fused candidates (Phase 2.1.2.2) |

---

## Retrieval Pipeline (Phase 2.1.2)

Document retrieval runs an optional **hybrid + re-rank** pipeline, each stage
independently toggleable via `configs/middleware.yaml` (or env overrides
`ENABLE_LEXICAL`, `ENABLE_RERANKER`, `RERANKER_BACKEND`, …) for A/B eval:

```
query
  │
  1. vector search (ChromaDB, cosine)   ── top `rerank_candidates`
  2. BM25 lexical search (rank_bm25)    ── top `rerank_candidates`
  3. RRF fusion (retriever.rrf_fuse)     ── one ranked list
  4. cross-encoder re-rank (Reranker)    ── top `rerank_top_n` (= top_k_documents)
  │
  ▼  documents (+ facts from SQLite) → prompt augmenter
```

- **Vector channel** — `ChromaStore.search` (cosine similarity), ticker-filtered
  when a ticker is detected.
- **Lexical channel** — `LexicalIndex` (BM25Okapi) over the same corpus
  (`ChromaStore.iter_documents()`). Tokenizer: lowercase + alphanumeric runs,
  so tickers/product codes (`MI300X`, `CRWD`) stay intact. Documents that share
  no query token are dropped; BM25's negative-IDF scores (common terms) are kept
  and used only for ranking.
- **RRF** — `rrf_fuse(vector, lexical, k=rrf_k)`; an id ranked highly by both
  channels beats one ranked highly by only one.
- **Re-ranker** — `Reranker` with two backends: `cross-encoder`
  (`sentence-transformers`, default `cross-encoder/ms-marco-MiniLM-L-6-v2`,
  downloads on first use) or `llm` (the TraceAlchemy model on `:8087` scoring
  `(query, doc)` pairs). Lazy-loaded, cached, and falls back to the fused order
  on any load/score failure — a query never fails because of the re-ranker.

The `/query` response reports which path ran via `retrieval_strategy`:
`vector` | `hybrid` | `hybrid+rerank`. `/search` document results carry
`fusion_score` / `rerank_score` for inspecting ranking quality.

### Lexical index freshness

The BM25 index is built from the corpus and goes stale as documents are added.
The chosen strategy is **lazy rebuild on corpus-count change**:

- Built eagerly on middleware startup (`Retriever.warm_lexical_index()` in the
  lifespan) so the first query doesn't pay the build cost.
- On every search, `LexicalIndex._ensure()` compares `ChromaStore.count()` to
  the count at build time; if it changed (ingestion added/removed docs), the
  index rebuilds before scoring.
- `POST /refresh/{ticker}` additionally calls `Retriever.refresh_lexical_index()`
  to force an immediate rebuild after an on-demand refresh. Scheduler-driven
  ingestion is covered by the lazy count-change check on the next query.

If the corpus is empty or the build fails, `LexicalIndex.search` returns `[]`
and the retriever falls back to vector-only — the query still succeeds.

---

## Analytical Tool Calling (Phase 2.1.4)

Tool definitions live in `src/middleware/tools/`:

- `base.py` owns `Tool`, `REGISTRY`, `register()`, `openai_schema()`, and
  `dispatch_tool()`.
- `data_tools.py` registers the model-facing data tools. Read tools query
  cached SQLite/Chroma-backed data; the single write tool, `refresh_data`,
  reuses the middleware per-ticker refresh path.

`_call_model()` in `src/middleware/app.py` runs a bounded tool loop when
`config.enable_tools` is true:

```
prompt + tool schema
  |
  v
model response
  |
  +-- no tool_calls --> final answer
  |
  +-- tool_calls ----> dispatch_tool(call, store, ToolContext)
                         |
                         v
                      append role=tool JSON result
                         |
                         v
                      next model round-trip
```

The loop is capped by `max_tool_iterations`. Each query gets a fresh
`ToolContext` containing `allow_write_tools`, `max_refreshes_per_query`, and
the current refresh count. `dispatch_tool()` centrally enforces write gating
and the refresh budget before a write handler can run, so individual handlers
do not duplicate those checks.

The session-level `_tools_supported` flag prevents repeated bad tool attempts
against model backends that do not support tool calling. If the first tool
request returns an HTTP error mentioning tools, or the first tool-enabled
response has neither content nor tool calls, the middleware disables tools for
the session and falls back to the plain pre-tool payload shape.

Registered tools:

| Tool | Write? | Purpose |
|------|--------|---------|
| `list_metrics` | No | List available metric names and tickers. |
| `query_facts` | No | Rank, filter, or threshold fundamentals across tickers. |
| `get_fundamentals` | No | Return selected fundamentals for one ticker. |
| `search_documents` | No | Search qualitative document chunks. |
| `get_macro_snapshot` | No | Return cached macro indicators. |
| `get_sentiment` | No | Return recent GDELT sentiment for one ticker. |
| `get_guidance` | No | Return latest extracted earnings guidance. |
| `check_freshness` | No | Report per-source freshness for one ticker. |
| `refresh_data` | Yes | Refresh stale or never-fetched sources for one ticker. |

`refresh_data` is intentionally narrow: source aliases are normalized through
`_normalize_sources()`, an explicitly invalid or unbounded request (for example
`"all"`, a universe scope, or more than eight sources) returns a structured
error, and the target is always one ticker. The handler preflights the same
source registry, durable request budgets, and persisted provider cooldowns
used by the scheduler. Scheduler-managed sources (`sec_filings`,
`earnings_transcripts`, `ir_pages`) are reported as skipped rather than being
expanded into a broad run. The human-facing `/refresh` endpoint keeps its
original wider behavior.

---

## Answer Policy

The middleware uses a graded grounding policy for `/query` answers. Retrieval
counts determine the grounding hint passed to the model: three or more combined
facts/documents is `grounded`, one or two is `partial`, and zero is `none`.

The model prompt then selects one of four response modes:

| Mode | When used | Required behavior |
|------|-----------|-------------------|
| `grounded` | Sufficient retrieved facts/documents are present | Answer from the retrieved data and cite sourced claims with `[Source: ...]`. |
| `partial` | Some relevant data is present | Answer only what the retrieved data supports and explicitly state what is missing. |
| `general` | No relevant data is present and `allow_general_fallback=true` | Use stable background knowledge only when clearly prefixed with `Not from your data - general knowledge:` and include a caveat to verify against a primary source. |
| `refused` | The ask is genuinely unknowable/unsafe, or no data is present while general fallback is disabled | Decline briefly instead of guessing. |

Hard rule across every mode: never invent specific numbers such as prices,
P/E ratios, targets, revenue, margins, growth rates, dates, or counts. Specific
figures must come from retrieved context or model-callable tools.

The policy is configurable in `configs/middleware.yaml` and environment
overrides:

- `answer_policy: graded|strict` controls whether the graded prompt is used.
  `strict` preserves the previous context-only system prompt for rollback and
  regression comparison.
- `allow_general_fallback: bool` lets deployments disable general-knowledge
  fallback while still allowing partial grounded answers.

The `/query` response reports the actual answer path as `grounding`:
`grounded`, `partial`, `general`, or `refused`.

> **Phase 2.2.4 augmentations (additive, flag-gated).** When
> `enable_evidence_sufficiency` is on, the count-based grounding hint above is
> replaced by a deterministic, route-aware sufficiency assessment
> (`sufficient|borderline|missing`) that can trigger one bounded corrective
> retrieval (`enable_corrective_retry`); the response carries an
> `evidence_sufficiency` block. When `answer_validation` is `report`/`enforce`,
> every model-visible fact/document is assigned a request-local `[E#]` id, a
> deterministic (stdlib + `Decimal`, no model call) validator checks citation
> support and specific financial numbers, and `evidence_citations` +
> `answer_validation` blocks are attached (see *Corrective Retrieval &
> Provenance* below). With every flag off, the answer path is byte-identical to
> the graded policy described here.

---

## Conversational & Adaptive Query Path (Phase 2.2.2–2.2.4)

These layers wrap the existing retrieval/answer pipeline. Each is feature-flagged
and falls soft to the legacy single-turn, single-plan path; all are **off by
default** (except `answer_validation: report`, which is metadata-only).

### Conversational query understanding (2.2.2)

The middleware stays stateless: the client (`scripts/chat.py`) owns a bounded
conversation history and sends it per request (`QueryRequest.history`, a flat
list of `ChatTurn`). The server uses at most `conversation_max_turns` /
`conversation_max_history_chars` of it and never persists anything; `session_id`
is tracing metadata only, never a server-side lookup key. The current question
has its own independent 16,000-char cap and is **never silently truncated** — an
over-limit question is an HTTP 422. When `enable_conversation_rewrite` is on and
history is present, the current turn + bounded history are compiled into a
**separate** standalone retrieval query (`retrieval_query`) while the raw
question is left byte-for-byte unchanged; deterministic entity/metric/timeframe
carryover populates `carried_context` / `resolved_*`, and
`enable_llm_rewrite_fallback` adds at most one bounded model call only when a
slot stays ambiguous. Responses carry a `conversation` block
(`history_turns_received/used`, `history_truncated`, `topic_reset`).

### Adaptive orchestration (2.2.3)

When `enable_adaptive_rag` is on, `src/middleware/adaptive_orchestrator.py`
routes a request through **fast / standard / complex** lanes under one shared
execution budget (≤ 3 subqueries incl. sq0, ≤ 2 retrieval rounds, ≤ 1 planning
call, ≤ 1 re-rank call, deterministic-tool cap) and one context-character
budget. Fast and standard lanes make no pre-answer model call; only the complex
lane may make one optional compact planning call
(`adaptive_enable_planning_call`). Deterministically recognized safe finance
operations are routed to the existing read-only tools **before** any model call
(`enable_deterministic_tool_routing`), and a write tool can never be selected by
that router. Every adaptive stage falls soft to `Retriever.retrieve()`; the
response carries an `orchestration` block of the **actual executed** counters
(not configured maxima).

### Corrective retrieval & provenance (2.2.4)

- **Evidence sufficiency & bounded retry (2.2.4.1).** A deterministic grader
  classifies coverage of the plan's obligations as `sufficient|borderline|
  missing`, answering immediately when covered, allowing **one** internal
  corrective retrieval when borderline, and returning an honest partial/refusal
  when missing. Total retrieval rounds stay ≤ 2.
- **Selective decomposition & weighted fusion (2.2.4.2).** In the complex lane,
  a genuinely compound or low-coverage plan is decomposed into at most two
  derived, drift-validated subqueries, retrieved through their appropriate
  modality and fused by weighted RRF (original query the strongest signal:
  `sq0=1.0`, derived `0.8`, planner `0.6`) with slot reservation.
- **Citation provenance & numeric validation (2.2.4.3).** Every model-visible
  fact/document/tool result/calculation gets a stable request-local `[E#]` id;
  the answer is expected to cite those ids. A stdlib+`Decimal` validator (never
  a model call) resolves `[E#]` and legacy `[Source: …]` citations and checks
  specific financial numbers (currency/percent/signed/ratio/K-M-B-T-scaled)
  against cited evidence. `answer_validation: report` attaches metadata without
  changing the answer; `enforce` additionally downgrades grounded→partial or
  refuses a wholly-unsupported answer. The validator always fails soft to
  `validation_status=report_unavailable`.

---

## Runtime: Streaming & Caching (Phase 2.2.6)

Additive, flag-gated, and local-safe; all off by default.

- **Tool-aware final streaming & progress events (2.2.6.1).** The bounded
  tool/planning rounds run non-streaming, then only the final answer synthesis
  is streamed, so enabling tools no longer disables streaming for the whole
  request. `enable_tool_final_streaming` makes `POST /query/stream` serve a
  tools-enabled request (off → it 404s while tools are enabled).
  `enable_stream_progress_events` emits versioned, **redacted** SSE progress
  events (`query_started`, `stage`, `tool_started`, `tool_completed`, `error`) —
  stage names, safe tool names, statuses, and optional row counts only, never
  prompts, tool arguments, or document text. `/health` advertises the honest
  effective capability (`streaming`, `streaming_tool_final`).
- **Versioned retrieval cache & prompt efficiency (2.2.6.2).** A monotonic
  `store_revision` is bumped before every model-visible mutation. When
  `enable_retrieval_cache` is on, a thread-safe LRU+TTL cache reuses a planned
  request's **pre-prompt** evidence keyed on the compiled query + validated plan
  + config/model fingerprint + revision, so any ingestion write (including a
  same-count section replacement) invalidates it exactly; a miss or internal
  error is always just a miss. Prompt sections are emitted in a stable order with
  a prefix digest for reuse, and `llama_cache_prompt` optionally sends
  llama-server's `cache_prompt: true` with a capability fallback. **No semantic
  final-answer cache exists** — volatile finance answers are never reused by
  similarity; only versioned retrieval evidence and immutable date-bounded
  results are cacheable.

---

## Configuration Profiles (Phase 2.3.7.4)

`src/middleware/config.py` resolves middleware settings in this order: code
defaults, a selected profile, explicit YAML values, and environment overrides.
The committed `configs/middleware.yaml` selects `recommended`, which matches
the currently enabled source-of-truth set. `legacy` is a complete Phase 2.2
rollback with Phase 2.2.3+ optional capabilities off; `evaluation` keeps the
recommended flags but records trace/progress metadata for promotion arms and is
not a production default. Profile files are validated as reviewed bundles;
explicit YAML/environment dependency violations are clamped off with a logged
warning. This keeps each arm reproducible while preserving one explicit,
tested rollback path.

## Data Sources

| Source | Module | Storage target | Scheduler cadence | TTL (hours) |
|--------|--------|----------------|-------------------|-------------|
| Yahoo Finance (fundamentals + news) | `src/ingestion/yfinance_ingestor.py` | SQLite (facts) + ChromaDB (news) | daily | 24 (fundamentals), 6 (news) |
| SEC EDGAR filings (10-K/10-Q/8-K) | `src/sec/` | SQLite (filings) + ChromaDB (filing text) | daily (discovery) + weekly (full pipeline) | 12 |
| FRED macro indicators | `src/macros/fred_ingestor.py` | SQLite (MACRO facts) | daily | 24 |
| GDELT global news | `src/macros/gdelt_ingestor.py` | ChromaDB + sentiment facts | hourly | 6 |
| Earnings transcripts | `src/macros/earnings_transcripts.py` | ChromaDB + guidance | weekly | 168 |
| Company IR pages | `src/macros/ir_ingestor.py` | ChromaDB | daily | 24 |
| Analyst estimates and price targets | `src/macros/estimates_ingestor.py` | SQLite (forward-dated facts) | daily | 24 |

TTLs are read from the `schedule` block in `configs/watchlist.yaml` and fall
back to built-in defaults in `Store._DEFAULT_TTLS` and
`UnifiedScheduler._load_ttls`.

### Forward-looking estimate facts

Analyst consensus data is stored in the same `fundamentals` table as realized
facts, but every row uses `source_type="estimates"` and
`period_type="estimate"`. Forecast periods carry an `E` suffix:

| Metric family | Period format | Example |
|---------------|---------------|---------|
| Quarterly revenue/EPS estimates | `YYYY-QnE` | `2026-Q3E` |
| Fiscal-year revenue/EPS estimates | `FY{year}E` | `FY2027E` |
| Price targets, analyst counts, recommendation mean | `YYYY-MME` | `2027-07E` |

The horizon is encoded in the metric name, for example
`estimate_revenue_current_q`, `estimate_revenue_next_y`, and
`price_target_mean`. This avoids mixing horizons under one metric, which is
important because latest-fact reads use string-sortable `MAX(period)` semantics.
The `E` suffix and distinct metric names keep analyst forecasts separate from
realized historical facts.

Quarter and year estimate periods are derived from the current calendar date.
That is a practical approximation for companies whose fiscal calendar differs
from the calendar year; source rows remain clearly labeled as estimates.

---

## Ingestion Flow

The `UnifiedScheduler` (invoked via `python -m src.scheduler <mode>`) drives
all ingestion:

- **Run modes** (`src/scheduler/__init__.py`):
  - `daily` → `yfinance`, `fred`, `sec_filings` (discovery), `ir_pages`
  - `hourly` → `gdelt`
  - `weekly` → `earnings_transcripts`, `sec_filings` (full pipeline, `deep_sec=True`)
  - `all` → every source whose TTL has expired
  - `status` → freshness/run report for every source
- **Staggered execution** — sources run sequentially in ascending `weight`
  order with an `inter_source_delay` (default 2.0 s) between them.
- **Freshness gating** — unless `--force` is passed, a source is skipped when
  its scheduler cache entry is still within TTL. Scheduler cadence is tracked
  in `cache_meta` under the synthetic ticker `"SCHEDULER"` with source
  identifiers of the form `unified:<source>` (e.g. `unified:yfinance`).
- **Partial-failure isolation** — one source failing does not abort the run;
  the failure is logged, the scheduler cache entry is marked stale, and (best
  effort) the item is pushed to the dead-letter queue.

Each ingestor marks its per-ticker cache entry fresh in `cache_meta` on
success, which feeds the freshness reports surfaced by the middleware.

### Operational run modes (Phase 2.3.4.3)

The four operational modes are deliberately separate:

| Mode | Behavior | Safety boundary |
|---|---|---|
| `bootstrap` | Explicit, resumable current-universe plus bounded SEC, company-news, market, and macro population | Partition checkpoints; no broad full periodic-filing backfill |
| `daily` / legacy incremental modes | TTL/cursor/overlap-driven refresh through the source registry | Missing keys and open circuits remain skipped/stale, never fresh |
| `repair` | Re-indexes stored pending/error narratives and stored SEC artifacts | No provider download; bounded filters and item limit |
| `retention` | Previews and explicitly applies configured narrative expiry | Daily refresh never invokes it; apply uses the displayed ID set |

Every operation writes one `scheduler_runs` row and one
`scheduler_run_sources` row per requested source. The rows retain policy and
registry revisions, timings, counts, cursor/quota/freshness snapshots, and a
redacted terminal error class/message. `bootstrap_partitions` is the durable
resume ledger: the manifest is created with the run header, and resumed source
counts are aggregated from completed checkpoints. Historical terminal runs
are pruned to the configured bound without deleting active/resumable runs.

`status` combines those rows with `cache_meta` compatibility freshness and the
SQLite-only `build_coverage_health()` projection. It reports active-security
denominators by index/scope/sector, recent news/market/SEC coverage, source
partition states, indexing backlog, corpus date bounds, and last successful
refresh. It does not invoke network, Chroma embedding, or provider code.

### Source registry, budgets, and provider circuits (Phase 2.3.4.1–2.3.4.2)

`configs/sources.yaml` replaced the scheduler's static source dictionary with
a validated, data-driven registry. `SourceRegistry.load()`
(`src/scheduler/source_registry.py`) parses each entry into a `SourceSpec`
(`capability_group`, `scope`, `cadence`/`run_modes`, `priority`,
`dependencies`, `cursor_kind`, `overlap`, `requests_per_minute/day/run`,
`batch_size`, `max_work_items_per_run`, `retry_policy`, `ttl_key`/`ttl_hours`,
optional `required_env`); an invalid entry or unknown dependency is converted
to a disabled `status: invalid_configuration` spec instead of raising, so one
bad definition never crashes registry load or another source's run
(`SourceSpec.is_available`). `UnifiedScheduler` reads `self.SOURCES` from the
loaded registry rather than a hard-coded dict.

- **Incremental cursors.** `CursorManager` (`src/scheduler/cursors.py`)
  persists one transactional `source_cursors` row per source+partition and
  orders partitions fair-oldest-first on resume; each source's `overlap`
  window (e.g. `6h`, `2d`) is re-fetched on top of the last cursor so a
  late-arriving record is never missed by an incremental run.
- **Durable budgets.** `RunBudget` (`src/scheduler/budget.py`) enforces
  per-run minute/day/run request and work-item caps *before* work starts, and
  persists attempted/successful counts across process restarts via
  `Store.get_source_budget_usage` / `record_source_budget_usage`. `--force`
  bypasses only freshness (TTL) checks — it never bypasses registry
  availability, request budgets, or an open provider circuit.
- **Error taxonomy.** `src/ingestion/errors.py` normalizes every adapter
  failure into one of eight `ErrorClass` values: `authentication`,
  `entitlement`, `rate_limited`, `quota_exhausted` (provider-wide — stop only
  that source), `transient`, `contract`, `item` (source-local), and
  `permanent`. Only `rate_limited`/`transient` are retryable. `ProviderError`
  carries a `safe_message` (credential/token-redacted before it reaches logs,
  `scheduler_run_sources`, or the DLQ) and a parsed retry window
  (`parse_retry_after`, accepting numeric seconds or an HTTP-date Retry-After
  header).
- **Provider circuits.** A provider-wide failure opens a per-source circuit
  persisted in `source_circuit_state` with its cooldown reset timestamp; the
  circuit is consulted on every later invocation, including a fresh process,
  until the reset passes — so an exhausted or unentitled provider stops being
  retried without ever blocking a different source in the same run. Failures
  are additionally recorded in a deduplicated bounded dead-letter queue.

---

## Storage Layer

### SQLite (`data/finance.db`)

WAL journaling and foreign keys are enabled per connection. The schema is
loaded from `docs/phase1.2/schema.sql` if present, otherwise from the inline
DDL in `SQLiteStore._inline_schema()`.

### Ordered schema migration and Phase 2.3 backfill

`SQLiteStore` distinguishes a brand-new database from an existing application
database. New databases bootstrap from the canonical (or inline fallback)
final DDL. Existing databases retain their Phase 2.2 tables and are upgraded by
five standard-library SQL migrations in dependency order:

```text
identities -> corpus bridge -> observations/events -> refresh state -> indexes
```

Each applied file is checksum-protected in `schema_migrations`. Conditional
legacy columns (`security_id`, filing index metadata, retention accounting)
are added transactionally by the runner because SQLite has no portable
`ADD COLUMN IF NOT EXISTS`. No migration deletes a row or Chroma family.

The explicit `scripts/migrate_phase2_3.py` command performs data backfill. It
creates deterministic legacy security/corpus IDs, attaches a CIK only when the
ticker-to-CIK mapping is unique, and links legacy structured tables through
nullable canonical security IDs. Chroma is read metadata-only and one SQLite
ledger row is created per stable family; unchanged text is never re-embedded.
Ambiguous/orphan records enter `identity_reconciliation_errors`. Stage cursors
and a Store revision increment commit with every bounded batch.

All Phase 2.3 rollout controls are additive and default off. Disabling a
capability stops new scheduling or selects the Phase 2.2 query/projection path;
it never hides or destroys stored evidence. Operational rollback is flags-only.

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `fundamentals` | Structured financial metrics | `ticker`, `metric`, `value`, `unit`, `period`, `period_type`, `source_type`, `source_url`, `ingested_at`; `UNIQUE(ticker, metric, period)` |
| `filings` | SEC filing index / processing state | `ticker`, `filing_type`, `filing_date`, `period`, `accession` (UNIQUE), `source_url`, `file_path`, `status` (`unprocessed`/`parsed`), `summary_embedding_id` |
| `cache_meta` | Per-(ticker, source) freshness/TTL | `ticker`, `source`, `metric_scope`, `last_updated`, `next_scheduled_update`, `status` (`fresh`/`stale`/`fetching`), `error_message`; `PRIMARY KEY (ticker, source, metric_scope)` |
| `scheduler_runs` | Bounded operation headers | `run_id`, `mode`, policy/config revisions, timings, source rollups, safe terminal error |
| `scheduler_run_sources` | Per-source operation summaries | counts, cursor before/after, quota remaining, freshness/next due/cooldown, safe error |
| `bootstrap_partitions` | Resumable bootstrap manifest/checkpoints | `run_id`, `source`, `partition_key`, status, attempts, item/new/updated/duplicate counts |
| `schema_migrations` | Ordered schema history | version, name, SHA-256 checksum, applied timestamp |
| `identity_reconciliation_errors` | Review queue for unresolved legacy identities | stable id, stage/table/row, identifier, issue type, candidates |
| `phase2_3_backfill_progress` | Resumable metadata migration cursors | stage, cursor, completion, processed count |
| `source_circuit_state` | Additive provider circuit/cooldown state | source, status, failures, cooldown, safe error metadata |
| `ingestion_log` | Audit log of ingestion runs | `run_id`, `ticker`, `source`, `status`, `items_processed`/`items_new`/`items_updated`, `started_at`, `completed_at`, `duration_seconds` |
| `dead_letter` | Persistently failing items (lazily created) | `source`, `item_key`, `error`, `failed_at`, `retry_count`, `last_error`; `PRIMARY KEY (source, item_key)` |

> The `dead_letter` table is created on first use by
> `DeadLetterQueue._ensure_table()` in `src/utils/resilience.py`, not by the
> base schema.

### ChromaDB (`data/chroma`)

- `PersistentClient` collection `tracealchemy_docs`, configured for cosine
  similarity (`hnsw:space: cosine`).
- Embeddings are produced by `TraceAlchemyEmbeddingFunction`, which POSTs text
  to the llama-server `/v1/embeddings` endpoint (`model: tracealchemy`,
  mean-pooled) in batches of 10.
- Chunking is **structure-aware by default** (Phase 2.1.3,
  `chunking.strategy: structural` in `configs/storage.yaml`): text is split on
  SEC section markers / markdown headings and whole sentences are packed up to
  `max_chars` (1000) with sentence-based overlap — no mid-sentence cuts. The
  legacy `fixed` strategy (a ~1000-char sliding window with 150-char overlap on
  whitespace boundaries) remains available for rollback. Each chunk is stored as
  `"{id}#{i}"` with `parent_id`, `chunk_index`, `chunk_count`, and (under the
  structural strategy) `section` metadata. Shorter documents are stored as a
  single entry under their original id, except SEC section families
  (`source == "sec_filing"`), which always keep the `#i` child scheme so the
  filing → section → child hierarchy (2.2.5.2) stays addressable.
- The embedding vector dimension is determined at runtime by the model served
  on `:8087` (ChromaDB infers it from the embedding function's first response).
  The `embedding_dimension` field in `configs/storage.yaml` is descriptive
  only and is not enforced by the code.

---

## Hierarchical Retrieval & Authoritative Facts (Phase 2.2.5)

Two additive, independently gated capabilities improve long-document evidence
and structured-fact authority without changing the default query path.

### Filing → section → child hierarchy

SEC filings are indexed as a natural hierarchy (2.2.5.2): a filing (`accession`)
holds sections (`parent_id = sec:{accession}:{section_key}`, `section_index`),
each split into child chunks (`chunk_index`). `src/middleware/hierarchical_retrieval.py`
(`expand_filing_hits`) reconstructs *just enough* local context around a precise
child hit:

- the exact hit and its section heading are always kept;
- at most `hierarchy_max_siblings` same-section neighbors are added when the hit
  begins or ends mid-sentence/table (one preceding, one following);
- at most `hierarchy_max_adjacent_sections` adjacent sections are added, only
  when a requested obligation matches the adjacent heading or the evidence grader
  reports missing local context;
- shared parent/sibling chunks are deduplicated across hits, expansion stops
  before the shared context character budget is exceeded, and **an entire filing
  parent is never returned**.

Every expanded item preserves the root hit's original retrieval score plus
`expansion_reason` and `root_hit_id`. The expander is wired into the adaptive
orchestrator's `EXPAND_PARENT_SECTION` corrective seam (upgraded from the
2.2.4.1 sibling-only reader) and exposed as `Retriever.hierarchical_expand`. All
store reads fail soft per item — a query never errors because an expansion read
failed. Feature flag: `enable_hierarchical_retrieval` (off by default).

### Authoritative CompanyFacts preference

When `configs/sec_companyfacts.yaml` is enabled, `Store.companyfacts_evidence`
projects filed GAAP observations into structured fact-evidence rows and
`evidence.reconcile_structured_facts` merges them into structured retrieval:
an exact SEC concept/unit/period/as-of match wins for filed GAAP facts,
estimates never overwrite realized facts, and legacy Yahoo fundamentals fill
unsupported or more-current market fields. Conflicting values remain **separate
evidence items** with source/period/unit; the grader discloses the conflict and
never averages. The merge happens only at retrieval/evidence normalization — the
legacy `fundamentals` table is never rewritten, so rollback stays possible.

### Backfill, promotion gates, and rollback

`scripts/index_sec_filing_text.py` migrates existing parsed artifacts into the
section index (dry-run-first, pilotable, resumable via a source-hash manifest,
idempotent, and `--backup`-reversible; no parser-model call). The offline
long-document eval (`eval/run_eval.py::evaluate_long_document_configs` over
`tests/fixtures/sec/hierarchical_corpus.json`) compares flat, flat-larger-top-k,
and hierarchical retrieval.

**Promotion gate** (`metrics.long_document_gate`) — enable hierarchical expansion
by default only when, on the long-document corpus:

- Recall@10 improves at least 8 points over the flat baseline;
- context precision regresses no more than 0.02;
- answer correctness regresses no more than 0.02;
- packed prompt characters are no higher than the larger-top-k config (the
  hierarchical path must reach that recall without the top-k prompt cost);
- retrieval p95 adds no more than 20% and the backfill restore is tested;
- no ingestion-time model call is added.

**Rollback** — set `enable_hierarchical_retrieval: false` to return to the
sibling-only corrective behavior; if the index itself must be reversed, restore
the Chroma backup snapshot taken by the backfill's `--backup`. CompanyFacts is an
independent additive source — disable it via `configs/sec_companyfacts.yaml`
(`enabled: false`) with no effect on the hierarchy path or vice versa.

---

## Model

| Attribute | Value |
|-----------|-------|
| Name | `TraceAlchemy-Gemma-4-E4B-Finance-IT` (id `tracealchemy`) |
| Family | Gemma 4 E4B |
| Architecture | Mixture-of-Experts (MoE) |
| Parameters | ~9B total / ~2.6B active |
| Quantization | Q8_0 |
| Backend | `llama-server` (TurboQuant `llama.cpp` build) |
| Host / port | `127.0.0.1:8087` |
| Max context | 131072 tokens |
| Modes | chat, completion, embedding (single server) |
| GPU | AMD RDNA3 (gfx1101), `-ngl 99` |
| Speculative decoding | Multi-Token Prediction (MTP) with an F16 draft head |

Endpoints used by this project:

- `POST /v1/chat/completions` — answer generation (`MiddlewareConfig.llama_endpoint`)
- `POST /v1/embeddings` — document embeddings (`MiddlewareConfig.embedding_endpoint`)
- `GET /health` — liveness probe used by `_check_model_health()`

The embeddings endpoint requires `--embeddings --pooling mean` on the server.

---

## Query Flow (`POST /query`)

1. **Intent parsing** — `IntentParser.parse()` extracts ticker (symbol or
   company name), financial metrics, question type (`fact_lookup`,
   `comparison`, `trend`, `explanation`, `sentiment`, `news`, `risk`, or
   `general`), and timeframe.
2. **Freshness check** — `_evaluate_and_refresh()` reads the per-ticker
   freshness report. If `request.refresh` is true (default), sources that are
   *present but expired* (`stale`) are re-ingested before answering;
   `never_fetched` sources are left to the scheduler. When `refresh` is false,
   stale sources are flagged in the response's `freshness.warning`.
3. **Hybrid retrieval** — `Retriever.retrieve()` selects a strategy from the
   intent and queries both stores:
   - `facts_only` — fact_lookup with ticker + metrics (SQLite)
   - `hybrid` — ticker present (SQLite facts + ChromaDB docs)
   - `documents_only` — sentiment / news / risk (ChromaDB)
   - `comparison` — multi-ticker (both stores per ticker)
   - `macro` / `macro_hybrid` — macro keywords detected (FRED facts ± company)
   - `broad` — no ticker detected (search everything)
4. **Prompt augmentation** — `PromptAugmenter.build_prompt()` assembles the
   retrieved facts and documents into a grounded prompt.
5. **Model call** — if `/health` on the model server returns 200, the prompt
   is sent to `/v1/chat/completions` with a finance-assistant system prompt.
   Inline `[Source: type/ticker]` citations are parsed out of the response; when
   `answer_validation` is on, request-local `[E#]` evidence-id citations are
   resolved and validated as well.
6. **Response** — `QueryResponse` returns the answer, citations, detected
   ticker/intent, counts of facts/documents used, latency, `model_available`,
   and the freshness metadata block, plus any flag-gated additive blocks
   (`conversation`, `orchestration`, `evidence_sufficiency`,
   `evidence_citations`, `answer_validation`, `retrieval_query`).

The steps above describe the legacy single-turn path, which remains the default.
When the Phase 2.2 flags are enabled, a conversational query compiler (2.2.2)
runs before step 1, adaptive lane selection (2.2.3) wraps steps 3–5, and the
evidence sufficiency grader / corrective retry / citation-numeric validator
(2.2.4) wrap steps 4–6 — see *Conversational & Adaptive Query Path* above.
`POST /query/stream` runs the same pipeline and streams the final answer (2.2.6.1).

---

## Failure Modes & Recovery

| Failure | Handling |
|---------|----------|
| Transient network error during ingestion | `retry_with_backoff` decorator (`src/utils/resilience.py`) — exponential backoff with jitter, configurable `max_attempts`/`base_delay`/`max_delay` |
| Repeatedly failing external API | `CircuitBreaker` — CLOSED → OPEN after `failure_threshold` consecutive failures; OPEN rejects calls until `recovery_timeout`, then HALF_OPEN test calls decide CLOSED/OPEN |
| Persistently failing item | `DeadLetterQueue` — stored in the `dead_letter` SQLite table with `retry_count` and `last_error`; reviewable via `get_pending()`, cleared via `retry()` on success |
| One ingestion source fails mid-run | `UnifiedScheduler._run_sources` isolates per-source failures: logs the error, marks the scheduler cache entry stale, best-effort DLQ push, continues with remaining sources |
| `llama-server` unavailable | `/query` checks `/health` first; on failure returns a **degraded** answer (`_format_degraded_answer`) built from raw retrieved facts/documents with `model_available=false` |
| Embedding server down during `/search` | `search()` exceptions are caught; the endpoint returns empty results instead of erroring |
| Stale data (past TTL) | Tracked in `cache_meta`. `/query` can auto-refresh stale sources; `/freshness/{ticker}` reports per-source status; `/refresh/{ticker}` triggers on-demand re-ingestion |
| SQLite backend down | `/health` reports `status: "degraded"` when `storage.sqlite` is false; the store `heartbeat()` reflects both backends |

Scheduler-managed refreshes (SEC filings, earnings transcripts, IR pages) are
routed through `UnifiedScheduler._run_source` so TTL tracking and error
handling stay consistent with cron-driven runs; other sources are ingested
directly (`_refresh_one_source_direct`).

## Live Retrieval Graph Observer (Phase 2.2.7)

A local, read-only observability side channel that visualizes the retrieval
pipeline without touching it. **Disabled by default** (`enable_graph_observer`),
served loopback-only. This is an *observability graph*, not a retrieval
architecture — it never changes what is retrieved or how the answer is produced.
See `docs/phase2.2/ARCHITECTURE-DECISION.md` for the explicit
observability-graph-vs-GraphRAG distinction.

### Observer side channel

- **One instrumentation seam.** The pipeline already emits a single stream of
  redacted `QueryEvent`s (`src/middleware/stream_events.py`, 2.2.6.1). The
  observer subscribes an extra callback to that same emitter
  (`make_event_observer`), so the pipeline is instrumented **once** and the same
  events feed both the chat SSE progress line and the graph. `/query` and
  `/query/stream` install the emitter (with the graph observer attached) only
  when the flag is on; otherwise no emitter is created and behavior is
  byte-identical.
- **Projection.** `event_graph_deltas` (`src/middleware/graph_observer.py`)
  projects each event into bounded, allowlisted graph deltas: nodes for the
  query, validated plan and its subqueries, executed stages, tools, retrieved
  evidence and its source, and the terminal answer/citations/validation; edges
  for the `compiled_to`/`contains`/`routed_to`/`retrieved`/`from_source`/
  `expanded_from`/`supports`/`cited_by`/`corrected_by`/`validated_as` relations.
  Node/edge ids are prefixed by `query_id`, so two concurrent queries never share
  elements.
- **TraceHub.** A bounded, in-memory, non-blocking store/broadcaster. It clamps
  trace count, total elements, and per-trace TTL; compacts repeated upserts;
  fans out deltas to SSE subscribers **without awaiting** them (a slow/full
  subscriber is marked for reset, never blocking a publish); and fails soft — if
  observation ever raises it disables itself and discards future deltas rather
  than affecting the query. Publishing is synchronous and measured p95 < 2 ms.
  Nothing is persisted.
- **Redaction by construction.** Questions become a bounded preview + SHA-256
  digest; evidence bodies are bounded excerpts; a metadata allowlist plus
  secret/local-path scrubbing runs on every node/edge; source links are kept only
  when `http`/`https`.

### Corpus projection

`src/middleware/corpus_graph.py` (`CorpusGraph`) projects the authoritative Store
inventory (sources, tickers, metrics, facts, filings, sections, document
families, freshness, scheduler sources) into the same bounded graph shape for the
explorer tab. It reads only the Store's read methods, never embeddings or the
model; every page is hard-bounded and keyed to `Store.retrieval_revision()` via
opaque cursors so a mid-browse ingestion write is detected (HTTP 409) rather than
silently mixing revisions. The overview is cached for a short TTL keyed by
revision.

### Phase 2.3 scale — source-aware Live Trace and Corpus Explorer

Both projections extend additively to the broad-universe corpus without
changing the observability-graph schema (`schema_version` stays `1`) or the
query path.

- **Source-aware Live Trace (2.3.5.1).** `evidence`/`source`/`citation` nodes
  carry additional allowlisted fields on top of the Phase 2.2 shape: security
  identity (`security_id`, `canonical_security`), index membership
  (`index_memberships`), `sector`, `source_category`, `authority_tier`,
  `coverage_tier`, provider vs. publisher (`provider`, `publisher`,
  `source_name`), `item_type`/`event_type`, filing `form`/`filing_item`/
  `exhibit`, date semantics (`published_at`, `effective_at`, `accessed_at`),
  and an `evidence_role` of `primary` or `corroborating` so the trace shows
  which source directly substantiates a claim versus which one corroborates
  it. Every field is an already-safe scalar, bounded date, or opaque id —
  never a key, payload, or full document body. The shared `stage:route` node
  identity is preserved through both the legacy and adaptive-fallback paths,
  and the Phase 2.2 golden trace fixture is untouched by the additive fields.
- **Aggregation-first Corpus Explorer (2.3.5.2–2.3.5.3).** The explorer opens
  on a bounded aggregate landing (market universe → index → sector → security
  → source category → item/event type → time bucket) instead of rendering the
  corpus directly, with a persistent, combinable facet rail backed by
  authoritative SQLite-only counts. `CorpusGraph`
  (`src/middleware/corpus_graph.py`) adds `aggregates()`, `facets()`,
  `groups()`, and `item_detail()` alongside the Phase 2.2.7.2 `overview()`/
  `search()`/`detail()`/`neighbors()`; every method is paged and
  revision-aware — a mid-browse ingestion write is detected and surfaced as
  HTTP 409 (`CorpusRevisionChanged`) rather than silently mixing revisions.
  URL-hash deep links hold the full filter state client-side (no server
  storage, no cookies). The bounded-scale fixture exercised in the offline
  gate seeds 600 securities across 11 sectors with overlapping index
  memberships and 100,000 corpus items, asserting warm p95 under 250 ms for
  overview/facet queries and under 300 ms for a filtered first page on the
  reference machine.

### Serving & security

`src/middleware/graph_api.py` mounts the read-only `/graph/api/*` router; the
static single-page UI (`src/middleware/static/graph/`, Cytoscape) is served from
the app. A single HTTP middleware is the chokepoint for every `/graph*` route:
it rejects non-loopback clients with 404 (no bypass flag) and stamps a strict
same-origin CSP + hardening headers. The UI is same-origin and self-contained
(no cookies/localStorage/service worker/analytics/third-party requests) and never
writes dynamic data through `innerHTML`. See `docs/API.md` for endpoints and the
event schema, and `docs/CONFIGURATION.md` for the flags and the explicit
no-remote-exposure rule.

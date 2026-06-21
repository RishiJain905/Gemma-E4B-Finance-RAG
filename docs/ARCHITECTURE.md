# Architecture

Gemma-E4B-Finance-RAG is a hybrid RAG system. Structured facts and free-text
documents are ingested from six sources into two storage backends, then served
to a fine-tuned Gemma 4 E4B model through a FastAPI middleware that performs
intent parsing, hybrid retrieval, and prompt augmentation.

```
                          6 data sources
   SEC EDGAR · Yahoo Finance · FRED · GDELT · Earnings transcripts · IR pages
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
| Unified scheduler | `src/scheduler/__init__.py` (`UnifiedScheduler`) | Orchestrates ingestion with staggered execution + TTL tracking |
| Ingestors | `src/ingestion/`, `src/macros/`, `src/sec/` | Per-source data fetching |
| Resilience utils | `src/utils/resilience.py` | Retry/backoff, circuit breaker, dead-letter queue |

---

## Data Sources

| Source | Module | Storage target | Scheduler cadence | TTL (hours) |
|--------|--------|----------------|-------------------|-------------|
| Yahoo Finance (fundamentals + news) | `src/ingestion/yfinance_ingestor.py` | SQLite (facts) + ChromaDB (news) | daily | 24 (fundamentals), 6 (news) |
| SEC EDGAR filings (10-K/10-Q/8-K) | `src/sec/` | SQLite (filings) + ChromaDB (filing text) | daily (discovery) + weekly (full pipeline) | 12 |
| FRED macro indicators | `src/macros/fred_ingestor.py` | SQLite (MACRO facts) | daily | 24 |
| GDELT global news | `src/macros/gdelt_ingestor.py` | ChromaDB + sentiment facts | hourly | 6 |
| Earnings transcripts | `src/macros/earnings_transcripts.py` | ChromaDB + guidance | weekly | 168 |
| Company IR pages | `src/macros/ir_ingestor.py` | ChromaDB | daily | 24 |

TTLs are read from the `schedule` block in `configs/watchlist.yaml` and fall
back to built-in defaults in `Store._DEFAULT_TTLS` and
`UnifiedScheduler._load_ttls`.

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

---

## Storage Layer

### SQLite (`data/finance.db`)

WAL journaling and foreign keys are enabled per connection. The schema is
loaded from `docs/phase1.2/schema.sql` if present, otherwise from the inline
DDL in `SQLiteStore._inline_schema()`.

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `fundamentals` | Structured financial metrics | `ticker`, `metric`, `value`, `unit`, `period`, `period_type`, `source_type`, `source_url`, `ingested_at`; `UNIQUE(ticker, metric, period)` |
| `filings` | SEC filing index / processing state | `ticker`, `filing_type`, `filing_date`, `period`, `accession` (UNIQUE), `source_url`, `file_path`, `status` (`unprocessed`/`parsed`), `summary_embedding_id` |
| `cache_meta` | Per-(ticker, source) freshness/TTL | `ticker`, `source`, `metric_scope`, `last_updated`, `next_scheduled_update`, `status` (`fresh`/`stale`/`fetching`), `error_message`; `PRIMARY KEY (ticker, source, metric_scope)` |
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
- Documents longer than `DEFAULT_CHUNK_CHARS` (1000 chars, 150-char overlap)
  are split into overlapping windows on whitespace boundaries; each chunk is
  stored as `"{id}#{i}"` with `parent_id`, `chunk_index`, and `chunk_count`
  metadata. Shorter documents are stored as a single entry under their original
  id.
- The embedding vector dimension is determined at runtime by the model served
  on `:8087` (ChromaDB infers it from the embedding function's first response).
  The `embedding_dimension` field in `configs/storage.yaml` is descriptive
  only and is not enforced by the code.

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
   Inline `[Source: type/ticker]` citations are parsed out of the response.
6. **Response** — `QueryResponse` returns the answer, citations, detected
   ticker/intent, counts of facts/documents used, latency, `model_available`,
   and the freshness metadata block.

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

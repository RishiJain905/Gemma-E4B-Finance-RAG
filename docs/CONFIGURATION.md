# Configuration

All runtime configuration lives in `configs/*.yaml`. Secrets are supplied via a
`.env` file in the project root and read from the process environment.

Most config loaders use a tolerant pattern: built-in defaults are applied
first, then any matching keys present in the YAML file override them. A missing
config file is therefore not fatal — the code falls back to its hard-coded
defaults.

---

## Environment Variables (`.env`)

| Variable | Required for | Used by | Notes |
|----------|--------------|---------|-------|
| `FRED_API_KEY` | FRED macro ingestion | `src/macros/fred_ingestor.py` | Resolution order: constructor arg → value of the `${FRED_API_KEY}` placeholder in `configs/fred.yaml` → `FRED_API_KEY` env var. Get a free key at <https://fred.stlouisfed.org/docs/api/api_key.html>. |
| `SEC_EDGAR_USER_AGENT` | SEC EDGAR access | `src/sec/edgar_fetcher.py` | SEC requires a descriptive User-Agent (name + contact email). Resolution order: constructor arg → `SEC_EDGAR_USER_AGENT` env var → `sec.user_agent` in `configs/storage.yaml` → built-in default. |

Example `.env`:

```
FRED_API_KEY=your_fred_api_key_here
SEC_EDGAR_USER_AGENT=Your Name your.email@example.com
```

---

## Symbol Catalog

`SymbolResolver` reads `data/symbol_catalog.json` when present. The file is a
generated cache with `generated_at`, `ttl_hours`, and SEC-backed entries. If the
catalog is missing, corrupt, or expired, query parsing still works from the
local ticker map; expired catalogs log a warning but remain usable.

Refresh the catalog with:

```bash
python scripts/refresh_symbol_catalog.py
```

The refresh command uses `SEC_EDGAR_USER_AGENT` first, then `sec.user_agent` in
`configs/storage.yaml`, then the built-in SEC fallback. Override fuzzy matching
with `RESOLVER_FUZZY_THRESHOLD=0.90` when stricter matching is desired.

---

## `configs/storage.yaml`

Storage layer configuration (SQLite, ChromaDB, embeddings, cache TTLs, SEC).

```yaml
sqlite:
  path: "data/finance.db"

chroma:
  path: "data/chroma"
  collection_name: "tracealchemy_docs"
  embedding_dimension: 2560   # informational; actual dim comes from the embedder

embedding:
  provider: "llama-server"
  endpoint: "http://127.0.0.1:8087/v1/embeddings"
  model: "tracealchemy"
  pooling: "mean"
  batch_size: 10

cache:
  default_ttl_hours: 24
  filing_ttl_hours: 168
  earnings_ttl_hours: 336

chunking:
  strategy: "structural"      # structural | fixed
  max_chars: 1000
  overlap_sentences: 1        # structural: sentences carried into the next chunk
  fixed_overlap_chars: 150    # fixed: legacy char overlap

sec:
  user_agent: "TraceAlchemy Research contact@tracealchemy.example.com"
```

| Field | Meaning |
|-------|---------|
| `sqlite.path` | Path to the SQLite database file. |
| `chroma.path` | ChromaDB persistence directory. |
| `chroma.collection_name` | Collection name used by `ChromaStore` (default `tracealchemy_docs`). |
| `chroma.embedding_dimension` | **Descriptive only.** ChromaDB infers the actual vector dimension at runtime from the embedding function's first response; this value is not enforced by the code. |
| `embedding.provider` / `endpoint` / `model` | Embedding backend (the local llama-server `/v1/embeddings`). |
| `embedding.pooling` | Must match the server's `--pooling` flag (`mean`). |
| `embedding.batch_size` | Texts per embedding request (default 10). |
| `cache.*_ttl_hours` | Descriptive cache TTL hints. Note: the operative per-source TTLs used by the scheduler and freshness logic come from `watchlist.yaml → schedule`, not this block. |
| `chunking.strategy` | **`structural` is the default** (Phase 2.1.3): splits on SEC section markers / markdown headings and packs whole sentences up to `max_chars` with sentence-based overlap — no mid-sentence cuts. `fixed` reproduces the legacy sliding character window (`max_chars` with `fixed_overlap_chars` overlap). |
| `chunking.max_chars` | Target chunk size in characters (default 1000). |
| `chunking.overlap_sentences` | Sentences carried into the next chunk under `structural` (default 1). |
| `chunking.fixed_overlap_chars` | Character overlap under `fixed` (legacy `DEFAULT_CHUNK_OVERLAP`, 150). |
| `sec.user_agent` | Fallback SEC EDGAR User-Agent if neither the constructor arg nor `SEC_EDGAR_USER_AGENT` is set. |

---

## `configs/sec.yaml`

SEC filing-text indexing is independently rollout-controlled. The default is
off, so filing processing retains its legacy behavior until explicitly enabled.

```yaml
sec:
  index_filing_text: false
  max_sections_per_filing: 200
  max_section_chars: 2000000
  index_forms: [10-K, 10-Q, 8-K]
```

| Field | Default | Meaning |
|-------|---------|---------|
| `sec.index_filing_text` | `false` | Persist and index actual SEC filing sections through the existing Chroma structural chunker. When false, the legacy filing path is unchanged. |
| `sec.max_sections_per_filing` | `200` | Safety bound on section parents accepted from one filing (hard-capped at 500). Excess sections are logged and skipped. |
| `sec.max_section_chars` | `2000000` | Parser sanity cap per section (hard-capped at 10,000,000). Oversized sections are skipped and logged; text is never silently truncated. |
| `sec.index_forms` | `[10-K, 10-Q, 8-K]` | Filing forms eligible for section indexing when the rollout flag is enabled. |

Parsed artifacts are stored beside the configured SQLite database under
`sec/parsed/`. A successful artifact whose vector write fails remains in the
additive `index_pending` state with its failure reason and is selected for a
later retry. Scheduler status reads persisted pending/section/chunk counters;
it does not call the embedding service.

### Backfill: `scripts/index_sec_filing_text.py` (Phase 2.2.5.3)

Migrates the **existing** parsed artifacts under `sec/parsed/` into the section
index — no parser-model call is made. `index_forms` above gates which forms are
eligible.

```bash
python scripts/index_sec_filing_text.py                     # dry run (default)
python scripts/index_sec_filing_text.py --ticker NVDA --limit 1 --apply
python scripts/index_sec_filing_text.py --apply \
    --resume-manifest data/sec/backfill-manifest.json --backup data/sec/backups
```

| Flag | Meaning |
|------|---------|
| *(none)* / `--dry-run` | Report eligible filings, parsed artifacts, estimated parent sections/chunks, missing files, and currently indexed section count. Writes nothing. |
| `--apply` | Index eligible filings one at a time through `Store.add_filing_sections`. |
| `--ticker` / `--accession` / `--limit` | Slice a pilot subset. |
| `--resume-manifest <path>` | Record each completed accession + source-file hash. A re-run skips unchanged input; a changed artifact replaces its section family. |
| `--backup <dir>` | Snapshot the Chroma collection directory (as `chroma-backup-<ts>/`) before the first write. |

Manifests (`*.backfill-manifest.json`) and backup snapshots (`chroma-backup-*/`)
are gitignored. An interruption leaves the last filing retryable and every prior
filing valid (the manifest is flushed after each filing; each family replace is
atomic). Rollback: disable `enable_hierarchical_retrieval` and, if the index
itself must be reversed, restore the Chroma backup snapshot.

---

## `configs/sec_companyfacts.yaml`

SEC CompanyFacts / XBRL structured ingestion and canonical metric projection
(Phase 2.2.5.1). Authoritative filed GAAP observations are stored in their own
`sec_companyfacts` table with full concept/unit/period/accession/fetch
provenance; the legacy `fundamentals` table is never rewritten. Disabled by
default — when off, the source makes zero HTTP calls and `Store.get_companyfacts`
returns `[]`.

```yaml
enabled: false
user_agent: "TraceAlchemy Research contact@tracealchemy.example.com"
allowed_taxonomies: [us-gaap]
allowed_forms: [10-K, 10-K/A, 10-Q, 10-Q/A]
timeout_seconds: 30
retries: 3
backoff_factor: 0.5
request_delay_seconds: 0.2
metrics:
  total_revenue:
    concepts: [RevenueFromContractWithCustomerExcludingAssessedTax, Revenues, SalesRevenueNet]
    units: [USD]
    period_kinds: [quarterly, ytd, annual]
  # ... net_income, diluted_eps, total_assets, total_liabilities,
  # stockholders_equity, operating_cash_flow, shares_outstanding
```

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `false` | Enable CompanyFacts ingestion + the canonical projection. Strict no-op when off. |
| `user_agent` | *(example)* | SEC-required descriptive User-Agent for `data.sec.gov`. |
| `allowed_taxonomies` | `[us-gaap]` | XBRL taxonomies accepted into the store. |
| `allowed_forms` | `[10-K, 10-K/A, 10-Q, 10-Q/A]` | Filing forms whose facts are ingested. |
| `timeout_seconds` / `retries` / `backoff_factor` / `request_delay_seconds` | `30` / `3` / `0.5` / `0.2` | HTTP timeout, retry count, backoff multiplier, and inter-request delay. |
| `metrics` | *(8 concepts)* | Canonical metric → ordered concept list (selection priority), accepted units, and period kinds. Raw observations are stored independently of these projection aliases. |

CompanyFacts preference at retrieval is described under **Hierarchical retrieval**
below and in `docs/ARCHITECTURE.md`; it needs no middleware flag. Rollback = set
`enabled: false`.

---

## `configs/middleware.yaml`

Loaded by `MiddlewareConfig` (`src/middleware/config.py`). Only keys that match
an existing attribute are applied; everything else is ignored.

```yaml
llama_endpoint: "http://127.0.0.1:8087/v1/chat/completions"
embedding_endpoint: "http://127.0.0.1:8087/v1/embeddings"
model_name: "tracealchemy"
default_temperature: 0.3
max_tokens: 2048
top_k_documents: 5
top_k_facts: 10
enable_citations: true
```

| Field | Default | Meaning |
|-------|---------|---------|
| `llama_endpoint` | `http://127.0.0.1:8087/v1/chat/completions` | Chat-completions endpoint for answer generation. |
| `embedding_endpoint` | `http://127.0.0.1:8087/v1/embeddings` | Embeddings endpoint passed through to the `Store`. |
| `model_name` | `tracealchemy` | Model id sent in the chat payload. |
| `default_temperature` | `0.3` | Temperature when the request does not override it. |
| `max_tokens` | `2048` | Max response tokens when the request does not override it. |
| `top_k_documents` | `5` | Max ChromaDB documents retrieved per query. |
| `top_k_facts` | `10` | Max SQLite facts retrieved per query. |
| `enable_citations` | `true` | Whether citation extraction is enabled. |

The remaining middleware keys are grouped by the phase that introduced them.
Every Phase 2.2.4–2.2.6 feature ships **disabled by default** (the one exception
is `answer_validation: report`, which is metadata-only and changes no answer).
For each, **rollback is to flip the flag back to its default** — none involves a
schema or storage migration. Any key can also be overridden by the environment
variable listed in its row (used for A/B evaluation).

#### Answer policy & grounding (Phase 2.1.7)

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `answer_policy` | `graded` | `ANSWER_POLICY` | `graded` uses the count-based grounding prompt; `strict` keeps the previous context-only system prompt for rollback/regression comparison. |
| `allow_general_fallback` | `true` | `ALLOW_GENERAL_FALLBACK` | Allow a clearly-labeled general-knowledge answer when no relevant data is retrieved; when `false`, an ungrounded ask is refused instead. |

#### Retrieval & re-ranking (Phase 2.1.2)

Hybrid document retrieval: a BM25 lexical channel fused with the vector channel
via RRF, then an optional cross-encoder / LLM re-ranker. Every stage fails soft —
a query never errors because a retrieval stage failed.

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `enable_lexical` | `true` | `ENABLE_LEXICAL` | Add the BM25 lexical channel and RRF fusion. |
| `rrf_k` | `60` | — | RRF constant; larger dampens the contribution of top ranks. |
| `enable_reranker` | `false` | `ENABLE_RERANKER` | Re-rank fused candidates. The `cross-encoder` backend downloads a HuggingFace model on first use. |
| `reranker_backend` | `cross-encoder` | `RERANKER_BACKEND` | `cross-encoder` (sentence-transformers) or `llm` (reuse TraceAlchemy on `:8087`, no download). |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `RERANKER_MODEL` | Cross-encoder model id. |
| `rerank_candidates` | `30` | `RERANK_CANDIDATES` | Candidate pool retrieved before re-ranking. |
| `rerank_top_n` | `5` | `RERANK_TOP_N` | Documents kept after re-ranking (= `top_k_documents`). |

#### Analytical tools (Phase 2.1.4)

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `enable_tools` | `true` | `ENABLE_TOOLS` | Advertise the read-mostly finance tools to the model on `/query`. |
| `max_tool_iterations` | `3` | `MAX_TOOL_ITERATIONS` | Hard cap on model tool-call rounds per query. |
| `allow_write_tools` | `true` | `ALLOW_WRITE_TOOLS` | Whether the single write tool (`refresh_data`) may run. |
| `max_refreshes_per_query` | `2` | `MAX_REFRESHES_PER_QUERY` | Per-query budget on refresh writes. |

#### Fetch-on-miss (Phase 2.1.6)

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `enable_fetch_on_miss` | `true` | `ENABLE_FETCH_ON_MISS` | Bounded live fetch for a never-seen ticker during a query. |
| `fetch_on_miss_timeout_s` | `10.0` | `FETCH_ON_MISS_TIMEOUT_S` | Timeout for that fetch. |
| `fetch_on_miss_per_query` | `1` | `FETCH_ON_MISS_PER_QUERY` | Max fetch-on-miss attempts per query. |

#### Streaming, timings & embedding cache (Phase 2.1.8)

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `enable_streaming` | `true` | `ENABLE_STREAMING` | Serve `POST /query/stream` (SSE). See tool-aware streaming (2.2.6.1) below for the tools-enabled interaction. |
| `return_timings` | `true` | `RETURN_TIMINGS` | Attach the per-stage `timings` breakdown to responses. |
| `embedding_cache_size` | `256` | `EMBEDDING_CACHE_SIZE` | LRU size of the in-process embedding cache. |

#### Conversation memory & follow-up rewriting (Phase 2.2.2)

The middleware stays stateless; the client owns the turns and sends them per
request. These bound how much of that client-sent history/question a single
request may use. Over-ceiling budget values are clamped with one warning
(`src/middleware/config.py`); an over-limit **question** is a validation error
(HTTP 422), never silently truncated.

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `conversation_max_turns` | `8` | `CONVERSATION_MAX_TURNS` | Max recent turns used per request (ceiling 50). |
| `conversation_max_history_chars` | `8000` | `CONVERSATION_MAX_HISTORY_CHARS` | Max total history chars used (ceiling 32000). |
| `conversation_max_question_chars` | `16000` | `CONVERSATION_MAX_QUESTION_CHARS` | Current-question hard cap (ceiling 16000; mirrors the pydantic `MAX_QUESTION_CHARS`). |
| `enable_conversation_rewrite` | `false` | `ENABLE_CONVERSATION_REWRITE` | Compile the current turn + bounded history into a **separate** retrieval query; the raw question is never changed. |
| `enable_llm_rewrite_fallback` | `false` | `ENABLE_LLM_REWRITE_FALLBACK` | One bounded, schema-validated model call only when deterministic compilation leaves a slot ambiguous. |
| `conversation_rewrite_timeout_s` | `15.0` | `CONVERSATION_REWRITE_TIMEOUT_S` | Timeout for that fallback call. |

#### Adaptive orchestration & deterministic tool routing (Phase 2.2.3)

When `enable_adaptive_rag` is on, a request runs through the fast/standard/complex
lanes under one shared execution budget and one context budget; every adaptive
stage falls soft to the legacy single-query `Retriever.retrieve()`. Budget values
outside the safe ranges are clamped with one warning.

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `enable_adaptive_rag` | `false` | `ENABLE_ADAPTIVE_RAG` | Route through the bounded adaptive orchestrator. |
| `adaptive_enable_planning_call` | `false` | `ADAPTIVE_ENABLE_PLANNING_CALL` | Allow one compact pre-answer planning model call (only when deterministic parsing cannot assign retrieval modes). |
| `adaptive_max_subqueries` | `3` | `ADAPTIVE_MAX_SUBQUERIES` | Max subqueries incl. sq0 (clamp 1–3). |
| `adaptive_max_retrieval_rounds` | `2` | `ADAPTIVE_MAX_RETRIEVAL_ROUNDS` | Max retrieval rounds incl. a corrective retry (clamp 1–2). |
| `adaptive_max_planning_calls` | `1` | `ADAPTIVE_MAX_PLANNING_CALLS` | Max planning calls (clamp 0–1). |
| `adaptive_max_context_chars` | `18000` | `ADAPTIVE_MAX_CONTEXT_CHARS` | Hard context character ceiling (clamp 1000–64000). |
| `adaptive_conditional_rerank` | `true` | `ADAPTIVE_CONDITIONAL_RERANK` | Let the complex lane invoke at most one re-rank call when justified. |
| `enable_deterministic_tool_routing` | `false` | `ENABLE_DETERMINISTIC_TOOL_ROUTING` | Route safe analytical/comparison/projection/calculation asks to read-only tools before any model call. |
| `max_deterministic_tools_per_query` | `3` | `MAX_DETERMINISTIC_TOOLS_PER_QUERY` | Cap on deterministic tool routes per query. |
| `enable_deterministic_answers` | `false` | `ENABLE_DETERMINISTIC_ANSWERS` | Let a fully-covered deterministic route answer from a template without a model call. |

#### Evidence sufficiency & corrective retrieval (Phase 2.2.4.1–2.2.4.2)

Replaces count-based grounding with a deterministic, route-aware sufficiency
assessment (`sufficient|borderline|missing`), allowing at most one internal
corrective retrieval when borderline. Total retrieval rounds stay capped at two.

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `enable_evidence_sufficiency` | `false` | `ENABLE_EVIDENCE_SUFFICIENCY` | Grade evidence coverage instead of item count; answer, retry once, or return an honest partial/refusal. |
| `enable_corrective_retry` | `false` | `ENABLE_CORRECTIVE_RETRY` | Permit the single bounded corrective retrieval when the grader reports `borderline`. |
| `max_corrective_retries` | `1` | `MAX_CORRECTIVE_RETRIES` | Hard-clamped to 0–1. |
| `enable_query_decomposition` | `false` | `ENABLE_QUERY_DECOMPOSITION` | In the complex lane, decompose a genuinely compound/low-coverage plan into ≤ 2 derived, drift-validated subqueries fused with the original (strongest signal). Not present in the committed YAML; defined in `src/middleware/config.py`. |

#### Citation provenance & numeric validation (Phase 2.2.4.3)

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `answer_validation` | `report` | `ANSWER_VALIDATION` | `off` = legacy byte-identical; `report` = render `[E#]` ids, run the deterministic (stdlib + `Decimal`) validator, attach validation metadata (no answer change); `enforce` = additionally downgrade grounded→partial or refuse a wholly-unsupported answer. Invalid value falls back to `off`. |
| `require_evidence_ids` | `false` | `REQUIRE_EVIDENCE_IDS` | Tighten enforcement so a specific-figure answer with no resolving `[E#]` counts as a violation. |

#### Tool-aware final streaming & progress events (Phase 2.2.6.1)

Keeps the bounded tool/planning rounds non-streaming, then streams the final
answer synthesis — so enabling tools no longer disables streaming for the whole
request. Both feature flags default off (behavior byte-identical to pre-2.2.6).

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `enable_tool_final_streaming` | `false` | `ENABLE_TOOL_FINAL_STREAMING` | Serve a tools-enabled `/query/stream` by running tool rounds non-streaming, then streaming only the final answer. Off → `/query/stream` 404s while tools are enabled. |
| `enable_stream_progress_events` | `false` | `ENABLE_STREAM_PROGRESS_EVENTS` | Emit versioned, redacted progress events (`query_started`, `stage`, `tool_started`, `tool_completed`, `error`) on the SSE stream. Off → only the legacy `token`/`metadata` events. |
| `stream_progress_include_counts` | `true` | `STREAM_PROGRESS_INCLUDE_COUNTS` | Include row/item counts on retrieve and `tool_completed` progress events. |

#### Versioned retrieval cache & prompt efficiency (Phase 2.2.6.2)

An in-memory cache of a planned request's **pre-prompt** evidence, keyed on the
compiled query + validated plan + config/model fingerprint + a monotonic
`store_revision`. Any ingestion write bumps the revision and invalidates the
cache exactly; a cache miss or internal error is always just a miss, never a
query failure. **No semantic final-answer cache exists** — volatile finance
answers (prices, news, estimates, "latest") are never reused by similarity.

| Field | Default | Env | Meaning |
|-------|---------|-----|---------|
| `enable_retrieval_cache` | `false` | `ENABLE_RETRIEVAL_CACHE` | Reuse pre-prompt retrieval evidence for an identical request at the same store revision. Explicit-refresh requests never look up the cache. |
| `retrieval_cache_max_entries` | `256` | `RETRIEVAL_CACHE_MAX_ENTRIES` | LRU entry cap. |
| `retrieval_cache_ttl_s` | `300` | `RETRIEVAL_CACHE_TTL_S` | Secondary staleness bound beneath the revision key (seconds). |
| `retrieval_cache_max_value_chars` | `200000` | `RETRIEVAL_CACHE_MAX_VALUE_CHARS` | Refuse to store a single larger evidence snapshot (bounds memory). |
| `llama_cache_prompt` | `false` | `LLAMA_CACHE_PROMPT` | Send llama-server's `cache_prompt: true` on the final answer request when the backend supports it; on rejection it disables for the process and retries the plain payload once (fail-soft). Off → the request JSON is byte-identical. |

### Hierarchical retrieval (Phase 2.2.5.3)

Bounded filing → section → child expansion of precise SEC hits. Off by default;
the corrective `EXPAND_PARENT_SECTION` seam keeps its 2.2.4.1 sibling-only
behavior until the long-document eval gates pass. An entire filing parent is
never injected into a prompt.

```yaml
enable_hierarchical_retrieval: false
hierarchy_max_siblings: 2            # clamp 0–4
hierarchy_max_adjacent_sections: 1  # clamp 0–3
hierarchy_max_expanded_items: 12    # clamp 0–50 (0 = bounded only by the budget)
```

| Field | Default | Meaning |
|-------|---------|---------|
| `enable_hierarchical_retrieval` | `false` | When on, a precise child hit is expanded with same-section neighbors and (obligation- or grader-gated) adjacent sections under the shared context character budget. |
| `hierarchy_max_siblings` | `2` | Max preceding/following same-section children added per hit to complete a sentence/table (clamp 0–4). |
| `hierarchy_max_adjacent_sections` | `1` | Max adjacent sections added per hit, only on an obligation-heading match or a grader missing-context signal (clamp 0–3). |
| `hierarchy_max_expanded_items` | `12` | Hard cap on total added neighbors across all hits, regardless of budget (clamp 0–50; `0` = bounded only by `adaptive_max_context_chars`). |

CompanyFacts preference (2.2.5.3) is governed by `configs/sec_companyfacts.yaml`
(`enabled: false` by default) and needs no middleware flag — when the source is
enabled, authoritative filed GAAP facts are merged into structured retrieval and
conflicting values are surfaced as separate evidence, never averaged.

---

## `configs/watchlist.yaml`

Tracked tickers and the per-source ingestion schedule. The `schedule` block is
the source of truth for TTLs used by both `UnifiedScheduler` and the `Store`
freshness logic (each falls back to built-in defaults if a key is missing).

```yaml
core:        [NVDA, AMD, AAPL, MSFT, META, CRWD]
extended:    [GOOGL, AMZN, TSLA, AVGO, ORCL, INTC, QCOM, PANW, PLTR, SNOW]
macro_tickers: [SPY, QQQ, TLT, GLD]

schedule:
  fundamentals: 24    # Re-fetch fundamentals daily
  news: 6             # Re-fetch news every 6 hours
  macro: 24           # Macro indicators daily
  sec_filings: 12     # Check for new SEC filings every 12 hours
  gdelt_news: 6       # GDELT global news every 6 hours
  transcripts: 168    # Earnings transcripts weekly (7 days)
  ir_pages: 24        # Company IR pages daily
```

| Field | Meaning |
|-------|---------|
| `core` | Tickers that get full ingestion (fundamentals + news + documents). |
| `extended` | Tickers that get fundamentals only, less frequently. |
| `macro_tickers` | Market-proxy ETFs fetched but not tied to a single company. |
| `schedule.<key>` | TTL in **hours** per logical source. Keys map to sources via `Store.FRESHNESS_SOURCES` and `UnifiedScheduler.SOURCES`. |

---

## `configs/fred.yaml`

FRED macro-indicator ingestion (`src/macros/fred_ingestor.py`).

```yaml
api_key: "${FRED_API_KEY}"   # placeholder resolved from the env var
request_delay: 0.25
max_retries: 3

indicators:
  GDP: "Gross Domestic Product"
  GDPC1: "Real GDP (Chained)"
  CPIAUCSL: "CPI (All Urban Consumers)"
  PCEPILFE: "Core PCE (Fed's preferred gauge)"
  T10YIE: "10-Year Breakeven Inflation Rate"
  FEDFUNDS: "Federal Funds Rate"
  DFF: "Federal Funds Rate (Daily)"
  DGS1: "1-Year Treasury Rate"
  DGS10: "10-Year Treasury Rate"
  DGS2: "2-Year Treasury Rate"
  T10Y2Y: "10-Year minus 2-Year Treasury Spread"
  UNRATE: "Unemployment Rate"
  PAYEMS: "Nonfarm Payrolls"
  IC4WSA: "Initial Jobless Claims (4-Week Avg)"
  HOUST: "Housing Starts"
  PERMIT: "Building Permits"
  UMCSENT: "Consumer Sentiment (U. of Michigan)"
  DSPIC96: "Real Disposable Personal Income"
  INDPRO: "Industrial Production"
  TCU: "Capacity Utilization"
```

| Field | Meaning |
|-------|---------|
| `api_key` | The literal `${FRED_API_KEY}` placeholder; the ingestor substitutes the env var of that name. |
| `request_delay` | Seconds between FRED API requests. |
| `max_retries` | Retry attempts on transient failure. |
| `indicators` | Map of FRED series id → human-readable label. Each is stored under the synthetic ticker `MACRO`. |

---

## `configs/gdelt.yaml`

GDELT 2.0 global-news ingestion (`src/macros/gdelt_ingestor.py`).

```yaml
max_records: 250
lookback_days: 7
request_delay: 5.0
max_retries_on_429: 2
gkg_max_files_per_fetch: 24
use_title_tone_fallback: true

max_finance_domains: 5
finance_domains: [reuters.com, bloomberg.com, cnbc.com, marketwatch.com,
  seekingalpha.com, finance.yahoo.com, wsj.com, ft.com, investing.com,
  barrons.com, economist.com, nytimes.com]

financial_topics: [FINANCE, MARKETS, ECON_MACRO, ECON_INDICATOR,
  CORPORATE, REGULATION, TRADE]
```

| Field | Meaning |
|-------|---------|
| `max_records` | Max articles per query (GDELT DOC API limit). |
| `lookback_days` | How far back to search on the initial run. |
| `request_delay` | Seconds between requests (GDELT DOC API: max 1 req / 5 s). |
| `max_retries_on_429` | Retries after HTTP 429, with exponential backoff (`delay * 2^n`). |
| `gkg_max_files_per_fetch` | Max GKG 15-minute CSV files per enrichment pass. |
| `use_title_tone_fallback` | Use a lexicon fallback when a GKG URL match is missing. |
| `max_finance_domains` | Caps the OR-clause size (the full domain list exceeds the DOC API query-length limit). |
| `finance_domains` | Candidate financial-news domains (only the first `max_finance_domains` are used per query). |
| `financial_topics` | GDELT GCAM topic codes used to filter results. |

---

## `configs/ir.yaml`

Company investor-relations page ingestion (`src/macros/ir_ingestor.py`).

```yaml
request_delay: 3.0
max_retries: 3
timeout: 30
user_agent: "TraceAlchemy Research (contact@tracealchemy.example.com)"

doc_types: [press_release, presentation, earnings_material, event]
max_items_per_ticker: 20
```

| Field | Meaning |
|-------|---------|
| `request_delay` | Seconds between per-ticker fetches. |
| `max_retries` | Retry attempts on transient failure. |
| `timeout` | HTTP request timeout in seconds. |
| `user_agent` | User-Agent header for IR-page requests. |
| `doc_types` | Document categories to store. |
| `max_items_per_ticker` | Cap on items fetched per ticker per run. |

---

## `configs/model.yaml` + `configs/model.local.yaml`

TraceAlchemy model + `llama-server` settings. Committed defaults live in
`configs/model.yaml`; machine-specific paths belong in
`configs/model.local.yaml` (gitignored — copy from `configs/model.example.yaml`).
The `serve_model.ps1` / `serve_model.sh` wrappers load and merge these files.

Resolution order: `model.yaml` → deep-merge `model.local.yaml` → substitute
`${ENV_VAR}` placeholders (`MAIN_MODEL_PATH`, `LLAMA_BUILD_DIR`,
`DRAFT_MODEL_PATH`). The middleware reads its endpoints from
`middleware.yaml`, not these files.

```yaml
model:
  name: "TraceAlchemy-Gemma-4-E4B-Finance-IT"
  id: "tracealchemy"
  backend: "llama-server"

  host: "127.0.0.1"
  port: 8087
  base_url: "http://127.0.0.1:8087"
  platform: "windows"

  family: "Gemma 4 E4B"
  architecture: "Mixture-of-Experts (MoE)"
  parameters_total: "~9B"
  parameters_active: "~2.6B"
  quantization: "Q8_0"

  gpu_vendor: "AMD"
  gpu_arch: "RDNA3 (gfx1101)"
  gpu_layers: 99
  gpu_memory: 16

  max_context: 131072
  max_tokens_default: 2048
  max_tokens_max: 8192
  supported_modes: [chat, completion, embedding]

  speculative_decoding:
    enabled: true
    type: "mtp"                  # Multi-Token Prediction
    draft_block_size: 3
    draft_quant: "F16"

  cache:
    type_key: "q8_0"
    type_value: "turbo4"

  cpu_threads: 8

  endpoints:
    chat_completions: "/v1/chat/completions"
    completions: "/v1/completions"
    embeddings: "/v1/embeddings"

  defaults: { temperature: 0.7, top_p: 0.95, top_k: 40, repeat_penalty: 1.1 }

  tasks:
    fact_extraction: { temperature: 0.1, max_tokens: 1024 }
    general_qa:      { temperature: 0.7, max_tokens: 512 }
    analysis:        { temperature: 0.5, max_tokens: 2048 }

  embedding:
    enabled: true
    endpoint: "http://127.0.0.1:8087/v1/embeddings"
    dimension: 2560       # informational; not enforced by code
    normalize: true
    pooling: "mean"

  paths:
    main_model: "${MAIN_MODEL_PATH}"   # override in model.local.yaml
    build_dir: "${LLAMA_BUILD_DIR}"
    binary: "bin/llama-server.exe"
```

| Section | Meaning |
|---------|---------|
| `name` / `id` / `backend` | Display name, API model id (`tracealchemy`), and serving backend. |
| `host` / `port` / `base_url` / `platform` | Network location of the model server. |
| `family` / `architecture` / `parameters_*` / `quantization` | Gemma 4 E4B MoE, ~9B total / ~2.6B active params, Q8_0. |
| `gpu_*` | GPU hints (`gpu_layers: 99` offloads all layers; adjust for your hardware). |
| `max_context` / `max_tokens_*` / `supported_modes` | 131072-token context; chat, completion, and embedding modes from one server. |
| `speculative_decoding` | Multi-Token Prediction (MTP) with an F16 draft head, block size 3. |
| `cache.type_key` / `type_value` | KV-cache quantization (`q8_0` keys / `turbo4` values). |
| `cpu_threads` | CPU threads for the server. |
| `endpoints` | OpenAI-compatible paths exposed by llama-server. |
| `defaults` / `tasks` | Default sampling params and per-task overrides. |
| `embedding.dimension` | `2560` — the TraceAlchemy embedding output dimension (informational; not enforced by code). |
| `paths` | Model/build/binary paths — set in `model.local.yaml`, not committed. |

> **Note on embedding dimension:** the served TraceAlchemy model emits
> **2560-dimensional** embeddings (verified against the running server).
> `storage.yaml` and `model.yaml` both record `2560` for reference, but the
> value is **informational only** — ChromaDB derives the true dimension from
> the embedding function's first response at runtime, so the configs are never
> enforced.

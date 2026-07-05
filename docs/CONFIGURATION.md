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
| `sec.user_agent` | Fallback SEC EDGAR User-Agent if neither the constructor arg nor `SEC_EDGAR_USER_AGENT` is set. |

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

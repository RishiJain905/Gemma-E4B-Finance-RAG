# Gemma-E4B-Finance-RAG

A hybrid Retrieval-Augmented Generation (RAG) system for financial research. It
ingests data from six sources, stores structured facts and document embeddings
in dual backends, and answers natural-language financial questions using a
fine-tuned **Gemma 4 E4B** model (codename *TraceAlchemy*) served locally by
`llama-server`.

Built on top of
[trjxter/TraceAlchemy-Gemma-4-E4B-Finance-IT-gguf](https://huggingface.co/trjxter/TraceAlchemy-Gemma-4-E4B-Finance-IT-gguf),
a finance-specialized GGUF model. The model server runs on **Windows + AMD
RDNA3** (gfx1101) via a TurboQuant build of `llama.cpp`. The FastAPI middleware
and ingestion pipeline are cross-platform (Windows/Linux).

---

## What it does

1. **Ingests** financial data from six sources (SEC EDGAR filings, Yahoo
   Finance, FRED macro indicators, GDELT global news, earnings-call
   transcripts, and company investor-relations pages).
2. **Stores** it in two complementary backends:
   - **SQLite** — structured facts, filing index, cache freshness, ingestion
     audit log, and a dead-letter queue.
   - **ChromaDB** — document embeddings produced by the TraceAlchemy
     `/v1/embeddings` endpoint (cosine similarity).
3. **Answers** questions through a FastAPI middleware that parses intent, runs
   hybrid retrieval (facts + documents), augments a prompt, and calls the model
   for a grounded, cited answer. When the model is unavailable it returns a
   **degraded** answer built from the raw retrieved data.

---

## Quick Start

### 1. Clone and create a virtual environment

```bash
git clone <repo-url> Gemma-E4B-Finance-RAG
cd Gemma-E4B-Finance-RAG
python -m venv .venv
```

Activate it:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root with:

```
FRED_API_KEY=your_fred_api_key_here
SEC_EDGAR_USER_AGENT=Your Name your.email@example.com
```

- `FRED_API_KEY` — required for FRED macro ingestion. Get one free at
  <https://fred.stlouisfed.org/docs/api/api_key.html>.
- `SEC_EDGAR_USER_AGENT` — SEC EDGAR requires a descriptive User-Agent that
  identifies the requester (name + contact email). See
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full resolution order.

### 4. Configure model paths (for `serve_model` scripts)

Copy the template and set paths for your machine:

```powershell
copy configs\model.example.yaml configs\model.local.yaml
```

Edit `configs/model.local.yaml` — at minimum set `paths.main_model`,
`paths.build_dir`, and (if using MTP) `speculative_decoding.draft_model_path`.
This file is gitignored. Alternatively, export `MAIN_MODEL_PATH`,
`LLAMA_BUILD_DIR`, and `DRAFT_MODEL_PATH` in your environment.

### 5. Start the model server (`llama-server`) on port 8087

The middleware expects an OpenAI-compatible `llama-server` on
`http://127.0.0.1:8087` with **embeddings enabled** (the same server provides
both chat completions and embeddings).

On Windows + AMD RDNA3, use the provided wrapper:

```powershell
.\scripts\serve_model.ps1 start
```

Or start it manually (any platform):

```bash
llama-server -m /path/to/gemma-4-E4B-it.Q8_0.gguf \
    --host 127.0.0.1 --port 8087 \
    -c 131072 -ngl 99 \
    --embeddings --pooling mean
```

> The `--embeddings` and `--pooling mean` flags are required — the
> `ChromaStore` embedding function posts to `/v1/embeddings` and expects
> mean-pooled vectors.

### 6. Start the middleware on port 8000

```powershell
# Windows: starts the FastAPI middleware (assumes llama-server is already up)
.\scripts\start_stack.ps1
```

```bash
# macOS / Linux (also starts llama-server if a model is found)
./scripts/start_stack.sh
```

Or run `uvicorn` directly:

```bash
uvicorn src.middleware.app:app --host 0.0.0.0 --port 8000
```

Interactive API docs are then available at <http://127.0.0.1:8000/docs>.

### 7. Ingest data

Run the unified scheduler to populate the stores. `daily` runs Yahoo Finance,
FRED, SEC filing discovery, and IR pages; `--force` skips the freshness checks:

```bash
python -m src.scheduler daily --force
```

Other modes: `hourly` (GDELT news), `weekly` (earnings transcripts + full SEC
pipeline), `all` (everything that is stale), and `status` (freshness report).

### 8. Ask a question

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was NVDA revenue last quarter?"}'
```

### Interactive chat (recommended)

For a single, centralized entry point, use the interactive client. It
**auto-starts the middleware** if it isn't already running, gives you a chat
loop over `/query` (streamed by default), and surfaces every Phase 2.1/2.2
feature — grounding mode, tool calls, retrieval strategy, fetch-on-miss,
resolved-ticker confirmation, bounded conversation history, multiline questions,
and streamed progress events — in one consistent answer renderer:

```bash
python scripts/chat.py
```

```
Capabilities: tools=on streaming=off answer_policy=graded

you> What was BB revenue last quarter?
  interpreting as BlackBerry Limited / BB

BlackBerry revenue is 143,000,000.00 usd (2026-Q1) [Source: yfinance/BB].
  [GROUNDED]
  ticker=BB intent=fact_lookup facts=10 docs=0 model_available=True latency=3803.2ms retrieval=hybrid
  used: query_facts

you> /refresh            # run ALL ingestion jobs (scheduler all --force)
you> /refresh NVDA       # refresh a single ticker via the API
you> /refresh daily      # run a specific scheduler mode
you> /health             # middleware + freshness summary
you> /tools              # list model-callable tools
you> /grounding strict   # force the strict answer policy for this session
you> /eval 10            # run the eval harness (10 cases) against this server
you> /ticker NVDA        # pin a ticker for following questions
you> And how about AMD?  # follow-ups reuse this session's bounded history
you> /history            # preview this conversation's turns (local, no API call)
you> /new                # start a fresh conversation (clear history, new session id)
you> /history off        # run single-turn (stop sending/recording history)
you> /graph              # open the live retrieval graph (if the observer is enabled)
you> /help               # full command list
you> /quit               # stops the middleware if this script started it
```

It still requires `llama-server` on `:8087`; it warns and falls back to
degraded answers if the model is unreachable. See
[`scripts/CHAT.md`](scripts/CHAT.md) for every command and metadata tag the
renderer can show.

### Live retrieval graph (optional, Phase 2.2.7)

An optional, **local read-only** visualization of the retrieval pipeline (query →
plan → stages → tools → evidence → answer) plus a corpus explorer. It is
**disabled by default** and served **loopback-only** with a strict same-origin
CSP — start the middleware with `ENABLE_GRAPH_OBSERVER=1` to enable it, then open
`http://127.0.0.1:8000/graph` or run `/graph` in the chat client (`--open-graph`
opens it at startup). It is an *observability* view and never changes retrieval
or answers. Remote exposure is intentionally unsupported. See
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) and [`docs/API.md`](docs/API.md).

### Broad-universe ingestion (optional, Phase 2.3.4–2.3.6)

Beyond the six deep-coverage sources above, ingestion can also run through a
validated **source registry** (`configs/sources.yaml`) that adds per-source
cadence, request/day budgets, incremental cursors, and provider circuit
breakers on top of the same failure-isolation guarantee — one exhausted,
rate-limited, or failing source never blocks another. It optionally extends
coverage to broad company events, company news, market data, and official
macro/regulatory/sector feeds across the wider S&P 500 / Nasdaq-100 universe.
Every broad-universe capability and the additive schema migration that backs
it are **disabled by default**; enabling a capability only adds new evidence,
it never changes the existing schema or query path.

The scheduler adds explicit `bootstrap` / `repair` / `retention` operational
modes alongside `daily`/`hourly`/`weekly`/`all`, plus a truthful `status`
report that never claims a disabled or rate-limited source is fresh:

```bash
python -m src.scheduler bootstrap --source sec_filings --since 2026-06-01
python -m src.scheduler status --json
python -m src.scheduler repair --source ir_pages --limit 50
python -m src.scheduler retention --preview
```

The optional live graph above also gains an **aggregation-first Corpus
Explorer** (market universe → index/sector/source groups → bounded on-demand
expansion, never a full-corpus render) and a **source-aware Live Trace** that
attributes each piece of evidence to its security identity, index membership,
authority tier, and provider-vs-publisher role. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the registry schema,
capability flags, and migration.

---

## Architecture

```
                          6 data sources
   SEC EDGAR · Yahoo Finance · FRED · GDELT · Earnings transcripts · IR pages
                               │
                       Ingestion pipeline
                  (UnifiedScheduler, resilience layer)
                               │
                ┌──────────────┴──────────────┐
                ▼                              ▼
          SQLite (fundamentals,          ChromaDB
          filings, cache_meta,           (document embeddings,
          ingestion_log,                  cosine similarity)
          dead_letter)
                └──────────────┬──────────────┘
                               ▼
                   FastAPI middleware (:8000)
        intent parsing → hybrid retrieval → prompt augmentation
                               │
                               ▼
              llama-server (:8087) — TraceAlchemy
              Gemma 4 E4B (chat + embeddings)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full breakdown of
components, the storage schema, the query flow, and failure-mode handling.

---

## Project Structure

```
Gemma-E4B-Finance-RAG/
├── configs/                  # YAML configuration (see docs/CONFIGURATION.md)
│   ├── storage.yaml          # SQLite + ChromaDB + embedding + SEC settings
│   ├── middleware.yaml       # Endpoints, model name, retrieval top-k
│   ├── watchlist.yaml        # Tracked tickers + per-source TTL schedule
│   ├── fred.yaml             # FRED indicators + request settings
│   ├── gdelt.yaml            # GDELT query/domain/topic filters
│   ├── ir.yaml               # Company IR-page ingestion settings
│   └── model.yaml            # TraceAlchemy model + llama-server settings
├── docs/                     # Documentation (this README links here)
│   ├── ARCHITECTURE.md
│   ├── CONFIGURATION.md
│   └── API.md
├── scripts/                  # Operational + smoke-test scripts
│   ├── serve_model.ps1 / .sh # Launch llama-server (TraceAlchemy)
│   ├── start_stack.ps1 / .sh # Launch the middleware (+ model on Unix)
│   ├── stop_stack.ps1 / .sh
│   ├── seed_test_data.py     # Seed demo data into the stores
│   └── validate_setup.py     # Environment / dependency sanity check
├── src/
│   ├── storage/              # Dual storage layer
│   │   ├── store.py          # Store facade over SQLite + ChromaDB
│   │   ├── sqlite_store.py   # Structured facts, filings, cache, audit log
│   │   └── chroma_store.py   # Vector store + TraceAlchemy embedding function
│   ├── middleware/           # FastAPI query router
│   │   ├── app.py            # Endpoints + query pipeline
│   │   ├── config.py         # MiddlewareConfig (from middleware.yaml)
│   │   ├── models.py         # Pydantic request/response schemas
│   │   ├── intent_parser.py  # Ticker / metric / question-type extraction
│   │   ├── retriever.py      # Hybrid retrieval strategy selection
│   │   └── prompt_augmenter.py
│   ├── scheduler/            # Unified ingestion orchestrator
│   │   ├── __init__.py       # UnifiedScheduler (daily/hourly/weekly/all)
│   │   └── __main__.py       # CLI: python -m src.scheduler <mode>
│   ├── ingestion/
│   │   └── yfinance_ingestor.py
│   ├── macros/
│   │   ├── fred_ingestor.py
│   │   ├── gdelt_ingestor.py
│   │   ├── ir_ingestor.py
│   │   └── earnings_transcripts.py
│   ├── sec/                  # SEC EDGAR discovery + parsing pipeline
│   │   ├── edgar_fetcher.py
│   │   ├── filing_parser.py
│   │   ├── filing_processor.py
│   │   └── scheduler.py      # FilingScheduler
│   └── utils/
│       ├── logging.py
│       └── resilience.py     # retry/backoff, CircuitBreaker, DeadLetterQueue
├── tests/                    # pytest suite
├── data/                     # SQLite DB + ChromaDB persistence (gitignored)
├── requirements.txt
└── pytest.ini
```

---

## Configuration

All runtime configuration lives in `configs/*.yaml`, with secrets supplied via
`.env` (`FRED_API_KEY`, `SEC_EDGAR_USER_AGENT`). Each config file and every
field is documented in **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.

---

## API Reference

The middleware exposes the following endpoints (full schemas and examples in
**[docs/API.md](docs/API.md)**):

| Method | Path                   | Purpose                                          |
|--------|------------------------|--------------------------------------------------|
| GET    | `/`                    | Service info + links                             |
| GET    | `/health`              | Storage, model, scheduler, freshness, capabilities |
| POST   | `/query`               | Full RAG pipeline — grounded, cited answer       |
| POST   | `/query/stream`        | SSE variant of `/query` (streamed final answer, 2.2.6.1) |
| POST   | `/search`              | Raw hybrid search (bypasses the model)           |
| GET    | `/freshness/{ticker}`  | Per-source freshness report for a ticker         |
| POST   | `/refresh/{ticker}`    | On-demand re-ingestion of stale sources          |
| GET    | `/macro/snapshot`      | Key macro indicators (cached FRED data)          |
| GET    | `/sentiment/{ticker}`  | GDELT sentiment summary                          |
| GET    | `/guidance/{ticker}`   | Latest earnings guidance                         |

---

## Development & Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

`tests/test_phase2_3_end_to_end.py` is an offline harness that drives a seeded
pilot slice (ORCL, NVDA, AAPL, plus one healthcare/automotive/government-
contractor security and a small macro catalog) through bootstrap, repeated
refresh, and simulated failure modes with no network or model, scored against
the Phase 2.3 golden set (`tests/fixtures/evaluation/phase2_3_golden.json`).
The golden set's `answerable` classes are asserted against the labeled
evidence ledger; its Phase 2.3.7-dependent classes are explicitly marked
`deferred` rather than silently omitted.

Tests that hit the live SEC EDGAR network and a running `llama-server` on
`:8087` are marked `live` (see `pytest.ini`). Skip them with:

```bash
pytest tests/ -v -m "not live"
```

Sanity-check your environment before running the stack:

```bash
python scripts/validate_setup.py
```

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for code style, branch, and PR
conventions.

---

## License

MIT

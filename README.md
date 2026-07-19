# Gemma-E4B-Finance-RAG

A hybrid Retrieval-Augmented Generation (RAG) system for financial research. It
uses a registry-driven ingestion pipeline spanning company, market, macro,
regulatory, sector, and deep-research sources; stores structured facts and
document embeddings in dual backends; and answers natural-language financial
questions using a fine-tuned **Gemma 4 E4B** model (codename *TraceAlchemy*)
served locally by `llama-server`.

Built on top of
[trjxter/TraceAlchemy-Gemma-4-E4B-Finance-IT-gguf](https://huggingface.co/trjxter/TraceAlchemy-Gemma-4-E4B-Finance-IT-gguf),
a finance-specialized GGUF model. The model server runs on **Windows + AMD
RDNA3** (gfx1101) via a TurboQuant build of `llama.cpp`. The FastAPI middleware
and ingestion pipeline are cross-platform (Windows/Linux).

---

## What it does

1. **Ingests** a source-aware research corpus for the S&P 500 / Nasdaq-100
   universe. Registered sources include SEC EDGAR and CompanyFacts, Yahoo
   Finance, Finnhub, Massive market data and Massive News, FRED, Federal
   Reserve, Treasury, BLS, BEA, EIA, New York Fed, CFTC, openFDA, NHTSA,
   USAspending, earnings-call transcripts, estimates, and company IR pages.
   The GDELT adapter and previously stored GDELT evidence remain supported,
   but scheduled GDELT ingestion is currently disabled. See
   [Phase 2.3 ingestion](#phase-23-broad-market-ingestion).
2. **Stores** it in two complementary backends:
   - **SQLite** — structured facts, filing index, cache freshness, ingestion
     audit log, a dead-letter queue, and a persistent **FTS5 lexical index**
     for hybrid search (startup in seconds, not minutes — no more in-memory
     BM25 materialization of the whole corpus).
   - **ChromaDB** — document embeddings produced by the TraceAlchemy
     `/v1/embeddings` endpoint (cosine similarity). A 100,000-chunk storage
     benchmark (Phase 2.3.7.6) passed all nine hard latency/correctness gates
     for this pairing, so no replacement store is planned — see
     `docs/phase2.3/2.3.7-rag-quality-speed-and-storage/STORAGE-BENCHMARK-RESULTS.md`.
3. **Answers** questions through a FastAPI middleware that parses intent,
   adaptively routes the query through one of four lanes (fast / standard /
   complex / catalog), runs hybrid retrieval (facts + documents), augments a
   prompt, and calls the model for a grounded, cited answer. Safe analytical
   questions are routed to read-only tools deterministically, and when a
   question's answer is fully covered by typed evidence the middleware skips
   generation entirely and returns a deterministic answer directly — measured
   ~94–96% faster than a generated answer on eligible queries. Capability
   questions ("what tickers/sources do you cover?") are answered
   deterministically from the `describe_coverage` tool rather than guessed by
   the model. When the model is unavailable, `/query` returns a **degraded**
   answer built from the raw retrieved data instead of failing.

---

## Quick Start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/RishiJain905/Gemma-E4B-Finance-RAG.git
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

```dotenv
FRED_API_KEY=your_fred_api_key_here
SEC_EDGAR_USER_AGENT=Your Name your.email@example.com

# Optional Phase 2.3 providers — each missing key disables only its source
FINNHUB_API_KEY=your_finnhub_api_key_here
MASSIVE_API_KEY=your_massive_api_key_here
BLS_API_KEY=your_bls_api_key_here
BEA_API_KEY=your_bea_api_key_here
EIA_API_KEY=your_eia_api_key_here
OPENFDA_API_KEY=your_openfda_api_key_here
```

- `FRED_API_KEY` — required for FRED macro ingestion. Get one free at
  <https://fred.stlouisfed.org/docs/api/api_key.html>.
- `SEC_EDGAR_USER_AGENT` — SEC EDGAR requires a descriptive User-Agent that
  identifies the requester (name + contact email). See
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full resolution order.
- The Phase 2.3 provider keys are optional. A missing key produces a truthful
  `disabled_missing_key` status for that adapter without blocking any other
  source.

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

For a new database, run the explicit bootstrap once to populate enabled sources,
then inspect the source-by-source result:

```bash
python -m src.scheduler bootstrap
python -m src.scheduler status
```

After bootstrap, use incremental refreshes. Cadence, TTLs, overlap windows,
request budgets, and cursors come from `configs/sources.yaml`:

```bash
python -m src.scheduler daily
python -m src.scheduler hourly
python -m src.scheduler weekly
```

| Mode | Current responsibility |
|---|---|
| `daily` | Universe membership, Yahoo Finance, SEC discovery/CompanyFacts, Finnhub, Massive market data/actions, official macro and sector feeds, IR pages, and estimates. |
| `hourly` | Massive News. GDELT is registered but intentionally reports `disabled`. |
| `weekly` | Full SEC pipeline, earnings transcripts, and CFTC. |
| `all` | Every enabled registered source that is due; disabled or unavailable sources skip independently. |
| `status` | Read-only freshness, run history, quotas, circuits, and denominator-explicit coverage health. |

`--force` bypasses freshness checks only; it still honors provider quotas and
circuit cooldowns. Use `bootstrap --resume` after an interrupted initial load,
or `repair` to re-index stored error/pending records without downloading them
again. See [Scheduler operations](#scheduler-operations) below.

### 8. Ask a question

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was NVDA revenue last quarter?"}'
```

### Interactive chat (recommended)

For a single, centralized entry point, use the interactive client. It
**auto-starts the middleware** if it isn't already running, gives you a chat
loop over `/query` (streamed by default), and surfaces every promoted
Phase 2.1–2.3.7 feature — grounding mode, tool calls, retrieval strategy,
fetch-on-miss, resolved-ticker confirmation, bounded conversation history,
multiline questions, streamed progress events, and the deterministic answer
fast path — in one consistent answer renderer:

```bash
python scripts/chat.py                # auto-start middleware if needed
python scripts/chat.py --no-start     # connect to an existing middleware
python scripts/chat.py --no-stream    # use POST /query instead of SSE
```

```
Capabilities: tools=on streaming=on answer_policy=graded history=on max_question_chars=16000 graph=on

you> What was BB revenue last quarter?
  interpreting as BlackBerry Limited / BB

BlackBerry revenue is 143,000,000.00 usd (2026-Q1) [Source: yfinance/BB].
  [GROUNDED]
  ticker=BB intent=fact_lookup facts=10 docs=0 model_available=True latency=3803.2ms retrieval=hybrid
  used: query_facts

you> /refresh            # run ALL ingestion jobs (scheduler all --force)
you> /refresh NVDA       # refresh a single ticker via the API
you> /refresh hourly     # run a specific scheduler mode
you> /ask                # compose a multiline research question
you> /health             # middleware + freshness summary
you> /tools              # list model-callable tools
you> /autorefresh on     # refresh stale sources before queries
you> /verbose on         # show stage timings and diagnostics
you> /grounding strict   # force the strict answer policy for this session
you> /eval 10            # run 10 single-turn evaluation cases
you> /eval conversations 10  # score conversation carryover/reset behavior
you> /ticker NVDA        # pin a ticker for following questions
you> And how about AMD?  # follow-ups reuse this session's bounded history
you> /history            # preview this conversation's turns (local, no API call)
you> /new                # start a fresh conversation (clear history, new session id)
you> /history off        # run single-turn (stop sending/recording history)
you> /graph              # open the local retrieval graph
you> /graph trace        # deep-link the graph to the latest query trace
you> /help               # full command list
you> /quit               # stops the middleware if this script started it
```

Conversation history is client-owned, bounded, and sent with each request; the
middleware does not persist it. `/ask` preserves multiline questions and
validates the advertised 16,000-character limit before sending. The client
still requires `llama-server` on `:8087`; it warns and falls back to degraded
answers if the model is unreachable. See
[`scripts/CHAT.md`](scripts/CHAT.md) for every command and metadata tag the
renderer can show.

### Live retrieval graph (local only, Phase 2.2.7 + 2.3.5)

An optional, **local read-only** visualization of the retrieval pipeline (query →
plan → stages → tools → evidence → answer) plus a corpus explorer. It is
enabled by the committed `recommended` profile and served **loopback-only** with
a strict same-origin CSP. Open `http://127.0.0.1:8000/graph`, run `/graph`, use
`/graph trace` to focus the latest query, or start chat with `--open-graph`.
Set `ENABLE_GRAPH_OBSERVER=0` to disable it. It is an *observability* view and
never changes retrieval or answers; remote exposure is intentionally
unsupported. See
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) and [`docs/API.md`](docs/API.md).

### Phase 2.3 broad-market ingestion

Phase 2.3 expands the original deep watchlist into a source-aware S&P 500 /
Nasdaq-100 research corpus. The validated **source registry**
(`configs/sources.yaml`) owns cadence, request/day budgets, incremental cursors,
overlap windows, and provider circuits. Its committed capability flags are
enabled; adapters that lack credentials or entitlement skip independently, so
one exhausted, rate-limited, or failing source never blocks another.

| Group | Current sources |
|---|---|
| Universe | Nasdaq-100 constituents, IVV holdings, SEC ticker/CIK mapping |
| Company and market | Yahoo Finance, Finnhub, Massive daily market data/actions, hourly Massive News |
| Macro and regulatory | FRED, Federal Reserve, Treasury, BLS, BEA, EIA, New York Fed, CFTC |
| Sector | openFDA, NHTSA, USAspending |
| Deep research | SEC filing text/CompanyFacts, earnings transcripts, IR pages, estimates |
| Disabled | GDELT scheduled ingestion; existing GDELT documents and sentiment remain queryable |

#### Scheduler operations

The scheduler provides explicit `bootstrap`, `repair`, and `retention` modes
alongside `daily`/`hourly`/`weekly`/`all`, plus a truthful `status` report that
never claims a disabled, rate-limited, or circuit-open source is fresh:

```bash
python -m src.scheduler bootstrap --source sec_filings --since 2026-06-01
python -m src.scheduler status --json
python -m src.scheduler repair --source ir_pages --limit 50
python -m src.scheduler retention --preview
```

For a continuously updating, read-only terminal view of the same persisted run
and freshness state:

```bash
python scripts/watch_scheduler.py
python scripts/watch_scheduler.py --once
```

The watcher keeps daily `massive` market data and hourly `massive_news` on
separate rows and displays GDELT as `disabled`. It reads SQLite only and never
starts ingestion, calls a provider, or invokes the model.

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
       Universe · company · market · macro · regulatory · sector · deep sources
                                      │
                            registry-driven ingestion
                       (budgets · cursors · circuits · DLQ)
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                              ▼
                 SQLite                         ChromaDB
       facts · events · provenance      narratives · document embeddings
       freshness · run history · FTS5        cosine similarity
                       └──────────────┬──────────────┘
                                      ▼
                          FastAPI middleware (:8000)
             intent → adaptive lane → deterministic tools/answers
                    → hybrid retrieval → grounded generation
                                      │
                                      ▼
                         llama-server (:8087)
                   TraceAlchemy Gemma 4 E4B
                       (chat + embeddings)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full breakdown of
components, the storage schema, the query flow, and failure-mode handling.

---

## Project Structure

```
Gemma-E4B-Finance-RAG/
├── configs/                  # YAML configuration (see docs/CONFIGURATION.md)
│   ├── storage.yaml          # SQLite + ChromaDB + embedding + SEC settings
│   ├── middleware.yaml       # Endpoints, model name, retrieval top-k, feature flags
│   ├── profiles/             # legacy / recommended / evaluation runtime profiles
│   │   ├── legacy.yaml       # Phase 2.2-compatible rollback (MIDDLEWARE_PROFILE=legacy)
│   │   ├── recommended.yaml  # Promoted-safe production default
│   │   └── evaluation.yaml   # recommended + eval trace/progress metadata
│   ├── watchlist.yaml        # Deprecated compatibility watchlist
│   ├── sources.yaml          # Authoritative source cadence, quotas, cursors, flags
│   ├── coverage.yaml         # Security registry / index membership settings
│   ├── universe.yaml         # Constituent providers + universe refresh policy
│   ├── official_sources.yaml # Official release catalog
│   ├── fred.yaml             # FRED indicators + request settings
│   ├── gdelt.yaml            # GDELT query/domain/topic filters
│   ├── ir.yaml               # Company IR-page ingestion settings
│   ├── model.yaml            # Committed TraceAlchemy + llama-server defaults
│   └── model.example.yaml    # Template for gitignored model.local.yaml
├── docs/                     # Documentation (this README links here)
│   ├── ARCHITECTURE.md
│   ├── CONFIGURATION.md
│   ├── API.md
│   └── phase2.3/             # Phase specifications, decisions, and measured results
├── scripts/                  # Operational + smoke-test scripts
│   ├── serve_model.ps1 / .sh # Launch llama-server (TraceAlchemy)
│   ├── start_stack.ps1 / .sh # Launch the middleware (+ model on Unix)
│   ├── stop_stack.ps1 / .sh
│   ├── chat.py / CHAT.md     # Interactive client + complete command reference
│   ├── watch_scheduler.py    # Read-only live scheduler monitor
│   ├── migrate_phase2_3.py   # Resumable migration for pre-2.3 databases
│   └── validate_setup.py     # Environment / dependency sanity check
├── src/
│   ├── storage/              # Dual storage layer
│   │   ├── store.py          # Store facade over SQLite + ChromaDB
│   │   ├── sqlite_store.py   # Structured facts, filings, cache, audit log
│   │   └── chroma_store.py   # Vector store + TraceAlchemy embedding function
│   ├── middleware/           # FastAPI query router
│   │   ├── app.py            # Endpoints + query pipeline
│   │   ├── config.py         # MiddlewareConfig (flags, profiles, from middleware.yaml)
│   │   ├── models.py         # Pydantic request/response schemas
│   │   ├── intent_parser.py  # Ticker / metric / question-type extraction
│   │   ├── adaptive_orchestrator.py  # Fast/standard/complex/catalog lane routing
│   │   ├── deterministic_router.py   # Deterministic tool routing (no model call)
│   │   ├── deterministic_answers.py  # Deterministic answer fast path (skips generation)
│   │   ├── tools/coverage_tools.py   # describe_coverage — read-only capability inventory
│   │   ├── lexical_index.py  # BM25 lexical index (persistent SQLite FTS5 or in-memory)
│   │   ├── retriever.py      # Hybrid retrieval strategy selection + RRF fusion
│   │   ├── reranker.py       # Optional cross-encoder/LLM re-rank stage
│   │   └── prompt_augmenter.py
│   ├── scheduler/            # Registry, budgets, cursors, run status, operations
│   │   ├── __init__.py       # UnifiedScheduler
│   │   └── __main__.py       # CLI: python -m src.scheduler <mode>
│   ├── ingestion/            # Company, market, universe, official + sector adapters
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
├── LICENSE                   # MIT license (third-party vendor licenses stay separate)
├── requirements.txt
└── pytest.ini
```

---

## Configuration

All runtime configuration lives in `configs/*.yaml`, with credentials supplied
through the gitignored `.env`. `configs/sources.yaml` is authoritative for
scheduler cadence/quotas/cursors; `configs/coverage.yaml` and
`configs/universe.yaml` own security coverage; and machine-specific model paths
belong in gitignored `configs/model.local.yaml`. Each file and field is
documented in **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.

The middleware selects one of three reviewed runtime profiles
(`configs/profiles/legacy|recommended|evaluation.yaml`) via `profile:` in
`middleware.yaml` or the `MIDDLEWARE_PROFILE` env var. `recommended` is the
production default (adaptive lane routing, deterministic tool routing,
deterministic answers, and the FTS5 lexical index all on); `legacy` is the
tested one-switch rollback to Phase 2.2-compatible behavior. See
[`docs/FEATURE-FLAGS.md`](docs/FEATURE-FLAGS.md) for the live inventory of
every optional flag, what's on by default, and why.

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
| GET    | `/sentiment/{ticker}`  | Sentiment from preserved GDELT evidence          |
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
evidence ledger. The older harness keeps its Phase 2.3.7-dependent classes
explicitly labeled `deferred`; those capabilities were subsequently completed
and measured by the dedicated Phase 2.3.7 gates and result set below.

Tests that hit the live SEC EDGAR network and a running `llama-server` on
`:8087` are marked `live` (see `pytest.ini`). Skip them with:

```bash
pytest tests/ -v -m "not live"
```

Sanity-check your environment before running the stack:

```bash
python scripts/validate_setup.py
```

Before committing a change, run the verify gate — it wraps `pytest`, lint,
and this repo's promotion gates, and prints a final `VERIFY: PASS`/`FAIL`
line:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1   # Windows
```

```bash
scripts/verify.sh                                              # macOS/Linux
```

An offline **eval harness** (`eval/run_eval.py`, `eval/score.py`, `eval/gate.py`)
scores answers against golden sets (`tests/fixtures/evaluation/`,
`eval/golden/`) with an LLM judge and deterministic metrics (faithfulness,
policy compliance, refusal rate, entity/metric carryover, latency), and gates
feature promotion — see
`docs/phase2.3/2.3.7-rag-quality-speed-and-storage/RESULTS.md` for the
measured results behind the current defaults. `python scripts/chat.py` also
exposes `/eval [N]` and `/eval conversations [N]` to run a quick scored pass
against a live server.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for code style, branch, and PR
conventions.

---

## License

[MIT](LICENSE). Bundled third-party graph assets retain their own license files
under `src/middleware/static/graph/vendor/`.

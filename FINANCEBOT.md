# FinanceBot — RAG as source of truth

This repo is a **hybrid finance RAG** (SQLite facts + Chroma documents + FTS5
lexical search). **FinanceBot** (a Grok Bot assistant — not Gemma) should treat
this service as the **source of truth** for finance answers. Use open-web
search **only** when the RAG reports a miss.

The original `/query` pipeline, hybrid retrieval, and **every registered RAG
tool** stay in place — including `classify_trade_bias`, which **must** be used
for long vs short (buy vs sell) questions instead of guessing.

## Remaining step for you

1. Copy `.env.example` → `.env` and fill in real keys (do not invent them here).
2. Populate / refresh the index with the commands below.
3. Point FinanceBot at `http://127.0.0.1:8000`.

## Required environment variables

Create `.env` in the project root. Do **not** commit it.

| Variable | Required? | Used for |
|----------|-----------|----------|
| `FRED_API_KEY` | For FRED macro ingestion | FRED API |
| `SEC_EDGAR_USER_AGENT` | For SEC EDGAR | Must be `Name email@example.com` |
| `FINNHUB_API_KEY` | Optional | Finnhub news; missing → that source disables itself |
| `MASSIVE_API_KEY` | Optional | Massive market data / news |
| `BLS_API_KEY` | Optional | BLS official series |
| `BEA_API_KEY` | Optional | BEA national accounts |
| `EIA_API_KEY` | Optional | EIA energy |
| `OPENFDA_API_KEY` | Optional | openFDA events |
| `ALPHA_VANTAGE_API_KEY` | Optional | OVERVIEW + income + news sentiment (~25 req/day free) |
| `FMP_API_KEY` | Optional | FMP annual income + TTM ratios via `/stable` (~250 calls/day free Basic) |
| `MARKETAUX_API_KEY` | Optional | Marketaux ticker news (~100 req/day; alias `MARKETAUX_API_TOKEN`) |
| `OPENFIGI_API_KEY` | Optional | OpenFIGI ticker↔FIGI enrichment (works without key; better limits with key) |
| `TWELVE_DATA_API_KEY` | Optional / unused by default | Future adapter only |

Placeholders live in `.env.example`. A missing optional key does not block other
sources (`disabled_missing_key` in `python -m src.scheduler status`).

### Free / freemium sources (added)

| Source | Registry name | What it stores | Free-tier caveat |
|--------|---------------|----------------|------------------|
| Alpha Vantage | `alpha_vantage` | Fundamentals/observations + news headlines/snippets/URLs | ~25 requests/day — deep coverage only |
| Financial Modeling Prep | `fmp` | Annual income + TTM ratios → fundamentals/observations | ~250 calls/day Basic |
| Marketaux | `marketaux` | Ticker news headlines/snippets/URLs (+ sentiment when present) | ~100 req/day, ~3 articles/req |
| OpenFIGI | `openfigi` | FIGI aliases on securities (`vendor_symbol`) | Key optional; helper also in `src/ingestion/openfigi.py` |
| GDELT | `gdelt` | News + tone (existing adapter) | Enabled on **daily/all** with safe budgets; hourly stays `massive_news` |

After keys are set:

```bash
python -m src.scheduler daily --source alpha_vantage
python -m src.scheduler daily --source fmp
python -m src.scheduler daily --source marketaux
python -m src.scheduler daily --source openfigi
python -m src.scheduler daily --source gdelt
python -m src.scheduler status
```

### Optional model / FinanceBot knobs (no secrets)

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLAMA_ENDPOINT` / `CHAT_ENDPOINT` | OpenAI-compatible chat URL | `http://127.0.0.1:8087/v1/chat/completions` |
| `EMBEDDING_ENDPOINT` | OpenAI-compatible embeddings URL | `http://127.0.0.1:8087/v1/embeddings` |
| `MODEL_NAME` / `EMBEDDING_MODEL` | Model id sent to those endpoints | `tracealchemy` |
| `FINANCEBOT_MIN_FACTS` | Facts needed for a RAG **hit** | `1` |
| `FINANCEBOT_MIN_DOCUMENTS` | Qualifying docs needed for a **hit** | `1` |
| `FINANCEBOT_MIN_DOCUMENT_SCORE` | Min fusion/rerank score for a doc hit | `0.0` |
| `MIDDLEWARE_PROFILE` | `legacy` / `recommended` / `evaluation` | from `configs/middleware.yaml` |

A local Gemma / llama-server process is **not required** for:

- `POST /financebot/rag` (hybrid retrieval + hit/miss)
- `POST /financebot/tools` (all registered tools, including long/short)
- SQLite facts and the FTS5 lexical channel

It **is** required to embed new documents into Chroma and for optional
`POST /query` generation. Point `EMBEDDING_ENDPOINT` at any OpenAI-compatible
`/v1/embeddings` server if you are not using Gemma.

## Populate / refresh the RAG (after `.env` exists)

```bash
# one-time (or interrupted) load of enabled sources
python -m src.scheduler bootstrap
python -m src.scheduler status

# incremental cadences
python -m src.scheduler daily
python -m src.scheduler hourly
python -m src.scheduler weekly

# rebuild lexical (FTS5) index if documents changed outside the scheduler
python scripts/rebuild_lexical_index.py --batch-size 100
```

Resume a partial bootstrap with `python -m src.scheduler bootstrap --resume`.

## Start the API

```bash
uvicorn src.middleware.app:app --host 127.0.0.1 --port 8000
```

Interactive docs: <http://127.0.0.1:8000/docs>

## How FinanceBot should call this RAG

### 1. Retrieval first — `POST /financebot/rag`

```bash
curl -s http://127.0.0.1:8000/financebot/rag \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is NVDA revenue?", "ticker": "NVDA"}'
```

Contract:

| `status` | `web_search_allowed` | What FinanceBot must do |
|----------|----------------------|-------------------------|
| `hit` | `false` | Answer **only** from `facts` / `documents`. Do not use the open web. |
| `miss` | `true` | No relevant indexed evidence. Open-web search is allowed. |

`model_generation` is always `false` on this endpoint (no local chat model).

### 2. Long vs short (required tool) — `POST /financebot/tools`

For “is this a long or a short?”, “buy or sell?”, or trade bias:

```bash
curl -s http://127.0.0.1:8000/financebot/tools \
  -H 'Content-Type: application/json' \
  -d '{"name": "classify_trade_bias", "arguments": {"ticker": "NVDA"}}'
```

Use `result.bias` (`long` / `short` / `neutral`). If `evidence_status` is
`miss`, `web_search_allowed` is true — do not invent a directional call.

The same tool is advertised to `/query` (OpenAI tool schema + deterministic
router + typed `trade_bias` answer template) so a local model **must** answer
long vs short from `classify_trade_bias` instead of guessing.

`POST /financebot/rag` also auto-runs this tool when the question is long vs
short / buy vs sell. Use `trade_bias.bias`. If `evidence_status` is `miss`,
open-web fallback is allowed — do not invent a directional call.

### 3. All other RAG tools stay attached

`GET /tools` and `GET /financebot/tools` list the full registry. Dispatch any
of them with `POST /financebot/tools` (`name` + `arguments`):

| Tool | Role |
|------|------|
| `classify_trade_bias` | Long vs short from indexed evidence (**required** for that question type) |
| `describe_coverage` | What tickers / sources / metrics are covered |
| `list_metrics` | Metric names actually in the DB |
| `query_facts` | Rank / filter / compare fundamentals |
| `get_fundamentals` | One ticker’s metrics |
| `search_documents` | Filings / news / transcripts |
| `get_macro_snapshot` | Cached FRED-style macros |
| `get_sentiment` | News tone for one ticker |
| `get_guidance` | Latest earnings-call guidance |
| `get_estimates` | Forward revenue / EPS consensus |
| `get_price_targets` | Targets + `recommendation_mean` |
| `check_freshness` | Per-source freshness |
| `refresh_data` | Write tool — only if `allow_write_tools` is on |

`POST /query` is unchanged: hybrid retrieval + these tools + optional local
generation. FinanceBot should prefer `/financebot/rag` and `/financebot/tools`
so Gemma is not required.

## Hit vs miss (retrieval)

A **hit** is at least `FINANCEBOT_MIN_FACTS` structured facts **or** at least
`FINANCEBOT_MIN_DOCUMENTS` documents whose score ≥ `FINANCEBOT_MIN_DOCUMENT_SCORE`.
Scores prefer `rerank_score`, then `fusion_score` / `fused_score` / `score`.
Chroma `distance` is converted to a descending score.

Hybrid retrieval (vector + BM25 FTS5 + RRF, optional rerank) is unchanged.

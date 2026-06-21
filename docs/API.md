# API Reference

The middleware is a FastAPI app (`src/middleware/app.py`) listening on
`:8000` by default. Request/response schemas are defined in
`src/middleware/models.py` (Pydantic). Interactive docs are available at
`http://127.0.0.1:8000/docs`.

All bodies are JSON. `{ticker}` path parameters are upper-cased server-side
where relevant.

---

## GET `/`

Service info and quick links.

**Response (200):**

```json
{
  "service": "Gemma-E4B-Finance-RAG Middleware",
  "docs": "/docs",
  "health": "/health",
  "query": "POST /query",
  "search": "POST /search"
}
```

---

## GET `/health`

Enhanced health check covering storage, model, scheduler, and per-ticker
freshness. `status` is `"ok"` when SQLite is reachable, otherwise `"degraded"`.
Scheduler and freshness blocks are best-effort (may be `null`/`"error"` if
unavailable).

**Response (`HealthResponse`):**

| Field | Type | Meaning |
|-------|------|---------|
| `status` | string | `"ok"` or `"degraded"`. |
| `storage` | object | `Store.heartbeat()` — `{sqlite, chroma, chroma_doc_count}`. |
| `model_available` | bool | Whether the llama-server `/health` returns 200. |
| `scheduler` | object \| null | `UnifiedScheduler.status_report()` per-source freshness. |
| `freshness` | object | Per-ticker overall freshness for the core watchlist. |
| `version` | string | API version (`"1.0.0"`). |

**Example (200):**

```json
{
  "status": "ok",
  "storage": { "sqlite": true, "chroma": true, "chroma_doc_count": 1428 },
  "model_available": true,
  "scheduler": {
    "sources": {
      "yfinance": { "status": "fresh", "age_hours": 3.2, "ttl_hours": 24, "error": null }
    },
    "timestamp": "2026-06-21T12:00:00+00:00"
  },
  "freshness": { "NVDA": "fresh", "AMD": "partial", "AAPL": "fresh",
                 "MSFT": "fresh", "META": "stale", "CRWD": "fresh" },
  "version": "1.0.0"
}
```

Returns **503** if the store is not initialized.

---

## POST `/query`

Full RAG pipeline: parse intent → check/refresh freshness → hybrid retrieval →
prompt augmentation → model call. If the model server is down, returns a
**degraded** answer built from raw retrieved data (`model_available: false`).

**Request (`QueryRequest`):**

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `question` | string | — (required) | 1–2000 chars |
| `ticker` | string \| null | `null` | Optional ticker override |
| `temperature` | float \| null | `null` | 0.0–2.0; falls back to config default (0.3) |
| `max_tokens` | int \| null | `null` | 64–8192; falls back to config default (2048) |
| `stream` | bool | `false` | Reserved |
| `refresh` | bool | `true` | Auto-refresh stale sources before answering |
| `include_sources` | bool | `true` | Include source citations |

**Response (`QueryResponse`):**

| Field | Type | Meaning |
|-------|------|---------|
| `answer` | string | Grounded answer (or degraded raw-data summary). |
| `citations` | array | `SourceCitation` objects parsed from `[Source: type/ticker]` markers. |
| `detected_ticker` | string \| null | Ticker the parser detected. |
| `detected_intent` | string \| null | Question type (`fact_lookup`, `comparison`, `trend`, `explanation`, `sentiment`, `news`, `risk`, `general`). |
| `facts_used` | int | Number of SQLite facts retrieved. |
| `documents_used` | int | Number of ChromaDB documents retrieved. |
| `latency_ms` | float | End-to-end latency. |
| `model_available` | bool | `false` when the answer was produced in degraded mode. |
| `freshness` | object | `{overall, refreshed_during_query, stale_sources_used, warning}`. |
| `timestamp` | string | UTC ISO timestamp. |

`SourceCitation` fields: `source_type`, `ticker`, optional `metric`, `value`,
`period`, `source_url`, `relevance_score`.

**Example request:**

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was NVDA revenue last quarter?"}'
```

**Example response (200, model available):**

```json
{
  "answer": "NVDA reported revenue of $29.8B in 2026-Q2 [Source: earnings_call/NVDA].",
  "citations": [{ "source_type": "earnings_call", "ticker": "NVDA" }],
  "detected_ticker": "NVDA",
  "detected_intent": "fact_lookup",
  "facts_used": 6,
  "documents_used": 3,
  "latency_ms": 842.5,
  "model_available": true,
  "freshness": {
    "overall": "fresh",
    "refreshed_during_query": [],
    "stale_sources_used": [],
    "warning": null
  },
  "timestamp": "2026-06-21T12:00:01.123Z"
}
```

**Example response (degraded mode, model unavailable):**

```json
{
  "answer": "⚠️ Model unavailable — showing raw retrieved data:\n\n**Structured Facts:**\n- total_revenue: 29.8 (2026-Q2)\n\nStart llama-server to get AI-grounded answers.",
  "citations": [],
  "detected_ticker": "NVDA",
  "detected_intent": "fact_lookup",
  "facts_used": 6,
  "documents_used": 3,
  "latency_ms": 14.2,
  "model_available": false,
  "freshness": { "overall": "fresh", "refreshed_during_query": [],
                 "stale_sources_used": [], "warning": null },
  "timestamp": "2026-06-21T12:00:01.123Z"
}
```

Returns **503** if the store is not initialized.

---

## POST `/search`

Raw hybrid search — returns retrieved facts and documents without calling the
model. If the embedding server is unavailable, returns empty results gracefully
rather than erroring.

**Request (`SearchRequest`):**

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `query` | string | — (required) | min 1 char |
| `ticker` | string \| null | `null` | Optional ticker filter |
| `n_results` | int | `5` | 1–20 |

**Response (`SearchResponse`):**

| Field | Type | Meaning |
|-------|------|---------|
| `documents` | array | ChromaDB matches (`id`, `document`, `metadata`, `distance`). |
| `facts` | array | SQLite `fundamentals` rows. |
| `ticker` | string \| null | Detected or provided ticker. |

**Example:**

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "datacenter revenue growth", "ticker": "NVDA", "n_results": 5}'
```

```json
{
  "documents": [
    { "id": "earnings_call/NVDA/Q2-2026", "document": "...", "metadata": { "ticker": "NVDA", "source": "earnings_call" }, "distance": 0.21 }
  ],
  "facts": [
    { "ticker": "NVDA", "metric": "datacenter_revenue_q2", "value": 26.1, "period": "2026-Q2", "source_type": "earnings_call" }
  ],
  "ticker": "NVDA"
}
```

Returns **503** if the store is not initialized.

---

## GET `/freshness/{ticker}`

Per-source freshness report for a ticker across all logical data sources.

**Response (`FreshnessResponse`):**

| Field | Type | Meaning |
|-------|------|---------|
| `ticker` | string | The requested ticker. |
| `overall` | string | `fresh`, `partial`, `stale`, or `never_fetched`. |
| `sources` | object | Per-source `{status, last_updated, age_hours, ttl_hours}`; status is `fresh`/`stale`/`never_fetched`. |
| `stale_sources` | array | Names of sources currently stale. |

Logical source keys: `yfinance_fundamentals`, `yfinance_news`, `sec_filings`,
`gdelt_news`, `earnings_transcripts`, `ir_pages`.

**Example:**

```bash
curl http://127.0.0.1:8000/freshness/NVDA
```

```json
{
  "ticker": "NVDA",
  "overall": "partial",
  "sources": {
    "yfinance_fundamentals": { "status": "fresh", "last_updated": "2026-06-21 08:00:00", "age_hours": 4.0, "ttl_hours": 24 },
    "gdelt_news": { "status": "stale", "last_updated": "2026-06-20 09:00:00", "age_hours": 27.0, "ttl_hours": 6 },
    "earnings_transcripts": { "status": "never_fetched", "last_updated": null, "age_hours": null, "ttl_hours": 168 }
  },
  "stale_sources": ["gdelt_news"]
}
```

Returns **503** if the store is not initialized.

---

## POST `/refresh/{ticker}`

Trigger on-demand re-ingestion for a ticker. If `sources` is omitted, all
currently-stale **and** never-fetched sources are refreshed. Scheduler-managed
sources (SEC filings, earnings transcripts, IR pages) are routed through the
`UnifiedScheduler`; others are ingested directly.

**Request (`RefreshRequest`, optional body):**

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `sources` | array \| null | `null` | Specific logical sources to refresh. Short aliases are accepted (`fundamentals`, `news`, `sec`, `gdelt`, `earnings`, `ir`) and mapped to their logical names. Omit to refresh all stale/never-fetched sources. |

**Response (`RefreshResponse`):**

| Field | Type | Meaning |
|-------|------|---------|
| `ticker` | string | The requested ticker. |
| `refreshed` | array | Sources successfully refreshed. |
| `skipped` | array | Sources not selected for refresh. |
| `errors` | array | `"<source>: <error>"` entries for failures. |
| `duration_s` | float | Wall-clock duration. |

**Example:**

```bash
curl -X POST http://127.0.0.1:8000/refresh/NVDA \
  -H "Content-Type: application/json" \
  -d '{"sources": ["gdelt", "news"]}'
```

```json
{
  "ticker": "NVDA",
  "refreshed": ["gdelt_news", "yfinance_news"],
  "skipped": ["yfinance_fundamentals", "sec_filings", "earnings_transcripts", "ir_pages"],
  "errors": [],
  "duration_s": 6.41
}
```

Returns **503** if the store is not initialized.

---

## GET `/macro/snapshot`

Snapshot of key macro indicators from cached FRED data in SQLite (no live FRED
call). Values are pulled from the synthetic `MACRO` ticker.

**Response (`MacroSnapshotResponse`):**

| Field | Source series | Meaning |
|-------|---------------|---------|
| `gdp` | `GDP` | Gross Domestic Product |
| `inflation_cpi` | `CPIAUCSL` | CPI (All Urban Consumers) |
| `fed_rate` | `FEDFUNDS` | Federal Funds Rate |
| `unemployment` | `UNRATE` | Unemployment Rate |
| `ten_year_treasury` | `DGS10` | 10-Year Treasury Rate |
| `ten_two_spread` | `T10Y2Y` | 10Y–2Y Treasury Spread |
| `timestamp` | — | UTC ISO timestamp |

Any indicator absent from the cache is returned as `null`.

**Example:**

```bash
curl http://127.0.0.1:8000/macro/snapshot
```

```json
{
  "gdp": 29350.1,
  "inflation_cpi": 318.5,
  "fed_rate": 4.33,
  "unemployment": 4.1,
  "ten_year_treasury": 4.28,
  "ten_two_spread": 0.45,
  "timestamp": "2026-06-21T12:00:00Z"
}
```

Returns **503** if the store is not initialized.

---

## GET `/sentiment/{ticker}`

GDELT sentiment summary for a ticker over a lookback window.

**Query parameters:**

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `days` | int | `7` | Lookback period in days |

**Response (`SentimentResponse`):**

| Field | Type | Meaning |
|-------|------|---------|
| `ticker` | string | The requested ticker. |
| `average_tone` | float \| null | Mean GDELT tone score. |
| `article_count` | int | Articles considered. |
| `positive_ratio` | float | Share of positive-tone articles. |
| `negative_ratio` | float | Share of negative-tone articles. |

**Example:**

```bash
curl "http://127.0.0.1:8000/sentiment/NVDA?days=7"
```

```json
{
  "ticker": "NVDA",
  "average_tone": 2.13,
  "article_count": 87,
  "positive_ratio": 0.62,
  "negative_ratio": 0.18
}
```

Returns **503** if the store is not initialized.

---

## GET `/guidance/{ticker}`

Latest earnings guidance from the most recent earnings transcript. The response
is a plain object (not a fixed Pydantic model).

**Response:**

| Field | Type | Meaning |
|-------|------|---------|
| `ticker` | string | The requested ticker. |
| `guidance` | object | Guidance payload (e.g. revenue range, EPS, margin). Empty `{}` when not found. |
| `status` | string | `"found"` or `"not_found"`. |

**Example (found):**

```bash
curl http://127.0.0.1:8000/guidance/NVDA
```

```json
{
  "ticker": "NVDA",
  "guidance": {
    "revenue_low": 32.0,
    "revenue_high": 34.0,
    "eps": 3.10,
    "gross_margin": 0.75,
    "period": "2026-Q3"
  },
  "status": "found"
}
```

**Example (not found):**

```json
{ "ticker": "XYZ", "guidance": {}, "status": "not_found" }
```

Returns **503** if the store is not initialized.

> The exact keys inside `guidance` are whatever
> `EarningsTranscriptIngestor.get_latest_guidance()` extracted for that ticker;
> the example above is illustrative.

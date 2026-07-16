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
| `scheduler` | object \| null | `UnifiedScheduler.status_report()`. Beyond per-source freshness (`status`, `age_hours`, `ttl_hours`, `error`), it also carries `runs` (recent `scheduler_runs` history), `coverage_health` (denominator-explicit active-security coverage by index/scope/sector, recent news/market/SEC coverage, partition states — Phase 2.3.4.3), and `registry_version` (`configs/sources.yaml` version). This is the same method behind `python -m src.scheduler status`; see `scripts/CHAT.md`. |
| `freshness` | object | Per-ticker overall freshness for the core watchlist. |
| `capabilities` | object \| null | Active deployment capabilities and effective limits. Always includes `{tools, streaming, answer_policy}`. `streaming` is the **effective** capability (2.2.6.1): whether `/query/stream` can be served at all — `false` when tools are enabled but tool-final streaming is off. `streaming_tool_final` is `true` only when a tools-enabled request streams its final synthesis after bounded non-streaming tool rounds. On conversation-memory builds (2.2.2) it also advertises `{history, multiline, max_question_chars, conversation_max_turns, conversation_max_history_chars}` so clients read the real limits instead of guessing. |
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

## GET `/tools`

Lists model-callable tools registered in the middleware and the current
tool-gating state.

**Response (200):**

| Field | Type | Meaning |
|-------|------|---------|
| `enabled` | bool | Whether `/query` is configured to advertise tools to the model. |
| `allow_write_tools` | bool | Whether state-changing tools may run. |
| `tools` | array | Registered tools with `name`, `description`, and `write` flag. |

**Example:**

```bash
curl http://127.0.0.1:8000/tools
```

```json
{
  "enabled": true,
  "allow_write_tools": false,
  "tools": [
    { "name": "describe_coverage", "description": "Read-only capability inventory...", "write": false },
    { "name": "query_facts", "description": "Rank, filter...", "write": false },
    { "name": "refresh_data", "description": "Use ONLY...", "write": true }
  ]
}
```

---

## `describe_coverage` tool

`describe_coverage` is a bounded, read-only inventory of the canonical Phase
2.3 security registry and corpus ledger. It does not call a provider, the
model, or ingestion. Callers use the Store facade; the tool never joins SQLite
tables or scans Chroma directly.

Supported operations are `summary`, `list_securities`, `contains_security`,
`security_sources`, `list_sources`, `list_item_types`, and `list_metrics`.
Arguments are `operation` plus optional `ticker`, `filters`, `limit`, `cursor`,
and `ticker_only`. Filters may include `index`, `sector`, `industry`,
`coverage_tier`, `item_type`, `source`, `source_category`, `active`, `as_of`,
`date_from`, and `date_to`.

Responses include `coverage_basis` (`canonical` or `legacy_partial`), registry
and corpus counts, membership tiers, stored evidence, configured/enabled/
available source capability, terminal source status, item types, metric
families, `data_revision`, and `universe_snapshot_at`. Detail pages are bounded
and set `complete: false` with an opaque `next_cursor` when more rows remain.
An explicit ticker-only inventory is complete when it fits the safe 1,000-ticker
bound. Missing canonical tables are disclosed through `coverage_basis` rather
than silently treated as complete canonical coverage.

---

## POST `/query`

Full RAG pipeline: parse intent → check/refresh freshness → hybrid retrieval →
prompt augmentation → model call. If the model server is down, returns a
**degraded** answer built from raw retrieved data (`model_available: false`).
Explicit capability-coverage questions are answered deterministically from
`describe_coverage`; an unavailable inventory produces an unavailable answer,
not a model-generated guess.
When `enable_tools` is on, `/query` may advertise registered tools to the model
and perform extra model round-trips before the final answer. Write tools are
additionally gated by `allow_write_tools` and per-query refresh limits.

**Request (`QueryRequest`):**

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `question` | string | — (required) | 1–16000 chars. Never silently truncated — an over-limit question is a **422** (`MAX_QUESTION_CHARS`, 2.2.2.1). |
| `history` | array | `[]` | Bounded client-owned conversation history (2.2.2.1): a flat list of `ChatTurn` `{role, content, turn_id?, context?}`. Additive — omit/empty preserves single-turn behavior. The server uses at most `conversation_max_turns` / `conversation_max_history_chars` of it and persists nothing. |
| `session_id` | string \| null | `null` | Opaque client id for tracing only (1–128 chars of `[A-Za-z0-9._:-]`); never a server-side lookup key. |
| `ticker` | string \| null | `null` | Optional ticker override |
| `temperature` | float \| null | `null` | 0.0–2.0; falls back to config default (0.3) |
| `max_tokens` | int \| null | `null` | 64–8192; falls back to config default (2048) |
| `stream` | bool | `false` | Hint only on `POST /query` (that endpoint always returns a full body). Streaming is served by `POST /query/stream` (below). |
| `refresh` | bool | `true` | Auto-refresh stale sources before answering |
| `include_sources` | bool | `true` | Include source citations |
| `answer_policy` | string \| null | `null` | Per-request override of the server default: `strict` or `graded`. |
| `include_evidence_trace` | bool | `false` | Attach the exact evidence trace (system/user prompts, usable facts/documents, tool results) used to produce the answer. Intended for evaluation (2.2.1.2); never populated for a degraded answer. |

**Response (`QueryResponse`):**

| Field | Type | Meaning |
|-------|------|---------|
| `answer` | string | Grounded answer (or degraded raw-data summary). |
| `answer_origin` | string \| null | Answer source; `deterministic_coverage` identifies a capability-inventory answer. |
| `coverage_metadata` | object \| null | Coverage basis, Store revision, snapshot, completeness, pagination, and applied filters for a deterministic coverage answer. |
| `citations` | array | `SourceCitation` objects parsed from `[Source: type/ticker]` markers. |
| `detected_ticker` | string \| null | Ticker the parser detected. |
| `detected_intent` | string \| null | Question type (`fact_lookup`, `comparison`, `trend`, `explanation`, `sentiment`, `news`, `risk`, `general`). |
| `facts_used` | int | Number of SQLite facts retrieved. |
| `documents_used` | int | Number of ChromaDB documents retrieved. |
| `grounding` | string | Actual answer path: `grounded`, `partial`, `general`, or `refused`. |
| `latency_ms` | float | End-to-end latency. |
| `timings` | object \| null | Per-stage latency breakdown (ms), incl. retrieval sub-timings; present when `return_timings` is on. |
| `model_available` | bool | `false` when the answer was produced in degraded mode. |
| `retrieval_strategy` | string \| null | Document retrieval path: `vector`, `hybrid`, or `hybrid+rerank`. |
| `tools_used` | array \| null | Names of middleware tools invoked while answering, if any. |
| `resolved_ticker` | object \| null | `{name, source}` when the resolver mapped a non-exact company name/typo (omitted for exact symbol / explicit override). |
| `freshness` | object | `{overall, refreshed_during_query, stale_sources_used, fetched_on_miss, warning}`. |
| `timestamp` | string | UTC ISO timestamp. |

**Additive, flag-gated blocks** — present only when the relevant feature ran;
older clients that ignore unknown fields are unaffected:

| Field | Type | Present when |
|-------|------|--------------|
| `conversation` | object \| null | The request carried history (2.2.2.1): `{history_turns_received, history_turns_used, history_truncated, topic_reset}`. |
| `retrieval_query` | string \| null | Follow-up rewriting ran (2.2.2.2): the compiled standalone retrieval query — retrieval input only, never the user's wording. |
| `carried_context` | object \| null | Rewriting ran (2.2.2.2): `{entities, metrics, timeframe, topic_reset, ambiguous_slots, resolution_sources}`. |
| `resolved_tickers` / `resolved_metrics` / `resolved_timeframe` | array / array / string \| null | Effective retrieval entities after carryover (2.2.2.2). |
| `orchestration` | object \| null | `enable_adaptive_rag` on (2.2.3.4): `{lane, reason_codes, subqueries_executed, retrieval_rounds, planning_calls, reranker_calls, deterministic_tools, context_chars, evidence_dropped, fallback_reason}` — **actual executed** counters, not maxima. |
| `evidence_sufficiency` | object \| null | `enable_evidence_sufficiency` on (2.2.4.1): `{status, reason_codes, covered_subqueries, missing_subqueries, corrective_action, retry_performed}`. |
| `evidence_citations` | array \| null | `answer_validation` is `report`/`enforce` (2.2.4.3): `EvidenceCitation` objects resolved against the model-visible ledger. |
| `answer_validation` | object \| null | `answer_validation` is `report`/`enforce` (2.2.4.3): `{validation_status, citation_support_rate, numeric_claims_supported, numeric_claims_unsupported, numeric_claims_ambiguous, mismatch_counts, enforcement, …}`. Validator errors report `validation_status=report_unavailable` rather than failing the query. |
| `evidence_trace` | object \| null | `include_evidence_trace: true` and a model call succeeded (2.2.1.2). |

`SourceCitation` fields: `source_type`, `ticker`, optional `metric`, `value`,
`period`, `source_url`, `relevance_score`.

`EvidenceCitation` fields (2.2.4.3): `evidence_id` (request-local `[E#]`, or
`null` for a legacy source label), `source_type`, `ticker`, `metric`, `period`,
`source_url`, and `support_status` (`supported` / `missing` / `malformed` — a
`missing`/`malformed` id is never converted into a real source citation).

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

## POST `/query/stream`

Server-Sent Events (SSE) variant of `/query`: streams the **final answer** as
`token` deltas, then a terminal `metadata` event carrying the same
`QueryResponse` fields (minus `answer`). Same request body as `/query`
(`QueryRequest`).

Availability is capability-gated (read `capabilities` from `/health`):

- `enable_streaming` off → **404** (`Streaming disabled`).
- `enable_tools` on **and** `enable_tool_final_streaming` off → **404**
  (`Streaming disabled while tools are enabled`). This is the documented signal
  the client caches to fall back to `POST /query`.
- `enable_tools` + `enable_tool_final_streaming` on (2.2.6.1) → the bounded
  tool/planning rounds run **non-streaming** first, then only the final answer
  synthesis streams.
- When `enable_stream_progress_events` is on, versioned, **redacted** progress
  events (`query_started`, `stage`, `tool_started`, `tool_completed`, `error`)
  precede the tokens — stage/tool names, statuses, and optional counts only,
  never prompts, tool arguments, or document text.

**Event stream (media type `text/event-stream`):**

| Event | Payload | Meaning |
|-------|---------|---------|
| `query_started` | `{version, seq, …}` | Emitted first when progress events are on. |
| `stage` | `{name, status, …}` | Pipeline stage transition (compile/route/retrieve/grade/generate…). |
| `tool_started` / `tool_completed` | `{name, status, count?}` | Safe tool lifecycle (progress events on). |
| `token` | `{token}` | A chunk of the streamed final answer. |
| `metadata` | `QueryResponse` sans `answer` | Terminal event with citations, grounding, counts, freshness, and any flag-gated blocks. |
| `error` | `{message}` | A redacted error notice; the stream still terminates cleanly. |

If the model is unavailable, the legacy path (both flags off) returns **404**;
with progress events or tool-final streaming on it degrades gracefully to a
streamed degraded answer instead of poisoning the client's streaming capability.

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

## Local retrieval-graph observer (Phase 2.2.7)

A bounded, **read-only, localhost-only** side channel that projects each query's
retrieval pipeline into a graph for the single-page UI at `GET /graph`. It is an
**observability** view — it never changes retrieval or the answer (see
`docs/phase2.2/ARCHITECTURE-DECISION.md` for why this is *not* GraphRAG).

**Enablement & access.** Every `/graph*` route (the UI, its static bundle, and
the `/graph/api/*` endpoints) is served only when `enable_graph_observer` is on
and only to loopback clients (`127.0.0.1`/`::1`). A non-loopback client, or any
request while the observer is disabled, gets **404** — there is no bypass flag.
Remote exposure is out of scope for Phase 2.2 and would need a separate design
covering authentication, TLS, proxy trust, and retention.

**Security headers.** Graph responses carry a strict same-origin CSP
(`default-src 'none'`; `script-src 'self'`; `style-src 'self' 'unsafe-inline'`;
`img-src 'self' data:`; `font-src 'self'`; `connect-src 'self'`;
`frame-ancestors 'none'`), plus `X-Content-Type-Options: nosniff`,
`Referrer-Policy: no-referrer`, and `Cross-Origin-{Opener,Resource}-Policy:
same-origin`. Trace/corpus detail responses are `Cache-Control: no-store`; the
SSE stream keeps `no-cache`. No cookies, localStorage, service worker, analytics,
or third-party requests are used.

**Redaction by construction.** Nodes/edges carry only allowlisted metadata.
Questions are stored as a bounded preview plus a SHA-256 digest; evidence bodies
are bounded excerpts; secrets (`api_key`/`token`/`secret`/`authorization`/
`cookie`) and local filesystem paths are stripped; source links are kept only
when `http`/`https`. Node ids never expose local paths.

### Query response field

When the observer is enabled, `POST /query` and the `/query/stream` terminal
`metadata` event include `graph_trace_id` — the opaque id of this request's
trace, used by the chat client's `/graph trace`. The field is **omitted** when
the observer is off, so the response stays byte-compatible for older clients.

### `GET /health` capabilities

When enabled, the `capabilities` block additionally advertises
`graph_observer: true`, `graph_url` (the effective same-origin `/graph` URL), and
`graph_observer_limits` (the bounded TraceHub limits). Older/observer-off servers
omit these keys.

### Live-trace endpoints (loopback only)

| Endpoint | Purpose |
|---|---|
| `GET /graph` | The single-page graph UI (static HTML/JS/CSS bundle). |
| `GET /graph/api/traces?limit=N` | Newest-first safe trace summaries. |
| `GET /graph/api/traces/{query_id}` | Reconstructed snapshot (nodes/edges) for one trace. |
| `GET /graph/api/traces/{query_id}/evidence/{evidence_id}` | One bounded evidence node. |
| `GET /graph/api/events` | SSE stream of graph deltas (supports `Last-Event-ID`/`last_sequence` replay and `once=true`). |
| `GET /graph/api/health` | Bounded observer counters + limits. |

### Corpus-explorer endpoints (loopback only)

Read-only projections of the Store inventory (sources, tickers, metrics, facts,
filings, sections, document families, freshness, scheduler). All responses are
hard-bounded (page/element caps), use opaque revision-keyed cursors, and never
invoke embeddings or a model.

| Endpoint | Purpose |
|---|---|
| `GET /graph/api/corpus/overview` | Cached source/ticker/freshness projection. |
| `GET /graph/api/corpus/search` | Label/metadata search (`q`, `kinds`, `sources`, `ticker`, `limit`, `cursor`). |
| `GET /graph/api/corpus/aggregates` | Bounded authoritative facet counts, grouped by `group_by` (`source_category` default, or `source`/`item_type`/`security`/`year`/`month`/`indexing_state`), narrowable by the same filter set. Never scans Chroma bodies (2.3.5.2). |
| `GET /graph/api/corpus/facets` | Bounded, cached facet counts for the current combined filter set — backs the persistent facet rail (2.3.5.2). |
| `GET /graph/api/corpus/groups` | One paged aggregate level (`group_by`) as drillable, counted nodes for the aggregation-first landing view (2.3.5.2). |
| `GET /graph/api/corpus/items/{node_id}` | One bounded corpus-item/event detail with safe provenance (2.3.5.3). |
| `GET /graph/api/corpus/nodes/{node_id}` | One node detail (at most one excerpt). |
| `GET /graph/api/corpus/nodes/{node_id}/neighbors` | One bounded neighbor page. |
| `GET /graph/api/corpus/filings/{accession}/sections` | Section-family page for a filing. |
| `GET /graph/api/corpus/refresh-status` | Persisted freshness/scheduler state (never refreshes). |

`aggregates`, `facets`, and `groups` accept the shared `CorpusFacetFilters`
query params plus `limit`/`cursor`; a filter change resets paging by changing
the opaque cursor scope, and a mid-browse ingestion write between pages
returns **409** (`CorpusRevisionChanged`) rather than mixing revisions.

### Graph event/delta schema

Deltas are versioned (`schema_version = 1`) with an `operation`
(`upsert_node`/`upsert_edge`/`remove`/`trace_complete`/`trace_evicted`/
`reset_required`). Nodes carry `kind` (`query`/`plan`/`subquery`/`stage`/`tool`/
`evidence`/`source`/`answer`/`citation`), `status`, `label`, bounded `summary`,
allowlisted `metadata`, and sequence numbers. Edges carry a `relation`
(`compiled_to`/`contains`/`routed_to`/`retrieved`/`returned`/`from_source`/
`expanded_from`/`supports`/`cited_by`/`corrected_by`/`validated_as`). The wire
shape carries **no** pixel/layout positions — layout is UI behavior; graph
meaning is the contract. A golden fixture pins this schema at
`tests/fixtures/graph/query_trace_v1.json`.

**Source-aware evidence (Phase 2.3.5.1).** `evidence`/`source`/`citation`
nodes may additionally carry `security_id`, `canonical_security`,
`index_memberships`, `sector`, `source_category`, `authority_tier`,
`coverage_tier`, `provider`, `publisher`, `source_name`, `item_type`,
`event_type`, `form`, `filing_item`, `exhibit`, `date_semantics`,
`published_at`, `effective_at`, `accessed_at`, and `evidence_role` (`primary`
| `corroborating`) — additive allowlisted fields that did not require bumping
`schema_version`. A mixed SEC/news/macro/market golden fixture covers this
shape alongside the unchanged Phase 2.2 fixture above.

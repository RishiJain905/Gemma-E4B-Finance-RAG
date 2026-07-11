# scripts/chat.py — command and metadata reference

`scripts/chat.py` is the interactive terminal client for the middleware. It
auto-starts the middleware if needed, streams `/query` answers by default,
and is the single front door for every Phase 2.1 feature: tool calls,
grounding mode, retrieval strategy, fetch-on-miss, and resolved-ticker
confirmation.

Every field in this doc is optional — the renderer is defensive, so
`chat.py` also works unchanged against an older middleware that doesn't
return some of these fields (it just omits the corresponding line).

## Starting the client

```bash
python scripts/chat.py                # start middleware (if needed) + chat
python scripts/chat.py --no-start     # don't auto-start; just connect
python scripts/chat.py --port 8000
python scripts/chat.py --no-stream    # disable SSE streaming, use POST /query
python scripts/chat.py --timeout 300  # HTTP request timeout in seconds
```

On startup the client checks the model server (`:8087`), connects to (or
starts) the middleware, then fetches `/health` once and prints a
`Capabilities:` line showing which optional features this deployment has
turned on (`tools`, `streaming`, `answer_policy`) — silently skipped if the
server doesn't report a `capabilities` block.

## Commands

| Command | Description |
|---|---|
| `<just type a question>` | Ask the RAG (`POST /query`, streamed by default). |
| `/refresh` | Run **all** ingestion jobs (`scheduler all --force`). |
| `/refresh daily\|hourly\|weekly\|all\|status` | Run that scheduler mode. |
| `/refresh NVDA` | Refresh one ticker via `POST /refresh/{ticker}`. |
| `/ticker NVDA` | Pin a ticker override for following questions. |
| `/ticker clear` | Clear the pinned ticker. |
| `/autorefresh on\|off` | Toggle auto-refresh of stale data per query. |
| `/verbose on\|off` | Toggle the server `timings` breakdown under answers. |
| `/grounding strict\|graded` | Send `answer_policy` with every query this session (server-side override). |
| `/grounding clear` | Stop overriding — use the server's configured default. |
| `/new` or `/clear` | Start a fresh conversation: clear this session's history and rotate its local `session_id`. Explicit settings (`/grounding`, `/verbose`, `/history off`, pinned `/ticker`, `/autorefresh`) are preserved. |
| `/history` | Preview this conversation's turns locally (turn number, role, short text). No API call. |
| `/history off\|on` | Stop / resume sending and recording conversation history (for privacy or single-turn comparisons). |
| `/health` | Show the middleware health + freshness summary. |
| `/tools` | List model-callable tools (`GET /tools`); degrades gracefully if the endpoint is missing or tools are disabled. |
| `/eval [N]` | Run `python eval/run_eval.py --limit N` (default 5) against this running server and print the tail. |
| `/help` | Show the full command list. |
| `/quit` or `/exit` | Leave (stops the middleware if this script started it). |

## Answer metadata

After every answer, the renderer prints whatever the server response
includes, in this order:

1. **Grounding tag** — a small colored `[LABEL]` for the response's
   `grounding` field: `[GROUNDED]` (green), `[PARTIAL]` (yellow),
   `[GENERAL]` (magenta), `[REFUSED]` (red). Color only appears when stdout
   is a real terminal (TTY); piped/redirected output is plain text.
2. **Metadata line** — `ticker=… intent=… grounding=… facts=… docs=…
   model_available=… latency=…ms retrieval=…`. `retrieval` is the document
   retrieval path used: `vector`, `hybrid`, or `hybrid+rerank`.
3. **`used: <tool names>`** — printed only when the response reports
   dispatched tool calls (e.g. `used: query_facts, get_price_targets`).
4. **`interpreting as <Name> / <TICKER>`** — printed only when the intent
   resolver mapped a company name or typo to a ticker (not for an exact
   symbol match or an explicit `/ticker` override).
5. **`fetched live data for <TICKER>`** — printed when the server had to
   fetch-on-miss data for a never-seen ticker during this query.
6. **`warning: …`** — a freshness warning (stale data used, or a
   fetch-on-miss failure).
7. **`refreshed: <sources>`** — sources that were refreshed during the
   query (only shown with `/autorefresh on`).
8. **`timings: …`** (only with `/verbose on`) — the server's per-stage
   latency breakdown, including retrieval sub-timings
   (`retrieval.embedding`, `retrieval.chroma`, `retrieval.sqlite`).

## Conversation memory (2.2.2.1)

Each `ChatSession` owns its own bounded conversation history and a local
`session_id`. The middleware stays **stateless**: every request carries the
recent turns and the id, and the server validates and uses them without
persisting anything. Nothing is written to disk.

- A turn is recorded **only** after the server accepts the request and returns
  a terminal, non-blank answer. Failed, cancelled, validation-error, or
  incomplete-streamed requests never enter history.
- Each assistant turn also stores a small structured context (detected/resolved
  ticker, intent, grounding, timeframe) so a later follow-up can resolve
  references like "and AMD?".
- The server bounds what it uses: at most `conversation_max_turns` recent turns
  and `conversation_max_history_chars` characters (see
  `configs/middleware.yaml`). The current question has its own independent
  16,000-character limit and is **never** silently truncated — an over-limit
  question is a validation error.
- Responses include a `conversation` block
  (`history_turns_received`, `history_turns_used`, `history_truncated`,
  `topic_reset`) whenever history was sent.
- `session_id` is tracing metadata only — never a server-side lookup key.

Use `/new` (or `/clear`) to start a fresh conversation, `/history` to preview
the local turns, and `/history off` to run single-turn (no history sent or
recorded).

## Capabilities block

At startup, and any time you run `/health`, the client reads (or shows)
`capabilities` from `GET /health`:

- `tools` — whether the model can call middleware tools this deployment.
- `streaming` — whether `/query/stream` is enabled (note: streaming is
  unavailable whenever tools are enabled, since the tool loop is
  multi-turn).
- `answer_policy` — the server's configured default answer policy
  (`strict` or `graded`), before any per-session `/grounding` override.

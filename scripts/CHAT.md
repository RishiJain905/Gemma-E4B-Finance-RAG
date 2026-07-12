# scripts/chat.py — command and metadata reference

`scripts/chat.py` is the interactive terminal client for the middleware. It
auto-starts the middleware if needed, streams `/query` answers by default,
and is the single front door for every Phase 2.1/2.2 feature: tool calls,
grounding mode, retrieval strategy, fetch-on-miss, resolved-ticker
confirmation, bounded conversation history (2.2.2), multiline questions
(2.2.2.3), and streamed progress events (2.2.6.1).

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
| `/ask` | Compose a **multiline** question (see below), then submit it once. |
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
| `/eval [N]` | Run the **single-turn** eval (default 5 cases, `--no-conversations`) against this running server and print the tail. |
| `/eval conversations [N]` | Run the **conversation** fixtures, then score deterministically and print carryover / topic-reset / subquestion-coverage / leakage metrics. Opt-in; needs an already-running stack. |
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

## Streaming progress (2.2.6.1)

When the server has `enable_stream_progress_events` on, a streamed answer is
preceded by redacted, versioned progress events (pipeline stages plus safe
tool starts/completions). On a terminal the client renders them as a single
line updated **in place**, then clears it when the answer begins:

```text
  resolve -> hybrid retrieval -> query_facts (12 rows) -> answer
```

Progress events carry only stage names, safe tool names, statuses, and row/item
counts — never prompts, tool arguments, or retrieved document text. The events
are cosmetic: piped/redirected (non-TTY) output shows **no** progress line and
no ANSI escapes, only the terminal answer and metadata. Under `/verbose on` the
single line is replaced by a per-event dim log that also shows stage timings and
corrective reasons. Unknown future event types are ignored.

With tools enabled and tool-final streaming on, the bounded tool/planning rounds
run non-streaming first (you may see `tool_started`/`tool_completed` progress),
and only the final answer synthesis streams token by token.

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

## Multiline questions (2.2.2.3)

Single-line questions are unchanged — type and press enter. For a structured,
multi-part research prompt, `/ask` opens a small composer (no new dependency;
it is just `input()` and a pure buffer):

```text
you> /ask
  multiline mode — end with /send, discard with /cancel, inspect with /preview.
... Compare NVDA and AMD revenue growth from FY2023 to FY2025.
... Include margin changes and summarize the strongest cited risk for each.
... /send
```

While composing:

- `/send` — join the lines with `\n`, validate locally, and submit **once**;
- `/cancel` — discard the draft; history is untouched;
- `/preview` — show the character count and the exact buffered text;
- **EOF (Ctrl+D) or Ctrl+C** — cancels the *buffer*, not the session.

Everything else you type is appended verbatim — newlines and punctuation are
preserved exactly, so the composed question reaches the middleware intact
rather than being split into several unrelated single-line questions.

### Limits and validation

The client reads the effective question ceiling from the `/health`
`capabilities` block (`max_question_chars`, default 16,000) and falls back to a
local default against older servers. It:

- shows a `current/max chars` usage line once a draft passes 80% of the limit;
- **rejects an over-limit draft locally** — nothing is sent, so you never get a
  server-side rejection for a question the client could see was too long;
- prints the **full structured** FastAPI `422` validation detail (every
  `loc`/`msg`/`type`), never a clipped 200-character preview, so the offending
  field and its limit are visible;
- treats a `422` (or `400/401/403/409`) as a per-request error only — it falls
  back to `POST /query` for that one request and **keeps streaming enabled**;
  streaming is disabled for the session solely on the documented `404/405`
  capability response.

## Session visibility (2.2.2.3)

The input prompt carries a compact session/turn suffix so you always know which
conversation you are in and how much history is attached:

```text
session 7f2a · 4 turns
you>
```

(`session <id4> · history off` when `/history off` is active.) After an answer,
the renderer also surfaces conversation state:

- **history truncated** — shown when the server used fewer turns than you sent
  (the conversation exceeded the server's bound).
- **topic reset** — shown when the server detected a new topic and did not carry
  earlier context into this question.
- **`context: AMD · revenue · FY2025`** — the entities/metrics/timeframe carried
  from earlier turns; shown **only** under `/verbose on`.
- **`retrieval query: …`** — the standalone rewritten retrieval query; a
  `/verbose`-only diagnostic. It is **never** echoed as if it were your wording.

`/history` previews only the local turn text (number, role, short preview); it
never prints hidden system prompts, tool schemas, or retrieved evidence.

## Capabilities block

At startup, and any time you run `/health`, the client reads (or shows)
`capabilities` from `GET /health`:

- `tools` — whether the model can call middleware tools this deployment.
- `streaming` — whether this deployment can serve `/query/stream` at all. With
  tools enabled it is available only when tool-final streaming is on (2.2.6.1);
  otherwise the endpoint is disabled and the client uses `POST /query`.
- `streaming_tool_final` — present and `true` when a tools-enabled request runs
  its bounded tool/planning rounds non-streaming and then streams only the final
  answer synthesis. Absent/false means tools and streaming don't combine here.
- `answer_policy` — the server's configured default answer policy
  (`strict` or `graded`), before any per-session `/grounding` override.
- `history` / `multiline` — whether this build honors bounded conversation
  history and accepts multi-line questions (2.2.2). Older servers omit these.
- `max_question_chars` — the effective per-question ceiling the `/ask` composer
  enforces locally (default 16,000 when the server doesn't advertise it).
- `conversation_max_turns` / `conversation_max_history_chars` — the server's
  bound on how much history it will actually use per request.

The startup `Capabilities:` line surfaces `history` and `max_question_chars`
alongside `tools`/`streaming`/`answer_policy` when the server reports them.

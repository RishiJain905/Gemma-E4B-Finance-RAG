# 2.1.4 Results — Tools On vs. Off (2.1.1 Harness)

Runs on 2026-07-05 against the live TraceAlchemy model (llama-server, `:8087`),
43 golden cases (42 original + `analytical-lowest-pe-001` added in 2.1.4.2),
middleware on `:8000`, `allow_write_tools=false` in both runs.

- **Tools OFF**: run `1783228468` — original server config.
- **Tools ON** (`ENABLE_TOOLS=true`): run `1783232334` — server restarted with
  `--jinja` (required for llama-server to render `tools` into the chat
  template), `-c 32768`, MTP draft head disabled (see Server Notes).

## Headline metrics

| Metric | Tools OFF | Tools ON | Δ |
|---|---|---|---|
| intent_accuracy | 0.907 | 0.907 | — |
| ticker_accuracy | 0.814 | 0.814 | — |
| retrieval_hit_rate | 1.000 | 1.000 | — |
| keyword_coverage | 0.630 | 0.630 | — |
| **refusal_rate** (lower better) | **0.535** | **0.372** | **−0.163** |
| answer_relevance (judge) | 0.749 | 0.830 | +0.081 |
| faithfulness (judge) | 0.905 | 0.757* | see caveat |
| avg latency / case | 28.0 s | 7.9 s* | confounded |

Regression gate: the tools-off run **passes** `eval/gate.py` against the
committed pre-2.1.4 `eval/baseline.json` (tools-off path unchanged).

## Per-category refusal rate (the metric 2.1.4 targets)

| Category | n | OFF | ON |
|---|---|---|---|
| analytical | 4 | 1.00 | 0.50 |
| sentiment | 2 | 1.00 | 0.00 |
| risk | 2 | 1.00 | 0.00 |
| news | 2 | 0.50 | 0.00 |
| macro | 5 | 0.20 | 0.00 |
| fact_lookup | 18 | 0.22 | 0.28 |
| comparison | 3 | 1.00 | 1.00 |
| explanation / trend / projection / hard | 6 | 1.00 | 0.83 |

The flagship case works end-to-end when the tool fires: *"Which tracked stock
has the lowest forward P/E?"* → model calls
`query_facts(metric=forward_pe, order=asc)` → **"META has the lowest forward
P/E at 15.93 [Source: query_facts/META]"** (4/4 in isolated smoke tests;
sanity rules exclude ETFs — TLT's junk −4288 — and the stray non-watchlist
ticker `E`).

## Caveats & findings

1. **Tool invocation is probabilistic.** At `temperature 0.3`, the 4B model
   calls a tool on aggregation questions roughly half the time inside the full
   pipeline (long augmented context makes it answer/refuse from context
   instead). The eval's analytical row (0.50 refusal) reflects this;
   isolated smokes of the golden case were 4/4. `tool_choice: "required"`
   is **not enforced** by the current turboquant llama-server build (probed:
   2/3 requests ignored it), so invocation cannot be forced yet.
2. **Faithfulness drop is mostly a judge artifact.** The 2.1.1 judge scores
   answers against *retrieved document context*; answers grounded in tool
   results (sentiment/news cases it scored 0.0) cite data the judge never
   sees. One case was also skipped (`judge unavailable`). Fixing the judge to
   include tool results in the context belongs to 2.1.5.
3. **Latency is not like-for-like.** The tools-on run used `-c 32768` and no
   MTP draft. Incidental finding: the MTP head's draft acceptance rate was
   ~7% (sampled), i.e. a net slowdown plus VRAM cost at current settings.
4. **VRAM ceiling.** The first tools-on attempt crashed llama-server with a
   ROCm OOM at a 6.6k-token tool-loop prompt under the original
   `-c 131072` config. Tool loops lengthen prompts; context size needs VRAM
   headroom (32k was stable).
5. **Comparison category unchanged** — the model does not yet map "compare X
   vs Y" phrasing to `query_facts`; candidate 2.1.5 work (few-shot tool
   examples or intent-conditioned prompting).

## Shipped defaults & how to enable

`enable_tools: false` stays the shipped default: the tools-off path is
byte-identical to pre-2.1.4 (regression-gated), and flipping the default
should come with the 2.1.5 judge fix plus a server config that always runs
`--jinja`. To use tools today:

1. Start llama-server with `--jinja` (and enough VRAM headroom, e.g.
   `-c 32768`). Without `--jinja` the middleware detects the empty-response
   signature once and falls back to the plain path for the session.
2. `ENABLE_TOOLS=true` (env) or `enable_tools: true` in
   `configs/middleware.yaml`; optionally `ALLOW_WRITE_TOOLS=true` for
   `refresh_data` (per-query budget `max_refreshes_per_query: 2`).
3. `scripts/chat.py` → `/tools` lists the registered suite (9 tools).

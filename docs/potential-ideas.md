# Phase 2 — Potential Ideas

Forward-looking upgrade ideas for taking Gemma-E4B-Finance-RAG to the next
level. Ordered by leverage within each tier. Nothing here is committed work —
it's a backlog to pull from when scoping Phase 2.

---

## Tier 1 — Quality (do these first)

Answer quality is the product. These have the highest leverage.

1. **Evaluation harness.**
   A golden set of Q&A plus automated scoring (faithfulness, answer relevance,
   retrieval hit-rate — RAGAS-style). You can't improve answer quality you
   don't measure, and every change below should be judged against it.

2. **Hybrid retrieval + re-ranking.**
   Add BM25 lexical search alongside the vector search, fuse the two with
   Reciprocal Rank Fusion (RRF), then run a cross-encoder re-ranker (e.g.
   `bge-reranker`) over the top candidates. Usually the single biggest
   answer-quality jump. The current retriever is vector search + a naive
   SQLite fact lookup, with no lexical channel and no re-ranking.

3. **Smarter chunking.**
   Replace the fixed ~1000-char windows in `ChromaStore` with structure-aware
   chunking (by filing section / sentence boundaries). Bad chunks cap how good
   retrieval can ever be.

4. **Structured & analytical tooling (the tool suite).**
   Vector retrieval finds *relevant text* — it cannot sort, rank, filter, or
   compute `MIN/MAX/AVG` across the data. Analytical questions ("which stock
   has the lowest forward P/E", "top 5 by revenue growth", "P/E under 20")
   require giving the model **tools it can call** over the structured store.
   Each tool is a function the middleware exposes; the model decides when to
   call one (depends on the function-calling mechanism in Tier 2 #4) and then
   reasons over the truthful result it gets back.

   Suggested tool suite (more tools doesn't hurt — favour many small, focused,
   read-only tools plus a guarded write tool):

   | Tool | What it does | Backs onto |
   |------|--------------|-----------|
   | `query_facts` | Aggregate / rank / filter over the `fundamentals` table — lowest/highest, top-N, thresholds, sorts. The core analytical tool ("lowest forward P/E"). | SQLite `fundamentals` (text-to-SQL or safe predefined queries) |
   | `get_fundamentals` | Fetch specific metric(s) for one or several tickers, incl. multi-ticker comparison ("NVDA vs AMD revenue"). | `Store.get_fundamentals_batch` |
   | `search_documents` | Hybrid semantic search over filings / news / IR for qualitative context, on demand. | `Store.search` / ChromaDB |
   | `list_metrics` / `describe_schema` | Introspection: which tickers and metric names actually exist, so the model picks valid arguments and never invents a metric. Grounds the other tools. | SQLite schema + distinct metrics |
   | `get_macro_snapshot` | Current FRED macro indicators (GDP, CPI, rates, unemployment). | `/macro/snapshot` |
   | `get_sentiment` | GDELT tone summary for a ticker over a window. | `/sentiment/{ticker}` |
   | `get_guidance` | Latest earnings guidance for a ticker. | `/guidance/{ticker}` |
   | `check_freshness` | Per-ticker / per-source freshness, so the model can decide whether data is stale before answering. | `/freshness/{ticker}` |
   | `refresh_data` | **State-changing** — trigger ingestion for a ticker/source mid-conversation (lets the model fetch missing/stale data). Should be guarded / confirmed. | `/refresh/{ticker}` + scheduler |
   | `compute` *(optional)* | Derived calculations not stored as facts (growth %, ratios, deltas). | in-process math over fetched values |

   Cross-cutting concerns for the suite:
   - **Data-sanity filtering.** `query_facts` must exclude non-equities and
     implausible values, or "lowest forward P/E" wrongly returns junk like
     `TLT` at `-4288` (a bond ETF with no real P/E). Filter ETFs/bonds and
     out-of-range values.
   - **Read-only vs. write.** Everything except `refresh_data` is read-only and
     safe to call freely; `refresh_data` mutates state and may be slow, so gate
     it behind a confirmation or rate limit.
   - **Grounding.** Expose `list_metrics` / `describe_schema` so the model
     calls tools with valid tickers/metric names instead of guessing.

5. **Forward-looking / projection data (analyst estimates & price targets).**
   Today the store only holds *historical/current* facts, so the model can't
   answer "what should I expect for next-quarter earnings / the stock price?"
   without inventing numbers. The fix is **not** to let the model freelance a
   forecast — it's to feed it **analyst consensus estimates** and have it
   *reason over and explain* them with caveats. That means a new data source,
   a new ingestion path, and a new tool.

   **Data sources (free-tier first):**

   | Source | Forward-looking data | Notes |
   |--------|----------------------|-------|
   | **yfinance** (already a dependency, no key) | Earnings/revenue estimates, analyst price targets (high/low/mean), recommendation trend | Zero-cost starting point; reuse the existing ingestor. Coverage/stability is best-effort. |
   | **Financial Modeling Prep (FMP)** | Financial Estimates API (projected revenue/EPS) + Price Target Consensus (high/low/median/consensus) | Clean JSON, generous free plan; best structured option. |
   | **Finnhub** | EPS/revenue estimates, recommendation trends, price targets | Free tier; good alternative/backstop to FMP. |
   | **Alpha Vantage** | Some estimate endpoints | Free key but tight rate limits (~25 req/day) — use as a fallback only. |

   **What it requires:**
   - **New ingestion path** storing consensus estimates as forward-dated facts
     (e.g. `estimate_revenue`, `estimate_eps`, `price_target_mean/high/low`,
     `num_analysts`, `recommendation`) under future periods like `FY2026E` /
     `2026-Q3E`, kept distinct from realized historicals.
   - **New tool** — `get_estimates` / `get_price_targets` (read-only) so the
     model can pull forward-looking consensus on demand and ground its
     "what to expect" answer in it. Pairs with the Tier 1 #4 tool suite.
   - **RAG / prompt tuning** — frame projections as *"analyst consensus expects
     X (range Y–Z, N analysts)"* plus the model's reasoning, always with an
     explicit "estimates, not guarantees / not financial advice" caveat. A
     dedicated `projection` / `outlook` intent can route these queries to the
     estimate tools and the right prompt.

   Truthfulness guardrail: the model reports and reasons over *sourced
   consensus*, it does not fabricate its own price target.

---

## Tier 2 — Capability

New things the system can do.

4. **Tool / function calling (agentic).**
   Let the model call `/search`, structured fact lookups, and `/refresh`
   itself, so it can decide to fetch or refresh data mid-answer. Directly
   enables "tell the model to run the jobs."

5. **Conversation memory.**
   Multi-turn context so `/query` is a real chat instead of stateless
   one-shots (each query currently does independent retrieval + answer).

6. **Streaming responses.**
   Stream tokens to the client (e.g. `scripts/chat.py`). The `stream` field is
   already reserved in `QueryRequest`. Big UX win.

---

## Tier 3 — Robustness / Ops

Reliability, trust, and operability.

7. **Self-critique / citation verification.**
   Verify the answer's `[Source: …]` citations actually exist in the retrieved
   context to cut hallucination.

8. **Prompt + sampling tuning per intent.**
   `model.yaml` already has per-task temperature / max_tokens; wire these in by
   question type and finish off the empty-completion edge case.

9. **Observability + caching.**
   Query/latency metrics, retrieval hit-rate logging, an embedding cache for
   repeated queries, and incremental ingestion (skip re-embedding unchanged
   documents).

---

## Cross-cutting: data & infrastructure

These feed everything above and are worth slotting in early.

### Better / more sources
Prefer **replacing scrapers with real free APIs** over simply adding sources —
this kills the fragility seen with earnings transcripts and IR pages.

- **SEC `companyfacts` XBRL API** — free, official, *structured* financials.
  Could replace fragile yfinance/scraped fundamentals with authoritative data.
- **Finnhub** (free tier, generous) — fundamentals, news, analyst estimates,
  insider transactions, earnings, via a clean API.
- **Financial Modeling Prep / Tiingo** (free tiers) — fundamentals + transcripts
  via API instead of scraping Seeking Alpha.
- **StockTwits / Reddit** — retail sentiment to complement GDELT.

### Vector DB memory / speed (defer until scale)
At current scale (hundreds of docs, ~3 MB of vectors) memory is **not** a
problem; HNSW in RAM is negligible. Revisit only at hundreds of thousands of
vectors. Real levers when that time comes:

- **Quantization** — int8 instead of float32 (≈4× smaller, tiny recall loss),
  or binary quantization (≈32× smaller). Biggest bang for buck.
- **Smaller dimensions** — 2560-dim is large; Matryoshka truncation (if the
  model supports it) or PCA to 768/1024 cuts memory and search time.
- **Leaner backend** — `sqlite-vec` (vectors in the existing SQLite file —
  collapses the two stores into one, very low memory), **LanceDB** (on-disk,
  memory-mapped), or **Qdrant** (production ANN with built-in quantization).

---

## Recommended sequence

1. Evaluation harness (Tier 1 #1)
2. Hybrid retrieval + re-ranking (Tier 1 #2)
3. Tool / function calling (Tier 2 #4)
4. Conversation memory + streaming (Tier 2 #5, #6)

Slot the **SEC XBRL** and **Finnhub** source upgrades in early, since they
improve the data everything else feeds on. Hold the vector-DB optimization
until scale actually demands it.

# Phase 2.2 — Adaptive Conversational Finance RAG Specs

Phase 2.2 evolves Gemma-E4B-Finance-RAG from a capable but single-turn,
single-plan hybrid pipeline into a **bounded Adaptive Conversational RAG**
system. It keeps the parts that already work — SQLite facts, Chroma documents,
BM25/vector RRF, optional re-ranking, finance tools, freshness handling, and
fail-soft behavior — and adds the missing query-understanding, evidence-quality,
and conversational layers around them. It also adds a local, live retrieval
graph that shows the actual query/evidence path and supports read-only corpus
exploration without turning retrieval into GraphRAG.

This is intentionally an incremental architecture change, not a rewrite. The
local TraceAlchemy model remains the final answer generator. Simple questions
must stay cheap, while complex questions may use a tightly bounded planning and
corrective path.

The detailed architecture comparison and decision rationale are in
[ARCHITECTURE-DECISION.md](ARCHITECTURE-DECISION.md).

---

## What the phase audit found

Every phase under `docs/` was reviewed before defining this roadmap.

| Phase | What it established | Important Phase 2.2 implication |
|---|---|---|
| 1.1 | Local `llama-server` chat and embedding endpoints | Extra model calls consume the same local GPU budget; they must be bounded. |
| 1.2 | SQLite + ChromaDB behind `Store` | Preserve the dual-store split: exact facts and semantic documents need different retrieval strategies. |
| 1.3 | Yahoo Finance fundamentals/news and TTL ingestion | Explicit ticker/metric questions became the easiest path, which helped create the current conciseness bias. |
| 1.4 | SEC filing discovery and parsing | Filing metadata exists, but Phase 2.1 results show full filing text is not actually represented in the vector corpus. |
| 1.5 | Stateless FastAPI query pipeline, intent parser, retriever, and prompt augmenter | The core request remains one question, one primary ticker, one intent label, and one retrieval plan. |
| 1.6 | FRED, GDELT, transcripts, macro/sentiment/guidance routes | Query planning must retain multiple source modalities for compound questions. |
| 1.7 | Unified scheduling, resilience, freshness, IR ingestion | Corrective retrieval should reuse freshness and fail-soft primitives rather than create a new orchestration stack. |
| 1.8 | Production hardening, degraded answers, documentation, broad test coverage | Phase 2.2 must preserve offline-safe verification and degraded behavior. |
| 2.1.1 | Golden-set evaluation and regression gate | The evaluator is single-turn and captures less evidence than generation receives; it must be repaired before judging new architecture work. |
| 2.1.2 | BM25 + vector RRF and optional cross-encoder re-ranking | The system is already Hybrid RAG. Selective query expansion can reuse RRF instead of replacing retrieval. |
| 2.1.3 | Structure/sentence-aware chunking | The existing corpus was short and contained no SEC filing text, so long-document retrieval still needs a real end-to-end path. |
| 2.1.4 | Nine read-mostly tools and a guarded refresh tool | Tools improve analytical answers, but model invocation is probabilistic; obvious finance operations should route deterministically. |
| 2.1.5 | Analyst estimates and projection handling | Future-looking evidence needs explicit period, source, analyst count, and estimate semantics in the common evidence contract. |
| 2.1.6 | Symbol resolution and bounded fetch-on-miss | Conversation state can reuse resolved entities, but must not carry them across unrelated sessions. |
| 2.1.7 | Graded grounding policy | Behavior improved, but its live faithfulness gate remains blocked by evaluator evidence-capture flaws and an inflated refusal-heavy baseline. |
| 2.1.8 | Timings, embedding cache, streaming, and TUI integration | More than 99.9% of measured latency was generation. Streaming improved UX but did not add conversation memory or compound-query understanding. |

### Current architecture classification

The repository is **not Naive RAG**. It is already a modular Hybrid RAG system:

- structured SQLite facts plus ChromaDB document retrieval;
- vector and BM25 channels fused with RRF;
- optional cross-encoder or LLM re-ranking;
- intent-selected facts, document, comparison, macro, and broad strategies;
- optional model tools and fetch-on-miss;
- graded grounding and fail-soft retrieval stages.

The remaining limitation is that those modules are coordinated by a stateless,
single-label query path. Phase 2.2 adds a conversational query compiler,
adaptive routing, bounded correction, and explicit evidence provenance.

### Confirmed causes of the “conciseness trap”

Static inspection found several concrete issues that make terse,
self-contained questions work better than natural research requests:

1. Chroma and BM25 return document bodies under `document`, while
   `PromptAugmenter` reads only `text`. Retrieved documents can therefore count
   as grounding while their bodies are blank in the model prompt.
2. `scripts/chat.py` sends only the current question. Neither the request model
   nor the middleware carries conversation history.
3. The TUI reads one input line and the API caps a question at 2,000
   characters.
4. Intent parsing returns one primary ticker and one question type. Compound
   asks can lose entities, timeframes, or an entire facts/document modality.
5. Grounding is based mainly on item count, not nonblank, relevant, fresh,
   coverage-complete evidence.
6. Timeframe is parsed but is not consistently applied by fact retrieval.
7. Prompt policy is duplicated between the system message and augmented user
   prompt, including repeated “be concise” and context-only instructions.

Phase 2.2.1 treats these as baseline correctness issues, not as behavior to hide
behind a more elaborate RAG architecture.

---

## Target architecture

```text
raw question + bounded client-owned history
                    |
                    v
      conversational query compiler
      - preserve raw question
      - resolve follow-up references
      - produce entities, intents, periods, operations
      - decompose only when needed
                    |
                    v
          adaptive route selection
       +------------+-------------+
       |            |             |
       v            v             v
  fast facts   standard hybrid   complex bounded
  + compute    vector + BM25     <= 3 subqueries
               optional rerank   <= 2 retrieval rounds
       +------------+-------------+
                    |
                    v
       normalized evidence ledger
       + coverage/sufficiency gate
                    |
          sufficient|borderline|missing
                    |     |       |
                    |     |       +--> honest partial/refusal
                    |     +----------> one internal corrective retry
                    v
        token-budgeted evidence pack
                    |
                    v
       one final answer generation call
                    |
                    v
 deterministic citation + numerical validation
                    |
                    +--> bounded trace hub --> localhost live graph
```

### Hard operational bounds

- Fast and standard lanes use no pre-answer model call.
- The complex lane may use at most one compact planner/rewrite call.
- A query plan contains at most three retrieval subqueries.
- Retrieval performs at most two rounds, including a corrective retry.
- Deterministically recognized finance tools execute in middleware without
  asking the model whether to call them.
- No uncontrolled web search is added to the request path.
- The original user question is never silently shortened. Any configured limit
  produces an explicit validation error.
- Every new stage is feature-flagged and fails back to the current path.
- Graph observation is a non-blocking side channel with bounded in-memory
  traces; it cannot change retrieval, tools, grounding, or answers.

---

## Feature specs

| Folder | Feature | Task files |
|---|---|---|
| `2.2.1-evaluation-fidelity-and-baseline-correctness` | Repair the document/prompt contract, capture exact evidence, and build trustworthy conversational/compound evaluation | 2.2.1.1 – 2.2.1.3 |
| `2.2.2-conversational-query-understanding` | Bounded request history, follow-up resolution, standalone retrieval queries, multiline chat, and session controls | 2.2.2.1 – 2.2.2.3 |
| `2.2.3-adaptive-rag-orchestration` | Multi-entity plans, deterministic finance tools, fast/standard/complex lanes, context budgets, and conditional re-ranking | 2.2.3.1 – 2.2.3.4 |
| `2.2.4-corrective-retrieval-and-provenance` | Evidence sufficiency, one bounded retry, selective query fusion, stable evidence IDs, and numerical/citation validation | 2.2.4.1 – 2.2.4.3 |
| `2.2.5-authoritative-and-long-document-data` | SEC CompanyFacts, actual filing-text indexing, natural section hierarchy, and reversible migration | 2.2.5.1 – 2.2.5.3 |
| `2.2.6-chat-runtime-and-phase-integration` | Tool-aware final streaming, freshness-versioned caching, prompt efficiency, documentation, and phase sign-off | 2.2.6.1 – 2.2.6.3 |
| `2.2.7-live-retrieval-knowledge-graph` | Truthful live query/evidence traces plus an interactive, expand-on-demand localhost corpus explorer | 2.2.7.1 – 2.2.7.4 |

`RESULTS.md` is created inside a feature folder only when that feature is
implemented and evaluated. Empty or speculative result files are not part of
this planning phase.

---

## Suggested build order

1. **2.2.1 Baseline correctness and evaluation fidelity.** Fix the blank
   document path and make the evaluator observe the exact evidence seen by the
   generator. No later metric is trustworthy until this lands.
2. **2.2.2 Conversational query understanding.** Add bounded client-owned
   history, deterministic follow-up resolution, a separate retrieval query,
   and practical multiline chat input.
3. **2.2.3 Adaptive orchestration.** Introduce explicit multi-entity query
   plans and fast/standard/complex lanes. Route safe finance operations
   deterministically.
4. **2.2.4 Corrective retrieval and provenance.** Grade evidence coverage,
   retry once only when justified, and validate evidence IDs, citations, and
   numeric claims.
5. **2.2.5 Authoritative and long-document data.** Add SEC CompanyFacts and
   make real filing text retrievable through a natural section hierarchy.
6. **2.2.6 Runtime and integration.** Stream the final post-tool synthesis,
   add safe versioned caches, reconcile documentation, and run the full gate.
7. **2.2.7 Live retrieval knowledge graph.** Publish bounded query trace
   events, project the Store on demand, serve the Cytoscape.js interface, and
   prove visualization never blocks or changes a query.

Tasks 2.2.5.1, 2.2.6.1, and 2.2.7.1 may be developed after 2.2.1 in parallel,
but final
integration remains ordered because they consume the common evidence and query
contracts.

---

## Phase-wide acceptance gates

Targets below are project acceptance criteria. They are not claims copied from
research papers, and they must be measured against TraceAlchemy and this
finance corpus before a feature is enabled by default.

### Correctness foundation

- 100% of production-shaped Chroma/BM25 document hits placed in generation
  contain the expected nonblank body.
- Blank document hits never increase grounding level.
- Facts retain their own ticker, metric, period, unit, source, and as-of data.
- The evaluator captures the exact facts, documents, tool results, and policy
  delivered to generation.

### Conversation and query planning

- Required ticker/entity/metric/period is preserved in at least 95% of labeled
  follow-up rewrites.
- No new entity is invented in at least 99% of rewrite cases.
- Multi-turn Recall@10 improves by at least 10 percentage points over the
  single-turn baseline.
- Router/plan classification accuracy is at least 90%.
- No history leaks between two `ChatSession` instances or unrelated API calls.

### Retrieval and answer quality

- Complex-query correctness improves by at least 10 percentage points.
- Single-turn retrieval nDCG@10 regresses by no more than 2 points.
- Unsupported numerical claims fall by at least 30% relative.
- Unanswerable/stale-query abstention F1 is at least 0.85.
- Citation IDs resolve to supplied evidence, with at least 95% citation support
  on the labeled set.

### Latency and resource safety

- Simple-query p95 latency increases by no more than 10%.
- Cold-query p95 latency increases by no more than 5% after caching work.
- Warm repeated-query time-to-first-token improves by at least 30% where
  llama-server reports reusable prompt tokens.
- No stale retrieval-cache hit occurs after a tested ingestion/freshness
  watermark change.
- Tests enforce the limits of three subqueries, two retrieval rounds, one
  optional planning call, and one final synthesis call.

### Long documents

- Long-document evidence Recall@10 improves by at least 8 points.
- The packed context token count is no greater than the flat-chunk baseline.
- The Phase 2.2 hierarchy performs no ingestion-time LLM summarization.

### Live retrieval graph

- Every displayed plan, stage, tool, evidence, source, citation, and status
  comes from an executed query object/event; no model call reconstructs it.
- Publishing a graph delta has offline p95 below 2 ms and never awaits a
  browser subscriber.
- The graph becomes visible within 250 ms of the first published localhost
  event in the implementation browser check.
- Trace, element, excerpt, queue, TTL, pagination, and visible-node limits are
  enforced under stress fixtures.
- Zero canary secrets, full prompts, model reasoning, credentials, or local
  paths appear in graph API/static UI output.
- Corpus exploration triggers no refresh, external network, embedding, model,
  or mutation operation.

---

## Deferred ideas

These remain useful experiments, but they are not default Phase 2.2
implementation work:

- **Full GraphRAG:** reconsider when ecosystem relationships or global corpus
  themes become a measured requirement. Start with a deterministic metadata
  graph before any LLM-derived community graph. The Phase 2.2 live graph is an
  observability/read projection and is not used to answer queries.
- **Self-RAG training/reflection tokens:** requires model training and a larger
  evaluation effort; repeated prompt-based self-critique would multiply the
  current generation bottleneck.
- **Default HyDE:** adds a generation and embedding call and can inject
  hypothetical finance details. Pilot only after rewrite/fusion misses are
  measured.
- **RAPTOR recursive summaries:** natural filing sections and parent/child
  expansion are the smaller first step. Add generated hierarchy summaries only
  if measured global/long-document questions still fail.
- **Open-ended ReAct/agent loops:** incompatible with the latency and
  predictability goals. Phase 2.2 uses explicit plans and hard iteration caps.
- **Multimodal filing RAG:** valuable for charts, slide decks, and complex
  tables, but it needs a separate corpus/model capability audit.
- **Vector database replacement:** the current corpus is too small for a
  backend migration to be the highest-leverage optimization.
- **Semantic final-answer caching:** unsafe for nearby but financially distinct
  entities, periods, prices, and news. Phase 2.2 caches only versioned retrieval
  and immutable, explicitly date-bounded results.

---

## Conventions

- Branch from `Rishi-Ghost`, one implementation branch per numbered feature.
- Follow the existing Phase 2.1 task format: Objective, Why This Matters,
  ordered steps with exact file paths, Testing, and Verification Checklist.
- Keep public APIs additive while a feature flag is off.
- New behavior requires deterministic offline tests. Mock network services and
  llama-server; live tests remain opt-in.
- Preserve per-source failure isolation and fail-soft retrieval behavior.
- Prefer small functions and existing modules over a framework or orchestration
  dependency.
- Frontend observer assets are pinned and served locally; no CDN, analytics,
  Node build chain, or third-party telemetry is required at runtime.
- Use the Phase 2.1 evaluator before/after each feature only after 2.2.1 has
  repaired its evidence fidelity.
- Each implemented feature must write `RESULTS.md` with setup, before/after
  metrics, caveats, regression-gate outcome, and an explicit ship/rollback
  decision.
- The repository completion gate remains
  `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1` on Windows or
  `bash scripts/verify.sh` on Unix. This offline gate must pass before a code
  task is complete; live GPU/model evaluation is a separate, explicitly timed
  activity.

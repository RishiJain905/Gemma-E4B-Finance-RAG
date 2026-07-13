# Phase 2.2 Architecture Decision — Bounded Adaptive Conversational RAG

**Status:** Proposed for Phase 2.2 implementation  
**Decision date:** 2026-07-10  
**Scope:** Query understanding, retrieval orchestration, evidence quality,
conversation UX, finance provenance, latency controls, and live retrieval
observability

---

## Context

The current query path is strong when a user asks a short, self-contained
question containing one recognizable ticker and one recognizable operation.
It becomes brittle when the user follows up conversationally, supplies several
requirements in one prompt, refers to more than one entity or period, or asks
for a mixture of exact facts and qualitative evidence.

This behavior is not evidence that the model simply needs shorter prompts. The
repository has concrete pipeline constraints:

- the request contains no conversation history;
- the query schema and intent parser produce one primary ticker and one intent;
- routing selects one main retrieval plan;
- the user question is capped at 2,000 characters and the TUI reads one line;
- production document hits use a `document` field while prompt assembly reads
  `text`, so qualitative evidence may be blank;
- item count, rather than usable evidence coverage, determines grounding;
- prompt policy is duplicated and repeatedly instructs the model to be
  concise;
- generation already dominates measured request latency.

The decision must improve natural-language tolerance, multi-turn usefulness,
retrieval precision, numerical trust, and perceived speed without creating an
unbounded loop around the same local model.

---

## Current architecture classification

The current system is best described as **modular Hybrid RAG with optional
agentic tools**, not Naive RAG.

It already contains:

- structured and semantic stores;
- dense vector retrieval plus sparse BM25 retrieval;
- Reciprocal Rank Fusion;
- optional cross-encoder/LLM re-ranking;
- intent-selected retrieval strategies;
- model-callable finance tools;
- freshness checks, fetch-on-miss, and fail-soft stages.

Its remaining “naive” property is the top-level single-turn orchestration:
one raw query is classified, retrieved once, placed into a mostly fixed prompt,
and answered. Phase 2.2 changes that coordination layer while preserving the
retrieval foundation.

---

## Decision

Adopt a **bounded Adaptive Conversational Hybrid RAG** architecture with a
lightweight Corrective RAG evidence gate.

The architecture has six explicit layers:

1. **Conversational query compiler** — accepts the raw question plus bounded,
   client-owned history; resolves follow-up references; and produces a separate
   retrieval query and structured finance plan without changing the raw user
   request.
2. **Adaptive router** — selects a fast structured lane, the existing standard
   hybrid lane, or a bounded complex lane based on entities, intents,
   operations, and retrieval complexity.
3. **Specialized retrieval execution** — runs SQLite, calculation, document,
   news, macro, estimate, or freshness operations per subquery, then normalizes
   results into a common evidence ledger.
4. **Corrective evidence gate** — checks nonblank content, relevance, entity,
   metric, period and subquery coverage, source authority, freshness, and
   conflicts. Borderline evidence may trigger one internal retry; missing
   evidence produces an honest partial answer or refusal.
5. **Grounded generation and validation** — packs evidence to a token budget,
   performs one final generation call, then deterministically validates
   evidence IDs, citations, and numerical claims.
6. **Live observability side channel** — publishes bounded graph deltas from
   executed query/evidence objects and projects corpus metadata on demand for a
   localhost interface. It never participates in route or answer decisions.

This is a composition of established RAG ideas, not a claim to implement any
paper verbatim.

---

## Decision drivers

1. **Fix the observed user problem first.** Conversation state, multiline
   input, follow-up rewriting, and compound plans directly reduce the need to
   phrase every question as a concise standalone command.
2. **Protect local latency.** Research techniques that add generation calls
   are used only on a measured complex path.
3. **Reuse existing strengths.** SQLite, Chroma, BM25/RRF, re-ranking, tools,
   freshness, resilience, and evaluation stay in place.
4. **Make finance operations deterministic.** Exact comparisons, rankings,
   filters, ratios, growth calculations, periods, currencies, and units should
   be computed or queried, not improvised by the model.
5. **Prefer evidence quality over context volume.** A larger model context is
   not a reason to fill it. Relevant, diverse, source-aware evidence should be
   packed deliberately.
6. **Keep failure behavior explicit.** A failed planner, re-ranker, cache, or
   corrective step falls back or returns a transparent partial result; it does
   not crash the query.
7. **Require measurable promotion.** No architecture label or paper result is
   sufficient to enable a feature against the local model without a repository
   evaluation.

---

## Query contract

### Raw question and history

The API remains stateless. Conversation state is owned by the client and sent
with each request as a bounded list of structured turns. A `session_id` may be
included for tracing, but it is not a server-side lookup key.

Recommended defaults:

- raw question: up to 16,000 characters;
- history: at most eight recent turns and at most 8,000 characters after
  deterministic selection;
- no silent truncation of the current raw question;
- clear validation metadata when configured budgets are exceeded;
- `scripts/chat.py` owns independent history per `ChatSession` and provides
  `/new` or `/clear`.

This choice avoids a conversation database, persistence policy, cleanup job,
and cross-user server state while making the same contract available to API
clients other than the TUI.

### Query plan

The compiler returns an explicit plan rather than another free-form prompt:

```json
{
  "raw_question": "And compare its margin trend with AMD over the last year",
  "retrieval_query": "Compare NVDA and AMD gross margin trend over the last year",
  "entities": ["NVDA", "AMD"],
  "intents": ["comparison", "trend"],
  "metrics": ["gross_margin"],
  "timeframe": {"kind": "relative", "value": "last_year"},
  "operations": ["compare", "trend"],
  "subqueries": [],
  "complexity": "complex",
  "resolution_source": "history+current_turn"
}
```

Deterministic entity, metric, period, and pronoun carry-forward runs first. An
optional planner model call is allowed only when rules cannot safely resolve an
ambiguous or compound request. The planner must return schema-valid JSON and
fall back to the deterministic/current pipeline on failure.

---

## Route contracts

### Fast lane

Use when the plan contains a supported exact operation such as one-ticker fact
lookup, rank/filter, deterministic comparison, calculation, freshness check,
or current estimate lookup.

- no planning generation;
- structured SQLite/tool execution first;
- no document retrieval unless the plan asks for explanation or qualitative
  support;
- one final answer call, with degraded formatting available if the model is
  down.

### Standard lane

Use for one coherent qualitative or mixed question with clear entities and no
multi-hop dependency.

- no planning generation;
- existing vector + BM25/RRF retrieval;
- re-ranking only when configured and justified by candidate ambiguity;
- one retrieval round and one final answer call.

### Complex lane

Use for compound, multi-entity, multi-period, ambiguous, comparison-plus-cause,
or evidence-synthesis questions.

- at most one planner/rewrite call;
- at most three subqueries;
- original/resolved query always retained and weighted strongest;
- independent structured/document retrieval per subquery;
- at most one corrective retry, for a total of two retrieval rounds;
- one final answer call.

The complex lane is not a ReAct loop. Tests enforce every cap.

---

## Common evidence contract

All retrieval and tool results are normalized before grounding or prompt
assembly. The contract must preserve at least:

```text
evidence_id
kind: fact | document | tool_result | calculation
ticker/entities
metric/operation
value and unit
period and as_of
source_type and source_url
text/document body
parent_id, section, and chunk position when applicable
retrieval/re-rank scores
freshness status and source watermark
subquery_ids covered
```

An empty document body is not evidence. A result is not “grounded” merely
because three records were returned.

The final prompt uses stable evidence IDs such as `[E1]`. The response
postprocessor verifies that cited IDs exist and that specific numerical claims
are present in supplied facts, calculations, or tool results. Validation does
not silently rewrite an unsupported claim into a different assertion; it
marks, removes, or downgrades the answer according to the configured policy.

---

## Evidence sufficiency and correction

The gate returns `sufficient`, `borderline`, or `missing` with machine-readable
reasons.

It checks:

- nonblank usable bodies or values;
- requested entity, metric, period, intent, and subquery coverage;
- minimum relevance or exact-match requirements appropriate to the modality;
- source authority for the question type;
- freshness/as-of requirements;
- contradictions or incompatible units/periods;
- evidence diversity after parent/chunk deduplication.

Actions are bounded:

- **sufficient:** pack and answer;
- **borderline:** broaden a filter, use a deterministic alias, expand a parent
  section, or issue one alternate internal query;
- **missing:** return a transparent partial/refusal, or expose the existing
  guarded refresh path when user policy allows it.

CRAG's web-search extension is not added. External data enters through audited
ingestors and existing freshness controls.

---

## Context and latency policy

- Rank and pack evidence by requested coverage, authority, relevance,
  freshness, and diversity rather than raw retrieval order alone.
- Deduplicate sibling/overlapping chunks and keep the parent heading and only
  the adjacent material needed for coherence.
- Keep policy in one system-prompt builder; the augmented user message contains
  conversation context, evidence, and the untouched raw question.
- Use intent-specific output limits, but do not globally command every answer
  to be terse.
- Batch embeddings and overlap independent SQLite/BM25/vector work where the
  local backend benefits.
- Cache query embeddings and retrieval results only with a key containing the
  model/config fingerprint, query plan, filters, top-k, and source freshness
  watermark.
- Never serve a cached volatile answer because it is semantically similar to
  another entity or period.
- Keep prompt prefixes stable and enable llama-server prompt reuse only after
  timing metadata proves reused tokens and correctness tests cover invalidation.

---

## Live retrieval graph decision

Add an optional localhost observer with two modes:

- **Live Trace** — query, conversation compilation, plan/subqueries, selected
  lane, retrieval/tool stages, evidence, sources, correction, citations, and
  validation status update as a real request executes.
- **Corpus Explorer** — read-only, paginated, expand-on-demand projection of
  sources, tickers, metrics, facts, filings, sections, document families, and
  freshness from the authoritative Store.

The observer uses a bounded in-memory trace hub, the existing FastAPI
application, SSE deltas, and REST snapshot/explorer endpoints. Slow or missing
browsers cannot block a query. Trace data is redacted by allowlist, excerpted,
TTL-limited, and not persisted by default.

### Renderer choice

Use **Cytoscape.js** for the initial interface.

- It is graph-native and supplies compound nodes, layouts, selection/events,
  filtering, and style selectors needed for evidence inspection.
- The visible graph is intentionally bounded and expanded on demand, so a
  general large-graph WebGL renderer is unnecessary.
- A clear 2D directed trace is easier to read, select, label, navigate by
  keyboard, and show on a side monitor than a 3D force scene.

Three.js is deferred because it is a lower-level 3D scene library and would
require custom graph layout, raycasting/selection, labels, filtering, and
accessibility work. Sigma.js is the preferred fallback only if measured corpus
views require tens of thousands of simultaneously visible elements; its WebGL
strength is unnecessary under the expand-on-demand contract.

Serve one pinned Cytoscape.js asset, fonts, licenses, and hashes locally. Do
not add React/Vue, a Node build chain, a CDN, analytics, or a graph database.

This feature is **not GraphRAG**. It visualizes the existing retrieval and Store
state; it neither creates LLM-derived graph communities nor changes what is
retrieved for an answer.

---

## Alternatives considered

### 1. Patch only the chat client

**Decision:** necessary first fixes, insufficient architecture.

Raising the input cap, adding multiline input, fixing the document field, and
adding history immediately improve usability. They do not solve the
single-intent router, missing timeframe semantics, probabilistic tools,
evidence-count grounding, or complex retrieval coverage. These repairs are
included in 2.2.1 and 2.2.2, followed by the adaptive layers.

### 2. Full GraphRAG

**Decision:** defer.

GraphRAG is designed for global questions over large private corpora by
extracting entity/relationship graphs and generating community summaries. It
can be valuable for questions such as cross-company ecosystem risks or themes
across thousands of filings. It is not a universal improvement for exact,
time-sensitive finance facts.

At the current scale it would add expensive indexing, summary staleness,
incremental graph maintenance, and additional generation calls before the
repository has fixed its direct evidence path. If relationship-heavy cases
become a measured priority, first build a deterministic metadata graph from
issuer, filing, period, industry, metric, and macro relationships.

### 3. Full Self-RAG

**Decision:** reject for Phase 2.2.

Self-RAG's published architecture trains models to emit special retrieval and
reflection tokens. Implementing it faithfully requires model training. A
prompt-only imitation would add repeated calls to the same local model and
depend on that model to detect its own errors. The bounded evidence gate gives
most of the operational benefit with deterministic controls.

### 4. Default HyDE

**Decision:** defer to an experiment.

HyDE generates a hypothetical relevant document, embeds it, and retrieves real
documents around that embedding. It is most compelling for weak zero-shot
dense retrieval. Here it adds generation plus embedding on the same server and
may introduce fabricated issuers, dates, or figures into the retrieval query.
Pilot only on a labeled subset where history-aware rewrite and selective fusion
still miss evidence.

### 5. RAG-Fusion on every query

**Decision:** use selectively.

Multi-query fusion fits the existing RRF implementation, but query variants can
drift and the extra candidates may be discarded by fixed re-ranking/context
budgets. Use it only for complex or low-coverage plans, retain the original
query with the strongest weight, and cap variants.

### 6. RAPTOR recursive summaries

**Decision:** defer generated hierarchy; implement natural hierarchy first.

SEC filings and transcripts already contain document, section, and paragraph
structure. Preserve and retrieve that hierarchy without ingestion-time LLM
summaries. Reconsider RAPTOR only if long-document/global questions remain weak
after the actual filing corpus is indexed and evaluated.

### 7. Open-ended agentic/ReAct retrieval

**Decision:** reject as the default.

Unbounded reasoning/retrieval loops conflict with local latency, predictable
resource use, and straightforward failure recovery. Phase 2.2 uses explicit
plans, deterministic tools, and one corrective retry.

### 8. Multimodal RAG

**Decision:** separate future phase.

Charts, tables, investor decks, and filing images can add important evidence,
but the ingestion formats, image/table extraction, embedding strategy, and
model modality support need their own audit. They do not fix the current
text-evidence and conversation defects.

### 9. Replace Chroma/vector storage

**Decision:** defer.

The current corpus is small and Phase 2.1 measured generation, not vector
search, as the dominant latency. Better query plans, evidence delivery, and
context budgets have higher leverage than a vector backend migration.

### 10. Three.js or Sigma.js for the live graph

**Decision:** Cytoscape.js first; keep explicit scale-based fallback criteria.

Three.js offers visual depth but increases implementation and accessibility
cost for an information-dense debugging tool. Sigma.js efficiently renders very
large graphs, while Phase 2.2 deliberately caps visible elements and needs
compound query/stage groups. Reconsider Sigma.js only if profiling shows the
bounded Cytoscape view cannot meet interaction targets.

---

## Consequences

### Positive

- Natural follow-ups and verbose multi-part questions become first-class
  inputs.
- Simple queries remain on a short deterministic path.
- Existing retrieval, tools, data sources, and resilience logic are reused.
- Evidence quality and coverage become observable before generation.
- Finance numbers, periods, units, and citations gain deterministic checks.
- Expensive techniques are isolated behind measured complex routes and flags.
- Users can inspect live query/evidence flow and corpus coverage on localhost
  without changing retrieval behavior.

### Costs and risks

- Query plans and evidence normalization introduce new API metadata and tests.
- Client-owned history requires every conversational client to send turns
  correctly.
- Deterministic rewriting can carry stale context after a topic shift; explicit
  topic-shift tests and `/new` are required.
- Rule-based sufficiency thresholds are not calibrated probabilities and must
  be evaluated per route.
- One planner call can still add meaningful latency; the router must prefer
  deterministic compilation.
- Citation/numeric validation can produce more partial answers until source
  coverage improves.
- Trace/event schemas and redaction limits become compatibility/security
  contracts; observer code must remain strictly non-blocking.

---

## Evaluation requirements

No Phase 2.2 capability is promoted based only on a paper result. The repaired
evaluation set must cover:

- standalone facts and explanations;
- later-turn references and topic shifts;
- verbose and multiline prompts;
- facts plus qualitative evidence in one request;
- multiple tickers, metrics, periods, and operations;
- analytical calculations and tool routes;
- stale, conflicting, missing, and unanswerable evidence;
- long SEC sections and adjacent-context requirements;
- unsupported citations and numerical claims;
- latency, prompt tokens, embedding calls, model calls, and route caps.
- live trace completeness, event overhead, redaction, reconnect/reset, corpus
  pagination, visible element caps, and observer-disabled noninterference.

The project-wide numeric gates are defined in `README.md`; each feature spec
adds its own focused checks and rollback criteria.

---

## Research basis

Primary papers and official project material used to evaluate the alternatives:

- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks](https://arxiv.org/abs/2005.11401).
- Ma et al., [Query Rewriting for Retrieval-Augmented Large Language
  Models](https://aclanthology.org/2023.emnlp-main.322/).
- Katsis et al., [mt RAG: A Multi-Turn Conversational Benchmark for Evaluating
  Retrieval-Augmented Generation Systems](https://aclanthology.org/2025.tacl-1.36/).
- Jeong et al., [Adaptive-RAG](https://arxiv.org/abs/2403.14403).
- Yan et al., [Corrective Retrieval Augmented
  Generation](https://arxiv.org/abs/2401.15884).
- Rackauckas, [RAG-Fusion](https://arxiv.org/abs/2402.03367).
- Gao et al., [Precise Zero-Shot Dense Retrieval without Relevance Labels
  (HyDE)](https://arxiv.org/abs/2212.10496).
- Asai et al., [Self-RAG](https://arxiv.org/abs/2310.11511).
- Sarthi et al., [RAPTOR](https://arxiv.org/abs/2401.18059).
- Edge et al., [From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/).
- Liu et al., [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/).
- Chen et al., [FinQA](https://aclanthology.org/2021.emnlp-main.300/).
- Islam et al., [FinanceBench](https://arxiv.org/abs/2311.11944).
- ggml-org, [llama-server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
- [Cytoscape.js documentation](https://js.cytoscape.org/).
- [Sigma.js documentation](https://v4.sigmajs.org/).
- [Three.js Raycaster documentation](https://threejs.org/docs/pages/Raycaster.html).
- FastAPI, [Server-Sent Events](https://fastapi.tiangolo.com/tutorial/server-sent-events/).

Published results commonly use different corpora, retrievers, and much larger
or specially trained generators. They motivate experiments; they do not
replace local evaluation.

---

## Phase 2.2.7 as-built confirmation (2.2.7.4)

The proposed live-retrieval-observability design shipped as specified across
2.2.7.1–2.2.7.4. This note ratifies the two load-bearing decisions.

**Cytoscape.js was used, as proposed.** One pinned Cytoscape bundle plus fonts,
licenses, and SHA-256 hashes are served locally from
`src/middleware/static/graph/` (no CDN, no Node build chain, no React/Vue, no
analytics, no graph database). The visible graph stays bounded and expands on
demand, so the general WebGL renderers (Sigma.js / Three.js) remained
unnecessary. The interface is keyboard-navigable and readable on a side monitor,
matching the rationale in *Renderer choice* above.

**This is an observability graph, not GraphRAG — the distinction held in the
implementation, not just the design.** Concretely:

- *No effect on retrieval or the answer.* The observer subscribes an extra
  callback to the single existing `QueryEvent` emitter (2.2.6.1) and projects
  events into a bounded graph. When the flag is off, no emitter is installed and
  the `/query` path is byte-identical; when on, the only response change is an
  optional `graph_trace_id`. It never re-ranks, re-retrieves, or re-prompts.
- *No LLM-derived graph.* Nodes/edges are the *actual* executed pipeline elements
  (query, validated plan, request-local subqueries, executed stages/tools,
  retrieved evidence and its real source, terminal answer/citations/validation)
  and, in the explorer, the *authoritative* Store inventory. There are no
  entity/relationship extraction passes, no community summaries, and no extra
  generation calls — the two things that define GraphRAG.
- *Direction of data flow is the opposite of GraphRAG.* GraphRAG builds a graph
  to *drive* retrieval; here retrieval drives the graph. The graph is a faithful
  read-out of what already happened, redacted and bounded for local viewing.

If relationship-heavy questions later become a measured priority, the deferred
path remains a deterministic metadata graph (issuer/filing/period/industry/
metric/macro) before any GraphRAG-style indexing — the corpus projector
(2.2.7.2) is a first, read-only step in exactly that direction.

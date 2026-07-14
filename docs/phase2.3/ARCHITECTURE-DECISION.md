# Phase 2.3 Architecture Decision — Capability-Layered Market Coverage

## Decision

Adopt a capability-layered ingestion architecture with a canonical security
universe, explicit coverage tiers, normalized provider records, dual-store
placement, bounded read projections for graph exploration, and an authoritative
query-catalog path for corpus capability questions.

Provider adapters remain small and independent. They do not define the database
schema, scheduling model, graph vocabulary, or retrieval policy.

## Alternatives considered

### 1. Expand the existing `core` watchlist to every constituent

This is the smallest apparent change, but it would cause IR scraping,
transcripts, CompanyFacts, GDELT, estimates, and other per-ticker work to fan out
to roughly 550 securities. Refresh time, quota pressure, and failures would grow
without a way to express which sources are appropriate for which securities.

**Rejected:** coverage scope must be source-specific.

### 2. Build one vertical pipeline and schema per provider

Provider-specific tables and scheduler branches are initially convenient, but
they push provider names into retrieval, graph, deduplication, and UI logic.
Adding or replacing a provider becomes a cross-system change and syndicated news
cannot be deduplicated consistently.

**Rejected:** provider adapters should normalize at the boundary.

### 3. Canonical universe plus normalized capability layers

The selected approach introduces a small shared contract:

- canonical security identity and index membership;
- `broad`, `deep`, and sector-specific coverage scopes;
- normalized document/event/observation/action records;
- persistent source identity, content hash, and incremental cursor;
- source-independent corpus facets and graph nodes.

This requires an additive schema migration, but it keeps provider differences at
the network boundary and makes refresh, retrieval, and visualization consistent.

**Selected:** best balance of coverage, maintainability, and failure isolation.

## Storage decision

- SQLite owns identities, memberships, observations, events/actions, corpus item
  metadata, cursors, run status, freshness, and deduplication keys.
- Chroma owns narrative text chunks and the metadata necessary for semantic
  filtering.
- SQLite FTS5 owns the persistent incremental lexical index; Chroma is not
  scanned into an in-process BM25 index at query startup.
- `corpus_items` is the metadata bridge between the stores. It is not a third
  content store and does not duplicate complete document bodies.
- `Store` remains the only public persistence facade and continues to bump one
  retrieval revision after committed corpus mutations.
- SQLite + Chroma remains the default through a reproducible 100,000-item
  benchmark. A replacement pilot starts only after a hard gate still fails after
  one bounded optimization pass or a new approved operational requirement cannot
  be met.

If a replacement pilot is triggered, LanceDB is the first embedded/local-first
candidate and Qdrant is considered only when operating a separate service is an
accepted requirement. A winning vector store does not automatically replace
SQLite's transaction-oriented identity, fact, cursor, and scheduler state.

## Query runtime decision

Capability, inventory, exact-count, and other completely covered structured
questions use explicit query-plan obligations and authoritative Store reads.
They do not rely on semantic top-k samples or model memory.

Validated deterministic answers may bypass local-model generation when every
obligation, provenance, freshness, and completeness requirement is satisfied.
Qualitative, ambiguous, conflicting, and partially covered questions continue
through grounded model generation. Persistent FTS5 lexical candidates and
Chroma dense candidates retain the existing bounded fusion/reranking contract.

## Refresh decision

Keep `UnifiedScheduler`, but move source metadata into a validated registry that
describes cadence, coverage scope, budget, cursor type, and dependencies. Do not
add Celery, Airflow, a message broker, or an async workflow framework.

Bootstrap/backfill and incremental refresh are separate commands. Daily refresh
uses cursors and overlap windows. A source-level circuit breaker opens only for
that source/provider and produces a skipped status for its remaining work.

## Graph decision

The Phase 2.2 graph remains an observability and read-projection system, not
GraphRAG. Phase 2.3 adds source-independent node metadata and faceted aggregate
groups. It does not use graph traversal to generate answers.

Corpus Explorer starts from aggregates:

```text
index -> sector -> security -> source category -> item/event type -> item
```

Users may enter through any facet, and each expansion is paged. Live Trace keeps
the existing query-to-answer stage rail and adds exact evidence type, source
category, event classification, coverage tier, and freshness metadata.

## Consequences

- New sources can be added without changing the graph UI vocabulary.
- Existing documents require a bounded metadata backfill into `corpus_items`.
- Source names remain available for provenance, while retrieval and UI grouping
  primarily use stable categories and item/event types.
- The corpus can grow significantly without attempting to draw every node.
- Provider-specific limitations remain visible in status and provenance.
- Complete inventory answers are independent of retrieval top-k and generation.
- Local-model latency is avoided for safely typed answers, while qualitative
  synthesis keeps the existing grounded path.
- Dormant RAG features cannot remain indefinitely disabled: each is promoted,
  assigned a Phase 2.3 expiry, or removed after controlled evaluation.

## Non-goals

- No commercial redistribution or public multi-user hosting.
- No full-text scraping of news publishers.
- No replacement of Chroma or SQLite without a failed benchmark gate, approved
  candidate pilot, and separate migration/rollback decision.
- No replacement of Cytoscape or the local model in this phase.
- No real-time tick streaming or intraday order-book storage.
- No LLM-generated entity graph, topic communities, or causal relationships.
- No automatic enabling of every deep source for every broad-universe ticker.

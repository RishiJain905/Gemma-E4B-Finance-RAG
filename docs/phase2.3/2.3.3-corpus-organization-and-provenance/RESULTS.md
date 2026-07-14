# 2.3.3 — Corpus Organization and Provenance — Results

## 2.3.3.3 Retrieval Taxonomy, Filtering, and Ranking

**Implemented:** 2026-07-14, branch `phase-2.3.3`, gpt-5.6-sol (effort high) via Codex, orchestrated by Claude (Fable 5).

### Setup

- `src/middleware/evidence_taxonomy.py` — versioned vocabulary
  `finance-evidence-taxonomy-v1`: 12 source categories, 14 item types, event
  types = `sec-events-v1` + stable macro/sector events. Provider labels stay
  metadata, never facets.
- Composable filters across Store search + normalized evidence:
  security/ticker/alias, index membership, sector/industry, source
  category/source, item/event type, form/item/exhibit, date ranges,
  freshness/indexing status, authority tier (seeded evaluation fixture
  `tests/fixtures/evaluation/phase2_3_sources.json`).
- Intent parsing recognizes financing / corporate-action / ownership /
  regulatory / macro-release / company-news requests without provider names
  (additive `evidence_topic`/`evidence_filters` on parse output; legacy adapter
  contract unchanged).
- Authority-aware ranking: normalized two-channel RRF relevance first, then
  bounded deterministic boosts — exact entity ≤0.040, item/event ≤0.030,
  requested-date ≤0.020, recency ≤0.050 (latest/news) / ≤0.015 (neutral) /
  ≤0.005 (historical, 7-day decay on the domain timestamp, never
  `ingested_at`), authority ≤0.025 by tier. Duplicate packing: primary + at
  most 2 materially different secondaries per event.
- Self-describing evidence: prompt + Live Trace metadata carry item/event
  type, authority tier, source, date semantics, canonical security, coverage
  tier; citations resolve to exact evidence ledger ids; graph nodes carry the
  same taxonomy (parity tested).

### Answer-quality regression check

Existing intent/retriever/prompt/reranker/graph/chat contract tests pass; only
one expectation changed (`test_legacy_adapter_matches_existing_parse_contract`
now compares the legacy key subset because parse output gained additive
fields — legacy adapter output itself unchanged). Fail-soft stage behavior,
degraded mode, and Phase 2.2 citation contracts intact. A fresh-context review
inside the run fixed four compatibility risks (incl. keeping candidate pools
untouched until final context selection) before handoff.

### Regression gate

Focused 153 passed; full offline suite 1,541 passed / 1 skipped / 8 deselected;
ruff clean; `scripts/verify.ps1` → **VERIFY: PASS** (implementer + independent
orchestrator re-run).

### Decision

**Ship.** Ranking boosts are bounded and applied after relevance, new behavior
is config-toggleable in `configs/middleware.yaml`, and the full offline suite —
including all pre-existing answer-pipeline tests — is green. Rollback = revert
the task commit or disable the new stage toggles.

## 2.3.3.2 Dual-Store Placement, Chunking, and Retention

**Implemented:** 2026-07-14, branch `phase-2.3.3`, gpt-5.6-sol (effort high) via Codex, orchestrated by Claude (Fable 5).

### Setup

- Placement-matrix enforcement with an explicit guard: numeric market/economic
  observations can never create a Chroma document; structured rows stay
  SQLite-only; narrative evidence (SEC sections, news headline+summary, release
  text) goes to Chroma with full facet metadata.
- Stable document families: one `document_family_id` per narrative
  `corpus_item_id`; chunks carry source/security/date/type/authority facets,
  parent/child ids + ordinal, content hash, normalization version. Exact replay
  → identical chunk ids; replacement is upsert-before-orphan-cleanup, atomic
  from the retrieval perspective (SEC section replacement converted to the same
  contract).
- `src/storage/retention.py` + `Store.run_retention(...)` (preview by default,
  max batch 1,000): news narratives 24 months (configurable) with SQLite
  tombstones after Chroma family removal; SEC/issuer/government/events/actions/
  memberships/observations permanent; failed metadata retained until repaired;
  raw payloads never retained. Deletions bump the revision once per run and are
  never query-triggered.
- `Store.get_corpus_accounting(group_by, ...)`: counts + approximate bytes by
  source category/source/item type/security/year/month/indexing state, computed
  entirely from SQLite metadata (no Chroma scans).

### Regression gate

Focused 69 passed; full offline suite 1,491 passed / 1 skipped / 8 deselected;
ruff clean; `scripts/verify.ps1` → **VERIFY: PASS** (implementer + independent
orchestrator re-run).

### Caveats

- Two legacy tests updated: one encoded delete-before-add replacement (now
  forbidden by the atomicity contract) and one used an outdated fake Chroma
  signature.

### Decision

**Ship.** Storage-side; retrieval sees identical or strictly-richer chunk
metadata, and Phase 2.2 SEC section hierarchy is preserved (existing pipeline
tests unchanged). Rollback = revert the task commit.

## 2.3.3.1 Normalized Record Contract and Deduplication

**Implemented:** 2026-07-14, branch `phase-2.3.3.1`, gpt-5.6-sol (effort high) via Codex, orchestrated by Claude (Fable 5).

### Setup

- Three frozen record families in `src/ingestion/records.py` —
  `NarrativeRecord`, `ObservationRecord`, `EventRecord` — plain dataclasses (no
  framework validation dependency) with required-field, timestamp,
  bounded-value, indexing-status, SHA-256, and metadata-allowlist validation.
  Additive `evidence_authority` field enforces that vendor-parsed filing text
  can never claim `direct_sec` authority.
- `src/ingestion/normalization.py`: canonical URL normalization, content
  hashing, conservative normalized-headline + six-hour publication-window key
  (syndicated news only), `normalization_version = "1"`.
- SQLite ledger: `corpus_items` (metadata bridge, bounded summaries only — no
  narrative bodies) + `corpus_item_sources` (multi-provider provenance) +
  `corpus_item_securities` (many-to-many links). Uniqueness layers:
  `(source, provider_record_id)` → `(source, canonical_url, published_at)` →
  `(source, content_hash)`.
- Dedup applied strictly in order: provider identity → normalized canonical URL
  → exact content hash → conservative headline/window (never embedding
  similarity or fuzzy title). Duplicates update access/provenance metadata and
  link the new origin; re-embedding happens only when canonical content changed.
- Atomic writes: metadata + links + indexing status commit together in SQLite;
  Chroma indexing follows with `pending|indexed|error|not_applicable` and failed
  embeddings remain retryable without refetching (tested with simulated Chroma
  failure).

### Regression gate

Focused 71 passed; full offline suite 1,422 passed / 1 skipped / 7 deselected;
ruff clean; `scripts/verify.ps1` → **VERIFY: PASS** (implementer run +
independent orchestrator re-run before commit).

### Caveats

- `src/ingestion/__init__.py` was the one Files-list expansion: the legacy
  `YFinanceIngestor` export became lazy to break a new ingestion-contract ↔
  Store circular import; public import compatibility preserved.
- No provider adapters are wired yet by design — 2.3.2.x implements against
  this contract.

### Decision

**Ship.** Contract + storage machinery only, exercised by tests/fixtures; no
retrieval or prompt surface touched, so answer quality is unchanged. Rollback =
revert the task commit; schema is additive.

# 2.3.3 — Corpus Organization and Provenance — Results

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

# 2.3.2 — Multi-Source Finance Ingestion — Results

## 2.3.2.1 SEC Event and Capital-Markets Ingestion

**Implemented:** 2026-07-14, branch `phase-2.3.2`, gpt-5.6-sol (effort high) via Codex, orchestrated by Claude (Fable 5).

### Setup

- `src/sec/daily_index.py`: index-driven incremental discovery — broad CIK set
  from the CoverageResolver/registry, each unprocessed daily index downloaded
  once, local filtering by CIK + configured forms, unseen accessions registered,
  cursor advanced only after registration commits (registration + checkpoint
  share one SQLite transaction). Bootstrap = one SEC submissions bulk download
  filtered locally (never 550 per-company submissions polls).
- Forms grouped by capability in `configs/sec.yaml`: periodic/material
  (10-K/10-Q/8-K/amendments/6-K), capital markets (S-1/S-3/S-3ASR/424B2/424B3/
  424B5/FWP), ownership/governance (SC 13D/SC 13G/3/4/5/DEF 14A).
- Exhibit policy: primary document + EX-99.1 + rule-referenced EX-10 +
  explicit allowlist only. Broad event indexing stays distinct from deep
  full-filing indexing.
- `src/sec/event_classifier.py` (`sec-events-v1`): deterministic rules over
  form / 8-K item / exhibit / bounded text for 15 event labels
  (debt_raise … beneficial_ownership_change), each with `rule_version` +
  `classification_reason`; financing terms extracted only when deterministic —
  null over guessed. No model calls during ingestion.
- Idempotence: accession = provider identity; event key =
  (accession, event_type, rule_version); replay produces zero duplicates.
  Partial document failure keeps the accession registered and retryable without
  stopping other accessions. Direct SEC provenance outranks vendor copies via
  `evidence_authority`.

### Regression gate

Task-scoped 71 passed / 3 deselected; full offline suite 1,449 passed /
1 skipped / 8 deselected; ruff clean; `scripts/verify.ps1` → **VERIFY: PASS**
(implementer + independent orchestrator re-run).

### Caveats

- `src/ingestion/records.py` gained two allowlisted metadata keys
  (`security_type`, `maturity`) so deterministic financing extraction flows
  through the 2.3.3.1 contract — additive, validated.
- Live SEC endpoints untouched in tests; 3 task-level live tests are opt-in.

### Decision

**Ship.** Ingestion-side only; retrieval/prompt surfaces unchanged, so no answer
regression path. Rollback = revert the task commit.

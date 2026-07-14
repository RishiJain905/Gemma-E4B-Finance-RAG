# 2.3.1 — Universe and Coverage Foundation — Results

## 2.3.1.2 Coverage Tiers and Source Policy

**Implemented:** 2026-07-14, branch `phase-2.3.1`, gpt-5.6-sol (effort high) via Codex, orchestrated by Claude (Fable 5).

### Setup

- `CoverageResolver` (`src/universe/coverage.py`): read-only policy engine over
  the 2.3.1.1 registry + `configs/coverage.yaml` — `tickers_for`, `scopes_for`,
  `is_enabled`, `explain`. Never contacts providers, never mutates membership.
- Scopes: `universe` / `broad` (active index union) / `deep` (explicit research
  list, additive) / `sector` / `global`, with the spec's default source policy so
  deep-only sources (full filing text, CompanyFacts, IR, transcripts, estimates,
  GDELT) can never fan out to the broad universe.
- Converted `src/scheduler/__init__.py`, `yfinance_ingestor`, `sec/scheduler`,
  `earnings_transcripts`, `estimates_ingestor`, `gdelt_ingestor`, `ir_ingestor`
  to resolve tickers through the policy. Legacy `core`/`extended` watchlist keys
  remain as a deprecated one-release compatibility mirror (warning + same deep
  list, tested).
- Unknown deep tickers are validation errors unless `allow_outside_indexes` is
  set. Policy decisions are deterministic and explainable (scope, inclusion
  reason, enabled capabilities, policy revision); no secrets/paths exposed.

### Before / after

| Metric | Before | After |
|---|---|---|
| Source scoping | one shared `core` ticker list for every source | per-source capability policy over registry scopes |
| Deep fan-out risk | `fetch_all_core()` reuse could hit every ticker | deep sources bounded to explicit list (tested invariant) |
| Offline test count | 1,387 | 1,400 passed / 1 skipped (57 focused policy tests) |

### Regression gate

Full offline suite 1,400 passed / 1 skipped / 7 deselected; ruff clean;
`scripts/verify.ps1` → **VERIFY: PASS** (implementer run + independent
orchestrator re-run before commit).

### Caveats

- Fresh-install fallback: with an empty securities registry, broad sources
  receive exactly `deep.tickers` (no broad additions) until the first universe
  snapshot lands — prevents accidental fan-out; documented in CONFIGURATION.md.
- Legacy watchlist keys are scheduled for removal after one release.

### Decision

**Ship.** No retrieval/prompt surface touched (ingestion scoping only), so
answer quality is unaffected; rollback = revert the task commit.

## 2.3.1.1 Security Universe and Membership History

**Implemented:** 2026-07-14, branch `phase-2.3.1`, gpt-5.6-sol (effort high) via Codex, orchestrated by Claude (Fable 5).

### Setup

- Additive schema: `securities`, `security_aliases`, `security_memberships`, plus a
  supporting `universe_errors` table for persisted reconciliation errors (the spec
  required `list_universe_errors(run_id)` but named no table; this is the one
  deliberate addition). Idempotent DDL + indexes in `docs/phase1.2/schema.sql` and
  `src/storage/sqlite_store.py`.
- Providers (`src/universe/providers.py`): Nasdaq-100 list, IVV holdings CSV
  (S&P 500 personal-use proxy), SEC `company_tickers.json` — records only, no
  persistence, validated for headers, plausible counts, duplicates,
  cash/derivative rows, and empty identifiers.
- `UniverseRegistry.refresh()` (`src/universe/registry.py`): single-transaction
  reconciliation — dot/dash share-class normalization, symbol/alias/exchange/CIK
  matching, create-only-when-no-safe-match, open/close (never delete)
  memberships, reconciliation errors recorded, one Store revision bump per
  changed snapshot.
- Store facade API: `list_securities`, `get_security`, `resolve_security`,
  `list_memberships`, `upsert_universe_snapshot`, `list_universe_errors`.
  `list_tickers()` behavior unchanged.
- Pinned offline fixtures in `tests/fixtures/universe/`.

### Before / after

| Metric | Before | After |
|---|---|---|
| Universe identity | hand-maintained ticker list (`configs/watchlist.yaml`) | canonical `securities` registry + alias + membership history |
| Membership history | none | historical open/close rows per index (`sp500`/`nasdaq100`) |
| Offline test count | 1,323 (baseline suite) | 1,387 passed / 1 skipped (64 new focused tests) |

### Regression gate

- Focused suite: 64 passed.
- Full offline suite: 1,387 passed, 1 skipped, 7 deselected.
- `ruff`: clean. `scripts/verify.ps1`: **VERIFY: PASS** (run by implementer and
  re-run independently by orchestrator before commit).

### Caveats

- IVV holdings are a personal-use proxy for S&P 500 membership, not a licensed
  constituent feed; provenance (`source`, `source_url`) is preserved on every
  membership row so this is never hidden.
- An in-run adversarial review surfaced four edge cases (same-name share-class
  collisions, quoted CSV headers among them) — all fixed with red-green coverage
  before the gate.

### Decision

**Ship.** Foundation for all Phase 2.3 fan-out; no answer-quality surface touched
(no retrieval/prompt changes), so no answer regression is possible from this task.
Rollback = revert the single task commit; schema is additive only.

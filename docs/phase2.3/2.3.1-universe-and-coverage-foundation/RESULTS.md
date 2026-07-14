# 2.3.1 — Universe and Coverage Foundation — Results

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

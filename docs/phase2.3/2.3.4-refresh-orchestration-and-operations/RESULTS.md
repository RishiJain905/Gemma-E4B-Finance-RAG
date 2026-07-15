# 2.3.4 — Refresh Orchestration and Operations: Results

Implemented on branch `phase-2.3.4` (from `Rishi-Ghost`), one commit per task.

## Setup

- Baseline at branch start: 1541 offline tests passing, `VERIFY: PASS`.
- Implementation: GPT-5.6 Codex (sol/high for 2.3.4.1–2.3.4.2 invariant-heavy work,
  luna/max for 2.3.4.3), orchestrated from Claude Code with independent gate
  verification before each commit.
- Gate for every task: `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1`
  (ruff + full offline pytest suite).

## Before / after

| Area | Before | After |
|---|---|---|
| Source scheduling | static source dictionary in `UnifiedScheduler` | validated registry (`configs/sources.yaml` + `src/scheduler/source_registry.py`): capability group, coverage scope, cadence, cursor kind, overlap, budgets, retry/circuit policy; invalid sources disabled visibly, never crash others |
| Incremental state | TTL freshness only (`cache_meta`) | separate transactional `source_cursors` per source+partition with overlap windows and fair oldest-first partition resume (`src/scheduler/cursors.py`) |
| Request limits | none | per-run minute/day budgets enforced before work, durable across restarts (`src/scheduler/budget.py`); `--force` bypasses TTL only |
| Provider failures | ad-hoc GDELT 429 handling | 8-class normalized taxonomy (`src/ingestion/errors.py`), bounded retries honoring Retry-After, provider circuits with persisted cross-invocation cooldowns, deduplicated bounded DLQ |
| Operations | single refresh path | distinct bootstrap (resumable manifests) / incremental / repair (no re-download) / retention (preview-then-apply) CLI modes; truthful run + per-source status and SQLite-only coverage health (`src/scheduler/status.py`) |
| Query-time refresh tool | loosely guarded | goes through registry/budget/circuit controls, per-security bounded, rejects unbounded requests, cannot invoke bootstrap/retention |
| Offline tests | 1541 | 1606 (+65 across 6 new test files + extended suites) |

## Caveats

- Live provider behavior (real 429/Retry-After/entitlement responses) is mocked;
  live validation is opt-in per the phase-wide bounds and deferred to 2.3.6.2.
- Historical run summaries are bounded by retention config; defaults chosen
  conservatively and can be tuned in `configs/sources.yaml`.
- One Codex turn for 2.3.4.3 was interrupted by client tooling mid-run and
  resumed via `codex exec resume`; no code impact (gate re-verified from scratch).

## Regression gate outcome

- `VERIFY: PASS` at every task commit (1574 → 1587 → 1606 tests).
- All pre-existing regression/e2e-marked offline suites pass unchanged; no
  answer-quality regressions observable in the offline contract tests.

## Decision

**Ship.** Merged back to `Rishi-Ghost` via PR. Rollback path: revert the three
task commits; schema additions are additive-only (`source_cursors`,
budget/run-status tables) and safe to leave in place on rollback.

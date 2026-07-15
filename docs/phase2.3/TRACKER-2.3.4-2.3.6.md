# Phase 2.3 — 2.3.4 through 2.3.6 build loop

Branch: `phase-2.3.4-2.3.6` (from `Rishi-Ghost`)

Covers the next three feature groups of the Phase 2.3 roadmap
([README-2.3.md](README-2.3.md)): refresh orchestration and operations, graph
and Corpus Explorer scale, and integration/migration/evaluation. Per the
documented build order, 2.3.4 promotes the bootstrap jobs left behind by
2.3.1–2.3.3 into bounded incremental refreshes before 2.3.5 extends the graph
against realistic seeded scale, and 2.3.6.1's additive schema/config migration
must seed the realistic corpus that 2.3.6.2's end-to-end gate measures.
2.3.6.2's final sign-off itself depends on the separate Phase 2.3.7 quality,
speed, and storage gates and is not the last work in Phase 2.3 as a whole — it
is the last task tracked here.

## 2.3.4 — Refresh orchestration and operations

- [x] 2.3.4.1 source-registry-cadence-and-incremental-cursors — **done:** validated data-driven registry (`configs/sources.yaml`), transactional `source_cursors` + fair partition scheduling, durable minute/day budgets; scheduler stays sequential/fail-soft; 1574 offline tests, VERIFY: PASS. — spec: `docs/phase2.3/2.3.4-refresh-orchestration-and-operations/2.3.4.1-source-registry-cadence-and-incremental-cursors.md` — scale the unified scheduler from a static source dictionary into a validated, budget-aware source registry (capability group, coverage scope, cadence, cursor kind, overlap window, request budgets, retry/circuit policy) while preserving the existing simple sequential, fail-soft execution model.
- [x] 2.3.4.2 rate-limits-circuit-breakers-and-failure-isolation — **done:** normalized 8-class provider error taxonomy (`src/ingestion/errors.py`), bounded retries + provider circuits with persisted cooldowns, deduplicated DLQ, adapters emit safe redacted failures; 1587 offline tests, VERIFY: PASS. — spec: `docs/phase2.3/2.3.4-refresh-orchestration-and-operations/2.3.4.2-rate-limits-circuit-breakers-and-failure-isolation.md` — generalize the current GDELT rate-limit handling into normalized provider error classes (authentication/entitlement/rate_limited/quota_exhausted/transient/contract/item/permanent) with provider-level circuit semantics so exhaustion on one source stops pointless work after bounded retries without ever stopping another source.
- [x] 2.3.4.3 refresh-status-backfills-and-operational-controls — **done:** distinct bootstrap/incremental/repair/retention modes with resumable manifests, truthful run+coverage status (`src/scheduler/status.py`, zero network), --force never bypasses quotas/circuits, bounded refresh tooling; 1606 offline tests, VERIFY: PASS. — spec: `docs/phase2.3/2.3.4-refresh-orchestration-and-operations/2.3.4.3-refresh-status-backfills-and-operational-controls.md` — separate bootstrap (explicit, resumable initial population from an empty/new schema) from incremental refresh (TTL/cursor/overlap/budget-driven) and add safe, truthful operational status and repair/backfill commands that never mix those workloads together.

## 2.3.5 — Graph and Corpus Explorer scale

- [ ] 2.3.5.1 source-aware-live-trace-contract — spec: `docs/phase2.3/2.3.5-graph-and-corpus-explorer-scale/2.3.5.1-source-aware-live-trace-contract.md` — extend Live Trace's `question -> plan/subquery -> retrieval/tool -> evidence -> source -> answer/citation` model to explain the expanded source set for one executed query, without provider-specific graph schemas and without turning scheduler/ingestion runs into query traces (that belongs in Corpus Explorer).
- [ ] 2.3.5.2 corpus-explorer-information-architecture — spec: `docs/phase2.3/2.3.5-graph-and-corpus-explorer-scale/2.3.5.2-corpus-explorer-information-architecture.md` — reorganize Corpus Explorer around an aggregation-first landing view (market universe → sector/index/source groups → on-demand expansion) that generalizes the Phase 2.2 freshness fold/aggregate pattern instead of rendering thousands of tickers/documents directly.
- [ ] 2.3.5.3 bounded-projection-ui-performance-and-accessibility — spec: `docs/phase2.3/2.3.5-graph-and-corpus-explorer-scale/2.3.5.3-bounded-projection-ui-performance-and-accessibility.md` — extend the read-only corpus projection and graph APIs (indexed bounded aggregate queries, Cytoscape zoom/pan-aligned stage lanes, rAF-coalesced scrubbing) so the new information architecture stays responsive, bounded, and accessible at Phase 2.3 scale.

## 2.3.6 — Integration, migration, and evaluation

- [ ] 2.3.6.1 schema-config-and-backward-compatible-migration — spec: `docs/phase2.3/2.3.6-integration-migration-and-evaluation/2.3.6.1-schema-config-and-backward-compatible-migration.md` — add a small standard-library ordered/idempotent migration runner and `schema_migrations` table (securities/aliases/memberships → corpus items/sources/links → observations/events/actions → cursors/circuit state → indexes) introducing Phase 2.3 tables/config without losing existing SQLite/Chroma data or requiring every new source to be enabled at once.
- [ ] 2.3.6.2 end-to-end-evaluation-rollout-and-phase-sign-off — spec: `docs/phase2.3/2.3.6-integration-migration-and-evaluation/2.3.6.2-end-to-end-evaluation-rollout-and-phase-sign-off.md` — pilot one representative slice (ORCL, NVDA, AAPL, one healthcare/automotive/government-contractor security, a small macro catalog) through bootstrap, repeated refresh, and simulated failure modes, reviewing normalized rows, retrieval, Live Trace, and Corpus Explorer before controlled-stage broad-universe rollout; final phase sign-off additionally depends on the separate Phase 2.3.7 gates.

## Rules

- Gate: `scripts\verify.ps1` → `VERIFY: PASS`.
- 2.3.4 builds on the completed 2.3.1–2.3.3 foundation (universe/coverage scopes, normalized record contract, dual-store placement) — do not start 2.3.4 work from a branch that predates that merge.
- 2.3.5 depends on 2.3.4's registry/status model existing so Live Trace and Corpus Explorer have real source-run state to project, per `README-2.3.md`'s build order.
- 2.3.6.1's additive schema/config migration must land, and seed the realistic corpus, before 2.3.6.2's end-to-end evaluation runs against it.
- 2.3.6.2 is the phase-wide integration gate tracked here, but final Phase 2.3 sign-off also depends on the separate Phase 2.3.7 RAG quality, speed, feature-disposition, and storage gates (tracked separately, not in this file).
- Two failed gate attempts on the same failure for one task → mark **BLOCKED** below with a one-line diagnosis, move to the next task. Don't loop on it.
- One commit per task, checked off here with a one-line result note as you go.
- Each implemented feature writes its own `RESULTS.md` inside the feature folder (setup, before/after metrics, caveats, regression-gate outcome, ship/rollback decision) — not created speculatively ahead of implementation.
- All new network tests are mocked and offline-safe; live tests are opt-in, per the phase-wide hard bounds in `README-2.3.md`.
- No push, no merge to `Rishi-Ghost` — leave the branch for review.

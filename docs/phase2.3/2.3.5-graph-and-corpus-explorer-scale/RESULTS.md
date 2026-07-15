# 2.3.5 — Graph and Corpus Explorer Scale: Results

Implemented on branch `phase-2.3.5` (from `Rishi-Ghost` after the 2.3.4 merge),
one commit per task.

## Setup

- Baseline at branch start: 1606 offline tests passing, `VERIFY: PASS`.
- Implementation: Claude Opus 4.8 (xhigh) agents — Claude-routed because all
  three tasks are contract/UI/taste-weighted — with independent gate
  verification before each commit.
- Gate for every task: `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1`.

## Before / after

| Area | Before | After |
|---|---|---|
| Live Trace evidence | Phase 2.2 metadata only | additive allowlisted source-aware fields (security identity, index membership, source category, authority tier, item/event type, provider vs publisher, date semantics, freshness); schema version held at 1; Phase 2.2 golden trace untouched; new mixed SEC/news/macro/market golden fixture |
| Trace truthfulness | n/a | primary vs corroborating evidence roles, bounded dedupe counts, shared `stage:route` identity preserved through legacy and adaptive-fallback paths |
| Corpus Explorer landing | source/ticker inventory rendering | aggregation-first landing (market universe → index → sector → security → source category → item/event type → time bucket), on-demand bounded page expansion, never full-corpus render |
| Faceting | q/ticker/kinds search only | persistent combinable facet rail with authoritative counts (`/corpus/aggregates`, `/corpus/facets`, `/corpus/groups`, `/corpus/items/{id}`), all SQLite-metadata-backed, paged, revision-aware, read-only |
| Deep links | none | full filter state in URL hash; 7 deterministic presets; no cookies/analytics/localStorage |
| Scale posture | untested beyond watchlist scale | deterministic fixture: 600 securities, 11 sectors, overlapping memberships, 100,000 corpus items; warm p95 asserted in the offline gate: overview/facet < 250 ms, filtered first page < 300 ms; bounded pages (50/100), 450 visible-node target with configured hard cap |
| Accessibility | Phase 2.2 baseline | keyboard-complete inventory/facets/inspector, aria-live status, WCAG 2.1 AA contrast/focus checks in UI contract tests, status never color-alone, reduced-motion preserved |
| Offline tests | 1606 | 1651 (+45 across the three tasks) |

## Caveats

- p95 targets are asserted on the developer machine inside the gate; the
  documented reference-machine measurements for phase sign-off happen in the
  Phase 2.3.7 gates.
- Playwright smoke was replaced by the stub-DOM runtime harness (no local
  Playwright Chromium); static UI contract tests cover the CSP/a11y/markup
  invariants.
- Corpus aggregate counts for index/sector use the security-registry join added
  in 2.3.5.3; earlier 2.3.5.2 UI shipped with those two facets uncounted for one
  commit.

## Regression gate outcome

- `VERIFY: PASS` at every task commit (1615 → 1630 → 1651 tests).
- Phase 2.2 guarantees explicitly regression-tested: route-node identity,
  freshness folding, label level-of-detail, stage-rail viewport projection,
  rAF-coalesced scrubbing, publish p95 < 2 ms, redaction canaries, CSP,
  loopback-only.

## Decision

**Ship.** Merged back to `Rishi-Ghost` via PR. Rollback path: revert the three
task commits; all API additions are additive and the graph schema version is
unchanged, so older UIs/fixtures remain valid.

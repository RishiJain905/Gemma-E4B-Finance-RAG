# 2.2.3 — Adaptive RAG Orchestration: Results

Feature 2.2.3 adds bounded adaptive orchestration on top of the Phase 2.2.1
evidence foundation and Phase 2.2.2 conversation layer. Four tasks, all landed
on `phase-2.2.3`:

- **2.2.3.1** — multi-entity `QueryPlan` contract + `parse_plan`/`resolve_all`.
- **2.2.3.2** — deterministic finance tool routing + whitelisted calculator.
- **2.2.3.3** — fast/standard/complex lanes, execution + context budgets,
  conditional re-ranking.
- **2.2.3.4** — `/query` + `/query/stream` integration behind
  `enable_adaptive_rag`, trace/metadata extensions, three-config eval
  scaffolding, and review hardening.

## Setup

All verification is **offline and deterministic** via `scripts\verify.ps1`
(ruff + `pytest -m "not live"`). Suite growth across the feature:
922 → 939 → 953 → **993 passed**. No live model evaluation was run during
implementation (per spec); the three-configuration live comparison is a
separately scheduled GPU activity with exact commands documented in
`eval/README.md` ("Adaptive orchestration comparison (2.2.3.4)").

## Offline quality gates (all met)

| gate | result |
|---|---|
| parser micro-F1 on compound fixture set (≥ 0.90) | pooled **0.965** (entities 1.000, intents 0.930, metrics 0.950, periods 1.000), zero raw-question mutations |
| routing accuracy on route matrix (≥ 95%) | **21/21 = 100%** |
| write-tool avoidance (100%) | **100%** — `refresh_data` structurally unreachable; refresh-worded requests abstain with `refresh_requested` |
| router latency p95 (< 5 ms) | **0.007 ms** |
| orchestration overhead (offline) | fast lane p95 0.008 ms; standard p95 ≈ 2.5 ms; both far under the 250 ms complex ceiling |
| budget caps (3 subqueries / 2 rounds / 1 planning / 1 re-rank) | enforced solely via `ExecutionBudget.consume()`; cap violations = 0 in `gate.adaptive_safety_checks` |
| fault injection | plan failure, budgeter failure, config failure, lane failure, deterministic-route error all fall soft with `adaptive_fallback` recorded |

## External review hardening

Two independent fresh-context reviews ran before integration completed:
gpt-5.5 (high) on the 2.2.3.1/2.2.3.2 foundations and gpt-5.6-luna (xhigh) on
the 2.2.3.3 orchestrator. Of 16 findings, **14 were confirmed and fixed** in
the 2.2.3.4 commit, including:

- `RouteDecision.complete` overclaiming coverage for multi-obligation plans
  (the fast lane now also has its own independent complex-obligation guard);
- refresh-worded requests routing as if data were fresh;
- ratio unit-equality and comparison period-equality checks in the calculator;
- an uncovered fail-soft boundary that could leak exceptions;
- a request-crossing concurrency bug in `retrieve_candidates()` channel state;
- context budgeting able to starve the only qualitative document.

**Deferred with rationale (known issues):**

1. `resolve_all` exact-match short-circuit suppresses fuzzy recovery for
   misspelled second entities ("NVDA and Micorsoft") — resolver redesign,
   outside this feature's surface.
2. `_calculate_rank()` has no tie-handling — currently unreachable from
   `route()`.
3. Conditional re-rank does not require `decision.requires_documents` —
   deliberately deferred: the ambiguity-signal gate is the substantive
   control, re-rank is hard-capped at 1/query, and the stricter gate would
   suppress legitimate standard-lane re-ranks for unroutable document queries.

## Caveats

- `enable_adaptive_rag=false` remains the shipped default; the legacy path is
  byte-for-byte preserved and is the tested rollback.
- Deterministic answers (`enable_deterministic_answers`) surface in metadata
  but do not yet bypass model generation — deliberate risk call, follow-up
  candidate.
- Comparative promotion gates (≥10 pp complex-query correctness and multi-turn
  Recall@10 improvement, ≤0.02 nDCG@10 regression, ≤10% simple-query p95
  regression, plan/router accuracy ≥0.90/0.95) are **pending the live
  three-config run**; the machine-checkable safety subset (zero write routes,
  zero cap violations) is enforced by `eval/gate.py` today.
- The eval baselines remain placeholders until the post-2.2.2 live re-baseline
  (see 2.2.1 RESULTS.md).

## Ship / rollback decision

**Ship on `phase-2.2.3` with the flag off.** The adaptive path is bounded,
observable, fail-soft, and review-hardened, with the legacy path intact as
rollback. **Do not enable `enable_adaptive_rag` by default** until the live
three-configuration evaluation passes the promotion gates listed above; run it
with the commands in `eval/README.md` during the next scheduled GPU session
and append the per-lane results and gate verdicts here.

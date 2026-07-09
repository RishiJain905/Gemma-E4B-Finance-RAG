# Phase 2.1 remaining — 2.1.7 + 2.1.8 overnight loop

Branch: `phase-2.1.7-2.1.8` (from `Rishi-Ghost`)

- [ ] **BLOCKED** 2.1.7.1 graded-grounding-answer-policy — spec: `docs/Phase2.1/2.1.7-answer-policy-graded-grounding/2.1.7.1-graded-grounding-answer-policy.md` — code landed & offline-verified (refusals 0.52→0.00, relevance +0.06, coverage +0.29, strict-mode rollback intact), but the eval faithfulness gate failed twice (0.89→0.74→0.72, tol 0.05): judge-context capture is lower-fidelity than the model prompt (3×600-char docs vs 5 full docs) + tool-sourced answers invisible to the judge + baseline inflated by trivially-faithful refusals. Diagnosis & unblock options in `2.1.7-answer-policy-graded-grounding/RESULTS.md`.
- [x] 2.1.8.1 latency-measurement-and-middleware-optimizations — spec: `docs/Phase2.1/2.1.8-chat-tui-optimization-and-integration/2.1.8.1-latency-measurement-and-middleware-optimizations.md` — per-stage `timings` in /query, health-check 10s TTL cache, /health scheduler+freshness cached, embedding LRU (repeat-query embed 69.5ms→0); measured: >99.9% of latency is model generation (RESULTS.md). VERIFY: PASS (772).
- [ ] 2.1.8.2 chat-client-performance-and-streaming — spec: `docs/Phase2.1/2.1.8-chat-tui-optimization-and-integration/2.1.8.2-chat-client-performance-and-streaming.md`
- [ ] 2.1.8.3 tui-integration-with-new-features — spec: `docs/Phase2.1/2.1.8-chat-tui-optimization-and-integration/2.1.8.3-tui-integration-with-new-features.md` — do last, must reflect 2.1.7's landed policy

## Rules
- Gate: `scripts\verify.ps1` → `VERIFY: PASS`.
- Two failed gate attempts on the same failure for one task → mark **BLOCKED** below with a one-line diagnosis, move to the next task. Don't loop on it.
- One commit per task, checked off here with a one-line result note as you go.
- No push, no merge to `Rishi-Ghost` — leave the branch for morning review.

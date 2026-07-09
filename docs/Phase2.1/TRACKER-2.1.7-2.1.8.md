# Phase 2.1 remaining — 2.1.7 + 2.1.8 overnight loop

Branch: `phase-2.1.7-2.1.8` (from `Rishi-Ghost`)

- [ ] 2.1.7.1 graded-grounding-answer-policy — spec: `docs/Phase2.1/2.1.7-answer-policy-graded-grounding/2.1.7.1-graded-grounding-answer-policy.md`
- [ ] 2.1.8.1 latency-measurement-and-middleware-optimizations — spec: `docs/Phase2.1/2.1.8-chat-tui-optimization-and-integration/2.1.8.1-latency-measurement-and-middleware-optimizations.md`
- [ ] 2.1.8.2 chat-client-performance-and-streaming — spec: `docs/Phase2.1/2.1.8-chat-tui-optimization-and-integration/2.1.8.2-chat-client-performance-and-streaming.md`
- [ ] 2.1.8.3 tui-integration-with-new-features — spec: `docs/Phase2.1/2.1.8-chat-tui-optimization-and-integration/2.1.8.3-tui-integration-with-new-features.md` — do last, must reflect 2.1.7's landed policy

## Rules
- Gate: `scripts\verify.ps1` → `VERIFY: PASS`.
- Two failed gate attempts on the same failure for one task → mark **BLOCKED** below with a one-line diagnosis, move to the next task. Don't loop on it.
- One commit per task, checked off here with a one-line result note as you go.
- No push, no merge to `Rishi-Ghost` — leave the branch for morning review.

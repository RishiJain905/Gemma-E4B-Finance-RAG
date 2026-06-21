# Phase 2.1 — Tier 1 Feature Specs

End-to-end implementation specs for the **Tier 1** ideas in
[../potential-ideas.md](../potential-ideas.md). Each folder is one feature,
broken into sequential task files in the same style as the Phase 1.x specs:
**Objective → Why → Steps (with file references) → Testing → Verification
Checklist**. Every task includes its own tests so the feature ships verified,
end to end.

| Folder | Feature | Task files |
|--------|---------|-----------|
| `2.1.1-evaluation-harness` | Evaluation harness (golden set + metrics) | 2.1.1.1 – 2.1.1.2 |
| `2.1.2-hybrid-retrieval-and-reranking` | BM25 + RRF + cross-encoder re-ranking | 2.1.2.1 – 2.1.2.3 |
| `2.1.3-smarter-chunking` | Structure/sentence-aware chunking | 2.1.3.1 – 2.1.3.2 |
| `2.1.4-structured-analytical-tooling` | Function-calling tool suite over the store | 2.1.4.1 – 2.1.4.4 |
| `2.1.5-forward-looking-projection-data` | Analyst estimates & price targets | 2.1.5.1 – 2.1.5.2 |
| `2.1.6-ticker-coverage-and-resolution` | Symbol resolution + fetch-on-miss | 2.1.6.1 – 2.1.6.2 |
| `2.1.7-answer-policy-graded-grounding` | Graded grounding answer policy | 2.1.7.1 |
| `2.1.8-chat-tui-optimization-and-integration` | Chat TUI perf + keep it in sync with all new features | 2.1.8.1 – 2.1.8.3 |

## Suggested build order

The **evaluation harness (2.1.1) goes first** — it's the scoreboard every
other feature is measured against. After that:

1. 2.1.1 Evaluation harness
2. 2.1.6 Ticker coverage (you hit this constantly today)
3. 2.1.7 Answer policy (quick prompt win, big helpfulness gain)
4. 2.1.2 Hybrid retrieval + re-ranking
5. 2.1.4 Structured tooling (depends on Tier 2 #4 function-calling plumbing)
6. 2.1.5 Projection data (uses the 2.1.4 tool pattern)
7. 2.1.3 Smarter chunking (re-embeds the corpus; do once retrieval is settled)
8. 2.1.8 Chat TUI optimization + integration (do its perf piece, 2.1.8.1–.2,
   early since it helps day-to-day; finalize 2.1.8.3 last so the TUI reflects
   every feature that landed)

## Conventions

- Branch from `Rishi-Ghost`, one branch per feature.
- New code under `src/`, new tests under `tests/` with the existing markers
  (`live`, `slow`, `integration`, `regression`, `network`).
- Keep `pytest tests/` green and coverage ≥ 80% on new modules.
- Run the 2.1.1 eval harness before/after each feature to confirm no
  answer-quality regression.

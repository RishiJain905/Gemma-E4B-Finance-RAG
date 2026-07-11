# Phase 2.2 — 2.2.1 through 2.2.3 build loop

Branch: `phase-2.2.1-2.2.3` (from `Rishi-Ghost`)

Covers the first three feature groups of the Phase 2.2 roadmap
([README-2.2.md](README-2.2.md)): baseline correctness/evaluation fidelity,
conversational query understanding, and adaptive RAG orchestration. 2.2.1 must
land first — no later metric in 2.2.2/2.2.3 is trustworthy until the
document/prompt evidence contract and evaluator fidelity are repaired.

## 2.2.1 — Evaluation fidelity and baseline correctness

- [x] 2.2.1.1 retrieved-document-contract-and-prompt-policy — spec: `docs/phase2.2/2.2.1-evaluation-fidelity-and-baseline-correctness/2.2.1.1-retrieved-document-contract-and-prompt-policy.md` — fix the `document` vs `text` blank-body bug (P0 evidence loss), establish one shared evidence contract and one authoritative prompt-policy builder. — **Done:** `evidence.py` + `prompt_policy.py` added; grounding/counts use usable evidence; per-fact provenance preserved; all model paths share one policy builder; also fixed collateral `truncate_if_needed` split bug and multi-ticker fact relabeling; VERIFY: PASS (794 passed).
- [ ] 2.2.1.2 exact-evidence-trace-and-policy-aware-scoring — spec: `docs/phase2.2/2.2.1-evaluation-fidelity-and-baseline-correctness/2.2.1.2-exact-evidence-trace-and-policy-aware-scoring.md` — capture the exact model-visible prompt/evidence/tool results per answer; replace the truncated re-retrieval judge pass; unblocks the 2.1.7 faithfulness gate.
- [ ] 2.2.1.3 conversational-and-compound-query-golden-set — spec: `docs/phase2.2/2.2.1-evaluation-fidelity-and-baseline-correctness/2.2.1.3-conversational-and-compound-query-golden-set.md` — extend golden set/runner with versioned fixtures for multi-turn, compound, multi-ticker, stale, and unanswerable cases.

## 2.2.2 — Conversational query understanding

- [ ] 2.2.2.1 request-history-contract-and-bounded-memory — spec: `docs/phase2.2/2.2.2-conversational-query-understanding/2.2.2.1-request-history-contract-and-bounded-memory.md` — add a bounded, client-owned `ChatTurn` history contract to `/query`; middleware stays stateless.
- [ ] 2.2.2.2 follow-up-rewriting-and-entity-carryover — spec: `docs/phase2.2/2.2.2-conversational-query-understanding/2.2.2.2-follow-up-rewriting-and-entity-carryover.md` — compile turn + history into a standalone retrieval query; deterministic ticker/metric/period carryover with at most one optional planner call.
- [ ] 2.2.2.3 multiline-chat-sessions-and-evaluation — spec: `docs/phase2.2/2.2.2-conversational-query-understanding/2.2.2.3-multiline-chat-sessions-and-evaluation.md` — `/ask` multiline composer + session controls in `scripts/chat.py`; offline integration test for verbose/follow-up questions.

## 2.2.3 — Adaptive RAG orchestration

- [ ] 2.2.3.1 multi-entity-intent-and-query-plans — spec: `docs/phase2.2/2.2.3-adaptive-rag-orchestration/2.2.3.1-multi-entity-intent-and-query-plans.md` — explicit `QueryPlan` contract (entities, intents, metrics, periods) behind a feature flag; `IntentParser.parse()` kept as compatibility view.
- [ ] 2.2.3.2 deterministic-finance-tool-routing — spec: `docs/phase2.2/2.2.3-adaptive-rag-orchestration/2.2.3.2-deterministic-finance-tool-routing.md` — rule-based router for safe analytical/comparison/projection/calculation requests; model no longer decides whether to invoke exact DB operations.
- [ ] 2.2.3.3 adaptive-lanes-context-budget-and-conditional-reranking — spec: `docs/phase2.2/2.2.3-adaptive-rag-orchestration/2.2.3.3-adaptive-lanes-context-budget-and-conditional-reranking.md` — bounded orchestrator selecting fast/standard/complex lanes under one shared execution budget (≤3 subqueries, ≤2 retrieval rounds, ≤1 planner call, ≤1 re-rank call).
- [ ] 2.2.3.4 integration-and-evaluation — spec: `docs/phase2.2/2.2.3-adaptive-rag-orchestration/2.2.3.4-integration-and-evaluation.md` — wire query plan + deterministic router + adaptive orchestrator into `/query` and `/query/stream` behind `enable_adaptive_rag`; promotion checkpoint comparing quality/latency vs. the legacy path.

## Rules

- Gate: `scripts\verify.ps1` → `VERIFY: PASS`.
- 2.2.1 blocks 2.2.2/2.2.3 metrics — land and verify all of 2.2.1 before trusting evaluation output from later tasks, per `README-2.2.md`'s suggested build order.
- Two failed gate attempts on the same failure for one task → mark **BLOCKED** below with a one-line diagnosis, move to the next task. Don't loop on it.
- One commit per task, checked off here with a one-line result note as you go.
- Each implemented feature writes its own `RESULTS.md` inside the feature folder (setup, before/after metrics, caveats, regression-gate outcome, ship/rollback decision) — not created speculatively ahead of implementation.
- No push, no merge to `Rishi-Ghost` — leave the branch for review.

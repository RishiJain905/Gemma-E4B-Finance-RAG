# 2.2.1 — Evaluation Fidelity and Baseline Correctness: Results

Feature 2.2.1 repairs the evaluation harness so that later Phase 2.2 work
(2.2.2 conversational understanding, 2.2.3 adaptive orchestration) is measured
against a trustworthy scoreboard. It spans three tasks, all landed on
`phase-2.2.1`:

- **2.2.1.1** — retrieved-document contract + single prompt-policy builder.
- **2.2.1.2** — exact evidence trace (schema v1) + policy-aware scoring.
- **2.2.1.3** — conversational & compound-query golden set + deterministic
  conversational metrics + Phase 2.2 acceptance gate.

## Setup

All measurement here is **offline and deterministic** — no live middleware,
model, or LLM-judge was run. The suite runs under
`scripts\verify.ps1` (ruff + `pytest -m "not live"`); the committed policy
baselines (`eval/baselines/{graded,strict}.json`) remain **placeholders** with
null metrics until a live stack scores them. The conversational metrics and the
Phase 2.2 gate are validated by unit tests with injected/fake responses, not by
a scored model run.

## 2.2.1.1 — Retrieved-document contract + prompt policy

- Fixed the P0 evidence-loss bug where document bodies read via `text`/`content`
  were blank (retriever emits `document`, Chroma naming); introduced
  `src/middleware/evidence.py` (`usable_facts`/`usable_documents`/
  `evidence_counts`, canonical `document_body`) and `src/middleware/
  prompt_policy.py` (one `build_system_prompt` used by every model path).
- Grounding level and response counts now derive from *usable* evidence;
  per-fact provenance preserved.
- Gate at completion: **VERIFY: PASS** (794 passed).

## 2.2.1.2 — Exact evidence trace + policy-aware scoring

- Opt-in `EvidenceTrace` (schema v1) captures the exact model-visible
  system/user prompts, usable facts/documents (untruncated), and tool results,
  request-scoped across plain, streaming, and tool-loop paths.
- The eval runner consumes the endpoint's trace verbatim (removed the second,
  truncated re-retrieval judge pass); the single `faithfulness` metric split
  into `grounded_faithfulness` (grounded/partial only) and `policy_compliance`
  (all four modes), each reporting `n_eligible`/`n_scored`/`n_skipped`.
- Added policy-specific baselines and hard pre-metric gate checks (policy /
  schema / dataset-digest mismatch, incomplete-trace, eligible-denominator drop).
- Gate at completion: **VERIFY: PASS** (809 passed).

## 2.2.1.3 — Conversational & compound-query golden set

### Before → after (capability)

| dimension | before | after |
|---|---|---|
| golden fixtures | 47 single-turn cases, one format | +38 single-turn Phase 2.2 cases (85 total) **and** a new versioned `finance_conversations.jsonl` (13 conversations / 31 turns) |
| conversational measurement | none | runner-owned per-conversation history; one result row per turn with `conversation_id`, `turn_index`, `history_sent`, expected-vs-resolved ticker/metric/timeframe, subquestion ids + coverage, exact evidence trace |
| conversational metrics | none | 8 deterministic metrics: entity/metric/timeframe carryover, verbose-paraphrase parity, compound-subquestion coverage, stale-disclosure rate, unanswerable numeric-hallucination rate, cross-session leakage rate — each with its own `n_eligible` denominator |
| acceptance gate | 2.2.1.2 pre-metric + baseline regression only | + `phase22_checks`: carryover ≥ 0.90, paraphrase parity ≥ 0.90, compound coverage ≥ 0.85, stale disclosure = 1.00, unanswerable hallucination = 0.00, leakage = 0.00; **fails when any required category has zero eligible fixtures** |

### Challenge-set coverage (from `run_eval.py --validate-fixtures`)

conversations 13 (min 12) · turns 31 (min 30) · paraphrase pairs 6 · compound 6
· multi-ticker 6 · timeframe 6 · stale 4 · unanswerable 4 — all minimums met,
no id collisions.

### Design notes

- No rewrite step exists until 2.2.2, so `retrieval_query == raw_question` and
  the resolved ticker/metric/timeframe fields are derived best-effort from the
  evidence trace, with an explicit `resolved_*` response field taking
  precedence when present. The carryover metrics therefore read low on a live
  run today — that is the scoreboard surfacing the follow-up gap 2.2.2 closes;
  no runner change is needed then.
- The live runner sends conversation history in a forward-compatible `history`
  field that the stateless middleware ignores (pydantic `extra="ignore"`) until
  2.2.2.1 honors it. `src/middleware` was not modified.
- `dataset_digest` now hashes both golden files; the placeholder baselines'
  pinned digest was recomputed to match (`87971fd8…`).

## Regression-gate outcome

- Offline suite: **VERIFY: PASS** (840 passed, 1 skipped, 6 deselected).
- The eval regression gate (`eval/gate.py`) is **not** run live: the committed
  baselines are null placeholders, so the Phase 2.2 thresholds are wired to
  activate only when a run scores the metrics. Gate pass/fail behavior is proven
  by unit tests (threshold boundaries, zero-eligible failure, absent-metric
  no-op, CLI exit codes).

## Caveats

- No live scored numbers exist for any 2.2.1 metric. Carryover / paraphrase /
  leakage scores require a live middleware + model run and an explicit
  `resolved_*` contract (arrives with 2.2.2) to move off their floor.
- The LLM-judge metrics (`grounded_faithfulness`, `policy_compliance`,
  `answer_relevance`) also remain unbaselined pending a live run.

## Ship / rollback decision

The evaluation-fidelity feature is **ready to land on `phase-2.2.1`** for
review: the evidence contract, exact trace, and conversational scoreboard are
in place and green offline, and every new behavior is covered by deterministic
offline tests. It is **not** yet possible to publish a real acceptance baseline
— that is a deliberate follow-up: after 2.2.2 populates the `resolved_*`
contract, run `eval/run_eval.py && eval/score.py --set-baseline` against a live
stack, review, and commit the baseline as a reviewed re-baseline. Strict mode
and the placeholder baselines are preserved for rollback.

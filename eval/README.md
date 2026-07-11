# Evaluation Harness (Phase 2.1.1, 2.2.1.2)

A repeatable scoreboard for the Gemma-E4B-Finance-RAG query pipeline. It turns
"this feels better" into numbers you can regress against:

> grounded_faithfulness went 0.71 → 0.86, retrieval hit-rate 0.6 → 0.9

The harness has three stages, run in order:

```
python eval/run_eval.py                # drive every golden case through the pipeline → eval/runs/<ts>.jsonl
python eval/score.py --policy graded    # score the latest run → eval/runs/<ts>.summary.json + console table
python eval/gate.py --policy graded     # compare latest summary vs eval/baselines/graded.json → exit 0 / 1
```

## Exact evidence trace (2.2.1.2)

Every run row carries an `evidence_trace`: the exact system/user prompts sent
to the model, the exact usable facts/documents (untruncated, full
provenance), and every tool result — captured by the middleware itself
(`src/middleware/evidence_trace.py`), not reconstructed afterwards. The live
runner requests it via `include_evidence_trace: true` on `POST /query` and
persists it verbatim; it never re-runs retrieval to build judge context, and
never truncates a document body for the judge (`PromptAugmenter` truncates
the *prompt* at 2000 chars for the model's context budget — the trace does
not). The direct/offline backend builds an equivalent trace in-process using
the same evidence helpers (`src/middleware/evidence.py`) and prompt builder
(`src/middleware/prompt_policy.py`) as the middleware.

A row whose model produced an answer (`model_available: true`) but whose
trace is missing or structurally incomplete (no captured system/user prompt)
is an **evaluation error**, not silently scorable context — it is excluded
from every judge metric's eligible pool and reported under
`summary.trace_errors` / `summary.n_trace_errors`.

## Metrics

- `intent_accuracy`, `ticker_accuracy`, `retrieval_hit_rate`, `keyword_coverage`,
  `refusal_rate` — cheap, deterministic (see `eval/metrics.py`).
- `grounded_faithfulness` — LLM-as-judge groundedness (0-1) of **only**
  `grounded`/`partial` answers, scored against the exact evidence trace
  (facts, documents, and tool results). `general`/`refused` answers and rows
  with a missing/incomplete trace never enter its denominator, so a strict
  refusal can no longer inflate it.
- `policy_compliance` — LLM-as-judge policy compliance (0-1) across **all
  four** grounding modes (`grounded`/`partial`/`general`/`refused`), checked
  against the rules for that mode (evidence-only claims, general-knowledge
  labeling + caveat, justified refusal).
- `answer_relevance` — LLM-as-judge relevance (0-1); needs only the question
  and answer, so every non-empty answer is eligible regardless of mode.

Every judge metric reports `n_eligible` (rows that passed mode/trace
filtering, regardless of judge availability), `n_scored`, `n_skipped`, and
`per_case` reasons — so a dropping eligible count is visible, not silently
absorbed into a smaller average.

### Conversational / compound metrics (2.2.1.3)

Deterministic metrics for the conversation and compound-query golden set. Each
reads captured structured fields (resolved vs. expected tickers/metrics/
timeframe, subquestion coverage, staleness/answerability) — never the model —
and reports its own `n_eligible` denominator and `n_pass`:

- `entity_carryover_accuracy` / `metric_carryover_accuracy` /
  `timeframe_carryover_accuracy` — over turns declaring that field under
  `expected_carryover`, the fraction whose resolved value matches the carried
  expectation.
- `verbose_paraphrase_parity` — fraction of `paraphrase_group`s whose members
  resolve to the same normalized retrieval plan (tickers, metrics, timeframe,
  intent).
- `compound_subquestion_coverage` — mean fraction of a compound question's
  `subquestions` addressed (all of a subquestion's `must_mention` terms present).
- `stale_disclosure_rate` — fraction of `requires_stale_disclosure` cases whose
  answer carries a visible freshness warning.
- `unanswerable_numeric_hallucination_rate` — fraction of `unanswerable` cases
  whose answer invents a financial figure (currency/percent/magnitude/multiple;
  bare years and quarters don't count). Lower is better.
- `cross_session_leakage_rate` — fraction of conversation turns that resolved a
  ticker belonging to a *different* conversation. Lower is better.

These blocks appear in the summary only when the run scored Phase 2.2 fixtures
(`metrics.has_phase22_fixtures`), plus `conversational_denominators` and
`conversational_by_category` for the per-category denominator view. The runner
owns conversation history (`run_conversation`); a single test interleaves two
conversations to prove histories never cross.

Because no rewrite step exists until 2.2.2, `retrieval_query == raw_question`
and the resolved fields are derived best-effort from the evidence trace — the
carryover metrics read low today and start rewarding real follow-up handling
once 2.2.2 populates them explicitly.

### Phase 2.2 acceptance gate

`eval/gate.py::phase22_checks` enforces absolute Phase 2.2 thresholds
(activated as soon as a run scores the metric, independent of the baseline):
entity/metric/timeframe carryover ≥ 0.90, verbose paraphrase parity ≥ 0.90,
compound subquestion coverage ≥ 0.85, stale disclosure rate = 1.00, unanswerable
numeric hallucination rate = 0.00, cross-session leakage rate = 0.00. The gate
also fails when any of these categories has **zero eligible fixtures** (a
missing category must never look like a pass). These run alongside — and never
weaken — the 2.2.1.2 pre-metric checks and the baseline regression comparison.

### From the chat client (2.2.2.3)

`scripts/chat.py` wraps this harness for convenience — it only shells out to
`run_eval.py`/`score.py`, and never loads or starts the model itself:

- `/eval [N]` — drive the first `N` **single-turn** golden cases
  (`run_eval.py --limit N --no-conversations`) and print the run tail.
- `/eval conversations [N]` — include the multi-turn **conversation** fixtures
  (`run_eval.py --limit N`), then score deterministically
  (`score.py --no-judge`) and print the carryover / topic-reset /
  subquestion-coverage / cross-session-leakage metrics.

Both are **opt-in and live**: they require an already-running stack (middleware
on `:8000`, and the model on `:8087` for the run to produce non-degraded
answers). They are a convenience wrapper, not a substitute for the full
`run → score → gate` flow above. The chat client's own unit tests inject a fake
runner, so no eval process, model, or network is touched offline.

## Layout

```
eval/
  golden/
    finance_qa.jsonl              # committed — single-turn golden dataset
    finance_conversations.jsonl   # committed — multi-turn conversations (2.2.1.3)
  runs/                 # gitignored — per-run raw + summary artifacts
  baselines/
    strict.json          # committed — the bar the gate compares against for answer_policy=strict
    graded.json           # committed — the bar the gate compares against for answer_policy=graded
  run_eval.py           # Stage 1 — runner
  metrics.py            # Stage 2 — scoring functions (incl. LLM-as-judge)
  judge_prompts.py      # LLM-judge prompt templates
  score.py              # Stage 2 — scoring CLI + report
  gate.py               # Stage 3 — regression gate
```

## The golden dataset

`eval/golden/finance_qa.jsonl` is one JSON object per line. Each case:

| key               | meaning                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `id`              | Stable unique id, e.g. `rev-nvda-001`.                                 |
| `question`        | The natural-language question asked of the pipeline.                    |
| `expected_ticker` | The ticker the parser **should** detect (`null` for macro/comparison).  |
| `expected_intent` | The question type the parser **should** return.                         |
| `must_mention`    | Terms a good answer should contain (drives `keyword_coverage`).        |
| `expected_sources`| Canonical sources a good retrieval should surface (`[]` = count fallback). |
| `category`        | Grouping label for the per-category report.                            |
| `notes`           | Optional — why a case exists / what it tracks.                         |

Canonical source names: `yfinance`, `sec`, `fred`, `gdelt`, `ir`, `earnings`.

The set spans every intent (`fact_lookup`, `comparison`, `trend`, `explanation`,
`sentiment`, `news`, `risk`, `general`) plus known hard cases:

- **analytical** — cross-ticker aggregation ("lowest forward P/E", "highest
  earnings growth", "average P/E"). Currently unanswerable without the 2.1.4
  tool suite; they track regression/uplift.
- **hard** — untracked ticker (Berkshire Hathaway) — should refuse, not
  hallucinate.
- **projection** — forward forecast ("revenue next year") — needs the 2.1.5
  projection tool; tracks refuse-vs-hallucinate behavior.

### Adding a case

Append one JSON line to `eval/golden/finance_qa.jsonl`. Pick an id prefixed by
intent (`rev-`, `cmp-`, `trend-`, `expl-`, `sent-`, `news-`, `risk-`,
`macro-`, `hard-`, `gen-`). Set `expected_intent`/`expected_ticker` to the
**ground-truth** labels (what the parser *should* return, not necessarily what
it does today — gaps are the scoreboard's job to surface). Leave
`expected_sources` empty when the ideal source isn't seeded yet, so the metric
falls back to "did retrieval surface anything".

Re-validate after editing:

```bash
.venv/Scripts/python.exe -m pytest tests/test_eval_harness.py::TestGoldenDataset -q
```

## Running

### One-shot: run → score → gate

```powershell
# Middleware must be up on :8000 (the runner prefers the live /query endpoint).
.\scripts\start_stack.ps1

.venv/Scripts/python.exe eval/run_eval.py
.venv/Scripts/python.exe eval/score.py --policy graded
.venv/Scripts/python.exe eval/gate.py --policy graded
```

Run under `answer_policy: strict` (e.g. `ANSWER_POLICY=strict` before starting
the stack) and gate with `--policy strict` to check the strict baseline instead.

`gate.py` exits non-zero if any metric regresses beyond tolerance vs.
`eval/baselines/<policy>.json`, or if any of the 2.2.1.2 pre-metric checks
fail first: policy/schema/dataset-digest mismatch, an answerable model row
missing a complete evidence trace, or a judge metric's eligible denominator
dropping versus the baseline. Wire it in before/after each Phase 2 feature
and (optionally) in CI.

### Runner backends

`run_eval.py` prefers the **live middleware** (`POST :8000/query`, with
`include_evidence_trace: true`) so it measures the real system using the
exact trace the endpoint captured — no second retrieval, no truncation. If
`:8000` is down it falls back to importing the pipeline directly
(`IntentParser → Retriever → PromptAugmenter → model`), building an
equivalent trace from the same in-process retrieval. If both are unavailable
it still emits a well-formed row with `model_available: false`.

Override the endpoint with `EVAL_QUERY_URL`; force the offline path with
`--offline`.

### Refreshing the baseline

Each baseline (`eval/baselines/strict.json` / `graded.json`) is a committed
snapshot, regenerated **deliberately** — never blindly. After an intentional
improvement, promote the latest summary for the policy you ran under:

```powershell
.venv/Scripts/python.exe eval/score.py --set-baseline --policy graded
git add eval/baselines/graded.json
```

## Baseline status

`eval/baselines/strict.json` and `eval/baselines/graded.json` currently ship
as **placeholders**: every metric is `null` pending a live run scored against
the exact-evidence-trace metrics (2.2.1.2). The pre-2.2.1.2 `eval/baseline.json`
scored `faithfulness` against a truncated, re-retrieved context string and
excluded tool results, so its numbers are not comparable to
`grounded_faithfulness`/`policy_compliance` and were not migrated. Populate a
real baseline with a live middleware + model run, review the numbers, then
commit the result — see "Refreshing the baseline" above.

## Testing

```bash
.venv/Scripts/python.exe -m pytest tests/test_eval_harness.py -q
```

Unit tests cover the dataset, metrics, and gate (pass + fail paths) with the
LLM-judge mocked. The live-runner test is marked `integration` and skips when
`:8000` is down.
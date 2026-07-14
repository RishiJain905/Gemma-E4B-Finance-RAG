# Golden Finance QA Dataset

Two committed fixture files, both one JSON object per line (JSONL):

- `finance_qa.jsonl` — single-turn cases.
- `finance_conversations.jsonl` — multi-turn conversations (2.2.1.3).

All ids (case ids, conversation ids, and turn ids) are unique and stable across
both files. Re-validate coverage after editing either file:

```powershell
.venv\Scripts\python.exe eval\run_eval.py --validate-fixtures
```

That prints a per-dimension count and fails if any minimum is unmet or an id
collides.

## `finance_qa.jsonl` (single-turn)

Common fields:

- `id`: stable unique case identifier.
- `question`: user-facing finance question.
- `category`: bucket used for coverage reporting.
- `expected_ticker`: expected resolved ticker (`null` for macro/comparison).
- `expected_intent`: expected intent label.
- `must_mention`: answer keywords (drives `keyword_coverage`).
- `expected_sources`: canonical sources a good retrieval should surface
  (`[]` = count fallback). Canonical names: `yfinance`, `sec`, `fred`, `gdelt`,
  `ir`, `earnings`.
- `tags`, `notes`: optional labels / maintainer context.

### Phase 2.2 optional fields (2.2.1.3)

Added so single-turn cases can express compound, multi-ticker, timeframe,
stale, paraphrase, and answerability expectations:

- `subquestions`: list of `{id, must_mention}` — one entry per independently
  scored part of a compound question. A subquestion is "addressed" when every
  `must_mention` term appears in the answer.
- `expected_tickers`: ordered ticker set for multi-company questions.
- `expected_metrics`: canonical metric tokens (e.g. `total_revenue`,
  `gross_margin`, `pe_ratio`, `free_cash_flow`).
- `expected_timeframe`: e.g. `2026-Q2`, `FY2025`, `latest`, `last_quarter`.
- `expected_grounding`: the grounding mode a good answer should land in
  (`grounded`/`partial`/`general`/`refused`).
- `requires_stale_disclosure`: `true` when a good answer must carry a visible
  freshness/staleness warning (drives `stale_disclosure_rate`).
- `answerability`: `answerable | partial | unanswerable`. An `unanswerable`
  case must not contain an invented financial figure (drives
  `unanswerable_numeric_hallucination_rate`).
- `paraphrase_group`: links a verbose question to its concise equivalent; all
  members of a group must resolve to the same retrieval plan
  (drives `verbose_paraphrase_parity`).

`category` values used by the challenge set: `paraphrase`, `compound`,
`multi_ticker`, `timeframe`, `stale`, `unanswerable` (plus the pre-2.2 intent
buckets). The validation report counts these categories directly, so one large
bucket cannot hide a missing dimension.

## `finance_conversations.jsonl` (multi-turn)

One conversation per line:

```json
{
  "id": "conv-followup-nvda-amd-001",
  "category": "follow_up",
  "turns": [
    {"id": "conv-followup-nvda-amd-001-t0", "question": "Show NVDA revenue for FY2025.",
     "expected_tickers": ["NVDA"], "expected_metrics": ["total_revenue"],
     "expected_timeframe": "FY2025"},
    {"id": "conv-followup-nvda-amd-001-t1", "question": "What about AMD?",
     "expected_tickers": ["AMD"], "expected_metrics": ["total_revenue"],
     "expected_timeframe": "FY2025",
     "expected_carryover": {"metrics": ["total_revenue"], "timeframe": "FY2025"}}
  ]
}
```

- `id` (conversation) and every turn `id` are unique and stable.
- Each turn may declare the same `expected_*` fields as a single-turn case.
- `expected_carryover` names the fields a turn should inherit from earlier
  context: `ticker` (a string), `metrics` (a list), `timeframe` (a string). It
  drives `entity_carryover_accuracy`, `metric_carryover_accuracy`, and
  `timeframe_carryover_accuracy` — a turn is eligible for a carryover metric
  only when it declares that field under `expected_carryover`.
- `context_reset: true` marks an explicit "start over / forget that" turn that
  should inherit nothing.

The runner (`eval/run_eval.py::run_conversation`) owns conversation history: it
starts empty at each conversation, appends `{question, answer}` after every
turn, and never shares it across conversations. Today `retrieval_query ==
raw_question` (no rewrite step exists until 2.2.2), so the resolved
ticker/metric/timeframe fields are derived best-effort from the evidence trace;
2.2.2 will populate them explicitly and the carryover metrics will start
rewarding real follow-up handling.

## Challenge-set minimums (2.2.1.3)

Enforced by `--validate-fixtures`:

| dimension            | minimum |
|----------------------|---------|
| conversations        | 12      |
| conversation turns   | 30      |
| paraphrase pairs     | 6       |
| compound cases       | 6       |
| multi-ticker cases   | 6       |
| timeframe cases      | 6       |
| stale cases          | 4       |
| unanswerable cases   | 4       |

Use stable, seeded expectations only — never values that change with live
markets.

### Adding a case

Append one JSON line to the right file. Pick an id prefixed by intent/dimension
(`rev-`, `cmp-`, `pg-`, `cmp2-`, `mt-`, `tf-`, `stale-`, `unans-`, `conv-`). Set
`expected_intent`/`expected_ticker` to the **ground-truth** labels (what the
parser *should* return, not what it does today — gaps are the scoreboard's job
to surface). Leave `expected_sources` empty when the ideal source isn't seeded
yet. Re-run `--validate-fixtures`, then:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_eval_harness.py::TestGoldenDataset -q
```

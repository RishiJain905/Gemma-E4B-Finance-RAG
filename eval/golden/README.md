# Golden Finance QA Dataset

`finance_qa.jsonl` stores one JSON object per line.

Common fields:

- `id`: stable unique case identifier.
- `question`: user-facing finance question.
- `category`: optional bucket used for coverage reporting.
- `expected_ticker`: optional expected resolved ticker.
- `expected_intent`: optional expected intent label.
- `must_mention`: optional answer keywords.
- `tags`: optional labels for phase-specific cases.
- `notes`: optional human-readable context for maintainers.

The Phase 2.1.1 eval runner already loads this dataset; broader fetch-on-miss
scoring and runner behavior are future work.

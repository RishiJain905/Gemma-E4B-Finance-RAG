# 2.3.6 — Integration, Migration, and Evaluation: Results

Implemented on branch `phase-2.3.6` (from `Rishi-Ghost` after the 2.3.5 merge),
one commit per task.

## Setup

- Baseline at branch start: 1651 offline tests passing, `VERIFY: PASS`.
- Implementation: 2.3.6.1 on GPT-5.6 Codex (sol/high — migration/backfill
  invariants); 2.3.6.2 on Claude Opus 4.8 (xhigh — cross-cutting evaluation
  design). Independent gate verification before each commit.

## Before / after

| Area | Before | After |
|---|---|---|
| Schema evolution | inline ad-hoc column migrations in `sqlite_store.py` | stdlib ordered/idempotent runner (`src/storage/migrations.py` + 5 checksum-protected SQL files) with `schema_migrations` history; canonical `schema.sql`, inline fallback, empty, and Phase 2.2 databases proven to converge on one schema |
| Legacy data | no Phase 2.3 identity/corpus metadata | resumable bounded backfill (`scripts/migrate_phase2_3.py`): securities from canonical tickers + known CIKs, `core` list preserved as deep coverage, `corpus_items` from existing Chroma families/filings with zero re-embedding, orphans recorded for review — never guessed or deleted |
| Configuration | scattered | documented `universe.yaml`, `coverage.yaml`, `official_sources.yaml` (+ existing `sources.yaml`); env vars by name only; status reports booleans, never values |
| Rollout control | none | 8 per-capability switches, default **off**; schema lands before sources enable; rollback = flags only, non-destructive |
| Integration proof | none | offline pilot-slice e2e (`tests/test_phase2_3_end_to_end.py`, 16 tests): bootstrap → double refresh (zero duplicates), provider-wide 429 bounded + isolated, missing key truthful, indexing failure repairable, truthful per-source terminal statuses, Live Trace + Corpus Explorer projections verified |
| Evaluation | Phase 2.2 golden set | `phase2_3_golden.json`: 15 answerable cases across 10 classes (financing/primary-filing, corroborating citations, corporate actions, broad news, openFDA/NHTSA/USAspending, macro releases, historical-as-of, structured fact, abstain, stale) + 4 classes explicitly deferred with named 2.3.7 gates; runner captures the exact evidence ledger delivered to generation |
| Offline tests | 1651 | 1674 (+23) |

## Live validation (2026-07-15)

- Provider keys: **9/9 PASS** (Finnhub, FRED, BLS, BEA, EIA, openFDA, SEC EDGAR
  UA, Massive, Twelve Data) via one cheap authenticated call each; BEA required
  key activation, verified after. No key values logged.
- Full-stack smoke on the real stores + llama-server (:8087): two `/query`
  requests returned grounded, cited, non-degraded answers through the complete
  intent → retrieval → augmentation → generation pipeline.

## Live rollout, stage 1 (2026-07-15)

After enablement, the first live bootstrap surfaced ten first-contact adapter
defects (iShares CSV preamble, Nasdaq bot-blocking, Massive path/pacing, five
official-feed endpoint/contract mismatches, SEC 403 misclassified as
entitlement). All were fixed and live-verified on `phase-2.3-live-fixes`:

- Universe landed: **103 Nasdaq-100 + 504 S&P 500 active memberships**, 10,410
  SEC identities; corpus items 2,541 → 15,057+.
- Repaired bootstrap run `cfa768d2` truthfully `partial`: 52,139 Massive market
  observations, 15,737 Finnhub news items (bounded, resumable), all official
  feeds green; SEC filings correctly `rate_limited` with persisted cooldown.
- First live `daily` run `3d6ea343` healthy: TTL fresh-skips, cursor+overlap
  dedup collapsing 7,952 duplicate news items, idempotent official replays,
  and only SEC still inside its provider-side cooldown (one bounded attempt,
  isolation preserved).

## Caveats

- **Phase sign-off is intentionally incomplete here:** the all-ticker inventory,
  deterministic-answer bypass, indirect-plan accuracy, seeded-scale lexical, and
  storage-benchmark acceptance gates belong to Phase 2.3.7 (tracked separately).
  `PHASE2.3-COMPLETION.md` is deliberately not created yet.
- Broad-universe ingestion remains **off by default**; enabling is an explicit
  operator action (flags + bootstrap), per the staged-rollout plan.
- Known pre-existing doc inconsistency (README "six sources" vs ARCHITECTURE
  "seven") left untouched as out of scope.

## Regression gate outcome

- `VERIFY: PASS` at both task commits (1658 → 1674 tests).
- Existing counts/query behavior preserved with all Phase 2.3 flags off
  (explicit migration test), and the live smoke showed no answer degradation.

## Decision

**Ship.** Merged back to `Rishi-Ghost` via PR. Rollback path: capability flags
off (non-destructive); additive tables retained; data removal only via the
separate authorized maintenance command.

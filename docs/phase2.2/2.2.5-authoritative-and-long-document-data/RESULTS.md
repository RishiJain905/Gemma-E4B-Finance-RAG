# 2.2.5 — Authoritative & Long-Document Data: Results

**Date:** 2026-07-11
**Branch:** `phase-2.2.5` (one commit per task: 2.2.5.1 `6d3babb`, 2.2.5.2 `890875b`, 2.2.5.3 `f1290d9`, plus a live-calibration reflow fix)

## Setup

- Offline gate: `scripts\verify.ps1` (ruff + `pytest tests/ -m "not live"`).
- Live pilots: real SEC EDGAR API (CompanyFacts + one filing fetch), llama-server embeddings on `:8087`, real `data/finance.db` + `data/chroma`. All feature flags in committed configs remain **off** (`sec_companyfacts.enabled: false`, `sec.index_filing_text: false`, `enable_hierarchical_retrieval: false`); pilots used temporary in-process/config overrides restored afterward.
- Implementation provenance: 2.2.5.1 and 2.2.5.2 by Codex gpt-5.6-sol/high; 2.2.5.3 by opus-xhigh (Codex credits exhausted — instant-fail signature; see CLAUDE.md fallback protocol added the same day).

## Offline regression gate

| Point | Result |
|---|---|
| Phase start (post-2.2.4 merge) | 1131 passed — VERIFY: PASS |
| After 2.2.5.1 | 1144 passed (+13) — VERIFY: PASS |
| After 2.2.5.2 | 1165 passed (+21) — VERIFY: PASS |
| After 2.2.5.3 | 1209 passed (+44) — VERIFY: PASS |
| After reflow calibration fix | 1211 passed (+2) — VERIFY: PASS |

One transient gate failure was environmental (llama-server answered 503 mid-model-reload, flipping two `network` tests from skip to fail); it passed on re-run with the server healthy.

## Live pilots (real data)

1. **CompanyFacts ingestion (NVDA):** 26,737 facts seen / 26,732 written / 5 skipped from the official `data.sec.gov` CompanyFacts endpoint, full concept/unit/period/accession/filed-at provenance (rows 2007→2026). Canonical projection returns `total_revenue` correctly across a concept change (NVDA moved from `RevenueFromContractWithCustomerExcludingAssessedTax` to `Revenues` — the ordered concept list handled it): latest Q `$81.615B` (10-Q filed 2026-05-20), FY2026 annual `$215.938B`. **As-of 2023-06-01 returns `$7.192B`/2023-04-30 (filed 2023-05-26) — zero look-ahead.** Disabled mode verified as a strict no-op (`enabled: false` → 0 HTTP calls, `get_companyfacts` → `[]`).
2. **Filing-text backfill dry run:** honest report — 117 tracked filings, 0 parsed artifacts on disk (pre-2.2.5.2 processing never persisted text), confirming Phase 2.1's "no filing text in the corpus" finding. Historical population therefore requires re-processing (fetch+parse), not artifact migration.
3. **One-filing indexing pilot (AAPL 10-Q 0000320193-26-000013):** first attempt produced *zero usable sections* and correctly recorded retryable `index_pending` (fail-soft proved live). Root cause: the legacy parser collapses the filing to one whitespace-joined line, so the line-oriented splitter found no headings. Fixed with a deterministic flat-text reflow (newline insertion before unambiguous `PART x` / `Item N. <Title>` markers when newline density is implausibly low) plus a TOC-stub body floor applied only to reflowed text. Retry then indexed **9 sections / 34 child chunks for MD&A alone** with full heading/accession/parent-child provenance, and the filing advanced `index_pending → parsed`.

## Caveats

- The parsed artifact still opens with iXBRL context noise before the prose; the reflow makes it splittable, but a cleaner upstream HTML text extraction would improve section body quality further.
- `FilingProcessor.process_ticker(ticker, limit=N)` applies `limit` to the *global* unprocessed fetch before filtering by ticker (pre-existing quirk) — use a generous limit or the 2.2.5.3 backfill script's `--ticker` slicing.
- Long-document promotion gates (Recall@10 +8pts, context precision, prompt-size parity) need the three-arm eval run against a populated section corpus — blocked until the backfill has been applied broadly (deliberate, separate GPU/network activity).
- Hierarchical expansion was validated offline (44 tests incl. budget enforcement, dedup, provenance) but not live, since the corpus had only the pilot filing.

## Regression-gate outcome

`VERIFY: PASS` at every task boundary and phase close. Default-config behavior unchanged (all new sources/paths flagged off; live smoke of `/query` was covered in the 2.2.4 RESULTS and the legacy path is untouched by this phase when flags are off).

## Ship/rollback decision

**Ship dark (merged, flags off).** Enable order when promoting: (1) `sec_companyfacts.enabled` (additive table, independently disableable), (2) `sec.index_filing_text` + backfill script pilot slice, (3) `enable_hierarchical_retrieval` only after the three-arm eval clears its gates. Rollback = flip flag(s); filing-section families are idempotently replaceable and the backfill supports `--backup` snapshots; CompanyFacts never touches `fundamentals`.

# 2.2.4 — Corrective Retrieval & Provenance: Results

**Date:** 2026-07-11
**Branch:** `phase-2.2.4` (three commits, one per task: 2.2.4.1 `4bc93aa`, 2.2.4.2 `7c1e841`, 2.2.4.3 `819ec92`, plus one live-calibration fix)

## Setup

- Offline gate: `scripts\verify.ps1` (ruff + `pytest tests/ -m "not live"`), Windows, Python 3.13 venv.
- Live smoke: llama-server (TraceAlchemy) on `:8087`, middleware `uvicorn src.middleware.app:app` on `:8000`, SQLite `data/finance.db` + ChromaDB `data/chroma` with real ingested corpus (25 tickers).
- Feature flags exercised live: `ENABLE_ADAPTIVE_RAG=1 ENABLE_EVIDENCE_SUFFICIENCY=1 ENABLE_CORRECTIVE_RETRY=1`. Defaults in `configs/middleware.yaml` remain **off** (`answer_validation: report` is the only on-by-default addition; it is metadata-only).

## Offline regression gate (before/after)

| Point | Result |
|---|---|
| Baseline (branch start, `994da3d`) | 993 passed, 1 skipped — VERIFY: PASS |
| After 2.2.4.1 | 1027 passed (+34 tests) — VERIFY: PASS |
| After 2.2.4.2 | 1067 passed (+40 tests) — VERIFY: PASS |
| After 2.2.4.3 | 1131 passed (+64 tests) — VERIFY: PASS |

No pre-existing test changed outcome at any step; legacy path is unchanged with flags off.

## Live smoke (quality spot-check, not the full golden-set eval)

1. **Simple fact query, default config** — `What is Apple's most recent revenue?` → grounded answer with correct store value + legacy `[Source: yfinance/AAPL]` citation; response now carries `answer_validation` metadata (`validation_status: supported`, `citation_support_rate: 1.0`). Behavior otherwise identical to pre-2.2.4. Server-side latency ~8.6 s (generation-dominated, consistent with Phase 2.1 measurements).
2. **Compound multi-entity query, flags on** — `Compare Apple and Microsoft revenue and explain any recent risks to each.`
   - **Before calibration fix:** complex lane, gate graded `borderline` (`missing_metric`), one corrective retry (`apply_validated_alias`), still insufficient → honest refusal. Root cause: the grader's metric alias catalog used generic names (`total_revenue`, `sales`) while the yfinance ingestor stores `revenue_ttm`/`gross_margin_ttm`/`eps_ttm` — revenue obligations could never match real rows.
   - **After calibration fix** (aliases extended with the ingestors' real store keys): same query → `borderline` → one `apply_validated_alias` retry → **grounded** two-company comparison citing `[E11]`/`[E12]` facts and `[E16]`-style document evidence. Retrieval rounds: 2 (cap respected), planning calls: 0.

## Caveats

- The numeric-claim extractor keys on symbols/suffixes (`$`, `%`, `x`, `K/M/B/T`); a value written as `451,442,016,256.00 usd` (word suffix) is not extracted. Conservative direction — no false unsupported flags — but claim coverage will undercount until word-unit patterns are added.
- Metric-alias calibration is catalog-based; period-suffixed store keys (e.g. `revenue_q2_2026`) are still not matchable as period-qualified obligations. Full calibration should be data-driven against the live store's metric vocabulary.
- Decomposition (`sq1`/`sq2`) did not trigger on the live compound query — the alias corrective action resolved coverage first, which is the designed one-action bound. Compound decomposition behavior is covered by offline tests; a live case that specifically exercises it still needs a labeled eval pass.
- Golden-set metrics (abstention F1, unsupported-number reduction, unnecessary-retry rate) require the repaired 2.2.1 evaluator run against this branch — a separate, explicitly timed GPU/model activity per the phase rules; not run here.

## Regression-gate outcome

`VERIFY: PASS` at every task boundary and at phase close. No answer-quality regression observed in live smoke with default config (legacy path byte-identical; only additive `report`-mode metadata).

## Ship/rollback decision

**Ship dark (merged, flags off).** `enable_evidence_sufficiency` / `enable_corrective_retry` stay off by default pending the golden-set evaluation (promotion gates in the specs: abstention F1 ≥ 0.85, ≥30% relative unsupported-number reduction, unnecessary-retry ≤ 0.15). `answer_validation: report` ships on by default: metadata-only, fail-soft, no behavior change to answers. Rollback for any regression = flip the flag(s) off; no schema or storage migration involved.

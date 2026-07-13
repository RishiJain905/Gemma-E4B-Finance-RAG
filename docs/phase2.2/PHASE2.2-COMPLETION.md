# Phase 2.2 (2.2.4 – 2.2.6) Completion — Offline Sign-Off

**Date:** 2026-07-11
**Scope:** the 2.2.4–2.2.6 build loop (`TRACKER-2.2.4-2.2.6.md`). 2.2.1–2.2.3 signed off in earlier merges (`TRACKER-2.2.1-2.2.3.md`). Path note: the 2.2.6.3 spec references `docs/plans/phase2.2/`; this repo keeps phase docs under `docs/phase2.2/`, so this file lives here.

## Feature status

| Feature | Tasks | Gate at close | RESULTS.md decision |
|---|---|---|---|
| 2.2.4 corrective retrieval & provenance | 3/3 done | VERIFY: PASS (1131) | Ship dark (flags off; `answer_validation: report` on, metadata-only) |
| 2.2.5 authoritative & long-document data | 3/3 done | VERIFY: PASS (1211) | Ship dark (flags off; enable order documented) |
| 2.2.6 chat runtime & phase integration | 3/3 done | VERIFY: PASS (1272 incl. doc tests) | Ship dark (flags off) |
| 2.2.7 live retrieval knowledge graph | 4/4 done | VERIFY: PASS (offline) | Ship disabled-by-default (`enable_graph_observer: false`; loopback-only when on) |

## Requirement audit (offline evidence)

- **Evidence-delivery/prompt-policy defects fixed** — 2.2.1 (merged earlier); `document_body()` contract + single-owner prompt policy verified by `tests/test_prompt_augmenter.py`, `tests/test_evidence_grader.py`.
- **Bounded history & multiline chat** — 2.2.2 (merged earlier); `tests/test_conversation.py`, `tests/test_chat_client.py`.
- **Lane/call/round caps** — `tests/test_adaptive_orchestrator.py` (budget exhaustion, ≤3 subqueries, ≤2 rounds, ≤1 planning call); corrective retry consumes the final round (2.2.4.1 tests).
- **Deterministic tools cannot select writes** — 2.2.3.2 router tests; refresh stays behind the guarded explicit path.
- **Sufficiency/correction/citations/numeric validation observable + fail-soft** — `tests/test_evidence_grader.py`, `tests/test_citations.py`, `tests/test_answer_validator.py`, `tests/test_answer_policy.py`; live smoke in 2.2.4 RESULTS.md.
- **CompanyFacts + filing hierarchy authoritative, reversible, provenance-preserving** — `tests/test_sec_companyfacts.py` (as-of, amendments, conflicts), `tests/test_filing_sections.py`, `tests/test_hierarchical_retrieval.py`, `tests/test_index_sec_filing_text.py` (dry-run/resume/backup); live pilots in 2.2.5 RESULTS.md (26.7k NVDA facts, zero look-ahead; AAPL 10-Q → 9 sections/34 chunks).
- **Streaming/caching bounded and local-safe** — `tests/test_streaming.py` (redaction, ordering, fallback-once, midstream terminal error), `tests/test_retrieval_cache.py` (zero stale hits incl. same-count replacement, injected clocks); live SSE smoke in 2.2.6 RESULTS.md.
- **Rollback flag per implemented feature** — every 2.2.4–2.2.6 behavior sits behind a default-off flag (or `report`-mode metadata); flags enumerated in `docs/CONFIGURATION.md` and validated against code by `tests/test_phase22_docs.py`.
- **Docs agree with code** — `docs/ARCHITECTURE.md` / `docs/API.md` / `docs/CONFIGURATION.md` / `README.md` / `scripts/CHAT.md` / `eval/README.md` reconciled; stale streaming/chunking/filing-text claims corrected; enforced by `tests/test_phase22_docs.py` (spec structure, link resolution, config-key cross-check).
- **No secrets/data/model paths committed** — `.env`, `configs/model.local.yaml`, `data/**`, backfill manifests/backups all gitignored; verified via `git status`/`.gitignore` at each commit.

## Final offline gate

`powershell -ExecutionPolicy Bypass -File scripts\verify.ps1` → **VERIFY: PASS** (see tracker per-task counts; final run recorded in the 2.2.6.3 commit).

## Live GPU/model evaluation — PENDING (scheduled, not run)

Per the 2.2.6.3 spec, the full golden-set live evaluation is a separately authorized GPU activity and was **not** silently run at sign-off. Interim targeted live smokes (recorded in each feature's RESULTS.md) exercised: default-path answer parity, flags-on corrective retry to a grounded compound answer, real CompanyFacts as-of queries, one-filing section indexing, and SSE stage/token streaming.

**Scheduled-run checklist (when GPU time is authorized):** record model build/context/flags/corpus watermark + baseline digest → repaired strict/graded single-turn baseline → conversation/compound, adaptive/corrective, long-document, citation, latency comparisons → raw artifacts outside git, summarized into feature RESULTS.md files → stop the stack. Until then every live-model promotion gate is **PENDING — GPU evaluation not authorized**, and no flag is enabled by default.

## Phase 2.2.7 — live retrieval graph (offline sign-off, 2.2.7.4)

- **Implemented (4/4):** query-trace event model + loopback API (2.2.7.1), Store-backed corpus projection/explorer (2.2.7.2), Cytoscape single-page UI (2.2.7.3), chat integration + security + evaluation (2.2.7.4).
- **Observability, not GraphRAG** — projects the existing pipeline/Store; never changes retrieval or the answer. See `ARCHITECTURE-DECISION.md` (as-built confirmation).
- **Local-safe by construction** — disabled by default; loopback-only for every `/graph*` route (no bypass flag); strict same-origin CSP + hardening headers; `no-store` on detail endpoints; allowlisted/redacted nodes (question digest + bounded preview, bounded excerpts, secret/local-path scrubbing, http(s)-only links); no cookies/localStorage/service worker/analytics/third-party requests.
- **Bounded & non-interfering** — one shared emitter seam (2.2.6.1); publish p95 < 2 ms, non-blocking fan-out, fail-soft observer; observer-disabled path is byte-compatible (only the optional `graph_trace_id` differs). Completeness/noninterference/overhead/redaction covered by `tests/test_graph_observer.py`, `tests/test_graph_api.py`, `tests/test_graph_ui_contract.py`, `tests/test_chat_client.py`, and the golden fixture `tests/fixtures/graph/query_trace_v1.json`.
- **Browser acceptance (implementation-time smoke)** — UI loads under the strict CSP with **zero console errors/CSP violations**; Cytoscape renders, SSE connects, `/graph#trace=<id>` deep-links a trace. No model/GPU required (degraded path produces a trace).
- **Ship decision:** ship **disabled by default**; see `docs/phase2.2/2.2.7-live-retrieval-knowledge-graph/RESULTS.md`.

## Deferred (not relabeled complete)

- Live promotion gates for: evidence sufficiency/corrective retry, decomposition, hierarchical retrieval, streaming TTFT, retrieval cache, llama prompt reuse.
- Broad filing-text backfill (`scripts/index_sec_filing_text.py --apply` beyond the one-filing pilot).
- Cleaner upstream HTML/iXBRL text extraction ahead of the flat-text reflow.

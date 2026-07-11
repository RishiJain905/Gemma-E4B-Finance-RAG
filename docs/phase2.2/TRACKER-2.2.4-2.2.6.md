# Phase 2.2 — 2.2.4 through 2.2.6 build loop

Branch: `Rishi-Ghost`

Covers the next three feature groups of the Phase 2.2 roadmap
([README-2.2.md](README-2.2.md)): corrective retrieval and provenance,
authoritative/long-document data, and chat runtime + phase integration. Build
order per the README's suggested sequence: 2.2.4 depends on the 2.2.1
evidence contract and 2.2.3 adaptive orchestration already landed
([TRACKER-2.2.1-2.2.3.md](TRACKER-2.2.1-2.2.3.md)); 2.2.5.1 may be developed in
parallel with 2.2.4 but final integration stays ordered since both consume the
common evidence/query contracts; 2.2.6 closes the phase and requires
2.2.4/2.2.5 to be in before doc reconciliation and sign-off are meaningful.

## 2.2.4 — Corrective retrieval and provenance

- [x] 2.2.4.1 evidence-sufficiency-and-bounded-retry — spec: `docs/phase2.2/2.2.4-corrective-retrieval-and-provenance/2.2.4.1-evidence-sufficiency-and-bounded-retry.md` — replace count-based grounding with a deterministic, route-aware evidence sufficiency assessment (`sufficient|borderline|missing`); answer immediately when obligations are covered, allow at most one internal corrective retrieval when borderline, return an honest partial/refusal when evidence is missing. **Done:** deterministic grader + one budget-bounded corrective round behind `enable_evidence_sufficiency`/`enable_corrective_retry` (default off); VERIFY: PASS (1027 passed, +34 tests).
- [x] 2.2.4.2 selective-query-decomposition-and-fusion — spec: `docs/phase2.2/2.2.4-corrective-retrieval-and-provenance/2.2.4.2-selective-query-decomposition-and-fusion.md` — decompose only genuinely compound or low-coverage plans into at most two derived finance-specific subqueries, retrieve each through its appropriate modality, and fuse/deduplicate while keeping the original resolved query as the strongest signal. **Done:** deterministic `decompose_plan` + single drift gate for deterministic/planner subqueries, weighted RRF (sq0=1.0/derived=0.8/planner=0.6) with slot reservation, RUN_DERIVED_SUBQUERIES seam wired; VERIFY: PASS (1067 passed, +40 tests).
- [x] 2.2.4.3 citation-provenance-and-numerical-validation — spec: `docs/phase2.2/2.2.4-corrective-retrieval-and-provenance/2.2.4.3-citation-provenance-and-numerical-validation.md` — assign stable evidence IDs to every model-visible fact/document/tool result/calculation, require answers to cite those IDs, and deterministically flag unsupported citations and financial numbers before returning a grounded answer. **Done:** request-local [E#] ledger + structured EvidenceCitation parsing (legacy [Source:] compatible), stdlib Decimal numeric-claim validator, `answer_validation: report` default with fail-soft `report_unavailable`; VERIFY: PASS (1131 passed, +64 tests).

## 2.2.5 — Authoritative and long-document data

- [x] 2.2.5.1 sec-companyfacts-structured-ingestion — spec: `docs/phase2.2/2.2.5-authoritative-and-long-document-data/2.2.5.1-sec-companyfacts-structured-ingestion.md` — ingest official SEC CompanyFacts/XBRL data as an authoritative structured source, preserving concept/unit/period/filing/accession/fetch provenance for deterministic as-of queries, without overwriting lower-priority historical `fundamentals` records. **Done:** provenance-preserving `sec_companyfacts` table + idempotent upserts, as-of/amendment/conflict-aware `Store.get_companyfacts`, flagged scheduler source (`enabled: false`), zero `fundamentals` writes; VERIFY: PASS (1144 passed, +13 tests).
- [ ] 2.2.5.2 sec-filing-text-and-parent-child-indexing — spec: `docs/phase2.2/2.2.5-authoritative-and-long-document-data/2.2.5.2-sec-filing-text-and-parent-child-indexing.md` — place actual SEC filing section text into ChromaDB with deterministic document/section/chunk relationships, preserving headings and filing provenance so retrieval can find precise child chunks and reconstruct surrounding sections.
- [ ] 2.2.5.3 hierarchical-retrieval-migration-and-evaluation — spec: `docs/phase2.2/2.2.5-authoritative-and-long-document-data/2.2.5.3-hierarchical-retrieval-migration-and-evaluation.md` — backfill existing parsed SEC filings into the section/child index, retrieve precise child hits with bounded parent/sibling expansion, merge CompanyFacts where appropriate, and prove recall improves without growing prompt size.

## 2.2.6 — Chat runtime and phase integration

- [ ] 2.2.6.1 tool-aware-streaming-and-progress-events — spec: `docs/phase2.2/2.2.6-chat-runtime-and-phase-integration/2.2.6.1-tool-aware-streaming-and-progress-events.md` — keep bounded tool/planning rounds non-streaming, then stream the final answer synthesis and emit safe progress events (`query_started`/`stage`/`tool_started`/`tool_completed`/`token`/`metadata`/`error`) so enabling tools no longer disables streaming for the whole request.
- [ ] 2.2.6.2 versioned-retrieval-cache-and-prompt-efficiency — spec: `docs/phase2.2/2.2.6-chat-runtime-and-phase-integration/2.2.6.2-versioned-retrieval-cache-and-prompt-efficiency.md` — add a small in-memory retrieval cache keyed on a persisted data revision/freshness watermark, make prompt prefixes reusable, and measure llama-server prompt-cache reuse before enabling it; never reuse volatile final finance answers via semantic similarity.
- [ ] 2.2.6.3 documentation-full-evaluation-and-phase-sign-off — spec: `docs/phase2.2/2.2.6-chat-runtime-and-phase-integration/2.2.6.3-documentation-full-evaluation-and-phase-sign-off.md` — reconcile `docs/ARCHITECTURE.md`/`docs/API.md`/`docs/CONFIGURATION.md`/`README.md`/`scripts/CHAT.md` with implemented Phase 2.2 behavior, run the full offline gate, schedule (not silently run) the GPU/model live evaluation, and require a results/rollback decision for every implemented feature before declaring Phase 2.2 complete.

## Rules

- Gate: `scripts\verify.ps1` → `VERIFY: PASS`.
- 2.2.4 and 2.2.5 both consume the 2.2.1 evidence contract and 2.2.3 query-plan/orchestration contracts — do not trust their metrics if those tasks regress.
- 2.2.5.1 may be developed in parallel with 2.2.4, but final integration (2.2.5.3, then 2.2.6) stays ordered per `README-2.2.md`'s suggested build order.
- 2.2.6.3 is the phase-close task — it must not run until 2.2.4/2.2.5 features have their own ship/rollback decisions to reconcile.
- Two failed gate attempts on the same failure for one task → mark **BLOCKED** below with a one-line diagnosis, move to the next task. Don't loop on it.
- One commit per task, checked off here with a one-line result note as you go.
- Each implemented feature writes its own `RESULTS.md` inside the feature folder (setup, before/after metrics, caveats, regression-gate outcome, ship/rollback decision) — not created speculatively ahead of implementation.
- No push, no merge to `Rishi-Ghost` — leave the branch for review.

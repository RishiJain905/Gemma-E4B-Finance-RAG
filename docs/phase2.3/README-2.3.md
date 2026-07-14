# Phase 2.3 — Broad Market Intelligence and Corpus Scale

Phase 2.3 expands the finance RAG from a small hand-maintained watchlist into a
source-aware research corpus covering, at minimum, the current S&P 500 and
Nasdaq-100 universes. It adds authoritative company events, broad company news,
market data, official macroeconomic releases, and selected sector/regulatory
feeds without making any one provider a dependency for the rest of a refresh.

The phase is organized by system capability, not by API key. Each provider keeps
a small adapter, but every adapter emits the same normalized records, uses the
same provenance and deduplication rules, and is scheduled through the same
bounded refresh contract.

This is an incremental extension of the existing SQLite + Chroma architecture.
It does not replace the `Store`, the unified scheduler, or the Phase 2.2 graph.
Storage replacement remains possible only when the Phase 2.3.7 benchmark proves
the optimized architecture misses an explicit scale or operational gate.

## Outcomes

- Maintain a dynamic, historical security universe for the S&P 500 and
  Nasdaq-100, reconciled to SEC CIKs and ticker aliases.
- Separate broad coverage from expensive deep-research ingestion.
- Detect financing and other material company events from SEC filings, including
  registrations, prospectus supplements, 8-K obligations, and exhibits.
- Ingest company news, daily market summaries, corporate actions, official macro
  releases, and selected sector/regulatory events.
- Put structured observations in SQLite and narrative evidence in Chroma while
  retaining one searchable provenance record for every corpus item.
- Run daily refreshes with per-source quotas, cursors, circuit breakers, and
  failure isolation. A rate-limited source must not block another source.
- Extend Live Trace and Corpus Explorer so the new evidence is understandable at
  roughly 550 securities and substantially larger document counts.
- Answer corpus-capability questions, including complete ticker inventories,
  from authoritative metadata rather than a top-k retrieval sample.
- Improve indirect-query routing by tracking answer obligations and set
  completeness explicitly.
- Skip local-model generation for validated, completely covered deterministic
  answers, and replace full-corpus in-memory BM25 with persistent incremental
  lexical search.
- Promote useful dormant Phase 2.2 features and fully remove rejected ones after
  measured quality, latency, and compatibility comparisons.

## Selected architecture

```text
constituent feeds + SEC identity map
                 |
                 v
      security universe registry
      broad | deep | sector scopes
                 |
       +---------+----------+----------------+
       |                    |                |
       v                    v                v
  SEC/company events   news/market data  official feeds
       |                    |                |
       +--------- normalized records --------+
                            |
                    provenance + dedupe
                            |
               +------------+-------------+
               |                          |
               v                          v
        SQLite structured state      Chroma narratives
               +------------+-------------+
                            |
              retrieval + graph projections
```

The rationale and rejected alternatives are documented in
[ARCHITECTURE-DECISION.md](ARCHITECTURE-DECISION.md).

## Feature specs

| Folder | Capability | Task files |
|---|---|---|
| `2.3.1-universe-and-coverage-foundation` | Dynamic constituents, security identity, membership history, and broad/deep/sector policies | 2.3.1.1–2.3.1.2 |
| `2.3.2-multi-source-finance-ingestion` | SEC events, company news, market data, macro/regulatory, and sector feeds | 2.3.2.1–2.3.2.3 |
| `2.3.3-corpus-organization-and-provenance` | Normalization, deduplication, dual-store placement, retention, and retrieval ranking | 2.3.3.1–2.3.3.3 |
| `2.3.4-refresh-orchestration-and-operations` | Source registry, cursors, budgets, retries, circuit breakers, backfills, and status | 2.3.4.1–2.3.4.3 |
| `2.3.5-graph-and-corpus-explorer-scale` | Source-aware query traces and an aggregation-first, faceted Corpus Explorer | 2.3.5.1–2.3.5.3 |
| `2.3.6-integration-migration-and-evaluation` | Additive schema/config migration, end-to-end evaluation, rollout, and sign-off | 2.3.6.1–2.3.6.2 |
| `2.3.7-rag-quality-speed-and-storage` | Authoritative capability inventory, indirect-query completeness, deterministic answers, feature cleanup, persistent lexical retrieval, and a measured storage gate | 2.3.7.1–2.3.7.7 |

No `RESULTS.md` file is created until its feature has actually been implemented
and evaluated.

## Source map

| Group | Sources | Default scope |
|---|---|---|
| Universe | [Nasdaq-100 companies](https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies), [IVV holdings](https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf), SEC ticker/CIK mapping | Universe |
| Company events | [SEC EDGAR APIs and bulk data](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Broad |
| Company news | [Finnhub](https://finnhub.io/docs/api/company-news), Massive news where entitled | Broad |
| Market data | Massive grouped daily market summary and corporate actions; Yahoo/Twelve Data as optional fallback | Broad |
| Macro/regulatory | Federal Reserve, Treasury, BLS, BEA, EIA, New York Fed, CFTC | Global/sector |
| Sector | openFDA, NHTSA, USAspending | Relevant sectors only |
| Existing deep sources | Full filing text, CompanyFacts, IR, transcripts, estimates, optional GDELT | Deep |

ETF holdings are a practical personal-use proxy for index membership, not an
official licensed S&P constituent feed. Membership provenance is displayed and
preserved so this distinction is never hidden.

## Build order

1. **2.3.1** establishes stable identities and coverage scopes. Nothing else may
   fan out to the broad universe before this exists.
2. **2.3.3.1** establishes the normalized record/provenance contract.
3. **2.3.2.1** expands SEC event discovery first because it provides the most
   authoritative signal for financings and material events.
4. **2.3.2.2–2.3.2.3** add vendor and official feeds through the same contract.
5. **2.3.3.2–2.3.3.3** finish placement, retention, taxonomy, and ranking.
6. **2.3.4** promotes bootstrap jobs into bounded incremental refreshes.
7. **2.3.5** extends the graph against realistic seeded scale.
8. **2.3.6.1** performs the additive schema/config migration and seeds the
   realistic corpus required for measurement.
9. **2.3.7** improves query completeness and speed, promotes or removes dormant
   features, and benchmarks the optimized storage architecture.
10. **2.3.6.2** runs final end-to-end gates and enables sources/features in
    controlled stages.

## Phase-wide hard bounds

- Broad coverage targets the union of both indexes, not a hard-coded count.
- Deep sources never automatically fan out to the entire broad universe.
- No provider adapter writes directly to Chroma or SQLite outside the `Store`.
- One source failure, exhausted quota, or 429 cannot stop another source.
- Incremental refreshes use stored cursors and overlap windows; they do not
  re-download full histories.
- Full publisher articles are not scraped. Store provider-supplied summaries,
  primary documents, public releases, metadata, and source links.
- Price rows and time-series observations are not embedded.
- Graph APIs remain read-only, bounded, redacted, loopback-only, and unable to
  trigger refreshes.
- Corpus Explorer never renders the entire corpus. It starts with aggregates and
  expands one bounded page at a time.
- Exhaustive inventory/count questions use authoritative Store metadata and
  never infer completeness from top-k retrieval.
- Database replacement requires a failed documented storage gate, an approved
  pilot, and a separate migration/rollback decision.
- All new network tests are mocked and offline-safe; live tests are opt-in.

## Phase-wide acceptance gates

- Every active S&P 500 and Nasdaq-100 member resolves to one canonical security
  or an explicit reconciliation error.
- Membership changes are historical; removed constituents are deactivated, not
  deleted.
- Replaying an identical provider payload produces zero duplicate corpus items,
  Chroma families, events, market bars, or corporate actions.
- A representative SEC financing fixture is discoverable and classified with
  its filing, form, item/exhibit, ticker, CIK, timestamp, and source URL intact.
- Every enabled source completes, skips, rate-limits, or fails independently and
  records a bounded status row.
- Daily refresh can cover the complete broad universe within configured free-tier
  budgets.
- Corpus overview and filtered searches remain bounded at a seeded scale of at
  least 600 securities and 100,000 corpus items.
- An explicit all-ticker question returns the exact active registry set and count
  with zero invented or omitted tickers.
- Eligible complete deterministic questions make zero model calls and improve
  end-to-end latency by at least 90% over their model-generated baseline.
- Warm lexical p95 is below 100 ms and warm hybrid retrieval p95 is below 250 ms
  at the 100,000-item seeded scale on the documented reference machine.
- Every implemented-but-disabled RAG feature is promoted, temporarily retained
  with an owner and Phase 2.3 expiry, or removed with its stale configuration and
  exclusive dependencies.
- Live Trace resolves all displayed evidence/source nodes to the exact evidence
  ledger used by the answer.
- Corpus Explorer supports index, sector, ticker, source, item/event type,
  freshness, and date filters without loading all matching nodes.
- Observer-disabled query behavior remains compatible with Phase 2.2.
- `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1` ends with
  `VERIFY: PASS` before any numbered implementation task is complete.

## Configuration and secrets

The specs use the existing `.env` contract and never expose key values:

```dotenv
FINNHUB_API_KEY=
MASSIVE_API_KEY=
BLS_API_KEY=
BEA_API_KEY=
EIA_API_KEY=
OPENFDA_API_KEY=
FRED_API_KEY=
SEC_EDGAR_USER_AGENT="Name email@example.com"
```

Optional fallbacks remain `TWELVE_DATA_API_KEY` and
`ALPHA_VANTAGE_API_KEY`. Missing optional keys disable only their adapters.

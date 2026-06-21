# Test Coverage Report

Generated: 2026-06-21 (Phase 1.8.3)

Command:
```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Result: **558 passed, 5 skipped** — overall coverage **84%**.

## Current Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| src/storage/store.py | 88% | freshness/staleness paths covered |
| src/storage/sqlite_store.py | 99% | |
| src/storage/chroma_store.py | 99% | |
| src/middleware/app.py | 71% | model-call + endpoint error branches need live model |
| src/middleware/intent_parser.py | 100% | all question types + edge cases |
| src/middleware/retriever.py | 77% | some ranking/error branches network-dependent |
| src/middleware/prompt_augmenter.py | 94% | |
| src/middleware/config.py | 100% | |
| src/middleware/models.py | 100% | |
| src/scheduler/__init__.py | 96% | all run modes + dispatch branches covered |
| src/scheduler/__main__.py | 0% | thin CLI shim (covered indirectly via main()) |
| src/utils/resilience.py | 99% | retry, circuit breaker, DLQ |
| src/utils/logging.py | 97% | file logging + structured logger |
| src/ingestion/yfinance_ingestor.py | 88% | |
| src/macros/fred_ingestor.py | 99% | |
| src/macros/gdelt_ingestor.py | 70% | GDELT HTTP/parse branches (network-heavy) |
| src/macros/ir_ingestor.py | 77% | RSS/HTML scraping branches (network-heavy) |
| src/macros/earnings_transcripts.py | 54% | transcript scraping/parsing (network-heavy) |
| src/sec/edgar_fetcher.py | 72% | SEC HTTP/retry branches (network-heavy) |
| src/sec/filing_parser.py | 81% | |
| src/sec/filing_processor.py | 96% | |
| src/sec/scheduler.py | 93% | |

## Priority Gaps

The modules below 80% are all **network-bound ingestors** whose uncovered
lines are HTTP fetch / HTML-scrape / API-parse branches that require live
upstream services (Yahoo Finance, GDELT, SEC EDGAR, company IR pages,
Seeking Alpha / Fool). Their orchestration, storage, and parsing-of-fixtures
paths are covered; the remaining lines exercise real-network error handling.

1. **earnings_transcripts.py (54%)** — transcript fetching/parsing from
   Seeking Alpha / Fool. Public read APIs (`get_latest_guidance`,
   `fetch_all_core`) are covered; raw scraping is not.
2. **gdelt_ingestor.py (70%)** — GDELT DOC/GKG HTTP paths. `get_sentiment_summary`
   and storage are covered.
3. **edgar_fetcher.py (72%)** — SEC EDGAR HTTP fetch/retry. Discovery
   orchestration via `src/sec/scheduler.py` (93%) is covered.
4. **ir_ingestor.py (77%)** — RSS parse + HTML scrape fallback.
   `fetch_for_ticker` / `fetch_all_core` orchestration is covered.
5. **app.py (71%)** — remaining lines are the real model-call body and
   per-endpoint exception branches; exercised by the live `integration`
   tests but not all error branches.

## Test Suites (Phase 1.8.3)

| File | Purpose | Markers |
|------|---------|---------|
| test_phase1_8_units.py | intent parser, store freshness, scheduler, resilience | — |
| test_phase1_8_coverage.py | file logging, source dispatch, degraded answer, endpoints | integration/network on endpoint class |
| test_scheduler_store_integration.py | scheduler -> store contract | integration |
| test_middleware_store_integration.py | middleware -> store path (live embedder) | integration, network |
| test_full_pipeline_integration.py | real ingest -> store -> search | slow, network, integration |
| test_regression.py | Phase 1.1-1.7 feature regression | regression (+network on live checks) |

## Timing

- Full suite (`pytest tests/`): ~85s with the model + embedding server up.
- Pure-unit subset (`-m "not slow and not network and not live"`): ~80s — the
  floor is dominated by pre-existing integration tests that each spin up a
  FastAPI `TestClient` and call the live embedder; further reducing this would
  require re-marking/optimising pre-1.8 tests.

Run categories:
```bash
pytest tests/ -m regression     # prior-phase regression
pytest tests/ -m integration    # cross-module integration
pytest tests/ -m "not slow"     # skip slow ingestion tests
pytest tests/ --live            # include live model/SEC network tests
```

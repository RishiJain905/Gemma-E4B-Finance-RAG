# Massive and Federal Reserve Daily Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair the two partial providers in the daily scheduler without changing other provider behavior.

**Architecture:** Add an opt-in unique-CIK tie-breaker to security resolution and enable it only in Massive. Make the official text helper BOM-safe, then correct the Federal Reserve feed catalog and provider-specific item matching.

**Tech Stack:** Python, SQLite, requests, ElementTree, pytest, PowerShell verification gate.

---

### Task 1: Massive ambiguous CBOE resolution

**Files:** `tests/test_universe_registry.py`, `tests/test_massive_ingestor.py`, `src/storage/sqlite_store.py`, `src/storage/store.py`, `src/ingestion/massive_ingestor.py`

1. Add failing tests proving default ambiguity remains and Massive can opt into a unique CIK-backed identity.
2. Add `prefer_cik: bool = False` through the Store facade and SQLite resolver.
3. Pass `prefer_cik=True` only from Massive normalization.
4. Re-run the focused tests.

### Task 2: Federal Reserve BOM and real feed shapes

**Files:** `tests/test_official_source_ingestors.py`, `src/ingestion/official/__init__.py`, `src/ingestion/official/federal_reserve.py`, `configs/official_sources.yaml`

1. Add failing tests for BOM-mojibake HTTP decoding, title/category matching, and RDF `dc:date`.
2. Prefer `utf-8-sig` decoding of response bytes with a text fallback.
3. Match Federal Reserve selectors against category plus title and accept `pubDate` or namespaced `date`.
4. Correct the five Federal Reserve feed endpoints/selectors.
5. Re-run the focused tests.

### Task 3: Repair the Chroma integration fixture

**Files:** `tests/test_middleware_store_integration.py`

1. Reproduce the empty `/search` result with the live embedding endpoint.
2. Trace the mismatch between the fixture store and the lifespan-created retriever.
3. Inject the temporary store, normal middleware config, and a reset retriever without starting the default lifespan store.
4. Run all three middleware/store integration tests.

### Task 4: Resumable Federal Reserve history

**Files:** `tests/test_official_source_ingestors.py`, `tests/test_scheduler_cli_phase2_3.py`, `src/ingestion/official/federal_reserve.py`, `src/scheduler/__init__.py`

1. Add failing tests for bounded history batches, durable resume, completion no-op, and bootstrap routing.
2. Prefer the RDF `about` identity for statistical release records.
3. Add a Federal-only history ingestion path that begins after the daily 50 and advances its cursor only after a clean committed batch.
4. Route explicit Federal bootstrap through the history path; leave daily and every other source unchanged.

### Task 5: Verification and live scheduler checks

1. Run scoped verification for Massive and official providers.
2. Run the full `scripts/verify.ps1` gate and require `VERIFY: PASS`.
3. Integrate the verified branch, confirm no scheduler run is active, and run `python -m src.scheduler daily --force`.
4. Inspect persisted Massive/Federal Reserve results and freshness.
5. Run `python -m src.scheduler hourly --force` separately and report GDELT without changing its code.

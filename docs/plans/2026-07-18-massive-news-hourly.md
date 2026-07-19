# Massive News Hourly Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hourly GDELT work with a separately monitored, quota-safe Massive news source.

**Architecture:** Register `massive_news` as its own hourly source and disable GDELT by configuration. Reuse `MassiveIngestor.ingest_news` with a durable timestamp cursor, while aliasing `massive_news` budget persistence to the daily `massive` provider bucket. Teach the read-only watcher to render configured-disabled sources explicitly.

**Tech Stack:** Python, YAML, SQLite, pytest, PowerShell verification gate.

---

### Task 1: Registry selection

**Files:**
- Modify: `configs/sources.yaml`
- Test: `tests/test_source_registry.py`
- Test: `tests/test_scheduler_store_integration.py`

1. Write failing tests that `massive_news` is the only available hourly source, has a one-hour TTL/two-hour timestamp overlap, and GDELT is `configured_disabled`.
2. Run the focused tests and confirm they fail because `massive_news` is absent and GDELT is enabled.
3. Add the `massive_news` registry entry and set `gdelt.enabled: false`.
4. Run the focused tests and require PASS.

### Task 2: Shared Massive quota accounting

**Files:**
- Modify: `src/scheduler/__init__.py`
- Test: `tests/test_scheduler.py`
- Test: `tests/test_source_budget.py`

1. Write a failing scheduler test that daily `massive` usage is loaded when creating the `massive_news` budget and hourly usage is persisted back to the `massive` day/minute windows.
2. Run it and confirm the current independent source name fails the assertion.
3. Add the smallest explicit budget alias map: `massive_news -> massive`.
4. Keep freshness/status identities unchanged; alias only durable quota reads/writes.
5. Run scheduler/budget tests and require PASS.

### Task 3: Durable hourly news cursor

**Files:**
- Modify: `src/ingestion/massive_ingestor.py`
- Test: `tests/test_massive_ingestor.py`

1. Write failing tests for a first news page, oldest-first provider parameters, a two-hour overlap on the next run, and cursor advancement after clean persistence.
2. Write a failing test proving a storage/indexing failure does not advance the cursor.
3. Run the focused tests and confirm the expected cursor/parameter failures.
4. Add `news_overlap_hours`, read `massive_news/US`, request one page with `limit=1000`, `sort=published_utc`, `order=asc`, and advance to the newest provider timestamp only on a clean batch.
5. Run all Massive ingestor tests and require PASS.

### Task 4: Hourly scheduler routing

**Files:**
- Modify: `src/scheduler/__init__.py`
- Test: `tests/test_scheduler.py`
- Test: `tests/test_scheduler_store_integration.py`

1. Write a failing test that hourly routing invokes only `MassiveIngestor.ingest_news`, with the budgeted/paced HTTP client and registry overlap.
2. Confirm daily Massive still invokes `ingest_all()` without news.
3. Add the `massive_news` source branch.
4. Run scheduler tests and require PASS.

### Task 5: Explicit disabled watcher state

**Files:**
- Modify: `scripts/watch_scheduler.py`
- Test: `tests/test_watch_scheduler.py`

1. Write failing tests that configured-disabled GDELT renders `state=disabled` even when its old cache row is stale, and that `massive_news` remains a separate freshness name.
2. Add a read-only YAML enabled-flag collector and merge it into freshness collection.
3. Render disabled rows dimly and leave fresh/stale calculations unchanged for enabled sources.
4. Run watcher tests and require PASS.

### Task 6: Verification and rollout

1. Run focused verification for Massive, scheduler, registry, budgets, and watcher.
2. Run `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1` and require final `VERIFY: PASS`.
3. Run `python -m src.scheduler hourly --force --json` live.
4. Confirm `massive_news` is successful/fresh, GDELT is configured-disabled, and `scripts\watch_scheduler.py --once` displays separate `massive` and `massive_news` rows plus disabled GDELT.
5. Check the Federal backfill process, cursor, and logs without interrupting it.
6. Commit all intended changes and push local `Rishi-Ghost` directly to `origin/Rishi-Ghost`.

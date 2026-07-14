# GDELT Rate-Limit Fail-Fast Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop the current GDELT source run after its third HTTP 429 while allowing the scheduler to continue other ingestion sources.

**Architecture:** Raise a typed exception at the point where GDELT exhausts its configured 429 retries. Let it cross query/ticker loops and rely on the scheduler's existing per-source exception boundary to record the failure and continue.

**Tech Stack:** Python, httpx, pytest, unittest.mock

---

### Task 1: Specify exhausted-rate-limit behavior

**Files:**
- Modify: `tests/test_gdelt_ingestor.py`

1. Add a test where all configured HTTP attempts return 429.
2. Assert the typed rate-limit error is raised after exactly three calls.
3. Run the test and confirm it fails because the current implementation returns `None`.

### Task 2: Specify batch fail-fast behavior

**Files:**
- Modify: `tests/test_gdelt_ingestor.py`

1. Add a ticker-level test whose first search raises the typed error.
2. Assert the next ticker alias is not searched.
3. Run the test and confirm the current broad exception handler swallows the error.

### Task 3: Implement the typed error

**Files:**
- Modify: `src/macros/gdelt_ingestor.py`

1. Define `GDELTRateLimitError` near the module logger.
2. Raise it after the final 429, including `Retry-After` when present.
3. Re-raise it explicitly from `_search_gdelt`; preserve fail-soft handling for other exceptions.
4. Run the focused regression tests and confirm they pass.

### Task 4: Verify scheduler isolation

**Files:**
- Modify: `tests/test_scheduler.py`

1. Add a scheduler test with GDELT failing and a later source succeeding.
2. Assert GDELT is recorded as an error and the later source runs.
3. Run the focused scheduler test and confirm it passes using the existing boundary.

### Task 5: Run the repository gate

**Files:**
- Verify: `scripts/verify.ps1`

1. Run `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1`.
2. Require the final line `VERIFY: PASS` before reporting completion.

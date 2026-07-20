# Fast Middleware Health Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make normal chat startup independent of Chroma HNSW hydration and full scheduler reporting while preserving detailed health behavior.

**Architecture:** `Store.heartbeat()` will derive `chroma_doc_count` from the application-owned lexical index state only when its revision agrees with Chroma's published corpus revision. Automatic chat probes will request `/health?details=false`, which keeps storage, model, and capability checks but skips the full scheduler report; detailed health remains the default for existing callers and the `/health` chat command.

**Tech Stack:** Python, SQLite, ChromaDB, pytest

---

### Task 1: Add the startup regression tests

**Files:**
- Modify: `tests/test_store.py`

**Step 1: Write the failing tests**

Add one test where SQLite reports a completed lexical state with row count 7
and Chroma publishes the same revision. Assert that heartbeat returns 7 and
never calls `chroma.count()`. Add a second test where revisions differ and
assert that the count is `None`, again without calling `chroma.count()`.

**Step 2: Run the focused tests to verify they fail**

Run: `pytest tests/test_store.py -k heartbeat -v`

Expected: FAIL because heartbeat still calls `chroma.count()`.

### Task 2: Use the lightweight revision-consistent count

**Files:**
- Modify: `src/storage/store.py:339-346`
- Test: `tests/test_store.py`

**Step 1: Implement the minimal helper**

Add a private helper that reads `SQLiteStore.get_lexical_index_state()` and
`ChromaStore.corpus_revision()`. Return the indexed row count only for matching
revisions with no rebuild cursor; otherwise return `None` and fail soft.

**Step 2: Switch heartbeat to the helper**

Keep the current response keys and backend heartbeat calls. Replace only the
native Chroma count call.

**Step 3: Run the scoped gate**

Run: `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 -TestPath tests\test_store.py`

Expected final line: `VERIFY: PASS`

### Task 3: Add lightweight startup health

**Files:**
- Modify: `src/middleware/app.py`
- Modify: `scripts/chat.py`
- Test: `tests/test_middleware.py`
- Test: `tests/test_chat_client.py`

**Step 1: Add failing lightweight-health tests**

Prove `details=false` does not call the scheduler summary builder and that every
automatic chat startup/capability probe uses the lightweight path.

**Step 2: Implement lightweight health mode**

Keep detailed health as the endpoint default. Skip only scheduler/freshness
expansion when `details=false`, and route automatic chat probes to that mode.

**Step 3: Run focused tests**

Run the affected middleware and chat client tests and confirm they pass.

### Task 4: Verify the repository

**Files:**
- Verify only

**Step 1: Run the full gate**

Run: `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1`

Expected final line: `VERIFY: PASS`

**Step 2: Manually time the live health path**

Start middleware with the existing command and confirm `/health` responds
without increasing process memory by loading the vector segment.

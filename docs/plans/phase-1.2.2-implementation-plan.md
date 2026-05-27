# Phase 1.2.2 Implementation Plan — SQLite Schema & Initialization

**Phase:** 1.2.2  
**Spec:** `docs/phase1.2/1.2.2-sqlite-schema-and-init.md`  
**Date:** 2026-05-27  
**Target Branch:** `Rishi-Ghost`  
**Orchestrator:** Context manager and integration owner  
**Delegated To:** 2 backend specialists  

---

## 1. Role & Operating Mode

**Orchestrator responsibilities:**
- Maintain global context across all workstreams
- Preserve the approved implementation plan
- Delegate all specialist implementation to backend subagents
- Identify dependencies, overlap, and execution order
- Own integration and merge strategy into `Rishi-Ghost`
- Ensure no approved requirements are lost across handoffs

**Who Does What:**
- Backend / data pipeline tasks: backend specialist
- Tests (creation or expansion): backend specialist (whoever owns the code)
- Integration and merging: orchestrator only
- Cross-stream conflict resolution: orchestrator only

**Orchestrator does not default to implementing specialist work directly.** Step in only when:
- The work is integration-owned by the orchestrator
- A blocking issue prevents safe delegation
- No suitable specialist role exists
- Subagent output requires intervention to unblock progress

---

## 2. Plan Intake

### 2.1 Spec Read
- Primary spec: `docs/phase1.2/1.2.2-sqlite-schema-and-init.md`
- Config reference: `configs/storage.yaml` (SQLite path: `data/finance.db`)
- Prior phase summary: `docs/p1.2.1-done.md`
- Current repo state: `src/storage/__init__.py` exists and is empty

### 2.2 Overlap & Dependencies
- **Overlap with 1.2.1:** `src/storage/__init__.py` exists but is empty. Must be updated to export `SQLiteStore`.
- **Shared file:** `src/storage/sqlite_store.py` — all Workstream B tasks target this single file.
- **Config dependency:** `configs/storage.yaml` already defines `sqlite.path: data/finance.db`. The `SQLiteStore` default path aligns with this.
- **Ordering dependency:** Schema file (`docs/phase1.2/schema.sql`) should exist before `SQLiteStore._init_schema` references it at runtime. The `_inline_schema` fallback mitigates boot failure if the file is missing.
- **Future dependency:** `src/storage/sqlite_store.py` will be imported by Phase 1.2.4 (Unified Storage Abstraction).

### 2.3 Verification Strategy
- **Syntax verification:** Run `sqlite3` against `schema.sql` to confirm DDL parseability.
- **Unit tests:** `pytest tests/test_sqlite_store.py` covering CRUD, upsert conflict resolution, cache lifecycle, and ingestion logging.
- **Integration smoke test:** `python -c "from src.storage.sqlite_store import SQLiteStore; s = SQLiteStore(); print(s.db_path)"` from project root.
- **WAL mode check:** Query `PRAGMA journal_mode` after connection.
- **Index check:** Query `sqlite_master` to confirm all 8 indexes exist.

---

## 3. Development

### Task Decomposition

#### Workstream A — Schema Definition
**Owner:** backend specialist 1

**Task A1: Create `docs/phase1.2/schema.sql`**
- **File path:** `docs/phase1.2/schema.sql`
- **Deliverable:** Full DDL for 4 tables (`fundamentals`, `filings`, `cache_meta`, `ingestion_log`) plus 8 indexes.
- **Cross-reference:** Spec section "The Schema"
- **Acceptance criteria:**
  - All 4 `CREATE TABLE IF NOT EXISTS` statements present
  - All columns match spec types and constraints
  - `UNIQUE(ticker, metric, period)` on `fundamentals`
  - `PRIMARY KEY (ticker, source, metric_scope)` on `cache_meta`
  - `UNIQUE` on `filings.accession`
  - 8 `CREATE INDEX IF NOT EXISTS` statements present
  - File runs successfully via `sqlite3 < schema.sql` without errors

#### Workstream B — SQLiteStore Python Module
**Owner:** backend specialist 2  
**Target file:** `src/storage/sqlite_store.py`

**Task B1: Class skeleton, connection setup, and schema initialization**
- **Deliverable:** `SQLiteStore` class with:
  - `SCHEMA_SQL` and `DEFAULT_DB_PATH` class attributes
  - `__init__(self, db_path=None)`
  - `_connect(self)` -> `sqlite3.Connection` (WAL mode, `row_factory=sqlite3.Row`, `foreign_keys=ON`)
  - `_init_schema(self)` (reads `docs/phase1.2/schema.sql` or falls back to `_inline_schema`)
- **Cross-reference:** Spec section "Python Module — src/storage/sqlite_store.py"
- **Acceptance criteria:**
  - `SQLiteStore()` creates `data/` directory if missing
  - `SQLiteStore()` creates `data/finance.db` if missing
  - `PRAGMA journal_mode` returns `wal`
  - `_connect` returns a connection with `row_factory` set
  - Schema initialization runs without `sqlite3.OperationalError`

**Task B2: Fundamentals CRUD**
- **Deliverable:**
  - `upsert_fundamental(self, ticker, metric, value, ...)` with `ON CONFLICT DO UPDATE`
  - `get_fundamental(self, ticker, metric, period=None)` -> `Optional[dict]`
  - `get_fundamentals_batch(self, ticker, metrics=None)` -> `dict`
- **Cross-reference:** Spec section "Fundamentals CRUD"
- **Acceptance criteria:**
  - Insert new row returns `total_changes > 0`
  - Upsert existing `(ticker, metric, period)` updates `value`, `source_url`, `source_accessed_at`
  - `get_fundamental` returns `None` for missing data
  - `get_fundamentals_batch` returns `{metric: value}` mapping
  - `dict(row)` is returned (not raw `sqlite3.Row`)

**Task B3: Filing Tracking**
- **Deliverable:**
  - `register_filing(self, ...)` -> `bool` (True if new, False if duplicate)
  - `mark_filing_parsed(self, accession, embedding_id=None)`
  - `get_unprocessed_filings(self, limit=10)` -> `list[dict]`
- **Cross-reference:** Spec section "Filing Tracking"
- **Acceptance criteria:**
  - Duplicate accession returns `False` (`INSERT OR IGNORE`)
  - `mark_filing_parsed` updates status to `parsed` and sets `parsed_at`
  - `get_unprocessed_filings` only returns rows with `status='unprocessed'`

**Task B4: Cache Management**
- **Deliverable:**
  - `get_cache_status(self, ticker, source)` -> `Optional[dict]`
  - `mark_cache_fresh(self, ticker, source, ttl_hours=24)`
  - `mark_cache_stale(self, ticker, source, error=None)`
  - `get_stale_cache_entries(self, limit=20)` -> `list[dict]`
- **Cross-reference:** Spec section "Cache Management"
- **Acceptance criteria:**
  - `mark_cache_fresh` uses `UPSERT` (`INSERT ... ON CONFLICT DO UPDATE`)
  - `get_stale_cache_entries` filters `next_scheduled_update < datetime('now')` and `status != 'fetching'`
  - `get_cache_status` returns `None` for unknown ticker/source pairs

**Task B5: Ingestion Logging**
- **Deliverable:**
  - `log_ingestion_start(self)` -> `str` (run_id UUID)
  - `log_ingestion_complete(self, run_id, status, items, new, updated)`
- **Cross-reference:** Spec section "Ingestion Logging"
- **Acceptance criteria:**
  - Returns valid UUID4 string
  - `log_ingestion_complete` calculates `duration_seconds` via `julianday`
  - Row exists in `ingestion_log` after both calls

**Task B6: Query Support & Inline Schema Fallback**
- **Deliverable:**
  - `search_facts(self, ticker=None, metric=None, source=None, limit=10)` -> `list[dict]`
  - `_inline_schema(self) -> str` (static method)
- **Cross-reference:** Spec section "Query Support" and "Schema String"
- **Acceptance criteria:**
  - `search_facts` builds dynamic `WHERE` clause safely (parameterized queries only)
  - Returns empty list when no matches
  - `_inline_schema` contains full DDL equivalent to `schema.sql`

#### Workstream C — Tests & Verification
**Owner:** backend specialist 2 (same as Workstream B, or parallel backend specialist 3 if available)

**Task C1: Create `tests/test_sqlite_store.py`**
- **Deliverable:** Comprehensive pytest suite covering:
  - Database initialization and WAL mode
  - `upsert_fundamental` insert + conflict update
  - `get_fundamental` with and without period
  - `get_fundamentals_batch`
  - `register_filing` idempotency
  - `mark_filing_parsed`
  - `get_unprocessed_filings`
  - `mark_cache_fresh` / `mark_cache_stale` / `get_stale_cache_entries`
  - `log_ingestion_start` / `log_ingestion_complete`
  - `search_facts`
  - `_inline_schema` correctness
- **Acceptance criteria:**
  - `pytest tests/test_sqlite_store.py -v` passes with 100% success
  - Uses a temporary database path (`tmp_path`) to avoid polluting `data/finance.db`

**Task C2: Smoke test from project root**
- **Deliverable:** One-liner confirming import and initialization from project root
- **Command:** `python -c "from src.storage.sqlite_store import SQLiteStore; store = SQLiteStore(); print('OK', store.db_path)"`
- **Acceptance criteria:** Prints `OK <path>` without exceptions

#### Workstream D — Package Integration
**Owner:** orchestrator

**Task D1: Update `src/storage/__init__.py`**
- **Deliverable:** `from .sqlite_store import SQLiteStore` in `src/storage/__init__.py`
- **Acceptance criteria:** `from src.storage import SQLiteStore` works from project root

**Task D2: Write `docs/p1.2.2-done.md`**
- **Deliverable:** Phase completion summary following `docs/p1.2.1-done.md` style
- **Acceptance criteria:** Lists all files created/modified, verification checklist, blockers/follow-ups

---

### Delegation Rules
- **Task A1, B1-B6:** Delegate to **backend specialist**
- **Task C1, C2:** Delegate to **backend specialist** (whoever owns the code under test)
- **Task D1, D2:** Orchestrator performs directly (integration-owned work)

---

### Parallel-First Execution Ordering

**Wave 1 (parallel, no dependencies):**
- A1: Create `schema.sql`
- B1: Class skeleton + connection setup

**Wave 2 (parallel, depends on Wave 1):**
- B2-B6: Implement CRUD, filing, cache, logging, query methods (all in `sqlite_store.py`). Since they target the same file, assign to a **single backend specialist** or have the orchestrator merge partial contributions in order.
- C1: Draft test file skeleton (can begin in parallel, though final assertions need B1-B6 complete).

**Wave 3 (depends on Wave 2):**
- D1: Update `__init__.py`
- C2: Run full smoke test + pytest suite
- D2: Write completion summary

**Why this ordering:**
- `schema.sql` and class skeleton are independent.
- All method implementations in `sqlite_store.py` share the same class and connection patterns defined in B1, but do not depend on each other internally.
- Tests need the complete module to pass, but test structure can be drafted earlier.
- `__init__.py` update is trivial and must happen after `sqlite_store.py` exists.

---

### Development Loop (per task)

1. **Delegate** — Orchestrator assigns task to backend specialist with explicit file path, acceptance criteria, and spec cross-reference.
2. **Implement** — Specialist writes code, commits to feature branch or reports patch.
3. **Verify** — Specialist runs:
   - Bug fix: reproduce first, confirm fail, fix, confirm pass.
   - Feature: run task-specific verification (for example, `python -c "..."` or `pytest <test_file>::<test_case> -v`).
   - If no test exists for the area, note it in the report.
4. **Report** — Specialist returns:
   - Completion status (done / blocked)
   - Files modified
   - Test results (pass/fail counts)
   - Any blockers or deviations from spec

---

## 4. Integration

**Integration Owner:** Orchestrator only.

**Merge Strategy:**
- **Approach:** Fast-forward or merge commit into `Rishi-Ghost` branch.
- **Why:** All changes are additive file creation and a small `__init__.py` edit. No rebase or cherry-pick complexity needed.

**Conflict Zones & Resolution:**
- **Zone 1:** `src/storage/sqlite_store.py` — single file written by one or more specialists.
  - **Resolution:** Orchestrator reviews the full file before merge. Ensure no duplicate method definitions, consistent imports, and that `_connect` is defined before methods that call it.
- **Zone 2:** `src/storage/__init__.py` — currently empty.
  - **Resolution:** Add `from .sqlite_store import SQLiteStore`. No conflicts expected.
- **Zone 3:** `docs/phase1.2/schema.sql` vs `_inline_schema()` string in Python.
  - **Resolution:** Orchestrator verifies `_inline_schema` DDL matches `schema.sql` exactly (same tables, columns, indexes).

**Ordering:**
1. Merge A1 (`schema.sql`) first.
2. Merge B1-B6 (`sqlite_store.py`) second.
3. Merge C1 (`tests/test_sqlite_store.py`) third.
4. Merge D1 (`__init__.py` update) last.

**Preservation Check:**
- Before claiming integration complete, orchestrator confirms:
  - All 4 tables exist in both `schema.sql` and `_inline_schema`
  - All 8 indexes defined
  - No stray print/debug statements
  - `data/finance.db` is git-ignored (confirmed from 1.2.1 `.gitignore`)

---

## 5. Verification & Cleanup

### 5.1 Full Test Suite
- **Run:** `pytest tests/test_sqlite_store.py -v`
- **Report:**
  - Total tests count
  - Pass / fail breakdown
  - Any stderr output

### 5.2 Smoke Tests
- **Command 1:** `python -c "from src.storage.sqlite_store import SQLiteStore; s = SQLiteStore(); print('OK', s.db_path)"`
- **Command 2:** `python -c "from src.storage import SQLiteStore; print('Import OK')"`
- **Command 3:** `sqlite3 data/finance.db ".schema"` (after initialization) — confirm 4 tables + 8 indexes

### 5.3 Spec Compliance Checklist
- `data/finance.db` creates successfully
- All 4 tables exist: `fundamentals`, `filings`, `cache_meta`, `ingestion_log`
- All 8 indexes created
- `upsert_fundamental` inserts a new row
- `upsert_fundamental` updates existing row without error
- `get_fundamental` returns latest value for ticker+metric
- `register_filing` returns `True` for new, `False` for duplicate
- `get_cache_status` returns `None` for unknown ticker, `dict` for known
- `log_ingestion_start` returns UUID, `log_ingestion_complete` updates row
- `search_facts` handles dynamic filters safely
- WAL mode enabled (`PRAGMA journal_mode == WAL`)

### 5.4 Cleanup
- Remove any temporary `data/finance.db` files created during testing (if tests used default path instead of `tmp_path`).
- Delete any feature branches after merge.
- Ensure no untracked `.pyc` or `__pycache__` files are committed.
- Do not claim completion without passing test results on the integrated `Rishi-Ghost` branch.

---

## Red Flags — Don't Do This
- Don't rewrite the approved plan — execute it
- Don't skip verification before claiming done
- Don't delegate integration — orchestrator only
- Don't implement outside the approved scope — flag scope creep instead
- Don't use `data/finance.db` for unit tests — use `tmp_path` or `:memory:`

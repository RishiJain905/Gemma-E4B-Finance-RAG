# Phase 1.3.2 Implementation Plan — Fundamentals Ingestion

**Phase:** 1.3.2
**Spec:** `docs/phase1.3/1.3.2-fundamentals-ingestion.md`
**Date:** 2026-05-30
**Target Branch:** `Rishi-Ghost`
**Orchestrator:** Context manager and integration owner
**Delegated To:** 1 backend specialist

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
- Primary spec: `docs/phase1.3/1.3.2-fundamentals-ingestion.md`
- Config references: `configs/watchlist.yaml`, `configs/storage.yaml`
- Prior phase summary: `docs/p1.3.1-done.md`
- Current repo state:
  - `src/ingestion/yfinance_ingestor.py` exists with skeleton class and `NotImplementedError` stubs for `ingest_fundamentals()`, `_ingest_ticker_fundamentals()`, `ingest_news()`, `ingest_macro()`
  - `src/storage/sqlite_store.py` exists with full implementation: `upsert_fundamental()`, `get_fundamental()`, `get_fundamentals_batch()`, `get_cache_status()`, `mark_cache_fresh()`, `mark_cache_stale()`, `get_stale_cache_entries()`
  - `src/storage/store.py` is a **shim** (`class Store: pass`) with no methods
  - `configs/watchlist.yaml` exists with 20 tickers and schedule block (`fundamentals: 24`, `news: 6`, `macro: 24`)
  - `requirements.txt` has `yfinance>=0.2.50`
  - `tests/test_yfinance_ingestor.py` exists with 17 tests from 1.3.1
  - `pytest.ini` has `pythonpath = .`

### 2.2 Overlap & Dependencies
- **Overlap with 1.3.1:** `yfinance_ingestor.py` is the only file being modified. The skeleton already has `_fetch_ticker()`, `all_tickers`, `core_tickers`, and `_load_watchlist()` — these are reused unchanged.
- **Shared file:** `src/storage/store.py` — currently a shim. The spec expects `self.store.save_fundamental(...)` and `self.store.mark_cache_fresh(...)`, but `Store` has no methods. `SQLiteStore` has the equivalent methods (`upsert_fundamental`, `mark_cache_fresh`).
- **Critical dependency:** The Store shim must be upgraded to a minimal working wrapper around `SQLiteStore` **before** or **concurrently with** the ingestor implementation, otherwise `YFinanceIngestor` will crash on attribute errors at runtime.
- **Future dependency:** `ingest_news()` will be implemented in 1.3.3; `ingest_macro()` in 1.3.4. Those stubs remain untouched.
- **No overlap with 1.2.x schema:** `sqlite_store.py` schema is already initialized with `fundamentals` and `cache_meta` tables.

### 2.3 Verification Strategy
- **Stage 1 — Spec compliance:** Confirm every method from the spec exists with the exact signatures and behavior (cache awareness, 18 metrics, normalization, period labeling).
- **Stage 2 — Code quality:** Confirm test coverage for the new logic, no stray debug statements, proper error handling, and clean separation between Store shim and ingestor.
- **Functional verification:**
  - Single ticker test (`NVDA`) writes ~18 rows to `fundamentals` and marks cache fresh.
  - Full `ingest_fundamentals()` iterates `core_tickers` correctly.
  - Re-run skips all tickers due to cache freshness.
  - TTL from `watchlist.yaml` schedule is respected.

---

## 3. Development

### Task Decomposition

#### Workstream A — Store Shim Upgrade
**Owner:** backend specialist

**Task A1: Upgrade `src/storage/store.py` to minimal working Store**
- **File path:** `src/storage/store.py`
- **Deliverable:** `Store` class that wraps `SQLiteStore` and exposes the methods needed by 1.3.2:
  - `save_fundamental(ticker, metric, value, unit, period, period_type, source_type)` → delegates to `self._sqlite.upsert_fundamental(...)`
  - `mark_cache_fresh(ticker, source, ttl_hours)` → delegates to `self._sqlite.mark_cache_fresh(...)`
  - `mark_cache_stale(ticker, source, error)` → delegates to `self._sqlite.mark_cache_stale(...)`
  - `get_fundamentals_batch(ticker, metrics=None)` → delegates to `self._sqlite.get_fundamentals_batch(...)`
  - `get_cache_status(ticker, source)` → delegates to `self._sqlite.get_cache_status(...)`
- **Constructor:** `__init__(self, sqlite_store=None)` — instantiate `SQLiteStore()` if none provided, or accept an injected instance for testing.
- **Cross-reference:** Spec Step 1 (implicit dependency), `src/storage/sqlite_store.py` method signatures
- **Acceptance criteria:**
  - `Store` class loads without `ImportError` or `SyntaxError`
  - `Store().save_fundamental("TEST", "pe_ratio_ttm", 25.0, "ratio", "2026-Q2", "ttm", "yfinance")` does not raise `AttributeError`
  - `Store().mark_cache_fresh("TEST", "yfinance_fundamentals", 24)` writes a row to `cache_meta`
  - `Store().get_cache_status("TEST", "yfinance_fundamentals")` returns a dict with `status` key
  - All delegated methods use the exact parameter names expected by the spec (`save_fundamental`, not `upsert_fundamental`, at the public API level)

#### Workstream B — YFinanceIngestor Fundamentals Implementation
**Owner:** backend specialist
**Target file:** `src/ingestion/yfinance_ingestor.py`

**Task B1: Replace `ingest_fundamentals()` stub with full implementation**
- **Deliverable:** Replace `raise NotImplementedError("Implement in 1.3.2")` with the method body from the spec.
- **Behavior:**
  - Iterate `self.core_tickers`
  - Call `self._fundamentals_fresh(ticker)` to check cache
  - Skip and count fresh tickers
  - Fetch via `_fetch_ticker()`, call `_ingest_ticker_fundamentals()`
  - Log final summary: `"Fundamentals ingestion complete: %d ingested, %d skipped (fresh)"`
- **Acceptance criteria:**
  - Method signature matches spec exactly: `def ingest_fundamentals(self):`
  - Does not raise `NotImplementedError`
  - Uses `logger.info` / `logger.debug` at appropriate levels

**Task B2: Replace `_ingest_ticker_fundamentals()` stub with full implementation**
- **Deliverable:** Replace `raise NotImplementedError("Implement in 1.3.2")` with the method body from the spec.
- **Behavior:**
  - Reads `t.info` from the passed yfinance Ticker object
  - Iterates `FUNDAMENTAL_METRICS` (18 metric tuples)
  - Skips `None` values gracefully with debug logging
  - Calls `self._normalize_value(raw_value)` before storage
  - Calls `self.store.save_fundamental(...)` with all spec parameters
  - Calls `self.store.mark_cache_fresh(...)` after successful save loop
- **Acceptance criteria:**
  - `FUNDAMENTAL_METRICS` class constant is defined exactly as in the spec
  - Saves 18 metrics when all are present; fewer when Yahoo returns sparse data
  - `period_label` comes from `_current_period_label()`
  - Cache is marked fresh exactly once per ticker, not per metric

**Task B3: Implement cache freshness check `_fundamentals_fresh()`**
- **Deliverable:** New method as specified.
- **Behavior:**
  - Reads `ttl_hours` from `self.watchlist["schedule"]["fundamentals"]` (default 24)
  - Calls `self.store.get_cache_status(ticker, "yfinance_fundamentals")`
  - Returns `True` only if status is `"fresh"` **and** `age_hours < ttl_hours`
  - Parses `accessed_at` / `last_updated` safely; returns `False` on any parse failure
- **Acceptance criteria:**
  - Returns `bool` always
  - No unhandled `TypeError` or `AttributeError` when cache row is missing
  - Uses `datetime.now(timezone.utc)` with fallback to `datetime.utcnow()` if needed

**Task B4: Implement value normalization `_normalize_value()`**
- **Deliverable:** New method as specified.
- **Behavior:**
  - Returns `None` for `None` input
  - Casts to `float` via `float(raw_value)`
  - Catches `TypeError`, `ValueError` and returns `None`
- **Acceptance criteria:**
  - `self._normalize_value("25.5") == 25.5`
  - `self._normalize_value(None) is None`
  - `self._normalize_value("N/A") is None`

**Task B5: Implement period labeling `_current_period_label()`**
- **Deliverable:** New method as specified.
- **Behavior:**
  - Returns `"YYYY-QN"` based on current calendar quarter
- **Acceptance criteria:**
  - Returns string in exact format `"2026-Q2"` for May 2026
  - Uses `datetime.now()` (no timezone required for quarter calculation)

#### Workstream C — Tests & Verification
**Owner:** backend specialist

**Task C1: Expand `tests/test_yfinance_ingestor.py` for fundamentals logic**
- **File path:** `tests/test_yfinance_ingestor.py`
- **Deliverable:** New pytest cases covering:
  - `_normalize_value` — float conversion, None passthrough, invalid string handling
  - `_current_period_label` — format correctness (mock `datetime.now` if needed)
  - `_fundamentals_fresh` — fresh cache returns True, stale cache returns False, missing cache returns False, TTL boundary cases
  - `_ingest_ticker_fundamentals` — mock yfinance Ticker with `info` dict, assert `save_fundamental` called for each metric present, assert `mark_cache_fresh` called once, assert `None` values are skipped
  - `ingest_fundamentals` — mock `_fetch_ticker` and `_ingest_ticker_fundamentals`, assert fresh tickers are skipped, assert summary log message format
- **Cross-reference:** Spec Step 2, Step 3, Step 4
- **Acceptance criteria:**
  - `pytest tests/test_yfinance_ingestor.py -v` passes with 100% success
  - Tests mock `yfinance` network calls; no live network required for CI
  - Tests mock `Store` / `SQLiteStore` methods where appropriate to avoid DB side effects, OR use `tmp_path` SQLite DB
  - New tests are grouped logically with docstrings

**Task C2: Single ticker smoke test**
- **Deliverable:** Manual or scripted verification run.
- **Command:**
  ```python
  from src.ingestion.yfinance_ingestor import YFinanceIngestor
  ingestor = YFinanceIngestor()
  t = ingestor._fetch_ticker('NVDA')
  if t:
      ingestor._ingest_ticker_fundamentals('NVDA', t)
      print('NVDA fundamentals ingested')
  from src.storage.store import Store
  store = Store()
  facts = store.get_fundamentals_batch('NVDA')
  for metric, value in facts.items():
      print(f'  {metric}: {value}')
  ```
- **Acceptance criteria:**
  - Prints `NVDA fundamentals ingested`
  - Prints ~18 metric lines with real numeric values (not `None`)
  - No `AttributeError` on `store.get_fundamentals_batch`

**Task C3: Full core tickers smoke test**
- **Deliverable:** Run `ingest_fundamentals()` across all core tickers.
- **Command:**
  ```python
  from src.ingestion.yfinance_ingestor import YFinanceIngestor
  ingestor = YFinanceIngestor()
  ingestor.ingest_fundamentals()
  ```
- **Acceptance criteria:**
  - Completes without exceptions
  - Log line shows `ingested` count equal to number of core tickers (6)
  - `skipped (fresh)` count is 0 on first run

**Task C4: Cache awareness smoke test**
- **Deliverable:** Re-run immediately after C3.
- **Command:** Same as C3.
- **Acceptance criteria:**
  - Log line shows `skipped (fresh)` count equal to number of core tickers (6)
  - `ingested` count is 0
  - Confirms cache TTL is respected on immediate re-run

---

### Delegation Rules
- **Task A1, B1–B5, C1–C4:** Delegate to **backend specialist**
- **Integration & merge:** Orchestrator only

---

### Parallel-First Execution Ordering

**Wave 1 (parallel, no dependencies):**
- A1: Upgrade `Store` shim in `src/storage/store.py`
- B4: Implement `_normalize_value()`
- B5: Implement `_current_period_label()`

**Wave 2 (parallel, depends on Wave 1):**
- B1: Replace `ingest_fundamentals()` stub
- B2: Replace `_ingest_ticker_fundamentals()` stub
- B3: Implement `_fundamentals_fresh()` (needs `mark_cache_fresh` / `get_cache_status` from A1)
- C1: Draft and complete expanded pytest suite (needs B1–B5 implementations)

**Wave 3 (depends on Wave 2):**
- C2: Single ticker smoke test (live network)
- C3: Full core tickers smoke test (live network)
- C4: Cache awareness re-run smoke test (live network + SQLite state)

**Why this ordering:**
- A1 is foundational — the ingestor cannot call `self.store.save_fundamental()` until `Store` exposes it.
- B4 and B5 are pure helper methods with no external dependencies; they can be written in parallel with A1.
- B1–B3 depend on A1 (Store methods available) and on each other (B1 calls B3 and B2; B2 calls B4 and B5).
- C1 tests need the full method graph from B1–B5 and A1.
- C2–C4 are live-network smoke tests that validate the integrated stack; they must run last to avoid polluting the DB before unit tests finish.

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
- **Why:** Changes are additive and confined to two files (`store.py` and `yfinance_ingestor.py`) plus test expansion. No rebase complexity needed.

**Conflict Zones & Resolution:**
- **Zone 1:** `src/storage/store.py` — previously a 10-line shim. Will become a real delegation class.
  - **Resolution:** Ensure no duplicate `Store` classes. Preserve docstring comment about 1.2.4 future replacement.
- **Zone 2:** `src/ingestion/yfinance_ingestor.py` — replacing `NotImplementedError` stubs with real bodies.
  - **Resolution:** Ensure `ingest_news()` and `ingest_macro()` stubs remain untouched (still raise `NotImplementedError`). Ensure `ingest_all()` continues to call all three sub-methods.
- **Zone 3:** `tests/test_yfinance_ingestor.py` — appending new tests to existing 17-test file.
  - **Resolution:** Append after existing tests; do not rewrite 1.3.1 tests. Verify all 17 original tests still pass after additions.
- **Zone 4:** `data/finance.db` — smoke tests will write real data.
  - **Resolution:** DB is git-ignored. No merge conflicts. Orchestrator verifies no stray `.db` files are staged.

**Ordering:**
1. Merge A1 (`store.py` upgrade) first.
2. Merge B1–B5 (`yfinance_ingestor.py` fundamentals methods) second.
3. Merge C1 (expanded pytest suite) third.
4. Run C2–C4 smoke tests on integrated `Rishi-Ghost` branch last.

**Preservation Check:**
- Before claiming integration complete, orchestrator confirms:
  - `src/storage/store.py` exposes `save_fundamental`, `mark_cache_fresh`, `mark_cache_stale`, `get_fundamentals_batch`, `get_cache_status`
  - `yfinance_ingestor.py` still has `ingest_news()` and `ingest_macro()` raising `NotImplementedError`
  - `FUNDAMENTAL_METRICS` contains exactly the 18 tuples from the spec, in the same order
  - No stray `print()` or debug statements in ingestor or store
  - `data/` and `*.db` remain git-ignored and unstaged
  - `requirements.txt` still has `yfinance>=0.2.50`

---

## 5. Verification & Cleanup

### 5.1 Full Test Suite
- **Run:** `pytest tests/test_yfinance_ingestor.py -v`
- **Report:**
  - Total tests count (original 17 + new fundamentals tests)
  - Pass / fail breakdown
  - Any stderr output or deprecation warnings from `yfinance`

### 5.2 Smoke Tests
- **Command 1 — Single ticker:**
  ```bash
  python -c "
  from src.ingestion.yfinance_ingestor import YFinanceIngestor
  ingestor = YFinanceIngestor()
  t = ingestor._fetch_ticker('NVDA')
  if t:
      ingestor._ingest_ticker_fundamentals('NVDA', t)
      print('NVDA fundamentals ingested')
  from src.storage.store import Store
  store = Store()
  facts = store.get_fundamentals_batch('NVDA')
  for metric, value in facts.items():
      print(f'  {metric}: {value}')
  "
  ```
- **Command 2 — Full core tickers:**
  ```bash
  python -c "
  from src.ingestion.yfinance_ingestor import YFinanceIngestor
  ingestor = YFinanceIngestor()
  ingestor.ingest_fundamentals()
  "
  ```
- **Command 3 — Cache awareness re-run:**
  ```bash
  python -c "
  from src.ingestion.yfinance_ingestor import YFinanceIngestor
  ingestor = YFinanceIngestor()
  ingestor.ingest_fundamentals()
  "
  ```
- **Command 4 — Store delegation quick-check:**
  ```bash
  python -c "
  from src.storage.store import Store
  s = Store()
  s.save_fundamental('TEST', 'pe_ratio_ttm', 25.0, 'ratio', '2026-Q2', 'ttm', 'yfinance')
  s.mark_cache_fresh('TEST', 'yfinance_fundamentals', 24)
  print('Store delegation OK')
  print(s.get_cache_status('TEST', 'yfinance_fundamentals'))
  "
  ```

### 5.3 Spec Compliance Checklist
- [ ] `yfinance` installed and Ticker objects return real data
- [ ] `Store` shim upgraded — `save_fundamental` and `mark_cache_fresh` do not raise `AttributeError`
- [ ] `_ingest_ticker_fundamentals()` saves ~18 metrics to SQLite
- [ ] `store.get_fundamentals_batch()` returns real values (not None)
- [ ] Cache awareness works — re-running skips fresh tickers
- [ ] Cache freshness respects the configured TTL from `watchlist.yaml`
- [ ] Single ticker test succeeds before running all core tickers
- [ ] `ingest_fundamentals()` iterates `core_tickers` and logs summary counts
- [ ] `_normalize_value()` handles `None`, numeric strings, and invalid values safely
- [ ] `_current_period_label()` returns `"YYYY-QN"` format
- [ ] All 17 original 1.3.1 tests still pass
- [ ] New fundamentals tests pass with mocked network and store dependencies

### 5.4 Cleanup
- Remove any temporary `.db` files created during testing (they are git-ignored, but verify they are not staged).
- Delete any feature branches after merge.
- Ensure no untracked `.pyc` or `__pycache__` files are committed.
- Do not claim completion without passing test results on the integrated `Rishi-Ghost` branch.

---

## Red Flags — Don't Do This
- Don't rewrite the approved plan — execute it
- Don't skip verification before claiming done
- Don't delegate integration — orchestrator only
- Don't implement outside the approved scope — flag scope creep instead
- Don't change `ingest_news()` or `ingest_macro()` stubs — those are 1.3.3 and 1.3.4
- Don't let the Store shim block fundamentals implementation — upgrade the shim (Option A) rather than bypassing it
- Don't hardcode metric names or TTL values inside the ingestor — use `FUNDAMENTAL_METRICS` and `watchlist.yaml` as the source of truth
- Don't skip cache-awareness on the first run — the `_fundamentals_fresh` check must run for every ticker every time
- Don't use live network calls in unit tests — mock `yfinance.Ticker` or `_fetch_ticker` for C1

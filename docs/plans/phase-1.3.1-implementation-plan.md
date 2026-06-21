# Phase 1.3.1 Implementation Plan — Yahoo Finance Setup & Module Skeleton

**Phase:** 1.3.1
**Spec:** `docs/phase1.3/1.3.1-yfinance-setup-and-module-skeleton.md`
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
- Primary spec: `docs/phase1.3/1.3.1-yfinance-setup-and-module-skeleton.md`
- Config references: `configs/storage.yaml`, `requirements.txt`
- Prior phase summaries: `docs/p1.2.2-done.md`, `docs/p1.2.3-section1-done.md`
- Current repo state:
  - `src/ingestion/__init__.py` exists but is empty (0 bytes)
  - `src/storage/__init__.py` exports `SQLiteStore` and `ChromaStore`
  - **There is no `src/storage/store.py` yet** (the unified Store class from Phase 1.2.4 is NOT implemented)
  - `requirements.txt` has `yfinance>=0.2.0` (needs bump to `>=0.2.50` per spec)
  - `configs/` has `model.yaml` and `storage.yaml`; no `watchlist.yaml` yet
  - `.gitignore` ignores `data/`, `*.db`, `.venv/`, `__pycache__/`

### 2.2 Overlap & Dependencies
- **Overlap with 1.2.x:** `src/ingestion/` directory exists with an empty `__init__.py`. No schema or ingestion code exists yet.
- **Shared file:** `configs/watchlist.yaml` — consumed by the ingestor at runtime. Must be created before module skeleton testing.
- **Config dependency:** `configs/storage.yaml` already exists. The ingestor does not read `storage.yaml` directly, but the `Store` abstraction (when it arrives in 1.2.4) will.
- **Ordering dependency:** The `YFinanceIngestor` constructor references `store = store or Store()`, where `Store` is imported from `src.storage.store`. That file **does not exist yet**. The spec provides a skeleton import (`from src.storage.store import Store`). The specialist must ensure the skeleton loads even if `Store` is a placeholder or the import is temporarily guarded.
- **Future dependency:** `YFinanceIngestor.ingest_fundamentals()` will be implemented in 1.3.2; `.ingest_news()` in 1.3.3; `.ingest_macro()` in 1.3.4.

### 2.3 Verification Strategy
- **Syntax verification:** `python -m py_compile src/ingestion/yfinance_ingestor.py` from project root.
- **Import smoke test:** `python -c "from src.ingestion.yfinance_ingestor import YFinanceIngestor; i = YFinanceIngestor(); print(len(i.all_tickers))"`
- **Config verification:** `configs/watchlist.yaml` loads via `yaml.safe_load` without errors.
- **Library check:** `import yfinance` succeeds after `pip install`.
- **Stub behavior check:** `ingest_all()` and the per-category stubs raise `NotImplementedError` as designed.

---

## 3. Development

### Task Decomposition

#### Workstream A — Dependency Installation & Config Setup
**Owner:** backend specialist

**Task A1: Bump `yfinance` in `requirements.txt`**
- **File path:** `requirements.txt`
- **Deliverable:** Update `yfinance>=0.2.0` → `yfinance>=0.2.50`
- **Cross-reference:** Spec Step 1
- **Acceptance criteria:**
  - Line reads `yfinance>=0.2.50`
  - No duplicate `yfinance` lines remain
  - `pip install -r requirements.txt` resolves `yfinance` successfully

**Task A2: Create `configs/watchlist.yaml`**
- **File path:** `configs/watchlist.yaml`
- **Deliverable:** YAML watchlist with:
  - `core` tickers (6 items)
  - `extended` tickers (10 items)
  - `macro_tickers` (4 items)
  - `schedule` block (`fundamentals: 24`, `news: 6`, `macro: 24`)
- **Cross-reference:** Spec Step 2
- **Acceptance criteria:**
  - File is valid YAML (parses with `yaml.safe_load`)
  - All 20 tickers present across `core`, `extended`, and `macro_tickers`
  - `schedule` keys exactly match `fundamentals`, `news`, `macro`

#### Workstream B — YFinance Ingestor Module Skeleton
**Owner:** backend specialist
**Target file:** `src/ingestion/yfinance_ingestor.py`

**Task B1: Create `src/ingestion/yfinance_ingestor.py`**
- **Deliverable:** Complete `YFinanceIngestor` class skeleton matching the spec:
  - Module docstring with usage examples
  - `DEFAULT_WATCHLIST_PATH` class attribute
  - `__init__(self, store=None, watchlist_path=None)` with fallback `Store()` placeholder
  - `_load_watchlist(self) -> dict` with safe fallback defaults
  - `all_tickers` and `core_tickers` properties
  - `_fetch_ticker(self, ticker)` with validation via `info.get("regularMarketPrice")`
  - `ingest_all()`, `ingest_fundamentals()`, `ingest_news()`, `ingest_macro()` — all stubs raising `NotImplementedError` with "Implement in 1.3.x" messages
  - `ingest_ticker(self, ticker)` — orchestrator stub
  - `_ingest_ticker_fundamentals(self, ticker, t)` — stub for 1.3.2
  - `_ingest_ticker_news(self, ticker, t)` — stub for 1.3.3
- **Cross-reference:** Spec Step 3
- **Acceptance criteria:**
  - Class loads without `ImportError` or `SyntaxError`
  - `_load_watchlist` falls back to hardcoded defaults when `watchlist.yaml` is missing
  - `all_tickers` returns a flat list of 20 tickers when config is present
  - `core_tickers` returns exactly the 6 core tickers
  - `_fetch_ticker("NVDA")` returns a valid `yfinance.Ticker` object (network permitting)
  - `ingest_all()` raises `NotImplementedError` with message "Implement in 1.3.2" (or matching stub label)
  - `ingest_fundamentals()` raises `NotImplementedError("Implement in 1.3.2")`
  - `ingest_news()` raises `NotImplementedError("Implement in 1.3.3")`
  - `ingest_macro()` raises `NotImplementedError("Implement in 1.3.4")`

**Task B2: Update `src/ingestion/__init__.py`**
- **File path:** `src/ingestion/__init__.py`
- **Deliverable:** `from .yfinance_ingestor import YFinanceIngestor`
- **Cross-reference:** Package convention from 1.2.x
- **Acceptance criteria:**
  - `from src.ingestion import YFinanceIngestor` works from project root
  - `__all__` updated if present

#### Workstream C — Tests & Smoke Verification
**Owner:** backend specialist

**Task C1: Create `tests/test_yfinance_ingestor.py`**
- **File path:** `tests/test_yfinance_ingestor.py`
- **Deliverable:** Pytest suite covering:
  - Import and class initialization
  - `_load_watchlist` with real config file
  - `_load_watchlist` fallback when file is missing
  - `all_tickers` / `core_tickers` property correctness
  - `_fetch_ticker` success path (mock or live, depending on CI policy)
  - `_fetch_ticker` failure path (invalid ticker or network error)
  - Stub methods raise `NotImplementedError` with expected messages
- **Cross-reference:** Spec Step 4
- **Acceptance criteria:**
  - `pytest tests/test_yfinance_ingestor.py -v` passes with 100% success
  - Tests use `tmp_path` for temporary watchlist files, never writing to `configs/watchlist.yaml`
  - If live network calls are used, they are marked with `@pytest.mark.network` and documented

**Task C2: Smoke test from project root**
- **Deliverable:** One-liner confirming import and initialization from project root
- **Command:**
  ```python
  python -c "
  from src.ingestion.yfinance_ingestor import YFinanceIngestor
  ingestor = YFinanceIngestor()
  print(f'Watchlist loaded: {len(ingestor.all_tickers)} tickers')
  print(f'Core tickers: {ingestor.core_tickers}')
  print(f'All tickers: {ingestor.all_tickers}')
  "
  ```
- **Acceptance criteria:**
  - Prints `Watchlist loaded: 20 tickers`
  - Prints `Core tickers: ['NVDA', 'AMD', 'AAPL', 'MSFT', 'META', 'CRWD']`
  - Prints full 20-ticker list without exceptions

#### Workstream D — Package Integration & Documentation
**Owner:** orchestrator

**Task D1: Verify `requirements.txt` consistency**
- **Deliverable:** Confirm `yfinance>=0.2.50` is present and no version conflicts exist
- **Acceptance criteria:** `pip install -r requirements.txt` completes without `yfinance` resolution errors

**Task D2: Write `docs/p1.3.1-done.md`**
- **Deliverable:** Phase completion summary following `docs/p1.2.2-done.md` style
- **Acceptance criteria:** Lists all files created/modified, verification checklist, blockers/follow-ups

---

### Delegation Rules
- **Task A1, A2, B1, B2, C1, C2:** Delegate to **backend specialist**
- **Task D1, D2:** Orchestrator performs directly (integration-owned work)

---

### Parallel-First Execution Ordering

**Wave 1 (parallel, no dependencies):**
- A1: Bump `yfinance` in `requirements.txt`
- A2: Create `configs/watchlist.yaml`
- B1: Create `YFinanceIngestor` class skeleton (`yfinance_ingestor.py`)

**Wave 2 (parallel, depends on Wave 1):**
- B2: Update `src/ingestion/__init__.py`
- C1: Draft and complete `tests/test_yfinance_ingestor.py` (needs B1 skeleton and A2 config)

**Wave 3 (depends on Wave 2):**
- C2: Run full smoke test + pytest suite
- D1: Verify `requirements.txt` consistency
- D2: Write completion summary

**Why this ordering:**
- A1, A2, and B1 are fully independent. A2 and B1 have a runtime relationship (ingestor loads watchlist), but B1's `_load_watchlist` includes a safe fallback, so either can proceed first.
- C1 needs the class skeleton (B1) and the YAML config (A2) to assert on real data.
- B2 (`__init__.py`) is trivial but must happen after `yfinance_ingestor.py` exists.
- D1 and D2 are integration and documentation tasks that close the phase.

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
- **Why:** All changes are additive (new files and one-line bumps). No rebase or cherry-pick complexity needed.

**Conflict Zones & Resolution:**
- **Zone 1:** `requirements.txt` — single-line version bump.
  - **Resolution:** Ensure no duplicate `yfinance` lines. Prefer `>=0.2.50`.
- **Zone 2:** `configs/watchlist.yaml` — new file.
  - **Resolution:** None expected. Verify YAML validity after merge.
- **Zone 3:** `src/ingestion/yfinance_ingestor.py` — single new file.
  - **Resolution:** None expected. Verify `Store` import is handled gracefully (placeholder, guard, or deferred to 1.2.4/1.3.2).
- **Zone 4:** `src/ingestion/__init__.py` — currently empty.
  - **Resolution:** Add `from .yfinance_ingestor import YFinanceIngestor`. No conflicts expected.

**Ordering:**
1. Merge A1 (`requirements.txt` bump) and A2 (`configs/watchlist.yaml`) first.
2. Merge B1 (`yfinance_ingestor.py`) second.
3. Merge B2 (`__init__.py` update) third.
4. Merge C1 (`tests/test_yfinance_ingestor.py`) fourth.
5. Merge D1 and D2 (integration checks + completion summary) last.

**Preservation Check:**
- Before claiming integration complete, orchestrator confirms:
  - `yfinance>=0.2.50` is the only `yfinance` entry in `requirements.txt`
  - `configs/watchlist.yaml` contains exactly 20 tickers across `core` + `extended` + `macro_tickers`
  - `src/ingestion/yfinance_ingestor.py` has no stray `print()` or debug statements
  - All stub methods raise `NotImplementedError` with correct phase references
  - `data/` and `*.db` remain git-ignored

---

## 5. Verification & Cleanup

### 5.1 Full Test Suite
- **Run:** `pytest tests/test_yfinance_ingestor.py -v`
- **Report:**
  - Total tests count
  - Pass / fail breakdown
  - Any stderr output

### 5.2 Smoke Tests
- **Command 1:**
  ```bash
  python -c "
  from src.ingestion.yfinance_ingestor import YFinanceIngestor
  ingestor = YFinanceIngestor()
  print(f'Watchlist loaded: {len(ingestor.all_tickers)} tickers')
  print(f'Core tickers: {ingestor.core_tickers}')
  print(f'All tickers: {ingestor.all_tickers}')
  "
  ```
- **Command 2:** `python -c "from src.ingestion import YFinanceIngestor; print('Import OK')"`
- **Command 3:** `python -c "import yfinance; print(yfinance.__version__)"` (confirms install)

### 5.3 Spec Compliance Checklist
- [ ] `pip install yfinance` succeeds (or `pip install -r requirements.txt`)
- [ ] `requirements.txt` updated with `yfinance>=0.2.50`
- [ ] `configs/watchlist.yaml` created with core, extended, and macro tickers
- [ ] `YFinanceIngestor` class loads without errors
- [ ] Watchlist loads the correct number of tickers (20)
- [ ] `ingest_all()` raises `NotImplementedError` (stubs are placeholders)
- [ ] `_fetch_ticker()` returns data for a known ticker like NVDA (network permitting)
- [ ] `src/ingestion/__init__.py` exports `YFinanceIngestor`
- [ ] `pytest tests/test_yfinance_ingestor.py -v` passes

### 5.4 Cleanup
- Remove any temporary `.yaml` or `.db` files created during testing.
- Delete any feature branches after merge.
- Ensure no untracked `.pyc` or `__pycache__` files are committed.
- Do not claim completion without passing test results on the integrated `Rishi-Ghost` branch.

---

## Red Flags — Don't Do This
- Don't rewrite the approved plan — execute it
- Don't skip verification before claiming done
- Don't delegate integration — orchestrator only
- Don't implement outside the approved scope — flag scope creep instead
- Don't implement the actual `ingest_fundamentals`, `ingest_news`, or `ingest_macro` logic in this phase — those are 1.3.2–1.3.4
- Don't hardcode the 20 tickers inside the Python class — the spec requires YAML config as the source of truth
- Don't let the missing `src/storage/store.py` block skeleton creation — handle the import gracefully (placeholder, guard, or temporary shim) and flag it for 1.2.4/1.3.2 integration

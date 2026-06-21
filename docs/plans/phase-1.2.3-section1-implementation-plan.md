# Phase 1.2.3 Implementation Plan — Section 1: Create the ChromaDB Module

**Phase:** 1.2.3  
**Spec:** `docs/phase1.2/1.2.3-chromadb-setup-and-embedding-function.md`  
**Section Scope:** Section 1 ONLY — "Create the ChromaDB Module"  
**Date:** 2026-05-27  
**Target Branch:** `Rishi-Ghost`  
**Orchestrator:** Context manager and integration owner  
**Delegated To:** 1 backend specialist  

---

## 0. Scope Boundary

**IN SCOPE for this plan:**
- Section 1: Create the ChromaDB Module (`src/storage/chroma_store.py`)
- Export `ChromaStore` from `src/storage/__init__.py`
- Minimal unit tests mocking the embedding endpoint
- `TraceAlchemyEmbeddingFunction` class implementation
- `ChromaStore` class with all CRUD, ticker-specific, and collection management methods

**OUT OF SCOPE (planned separately):**
- Section 2: Verify the Embedding Dimension
- Section 3: Test the Full Pipeline (requires llama-server running)
- Section 4: Check the Cosine Similarity Quality

**Explicit note:** Sections 2–4 require a live llama-server on `http://127.0.0.1:8087/v1/embeddings`. That live-server verification is NOT part of this plan. This plan covers code creation and mock-based unit tests only.

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
- Primary spec: `docs/phase1.2/1.2.3-chromadb-setup-and-embedding-function.md`, Section 1 only
- Config reference: `configs/storage.yaml`:
  - `chroma.path`: `"data/chroma"`
  - `chroma.collection_name`: `"tracealchemy_docs"`
  - `chroma.embedding_dimension`: `2048`
  - `embedding.endpoint`: `"http://127.0.0.1:8087/v1/embeddings"`
  - `embedding.model`: `"tracealchemy"`
  - `embedding.batch_size`: `10`
- Prior phase summary: `docs/p1.2.2-done.md` (SQLite schema completed)
- Current repo state: `src/storage/__init__.py` exports only `SQLiteStore`

### 2.2 Overlap & Dependencies
- **Overlap with 1.2.2:** `src/storage/__init__.py` already exports `SQLiteStore`. This plan must update it to also export `ChromaStore` without breaking the SQLiteStore import.
- **Shared file:** `src/storage/__init__.py` — the only shared file between 1.2.2 and 1.2.3.
- **Config dependency:** `configs/storage.yaml` already defines chroma and embedding values. The `ChromaStore` constructor should accept override parameters but default to values aligned with `storage.yaml`.
- **Ordering dependency:** `chromadb` package must be installed in `.venv` before code runs. This is a runtime dependency, not a code-creation dependency.
- **No live-server dependency at code-creation time:** The `TraceAlchemyEmbeddingFunction` calls `httpx.Client.post()` to `http://127.0.0.1:8087/v1/embeddings`. The llama-server does **not** need to be running to write or import the module. It only needs to run for live embedding tests (Section 3 — OUT OF SCOPE).
- **Future dependency:** `src/storage/chroma_store.py` will be imported by Phase 1.2.4 (Unified Storage Abstraction).

### 2.3 Verification Strategy (Section 1 Scope Only)
- **Syntax verification:** `python -m py_compile src/storage/chroma_store.py` passes.
- **Import verification:** `python -c "from src.storage.chroma_store import ChromaStore, TraceAlchemyEmbeddingFunction"` from project root.
- **Package export verification:** `python -c "from src.storage import ChromaStore, SQLiteStore; print('OK')"` passes.
- **Unit tests:** `pytest tests/test_chroma_store.py` passes with mocked `httpx.Client.post` — no llama-server required.
- **Method stub verification:** Instantiate `ChromaStore()` with mocked embedding, confirm `add_document`, `search`, `get_document`, `delete_document`, `count`, `search_by_ticker`, `get_ticker_documents`, `reset_collection`, and `heartbeat` all exist and are callable.

---

## 3. Development

### Task Decomposition

#### Workstream A — ChromaDB Module (`src/storage/chroma_store.py`)
**Owner:** backend specialist  
**Target file:** `src/storage/chroma_store.py`

**Task A1: `TraceAlchemyEmbeddingFunction` class**
- **Deliverable:** Embedding function implementing `chromadb.EmbeddingFunction`.
- **Cross-reference:** Spec Section 1, `TraceAlchemyEmbeddingFunction` block.
- **Acceptance criteria:**
  - `__init__(self, endpoint, model, batch_size, timeout)` stores all parameters.
  - `self._client` is an `httpx.Client(timeout=timeout)`.
  - `__call__(self, input)` accepts `Documents` (list of strings).
  - Batches input by `self.batch_size`.
  - Payload uses `"input"` key (single string if batch size 1, else list).
  - Response parsed as JSON; extracts `data[i]["embedding"]` for each item.
  - Returns `Embeddings` (list of list of floats).
  - `__del__` closes `self._client` if it exists.
  - No hard dependency on a running llama-server at import time.

**Task A2: `ChromaStore` class — initialization**
- **Deliverable:** `ChromaStore` class with constructor and persistent client setup.
- **Cross-reference:** Spec Section 1, `ChromaStore.__init__` block.
- **Acceptance criteria:**
  - `DEFAULT_PATH = Path(__file__).parent.parent.parent / "data/chroma"`.
  - `__init__(self, persist_directory, collection_name, embedding_endpoint)` accepts optional overrides.
  - Persists directory created with `mkdir(parents=True, exist_ok=True)`.
  - `self.embedding_fn = TraceAlchemyEmbeddingFunction(endpoint=embedding_endpoint)`.
  - `self.client = chromadb.PersistentClient(path=str(self.persist_directory))`.
  - `self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"})`.

**Task A3: `ChromaStore` class — CRUD operations**
- **Deliverable:** `add_document`, `add_documents_batch`, `search`, `get_document`, `delete_document`, `count`.
- **Cross-reference:** Spec Section 1, CRUD Operations block.
- **Acceptance criteria:**
  - `add_document` builds metadata dict, injecting `ticker` (uppercased), `source`, `date` if provided.
  - `add_documents_batch` accepts parallel lists of `ids`, `texts`, optional `metadatas`.
  - `search` calls `collection.query`, returns list of dicts with keys: `id`, `document`, `metadata`, `distance`.
  - `get_document` calls `collection.get`, returns single dict or `None`.
  - `delete_document` calls `collection.delete` by id.
  - `count` returns `self.collection.count()`.

**Task A4: `ChromaStore` class — ticker-specific operations**
- **Deliverable:** `search_by_ticker`, `get_ticker_documents`.
- **Cross-reference:** Spec Section 1, Ticker-Specific Operations block.
- **Acceptance criteria:**
  - `search_by_ticker` delegates to `self.search` with `filter_dict={"ticker": ticker.upper()}`.
  - `get_ticker_documents` builds `where` filter with optional `source`, calls `collection.get`, returns formatted list of dicts.

**Task A5: `ChromaStore` class — collection management**
- **Deliverable:** `reset_collection`, `heartbeat`.
- **Cross-reference:** Spec Section 1, Collection Management block.
- **Acceptance criteria:**
  - `reset_collection` deletes and recreates the collection with the same name, embedding function, and cosine metadata.
  - `heartbeat` calls `self.client.heartbeat()` inside a try/except and returns `True`/`False`.

#### Workstream B — Unit Tests
**Owner:** backend specialist (same as Workstream A, or parallel if capacity allows)

**Task B1: Create `tests/test_chroma_store.py`**
- **Deliverable:** pytest suite with mocked `httpx.Client` and `chromadb.PersistentClient`.
- **Acceptance criteria:**
  - All tests pass without requiring `llama-server`.
  - `test_embedding_function_call` mocks `httpx.Client.post` to return a fake embedding vector (list of 2048 floats), verifies `TraceAlchemyEmbeddingFunction(["test"])` returns correct shape.
  - `test_chroma_store_init` mocks `chromadb.PersistentClient` and `get_or_create_collection`, verifies `ChromaStore()` initializes without error.
  - `test_add_document` mocks `collection.add`, verifies `add_document` constructs correct `metadatas` dict.
  - `test_search` mocks `collection.query`, verifies formatted result structure (keys: `id`, `document`, `metadata`, `distance`).
  - `test_get_document` mocks `collection.get`, verifies return when found and when missing.
  - `test_delete_document` mocks `collection.delete`.
  - `test_count` mocks `collection.count`.
  - `test_search_by_ticker` verifies `filter_dict` passed correctly.
  - `test_get_ticker_documents` verifies `where` clause includes `ticker` uppercase and optional `source`.
  - `test_reset_collection` verifies delete + recreate sequence.
  - `test_heartbeat` verifies `True` on success, `False` on `Exception`.
  - Uses `tmp_path` for `persist_directory` to avoid polluting `data/chroma`.
  - Uses `monkeypatch` or `unittest.mock.patch` for `httpx.Client`.

#### Workstream C — Package Integration
**Owner:** orchestrator

**Task C1: Update `src/storage/__init__.py`**
- **Deliverable:** Export `ChromaStore` alongside `SQLiteStore`.
- **Cross-reference:** Current `src/storage/__init__.py` from 1.2.2.
- **Acceptance criteria:**
  - `from .chroma_store import ChromaStore` added.
  - `__all__ = ["SQLiteStore", "ChromaStore"]`.
  - `from src.storage import ChromaStore` works from project root.
  - `from src.storage import SQLiteStore` still works (no regression).

---

### Delegation Rules
- **Task A1–A5:** Delegate to **backend specialist**
- **Task B1:** Delegate to **backend specialist** (whoever owns the code under test)
- **Task C1:** Orchestrator performs directly (integration-owned work)

---

### Parallel-First Execution Ordering

**Wave 1 (parallel, no dependencies):**
- A1–A5: Implement `src/storage/chroma_store.py`
- B1: Draft `tests/test_chroma_store.py` (can begin in parallel because tests mock the embedding endpoint and ChromaDB client)

**Wave 2 (depends on Wave 1):**
- C1: Update `src/storage/__init__.py` (needs `chroma_store.py` to exist to avoid import errors)
- B1 finalization: Run full pytest suite after A1–A5 are complete and merged into the working branch

**Why this ordering:**
- `chroma_store.py` and its test file can be developed in parallel:
  - Tests mock `httpx.Client`, so no live server needed.
  - Tests mock `chromadb.PersistentClient`, so no persistent directory creation needed during test-writing.
- `__init__.py` update is trivial but must happen after `chroma_store.py` exists to avoid broken imports.
- Section 1 scope intentionally excludes live-server tests (Section 3), so there is no dependency on `llama-server` being up.

---

### Development Loop (per task)

1. **Delegate** — Orchestrator assigns task to backend specialist with explicit file path, acceptance criteria, and spec cross-reference.
2. **Implement** — Specialist writes code, commits to feature branch or reports patch.
3. **Verify** — Specialist runs:
   - `python -m py_compile src/storage/chroma_store.py`
   - `pytest tests/test_chroma_store.py -v` (with mocks, no server)
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
- **Zone 1:** `src/storage/chroma_store.py` — single new file. No conflicts expected.
- **Zone 2:** `src/storage/__init__.py` — currently exports only `SQLiteStore`.
  - **Resolution:** Add `from .chroma_store import ChromaStore` and append `"ChromaStore"` to `__all__`. Preserve existing `SQLiteStore` export. No conflicts expected.
- **Zone 3:** `data/chroma` directory — created at runtime by `ChromaStore.__init__`.
  - **Resolution:** Ensure `data/chroma` is git-ignored (verify `.gitignore` from 1.2.1). Do not commit runtime persistent database files.

**Ordering:**
1. Merge A1–A5 (`src/storage/chroma_store.py`) first.
2. Merge B1 (`tests/test_chroma_store.py`) second.
3. Merge C1 (`__init__.py` update) last.

**Preservation Check:**
- Before claiming integration complete, orchestrator confirms:
  - `src/storage/chroma_store.py` contains both `TraceAlchemyEmbeddingFunction` and `ChromaStore` classes.
  - All methods from spec Section 1 are present: `add_document`, `add_documents_batch`, `search`, `get_document`, `delete_document`, `count`, `search_by_ticker`, `get_ticker_documents`, `reset_collection`, `heartbeat`.
  - No stray print/debug statements.
  - `data/chroma` is git-ignored.

---

## 5. Verification & Cleanup

### 5.1 Full Test Suite
- **Run:** `pytest tests/test_chroma_store.py -v`
- **Report:**
  - Total tests count
  - Pass / fail breakdown
  - Any stderr output

### 5.2 Smoke Tests
- **Command 1:** `python -c "from src.storage.chroma_store import ChromaStore, TraceAlchemyEmbeddingFunction; print('Import OK')"`
- **Command 2:** `python -c "from src.storage import ChromaStore, SQLiteStore; print('Package OK')"`
- **Command 3:** `python -m py_compile src/storage/chroma_store.py`

### 5.3 Spec Compliance Checklist (Section 1 Only)
- `TraceAlchemyEmbeddingFunction` class exists with `__init__`, `__call__`, `__del__`
- `__call__` batches input by `batch_size` (default 10)
- `__call__` extracts embeddings from `data[i]["embedding"]`
- `ChromaStore.__init__` creates `data/chroma` directory if missing
- `ChromaStore.__init__` initializes `chromadb.PersistentClient`
- `ChromaStore.__init__` calls `get_or_create_collection` with `metadata={"hnsw:space": "cosine"}`
- `add_document` injects `ticker` (uppercase), `source`, `date` into metadata
- `add_documents_batch` accepts parallel lists
- `search` returns formatted list of dicts with `id`, `document`, `metadata`, `distance`
- `get_document` returns dict or `None`
- `delete_document` removes by id
- `count` returns integer
- `search_by_ticker` applies `{"ticker": ticker.upper()}` filter
- `get_ticker_documents` applies `where` with optional `source`
- `reset_collection` drops and recreates collection
- `heartbeat` returns `True` on responsive client, `False` on exception
- `src/storage/__init__.py` exports both `SQLiteStore` and `ChromaStore`

### 5.4 Cleanup
- Ensure `data/chroma` runtime files are not committed (verify `.gitignore`).
- Delete any feature branches after merge.
- Ensure no untracked `.pyc` or `__pycache__` files are committed.
- Do not claim completion without passing test results on the integrated `Rishi-Ghost` branch.

---

## Red Flags — Don't Do This
- Don't rewrite the approved plan — execute it
- Don't skip verification before claiming done
- Don't delegate integration — orchestrator only
- Don't implement outside the approved scope — Sections 2–4 are OUT OF SCOPE and will be planned separately
- Don't require llama-server for code creation or unit tests — mock the endpoint
- Don't commit `data/chroma` persistent files to git

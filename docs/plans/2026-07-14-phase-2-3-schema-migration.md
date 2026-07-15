# Phase 2.3 Schema Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Formalize the already-merged Phase 2.3 storage evolution with ordered migrations and safely backfill legacy identities and corpus metadata.

**Architecture:** New databases bootstrap from the canonical or inline final schema and record all migrations. Existing databases apply five ordered, transaction-scoped standard-library migrations. A metadata-only backfill uses deterministic identities and persisted per-stage cursors; every bounded committed batch advances the Store revision and never mutates existing Chroma text.

**Tech Stack:** Python standard library, SQLite, existing Chroma facade, YAML configuration, pytest.

---

### Task 1: Migration contract

**Files:**
- Create: `tests/test_storage_migrations.py`
- Create: `src/storage/migrations.py`
- Create: `src/storage/migrations/001_identities.sql`
- Create: `src/storage/migrations/002_corpus.sql`
- Create: `src/storage/migrations/003_observations_events.sql`
- Create: `src/storage/migrations/004_refresh_state.sql`
- Create: `src/storage/migrations/005_indexes.sql`
- Modify: `src/storage/sqlite_store.py`
- Modify: `docs/phase1.2/schema.sql`

1. Write tests for empty, Phase 2.2, canonical, and inline initialization plus double application and repeated reopen.
2. Run the tests and confirm missing migration behavior fails.
3. Implement discovery, checksum validation, schema history, transactional ordered application, and additive compatibility columns.
4. Route `SQLiteStore` initialization through the runner and converge all schema sources.
5. Run the scoped migration tests.

### Task 2: Resumable metadata backfill

**Files:**
- Create: `tests/test_phase2_3_backfill.py`
- Create: `src/storage/phase2_3_backfill.py`
- Create: `scripts/migrate_phase2_3.py`
- Modify: `src/storage/store.py`

1. Write tests for bounded interruption/resume, stable security/CIK linkage, Chroma metadata-only family creation, filing metadata, reconciliation errors, count preservation, and revision increments.
2. Run the tests and confirm the backfill API is absent.
3. Implement deterministic stage processing and the CLI wrapper.
4. Expose the operation through the Store facade without introducing a second persistence path.
5. Run the scoped backfill tests.

### Task 3: Rollout configuration and documentation

**Files:**
- Modify: `configs/universe.yaml`
- Modify: `configs/coverage.yaml`
- Modify: `configs/sources.yaml`
- Modify: `configs/official_sources.yaml`
- Modify: `configs/middleware.yaml`
- Modify: `src/middleware/config.py`
- Modify: `scripts/validate_setup.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/ARCHITECTURE.md`

1. Add tests proving all eight additive switches default off and status reports only `configured` booleans for named environment variables.
2. Add the smallest configuration/loading changes needed to pass them.
3. Document non-destructive rollback, ownership, and secret-reporting rules.
4. Run scoped tests and lint.

### Task 4: Verification

1. Run both new test files through `scripts/verify.ps1 -TestPath`.
2. Inspect the complete diff for unintended data paths, dependencies, or destructive operations.
3. Run `powershell -ExecutionPolicy Bypass -File scripts/verify.ps1` until its final line is `VERIFY: PASS`.


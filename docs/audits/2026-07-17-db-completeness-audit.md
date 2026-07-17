# Finance RAG data-completeness audit

Read-only audit of `data/finance.db` (SQLite) and `data/chroma` (ChromaDB). All DB access via `mode=ro` URIs. A `python -m src.scheduler all --force` bootstrap was **running throughout** this audit (scheduler_runs shows mode=`all`, status=`running`, started 2026-07-17T06:07Z), so live counts drift upward; each count below is timestamped. Code reading was done first, counts taken near the end.

## TL;DR verdicts

| # | Finding | Verdict | Severity |
|---|---------|---------|----------|
| 1 | 147 `filings` rows `unprocessed`, `file_path`/`summary_embedding_id`/`parsed_at` all NULL | WORKING AS DESIGNED (queue) + DESIGN GAP (only `weekly` mode drains it) | Low–Med |
| 2 | Structured facts (companyfacts / fundamentals / observations) | WORKING — 0% null payload | — |
| 3 | News + official_release narratives present in Chroma + FTS | WORKING | — |
| 4 | **Chroma `corpus_revision` never published → BM25 lexical fusion silently disabled on every query** | **DESIGN/CODE BUG (retrieval quality)** | **HIGH** |
| 5 | ~5.8k+ `yfinance_news` Chroma docs have no `corpus_items` ledger row | DESIGN GAP (accounting/routing, not retrieval) | Med |
| 6 | Phase 2.3 backfill (`chroma_documents` stage) marked complete at 670 rows, stale | DESIGN GAP | Med |
| 7 | 117 SEC 10-K filings: full narrative text not vector-indexed anywhere | WORKING AS DESIGNED (flag-gated) | Low–Med |
| 8 | Chroma distance function = cosine, matches retrieval assumption | WORKING (lead hypothesis refuted) | — |

---

## 1. filings table — unprocessed queue with NULL payload columns (the user's original concern)

Counts @ 02:16: `filings` = 148 rows → `parsed`=1, `unprocessed`=147 (116 deep + 31 broad). For **all 147 unprocessed**: `file_path` NULL, `summary_embedding_id` NULL, `parsed_at` NULL, `index_section_count`/`index_chunk_count` = 0. `index_error` is NULL on all rows (no failures recorded).

This is a **discovery queue**, not lost data. Lifecycle (from `src/sec/filing_processor.py`, `src/storage/sqlite_store.py`):
- Discovery (`SECDailyIndexDiscovery.bootstrap_from_submissions_bulk` / `register_sec_filings`) inserts rows with `status='unprocessed'` and NULL payload columns by construction (`_insert_filing_rows`, sqlite_store.py:865).
- Processing (`FilingProcessor.process_pending_filings` → `_process_single_filing`) downloads text from EDGAR, parses, stores, then `mark_filing_parsed` sets `file_path`, `summary_embedding_id`, `parsed_at`, `status='parsed'`. `file_path` is **only ever written at parse time** (mark_filing_parsed / mark_filing_index_pending), so NULL is correct for an unprocessed row.

**The DESIGN GAP:** the unified scheduler drains this queue **only in `weekly` mode**. `_run_mode` sets `deep_sec = (mode == "weekly")` (scheduler/__init__.py:1521); `_run_source` calls `run_full_pipeline` (discovery **+** processing) only when `deep=True`, otherwise `run_discovery` (discovery only) (scheduler/__init__.py:483-485). `bootstrap` mode's SEC partition is discovery-only too (`_run_bootstrap_partition`, :1640-1649). So `all`, `daily`, `hourly`, `bootstrap` register filings but **never process them**. The user browsed the DB after `all`/bootstrap runs → sees unprocessed rows with NULLs → **expected**.

Note: even after processing, raw 10-K text is not persisted to a `file_path` unless the filing-text rollout is on. `index_filing_text` defaults `False` (filing_processor.py:99); with it off, deep filings store structured facts + a whole-doc vector via `process_filing`; broad filings store event narratives via `_process_event_filing`. `file_path` is only set when `index_filing_text=True` (parsed artifact written to `data/sec/parsed/`).

**Fix to populate:** `python -m src.scheduler weekly --force` with llama-server up (:8087) + network. Processes 50 filings/run (`process_pending_filings(limit=50)`), so ~3 runs for the 147 backlog.

## 2. Structured facts — fully present (0% null payload)

Null-payload sweep @ 02:12 (all value columns NULL):
- `sec_companyfacts` (133,705 rows): 0 rows with both `value_numeric` and `value_text` NULL (0.00%).
- `fundamentals` (9,612): 0 (0.00%).
- `corpus_observations` (51,678): 0 (0.00%).

The actual financial numbers are all there. `corpus_observations` are structured-only and **intentionally not** in Chroma (`ChromaStore.STRUCTURED_ONLY_ITEM_TYPES` / `add_document` raises on them). No gap.

## 3. Narrative content present & retrievable

`corpus_items` (24,413 @ 02:16), all `indexing_status='indexed'`. By item_type: news 23,505; legacy_document 560; official_release 231; sec_filing 117. Cross-store check @ 02:12: **news (23,505) and official_release (231) map 1:1 to Chroma parents** (0 missing). `corpus_items` has **no body column** — narrative text lives only in Chroma `documents` (SQLite keeps title/summary/metadata + `narrative_bytes`). Lexical parity: `corpus_fts` = `lexical_chunk_meta` = 29,563 (≈ Chroma count), so both retrieval channels (vector + BM25) index the same corpus.

## 4. HIGH: Chroma corpus_revision never published → BM25 lexical fusion silently OFF

This is the real retrieval-quality bug behind lead concern #5 (but it is **not** the distance function).

**Evidence (@ 02:18):**
- SQLite `store_revision` = 242164; `lexical_index_state.indexed_revision` = 242164 (lexical index in sync on the SQLite side).
- `store.corpus_revision()` returns `indexed_revision` = 242164 (store.py:439).
- Chroma `collection_metadata` contains **only** `hnsw:space=cosine` — **no `corpus_revision` key**. `ChromaStore.corpus_revision()` reads `self.collection.metadata.get("corpus_revision")` → **None**.

**The gate** (`src/middleware/lexical_index.py` `_search_fts5`, :201-211):
```
if int(state["indexed_revision"]) != revision or chroma_revision != revision:
    self.last_status = {"reason": "revision_mismatch", ...}
    return []          # <-- lexical/BM25 contributes nothing
```
With `revision`=242164 and `chroma_revision`=None, `None != 242164` is always True → **every query returns revision_mismatch and skips BM25**. Hybrid retrieval silently degrades to vector-only. This is permanent (not transient write drift): Chroma's revision is unset, so it can never equal a positive SQLite revision.

**Root cause** (`src/storage/chroma_store.py` `mark_corpus_revision`, :592-599):
```
metadata = dict(self.collection.metadata or {})   # includes 'hnsw:space': 'cosine'
metadata["corpus_revision"] = int(revision)
self.collection.modify(metadata=metadata)          # rejected by newer chromadb
```
Newer ChromaDB rejects `collection.modify(metadata=...)` when metadata carries any reserved `hnsw:*` key, raising *"Changing the distance function of a collection once it is created is not supported"* — the exact error the lead saw. `Store._mark_chroma_revision` swallows it (store.py:334-337, comment literally: *"mismatch safely disables lexical fusion"*). So corpus_revision is never persisted to Chroma, and the swallow makes it invisible.

**Fix:** in `mark_corpus_revision`, strip reserved keys before modify, e.g. build metadata from `{k: v for k, v in (self.collection.metadata or {}).items() if not k.startswith("hnsw:")}` then set `corpus_revision`. Then re-verify a query reports `mode: "fts5"` (not `revision_mismatch`) and that BM25 results appear. Consider making `_mark_chroma_revision`'s swallow log at ERROR, since it disables a shipped retrieval feature.

## 5. Med: yfinance news bypasses the corpus_items ledger

Cross-store @ 02:14: **5,835 Chroma parents (all `news/*`) have no `corpus_items` row** (was 5,977 @ 02:12; growing live — corpus_items flat at 24,413 while Chroma climbed 29,563 → 30,594 during the run).

Root cause: `YFinanceIngestor.ingest_news` calls `store.save_document(source="yfinance_news", ...)` (yfinance_ingestor.py:622) with legacy metadata (`title`/`publisher`/`link`/`type`). `save_document` (store.py:1350) writes **Chroma + lexical FTS only** — no `upsert_narrative`, no `corpus_items` row. Sample orphan `news/VZ/...` has `source=yfinance_news` and old-schema metadata, distinct from the 23,505 tracked news (full corpus schema with `corpus_item_id`).

**Impact:** content IS fully retrievable (both vector and BM25 index it — modulo finding #4). But it is invisible to `corpus_items`-based accounting: the 2.3.7.1 capability/coverage inventory, corpus-explorer SQLite views, deterministic fast-path, and 2.3.7.2 completeness routing. A completeness-routing decision that counts `corpus_items` will undercount news coverage by thousands of docs.

**Fix options:** route yfinance news through `upsert_narrative` (NarrativeRecord), or extend the Phase 2.3 backfill to register `save_document`-origin Chroma docs and run it on a cadence (see #6).

## 6. Med: Phase 2.3 backfill is complete-but-stale

`phase2_3_backfill_progress`: `chroma_documents` stage `completed=1`, `cursor_value=670`, `rows_processed=670` (updated 2026-07-15 05:17). It registered 560 legacy_document + 117 legacy-filing rows (677 total, all `narrative_bytes=0`, `normalization_version='phase2_3_backfill_v1'`). Because it self-marked complete at 670 rows, it will **not** re-scan and will never register news added since (finding #5). It is a one-shot migration with no re-run trigger as new direct-to-Chroma docs accumulate.

## 7. Low–Med: SEC 10-K narrative text not vector-indexed

The 117 `sec_filing` ledger rows have `document_family_id` = `legacy-filing-5`..`legacy-filing-121`; **none exist in Chroma** (0/117). These are backfill placeholders for the same accessions now sitting in the unprocessed queue (finding #1). The 10-K **numbers** live in `sec_companyfacts` (133,705 rows, fully populated); the 10-K **prose** is not semantically searchable anywhere. WORKING AS DESIGNED (`index_filing_text=False`), but a coverage limitation for narrative/MD&A-style questions. Enabling `sec.index_filing_text=true` + a weekly run would index section text.

## 8. Chroma distance function = cosine (lead hypothesis refuted)

`collection_metadata` → `hnsw:space=cosine`; `collections.config_json_str` → `"space":"cosine"` on the `#embedding` vector index. Matches `TraceAlchemyEmbeddingFunction`/retrieval's cosine assumption. The swallowed `modify` ValueError did **not** change distance semantics; its real damage is finding #4 (corpus_revision). L2-vs-cosine score corruption is **not** occurring.

---

## Cross-store reconciliation summary (@ ~02:14, mid-bootstrap)
- Chroma embeddings: 30,594 ids / ~30,131 distinct parents (growing).
- `corpus_items`: 24,413 (flat during run).
- Chroma parents covered by ledger (corpus_item_id OR document_family_id): 24,296.
- Chroma parents untracked by ledger: 5,835 — **100% `news/` (yfinance_news)** → finding #5.
- Ledger rows with no Chroma content: 117 sec_filing placeholders → finding #7.
- Structured tables: 0% null payload → finding #2.

## Recommended actions (priority order)
1. **[HIGH] Fix `mark_corpus_revision`** to not echo `hnsw:*` keys into `collection.modify`; re-verify BM25 lexical fusion actually fires (query `last_status.mode == "fts5"`). Restores half of the shipped hybrid pipeline.
2. **[MED] Register yfinance news in the corpus ledger** (route through `upsert_narrative`, or a recurring Chroma→corpus_items backfill) so coverage/completeness accounting is correct.
3. **[MED] Make the Phase 2.3 chroma_documents backfill re-runnable** (or fold its intent into #2) so the ledger stops diverging from Chroma.
4. **[LOW-MED] Run `weekly --force`** to drain the 147-filing queue; decide whether to enable `sec.index_filing_text` for 10-K prose search.

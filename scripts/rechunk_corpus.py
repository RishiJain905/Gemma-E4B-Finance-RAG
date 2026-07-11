"""
scripts/rechunk_corpus.py — Re-chunk + re-embed the ChromaDB corpus under the
new structure-aware chunker (Phase 2.1.3.2).

For every document group (entries sharing a `parent_id`, falling back to the
entry id for single-chunk docs):
  1. reconstruct the original text by joining the old chunks in chunk_index order
  2. (dry-run) compute the new chunk count; or
     (real)    delete the old chunk entries and re-add via
               `ChromaStore.add_document`, which re-chunks + re-embeds on :8087.

Idempotent (deterministic re-chunking → re-running yields the same chunks) and
resumable (a manifest of completed parent_ids is appended as each group is
migrated and skipped on the next run). Per-parent processing keeps the store
consistent on a mid-run failure: a parent is either fully re-added or, if it
failed, re-processed on resume.

Usage:
    python scripts/rechunk_corpus.py --dry-run            # report old vs new counts
    python scripts/rechunk_corpus.py --source sec         # one source
    python scripts/rechunk_corpus.py --ticker NVDA        # one ticker
    python scripts/rechunk_corpus.py --limit 50           # incremental batch
    python scripts/rechunk_corpus.py --backup             # full run, back up first
    python scripts/rechunk_corpus.py                      # full migration

Re-embedding hits http://127.0.0.1:8087/v1/embeddings (batched by the store's
embedding batch size). Back up `data/chroma/` before a full run (see --backup).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.storage.chroma_store import ChromaStore

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Chunk-specific metadata keys that add_document regenerates — strip them from
# the preserved base metadata before re-adding.
_CHUNK_META_KEYS = ("parent_id", "chunk_index", "chunk_count", "section")


def _load_store() -> "ChromaStore":
    """Build a ChromaStore using the configured (storage.yaml) chunking params."""
    from src.storage.chroma_store import ChromaStore
    return ChromaStore()


def _group_documents(store, source_filter: Optional[str],
                     ticker_filter: Optional[str]) -> list[tuple[str, dict]]:
    """Group collection entries by parent_id (fallback: the entry id).

    Returns [(parent_id, {"entries": [(chunk_index, id, text, meta)], "meta": base_meta})].
    `base_meta` is the first entry's metadata (representative ticker/source/date).
    """
    res = store.collection.get(include=["metadatas", "documents"])
    ids = list(res.get("ids") or [])
    docs = list(res.get("documents") or [])
    metas = list(res.get("metadatas") or [])

    groups: dict[str, dict] = {}
    order: list[str] = []
    for i, entry_id in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        parent = (meta or {}).get("parent_id") or entry_id
        chunk_index = (meta or {}).get("chunk_index", 0)
        if parent not in groups:
            groups[parent] = {"entries": [], "meta": dict(meta or {})}
            order.append(parent)
        groups[parent]["entries"].append((chunk_index, entry_id,
                                          docs[i] if i < len(docs) else "", meta or {}))

    out: list[tuple[str, dict]] = []
    for parent in order:
        g = groups[parent]
        base = g["meta"]
        src = (base.get("source") or "").lower()
        tkr = (base.get("ticker") or "").upper()
        if source_filter and src != source_filter.lower():
            continue
        if ticker_filter and tkr != ticker_filter.upper():
            continue
        out.append((parent, g))
    return out


def _reconstruct_text(group: dict) -> str:
    """Join old chunks in chunk_index order to approximate the original text."""
    entries = sorted(group["entries"], key=lambda x: x[0])
    return "\n".join((e[2] or "").strip() for e in entries if (e[2] or "").strip())


def _base_meta(group: dict) -> dict:
    """Base metadata to preserve (chunk-specific keys stripped)."""
    return {k: v for k, v in (group["meta"] or {}).items()
            if k not in _CHUNK_META_KEYS}


def _new_chunk_count(store, text: str, base_meta: dict) -> int:
    """Count the chunks the new chunker would produce (no embedding)."""
    from src.storage.chunking import chunk_document
    strategy = getattr(store, "chunk_strategy", "structural")
    chunks = chunk_document(
        text, source=base_meta.get("source"),
        max_chars=getattr(store, "chunk_chars", 1000),
        overlap_sentences=getattr(store, "overlap_sentences", 1),
        strategy=strategy,
    )
    # The "fixed" strategy uses char overlap; mirror add_document's fixed path.
    if strategy == "fixed":
        from src.storage.chroma_store import ChromaStore
        chunks = ChromaStore._chunk_text(
            text, getattr(store, "chunk_chars", 1000),
            getattr(store, "chunk_overlap", 150))
    return max(len(chunks), 1)  # add_document stores ≥1 entry for non-empty text


def _load_manifest(path: Optional[Path]) -> set[str]:
    if not path or not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Re-chunk + re-embed the chroma corpus.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report old vs new chunk counts; write nothing.")
    p.add_argument("--source", default=None, help="Only migrate this source.")
    p.add_argument("--ticker", default=None, help="Only migrate this ticker.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N parent groups (incremental batch).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Resume manifest file (completed parent_ids are skipped).")
    p.add_argument("--backup", action="store_true",
                   help="Back up data/chroma/ before a real (non-dry-run) run.")
    args = p.parse_args(argv)

    store = _load_store()
    groups = _group_documents(store, args.source, args.ticker)
    if args.limit:
        groups = groups[: args.limit]

    if not groups:
        print("No matching documents to migrate.")
        return 0

    done = _load_manifest(args.manifest)
    print(f"{len(groups)} parent groups to process "
          f"({len(done)} already in manifest). strategy={getattr(store, 'chunk_strategy', 'structural')}, "
          f"max_chars={getattr(store, 'chunk_chars', 1000)}.")

    # Backup before a real full run.
    if args.backup and not args.dry_run:
        backup = store.persist_directory.parent / f"chroma.bak.{int(time.time())}"
        shutil.copytree(store.persist_directory, backup)
        print(f"Backed up {store.persist_directory} -> {backup}")

    changed = skipped = 0
    old_total = new_total = 0
    t0 = time.time()
    manifest_fp = None
    if not args.dry_run and args.manifest:
        manifest_fp = args.manifest.open("a", encoding="utf-8")

    try:
        for idx, (parent, group) in enumerate(groups, 1):
            if parent in done:
                skipped += 1
                continue
            text = _reconstruct_text(group)
            base_meta = _base_meta(group)
            old_count = len(group["entries"])
            new_count = _new_chunk_count(store, text, base_meta) if text else 0
            old_total += old_count
            new_total += new_count

            if args.dry_run:
                print(f"[dry-run] {parent}: old={old_count} new={new_count} "
                      f"chars={len(text)}")
                changed += 1
                continue

            # Real migration: delete old entries, re-add (re-chunk + re-embed).
            old_ids = [e[1] for e in group["entries"]]
            store.collection.delete(ids=old_ids)
            store.add_document(
                document_id=parent, text=text,
                ticker=base_meta.get("ticker"), source=base_meta.get("source"),
                date=base_meta.get("date"),
                metadata={k: v for k, v in base_meta.items()
                          if k not in ("ticker", "source", "date")},
            )
            if manifest_fp:
                manifest_fp.write(parent + "\n")
                manifest_fp.flush()
            changed += 1
            if changed % 25 == 0:
                print(f"  ... {changed}/{len(groups)} parents "
                      f"({time.time() - t0:.0f}s)")
    finally:
        if manifest_fp:
            manifest_fp.close()

    print(f"\nProcessed {changed} parent(s), skipped {skipped}. "
          f"Old chunks: {old_total} -> New chunks: {new_total}. "
          f"Elapsed {time.time() - t0:.0f}s.")
    if args.dry_run:
        print("(dry-run: no writes were made.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

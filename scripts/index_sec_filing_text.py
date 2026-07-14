"""
scripts/index_sec_filing_text.py
Idempotent backfill of already-parsed SEC filings into the section/child index
(Phase 2.2.5.3, Step 3).

Migrates EXISTING parsed filing artifacts — the ``<db>/sec/parsed/<accession>.txt``
files written by the filing pipeline — into the hierarchical section index via the
same ``split_filing_sections`` + ``Store.add_filing_sections`` path the live
processor uses. No parser model call is ever made here; this only re-indexes text
that was already produced.

Safety / idempotency contract (spec Step 3):
  - default ``--dry-run`` reports eligible filings, parsed artifacts, estimated
    parent sections / chunks, missing files, and current indexed counts;
  - ``--apply`` indexes one filing at a time through the Store APIs;
  - ``--ticker`` / ``--accession`` / ``--limit`` slice a pilot;
  - ``--resume-manifest`` records each completed accession plus its source-file
    hash, so a re-run skips unchanged input and a changed artifact replaces its
    section family (``add_filing_sections`` deletes then re-adds);
  - ``--backup`` snapshots the affected Chroma collection directory before the
    first write;
  - an interruption leaves the last filing retryable and every prior filing valid
    (the manifest is flushed after each filing; each section family replace is
    atomic).

``print`` is used intentionally — this is an operator script under ``scripts/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Chunk-size estimate for the dry-run report only (the real chunk count comes
# from the store on apply). Mirrors ChromaStore.DEFAULT_CHUNK_CHARS.
_ESTIMATE_CHUNK_CHARS = 1000
_DEFAULT_INDEX_FORMS = ("10-K", "10-Q", "8-K")


# ── Parsed-artifact + filing discovery ─────────────────────

def parsed_dir_for(store) -> Path:
    """The parsed-artifact directory for a store (``<db-dir>/sec/parsed``)."""
    raw_db_path = getattr(store.sqlite, "db_path", None)
    db_path = Path(raw_db_path) if isinstance(raw_db_path, (str, Path)) else Path("data/finance.db")
    return db_path.parent / "sec" / "parsed"


def artifact_path(parsed_dir: Path, accession: str) -> Path:
    """Path to one filing's parsed artifact (matches the pipeline's naming)."""
    safe = "".join(ch for ch in str(accession) if ch.isalnum() or ch in "-_")
    return Path(parsed_dir) / f"{safe}.txt"


def file_hash(path: Path) -> str:
    """SHA-256 of a source artifact (records what a manifest entry migrated)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_filings(
    store,
    *,
    ticker: Optional[str] = None,
    accession: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Return filing rows to consider, filtered by the pilot-slice options."""
    sql = "SELECT * FROM filings"
    conditions: list[str] = []
    params: list = []
    if ticker:
        conditions.append("UPPER(ticker) = ?")
        params.append(ticker.upper())
    if accession:
        conditions.append("accession = ?")
        params.append(accession)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY filing_date DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    with store.sqlite._connect() as conn:  # noqa: SLF001 - operator script read
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


# ── Planning (dry-run) ─────────────────────────────────────

def _index_forms(config: Optional[dict]) -> set[str]:
    forms = (config or {}).get("index_forms") or _DEFAULT_INDEX_FORMS
    return {str(form).upper() for form in forms}


def _estimate_chunks(sections: list) -> int:
    total = 0
    for section in sections:
        length = len(section.text or "")
        total += max(1, (length + _ESTIMATE_CHUNK_CHARS - 1) // _ESTIMATE_CHUNK_CHARS)
    return total


def plan_filing(
    filing: dict, parsed_dir: Path, index_forms: set[str],
) -> dict:
    """Assess one filing for backfill without writing anything.

    Returns eligibility, whether the parsed artifact is present, and the
    estimated parent-section / chunk counts (from a deterministic split of the
    existing artifact — never a parser-model call).
    """
    from src.sec.filing_sections import split_filing_sections

    accession = filing.get("accession", "")
    form = str(filing.get("filing_type") or filing.get("form") or "").upper()
    path = artifact_path(parsed_dir, accession)
    plan = {
        "accession": accession,
        "ticker": filing.get("ticker"),
        "form": form,
        "artifact": str(path),
        "artifact_exists": path.exists(),
        "eligible_form": form in index_forms,
        "sections": 0,
        "estimated_chunks": 0,
        "error": None,
    }
    if not plan["eligible_form"] or not plan["artifact_exists"]:
        return plan
    try:
        text = path.read_text(encoding="utf-8")
        sections = split_filing_sections(
            text, {**filing, "file_path": str(path)})
        plan["sections"] = len(sections)
        plan["estimated_chunks"] = _estimate_chunks(sections)
    except Exception as error:  # noqa: BLE001 - report, never abort the whole run
        plan["error"] = str(error)
    return plan


# ── Manifest ───────────────────────────────────────────────

def load_manifest(path: Optional[Path]) -> dict:
    if path is None or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - a corrupt manifest never blocks a re-run
        print(f"WARNING: could not read manifest {path}; starting fresh")
        return {}


def save_manifest(path: Optional[Path], data: dict) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


# ── Backup ─────────────────────────────────────────────────

def backup_collection(store, backup_root: Optional[Path]) -> Optional[Path]:
    """Snapshot the Chroma persist directory before the first write."""
    if backup_root is None:
        return None
    source = Path(getattr(store.chroma, "persist_directory", "data/chroma"))
    backup_root = Path(backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)
    dest = backup_root / f"chroma-backup-{int(time.time())}"
    shutil.copytree(source, dest)
    print(f"Backed up Chroma collection: {source} -> {dest}")
    return dest


# ── Apply ──────────────────────────────────────────────────

def apply_filing(store, filing: dict, parsed_dir: Path) -> dict:
    """Index one filing's existing parsed artifact into the section index.

    Replaces the filing's section families (delete-then-add is atomic per family
    in ``Store.add_filing_sections``). Returns the store's write counts. Raises
    only on a genuine indexing failure so the caller can leave this filing
    retryable while every prior filing stays valid.
    """
    from src.sec.filing_sections import split_filing_sections

    accession = filing.get("accession", "")
    path = artifact_path(parsed_dir, accession)
    text = path.read_text(encoding="utf-8")
    sections = split_filing_sections(text, {**filing, "file_path": str(path)})
    if not sections:
        return {"sections_written": 0, "chunks_written": 0,
                "replacements": 0, "skipped": 0}
    counts = store.add_filing_sections(sections)
    store.sqlite.mark_filing_parsed(
        accession,
        embedding_id=f"sec:{accession}",
        file_path=str(path),
        section_count=counts["sections_written"],
        chunk_count=counts["chunks_written"],
    )
    return counts


def run_apply(
    store,
    filings: list[dict],
    parsed_dir: Path,
    *,
    index_forms: set[str],
    manifest_path: Optional[Path] = None,
    backup_root: Optional[Path] = None,
) -> dict:
    """Index eligible filings one at a time, honoring the resume manifest.

    Unchanged input (same source-file hash) is skipped; a changed artifact
    replaces its section family. The manifest is flushed after every filing so an
    interruption leaves prior filings recorded and the last filing retryable. The
    Chroma backup, when requested, is taken once before the first write.
    """
    manifest = load_manifest(manifest_path)
    summary = {
        "applied": 0, "skipped_unchanged": 0, "skipped_ineligible": 0,
        "missing_artifact": 0, "failed": 0,
        "sections_written": 0, "chunks_written": 0, "replacements": 0,
        "errors": [],
    }
    backed_up = False
    for filing in filings:
        accession = filing.get("accession", "")
        form = str(filing.get("filing_type") or filing.get("form") or "").upper()
        path = artifact_path(parsed_dir, accession)
        if form not in index_forms:
            summary["skipped_ineligible"] += 1
            continue
        if not path.exists():
            summary["missing_artifact"] += 1
            continue
        current_hash = file_hash(path)
        prior = manifest.get(accession)
        if isinstance(prior, dict) and prior.get("hash") == current_hash:
            summary["skipped_unchanged"] += 1
            continue

        if not backed_up:
            backup_collection(store, backup_root)
            backed_up = True

        try:
            counts = apply_filing(store, filing, parsed_dir)
        except Exception as error:  # noqa: BLE001 - isolate per filing, keep going
            summary["failed"] += 1
            summary["errors"].append(f"{accession}: {error}")
            print(f"FAILED {accession}: {error}")
            continue

        summary["applied"] += 1
        summary["sections_written"] += counts.get("sections_written", 0)
        summary["chunks_written"] += counts.get("chunks_written", 0)
        summary["replacements"] += counts.get("replacements", 0)
        manifest[accession] = {
            "hash": current_hash,
            "sections": counts.get("sections_written", 0),
            "chunks": counts.get("chunks_written", 0),
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # Flush after each filing: an interruption keeps prior filings recorded.
        save_manifest(manifest_path, manifest)
        print(f"Indexed {accession}: {counts.get('sections_written', 0)} sections, "
              f"{counts.get('chunks_written', 0)} chunks")
    return summary


def run_dry_run(
    store, filings: list[dict], parsed_dir: Path, *, index_forms: set[str],
) -> dict:
    """Assess every candidate filing and print the backfill plan."""
    plans = [plan_filing(f, parsed_dir, index_forms) for f in filings]
    eligible = [p for p in plans if p["eligible_form"] and p["artifact_exists"]]
    missing = [p for p in plans if p["eligible_form"] and not p["artifact_exists"]]
    ineligible = [p for p in plans if not p["eligible_form"]]
    try:
        indexed_now = store.count_filing_sections()
    except Exception:  # noqa: BLE001 - report best-effort
        indexed_now = None

    print("=== SEC filing-text backfill (dry run) ===")
    print(f"  candidate filings:   {len(filings)}")
    print(f"  eligible w/ artifact: {len(eligible)}")
    print(f"  missing artifacts:   {len(missing)}")
    print(f"  ineligible form:     {len(ineligible)}")
    print(f"  est. parent sections: {sum(p['sections'] for p in eligible)}")
    print(f"  est. child chunks:    {sum(p['estimated_chunks'] for p in eligible)}")
    print(f"  currently indexed sections: {indexed_now}")
    if missing:
        print("  -- missing parsed artifacts --")
        for p in missing:
            print(f"     {p['accession']} ({p['ticker']} {p['form']}) -> {p['artifact']}")
    return {
        "candidates": len(filings),
        "eligible": len(eligible),
        "missing_artifact": len(missing),
        "ineligible": len(ineligible),
        "estimated_sections": sum(p["sections"] for p in eligible),
        "estimated_chunks": sum(p["estimated_chunks"] for p in eligible),
        "currently_indexed_sections": indexed_now,
        "plans": plans,
    }


# ── CLI ────────────────────────────────────────────────────

def _load_sec_config() -> dict:
    from src.sec.filing_processor import FilingProcessor
    return FilingProcessor._load_sec_config()  # noqa: SLF001 - reuse the loader


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill parsed SEC filings into the hierarchical section index.")
    parser.add_argument("--apply", action="store_true",
                        help="Index eligible filings (default is a dry run).")
    parser.add_argument("--ticker", help="Limit to one ticker.")
    parser.add_argument("--accession", help="Limit to one filing accession.")
    parser.add_argument("--limit", type=int, help="Cap the number of filings.")
    parser.add_argument("--resume-manifest", type=Path,
                        help="Manifest path recording completed accession + hash.")
    parser.add_argument("--backup", type=Path,
                        help="Directory to snapshot the Chroma collection into "
                             "before the first write.")
    args = parser.parse_args(argv)

    from src.storage.store import Store
    from src.utils.env import load_env
    load_env()

    store = Store()
    parsed_dir = parsed_dir_for(store)
    index_forms = _index_forms(_load_sec_config())
    filings = load_filings(
        store, ticker=args.ticker, accession=args.accession, limit=args.limit)

    if not args.apply:
        run_dry_run(store, filings, parsed_dir, index_forms=index_forms)
        print("\nDry run only. Re-run with --apply to index.")
        return 0

    summary = run_apply(
        store, filings, parsed_dir, index_forms=index_forms,
        manifest_path=args.resume_manifest, backup_root=args.backup)
    print("\n=== Apply summary ===")
    for key in ("applied", "skipped_unchanged", "skipped_ineligible",
                "missing_artifact", "failed", "sections_written",
                "chunks_written", "replacements"):
        print(f"  {key:<20} {summary[key]}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

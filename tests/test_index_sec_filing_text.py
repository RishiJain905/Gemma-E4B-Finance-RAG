"""
tests/test_index_sec_filing_text.py
Offline tests for the SEC filing-text backfill CLI (Phase 2.2.5.3, Step 3).

Runs without ChromaDB, the model, or the network: a real SQLite ``filings`` table
(tmp), a small in-memory ``FakeChroma`` that tracks section families the way the
real store does, and parsed ``*.txt`` artifacts on disk. The tests assert the
dry-run report, one-filing-at-a-time apply, resume-manifest skip/replace, missing
artifacts, backup snapshotting, and interruption safety.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

if "chromadb" not in sys.modules:
    _mock_chroma = type(sys)("chromadb")
    _mock_chroma.EmbeddingFunction = object
    _mock_chroma.Documents = list
    _mock_chroma.Embeddings = list
    _mock_chroma.PersistentClient = MagicMock
    sys.modules["chromadb"] = _mock_chroma
    sys.modules["chromadb.api"] = MagicMock()

from scripts import index_sec_filing_text as B  # noqa: E402
from src.storage.store import Store  # noqa: E402


# ── In-memory section-family store ─────────────────────────

class FakeChroma:
    """Tracks SEC section families the way ChromaStore does (one child per ~1KB)."""

    def __init__(self, persist_directory):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        (self.persist_directory / "chroma.sqlite3").write_text("db", encoding="utf-8")
        self.families: dict[str, list[str]] = {}
        self.meta: dict[str, dict] = {}

    def count_filing_section_chunks(self, parent_id):
        return len(self.families.get(parent_id, []))

    def delete_filing_section_family(self, parent_id):
        self.families.pop(parent_id, None)
        self.meta.pop(parent_id, None)

    def add_document(self, document_id, text, ticker=None, source=None,
                     date=None, metadata=None):
        n = max(1, (len(text) + 999) // 1000)
        self.families[document_id] = [f"{document_id}#{i}" for i in range(n)]
        self.meta[document_id] = dict(metadata or {})

    def count_filing_sections(self, accession=None):
        parents = [
            pid for pid, m in self.meta.items()
            if accession is None or m.get("accession") == accession
        ]
        return len(parents)


@pytest.fixture
def store(tmp_path):
    with patch("src.storage.store.ChromaStore") as mock_cls:
        mock_cls.return_value = MagicMock()
        s = Store(db_path=tmp_path / "finance.db", chroma_path=tmp_path / "chroma")
    s.chroma = FakeChroma(tmp_path / "chroma")
    return s


# A tiny synthetic 10-K body with two recognizable Item headings + real content.
_FILING_TEXT = (
    "ITEM 1. BUSINESS\n"
    "The company designs accelerated computing platforms for data centers.\n"
    "Revenue is concentrated in the data-center segment.\n"
    "ITEM 1A. RISK FACTORS\n"
    "Demand for our products may decline if competition intensifies.\n"
    "Supply constraints could limit our ability to meet demand.\n"
)


def _register(store, accession, ticker="NVDA", form="10-K"):
    store.register_filing(ticker, form, "2026-01-15", "FY2025", accession,
                          f"https://sec.gov/{accession}")


def _write_artifact(parsed_dir: Path, accession: str, text=_FILING_TEXT) -> Path:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    path = B.artifact_path(parsed_dir, accession)
    path.write_text(text, encoding="utf-8")
    return path


# ── Dry run ────────────────────────────────────────────────

def test_dry_run_reports_eligible_and_missing(store, tmp_path):
    parsed = tmp_path / "parsed"
    _register(store, "acc-present")
    _register(store, "acc-missing")
    _write_artifact(parsed, "acc-present")

    filings = B.load_filings(store)
    report = B.run_dry_run(store, filings, parsed, index_forms={"10-K"})

    assert report["candidates"] == 2
    assert report["eligible"] == 1
    assert report["missing_artifact"] == 1
    assert report["estimated_sections"] >= 2  # Item 1 + Item 1A
    # Dry run writes nothing.
    assert store.count_filing_sections() == 0


def test_dry_run_marks_ineligible_forms(store, tmp_path):
    parsed = tmp_path / "parsed"
    _register(store, "acc-8k", form="8-K")
    _write_artifact(parsed, "acc-8k")

    filings = B.load_filings(store)
    report = B.run_dry_run(store, filings, parsed, index_forms={"10-K", "10-Q"})

    assert report["ineligible"] == 1
    assert report["eligible"] == 0


# ── Apply ──────────────────────────────────────────────────

def test_apply_indexes_one_filing(store, tmp_path):
    parsed = tmp_path / "parsed"
    _register(store, "acc-1")
    _write_artifact(parsed, "acc-1")

    filings = B.load_filings(store)
    summary = B.run_apply(store, filings, parsed, index_forms={"10-K"})

    assert summary["applied"] == 1
    assert summary["sections_written"] >= 2
    assert store.count_filing_sections("acc-1") >= 2


def test_apply_skips_missing_and_ineligible(store, tmp_path):
    parsed = tmp_path / "parsed"
    _register(store, "acc-ok")
    _register(store, "acc-missing")
    _register(store, "acc-8k", form="8-K")
    _write_artifact(parsed, "acc-ok")
    _write_artifact(parsed, "acc-8k")

    filings = B.load_filings(store)
    summary = B.run_apply(store, filings, parsed, index_forms={"10-K"})

    assert summary["applied"] == 1
    assert summary["missing_artifact"] == 1
    assert summary["skipped_ineligible"] == 1


# ── Resume manifest ────────────────────────────────────────

def test_resume_skips_unchanged_and_replaces_changed(store, tmp_path):
    parsed = tmp_path / "parsed"
    manifest = tmp_path / "manifest.json"
    _register(store, "acc-1")
    _write_artifact(parsed, "acc-1")
    filings = B.load_filings(store)

    first = B.run_apply(store, filings, parsed, index_forms={"10-K"},
                        manifest_path=manifest)
    assert first["applied"] == 1
    assert manifest.exists()

    # Re-run, unchanged input → skipped.
    second = B.run_apply(store, filings, parsed, index_forms={"10-K"},
                         manifest_path=manifest)
    assert second["applied"] == 0
    assert second["skipped_unchanged"] == 1

    # Change the artifact → the section family is replaced on the next run.
    _write_artifact(parsed, "acc-1", text=_FILING_TEXT + "ITEM 2. PROPERTIES\n"
                    "We lease offices worldwide for research and operations.\n")
    third = B.run_apply(store, filings, parsed, index_forms={"10-K"},
                        manifest_path=manifest)
    assert third["applied"] == 1
    assert third["replacements"] >= 1


# ── Backup ─────────────────────────────────────────────────

def test_backup_snapshots_before_first_write(store, tmp_path):
    parsed = tmp_path / "parsed"
    backup_root = tmp_path / "backups"
    _register(store, "acc-1")
    _write_artifact(parsed, "acc-1")
    filings = B.load_filings(store)

    B.run_apply(store, filings, parsed, index_forms={"10-K"},
                backup_root=backup_root)

    snapshots = list(backup_root.glob("chroma-backup-*"))
    assert snapshots, "expected a Chroma backup snapshot"
    assert (snapshots[0] / "chroma.sqlite3").exists()


def test_backup_not_taken_when_nothing_to_write(store, tmp_path):
    parsed = tmp_path / "parsed"
    backup_root = tmp_path / "backups"
    _register(store, "acc-missing")  # no artifact on disk
    filings = B.load_filings(store)

    B.run_apply(store, filings, parsed, index_forms={"10-K"},
                backup_root=backup_root)

    assert not list(backup_root.glob("chroma-backup-*"))


# ── Interruption safety ────────────────────────────────────

def test_interruption_leaves_prior_valid_and_last_retryable(store, tmp_path):
    parsed = tmp_path / "parsed"
    manifest = tmp_path / "manifest.json"
    _register(store, "acc-1")
    _register(store, "acc-2")
    _write_artifact(parsed, "acc-1")
    _write_artifact(parsed, "acc-2")
    # load_filings orders by filing_date DESC; both share a date, so order is by
    # insertion — fetch and index acc-1 first, fail acc-2.
    filings = sorted(B.load_filings(store), key=lambda f: f["accession"])

    real_add = store.add_filing_sections
    calls = {"n": 0}

    def flaky_add(sections):
        calls["n"] += 1
        if calls["n"] == 2:  # second filing blows up mid-run
            raise RuntimeError("chroma write interrupted")
        return real_add(sections)

    store.add_filing_sections = flaky_add
    summary = B.run_apply(store, filings, parsed, index_forms={"10-K"},
                          manifest_path=manifest)

    assert summary["applied"] == 1
    assert summary["failed"] == 1
    # Prior filing is recorded and valid; the failed one is absent → retryable.
    data = B.load_manifest(manifest)
    assert "acc-1" in data
    assert "acc-2" not in data
    assert store.count_filing_sections("acc-1") >= 2
    assert store.count_filing_sections("acc-2") == 0


def test_ticker_and_accession_slicing(store, tmp_path):
    _register(store, "acc-nvda", ticker="NVDA")
    _register(store, "acc-amd", ticker="AMD")

    assert {f["accession"] for f in B.load_filings(store, ticker="AMD")} == {"acc-amd"}
    assert {f["accession"] for f in B.load_filings(store, accession="acc-nvda")} == {"acc-nvda"}
    assert len(B.load_filings(store, limit=1)) == 1

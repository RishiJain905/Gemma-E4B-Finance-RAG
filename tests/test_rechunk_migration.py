"""
tests/test_rechunk_migration.py
Phase 2.1.3.2 — re-chunk + re-embed migration script.

Uses a fake in-memory collection (no live chromadb, no :8087 embeddings) so the
migration logic is tested deterministically. A live re-embed exercise is marked
``integration``/``slow`` and skipped by default.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import scripts.rechunk_corpus as rc


class FakeCollection:
    """In-memory chroma collection stand-in."""

    def __init__(self, docs):
        # docs: list of (id, text, meta)
        self._docs = {i: (t, m) for i, t, m in docs}
        self.add_calls: list[tuple] = []
        self.delete_calls: list[list] = []

    def get(self, include=None, where=None, limit=None):
        ids = list(self._docs.keys())
        return {
            "ids": ids,
            "documents": [self._docs[i][0] for i in ids],
            "metadatas": [self._docs[i][1] for i in ids],
        }

    def delete(self, ids):
        self.delete_calls.append(list(ids))
        for i in ids:
            self._docs.pop(i, None)

    def add(self, documents, metadatas, ids):
        self.add_calls.append((list(documents), list(metadatas), list(ids)))
        for i, d, m in zip(ids, documents, metadatas):
            self._docs[i] = (d, m)

    def count(self):
        return len(self._docs)


@pytest.fixture
def make_store(tmp_path):
    """Return a factory building a ChromaStore backed by a FakeCollection."""
    def _make(docs):
        with patch("src.storage.chroma_store.httpx.Client"), \
                patch("src.storage.chroma_store.chromadb.PersistentClient") as pc:
            pc.return_value.get_or_create_collection.return_value = FakeCollection(docs)
            from src.storage.chroma_store import ChromaStore
            store = ChromaStore(persist_directory=tmp_path / "chroma")
        # Replace with a populated FakeCollection for assertions.
        store.collection = FakeCollection(docs)
        store.persist_directory = tmp_path / "chroma"
        store.persist_directory.mkdir(parents=True, exist_ok=True)
        return store
    return _make


def _run(store, argv, monkeypatch):
    monkeypatch.setattr(rc, "_load_store", lambda: store)
    return rc.main(argv)


# ── Required tests ─────────────────────────────────────────────────────

def test_dry_run_no_writes(make_store, monkeypatch):
    docs = [
        ("news/NVDA/a", "NVIDIA revenue grew. Datacenter surged.", {"ticker": "NVDA", "source": "yfinance_news"}),
        ("ir/AMD/b", "AMD launched a new chip. It is fast.", {"ticker": "AMD", "source": "ir"}),
    ]
    store = make_store(docs)
    coll = store.collection
    code = _run(store, ["--dry-run"], monkeypatch)
    assert code == 0
    assert coll.add_calls == []      # no writes
    assert coll.delete_calls == []
    # Corpus unchanged.
    assert {i for i in coll.get()["ids"]} == {"news/NVDA/a", "ir/AMD/b"}


def test_parent_grouping_replaces_as_unit(make_store, monkeypatch):
    parent = "sec/NVDA/10-Q-2026"
    docs = [
        (f"{parent}#0", "First chunk text about revenue. ", {"parent_id": parent, "chunk_index": 0, "chunk_count": 2, "ticker": "NVDA", "source": "sec"}),
        (f"{parent}#1", "Second chunk text about margins. ", {"parent_id": parent, "chunk_index": 1, "chunk_count": 2, "ticker": "NVDA", "source": "sec"}),
        ("news/AMD/x", "AMD news snippet.", {"ticker": "AMD", "source": "yfinance_news"}),
    ]
    store = make_store(docs)
    coll = store.collection
    _run(store, [], monkeypatch)  # full migration

    # The two old chunks of the parent were deleted together.
    assert any(f"{parent}#0" in d and f"{parent}#1" in d for d in coll.delete_calls)
    # And re-added as a unit under the parent id (new chunk ids).
    added_ids = [i for _, _, ids in coll.add_calls for i in ids]
    assert any(i.startswith(f"{parent}#") for i in added_ids) or parent in added_ids
    # The single AMD doc was also processed (deleted + re-added).
    assert any("news/AMD/x" in d for d in coll.delete_calls)


def test_migration_idempotent(make_store, monkeypatch, tmp_path):
    docs = [
        ("news/NVDA/a", "NVIDIA revenue grew. Datacenter surged. Margins improved.", {"ticker": "NVDA", "source": "yfinance_news"}),
        ("ir/AMD/b", "AMD launched a new chip. It is fast.", {"ticker": "AMD", "source": "ir"}),
    ]
    manifest = tmp_path / "manifest.txt"
    store = make_store(docs)

    _run(store, ["--manifest", str(manifest)], monkeypatch)
    state1 = sorted(coll_state(store.collection))
    # Second run with the same manifest -> everything skipped, no new writes.
    n_adds_before = len(store.collection.add_calls)
    _run(store, ["--manifest", str(manifest)], monkeypatch)
    state2 = sorted(coll_state(store.collection))
    assert state1 == state2
    # No new add calls on the second run (all skipped via manifest).
    assert len(store.collection.add_calls) == n_adds_before


def test_resume_skips_completed(make_store, monkeypatch, tmp_path):
    parent_a = "news/NVDA/a"
    parent_b = "ir/AMD/b"
    docs = [
        (parent_a, "NVIDIA revenue grew. Datacenter surged.", {"ticker": "NVDA", "source": "yfinance_news"}),
        (parent_b, "AMD launched a new chip. It is fast.", {"ticker": "AMD", "source": "ir"}),
    ]
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(parent_a + "\n", encoding="utf-8")  # a already done
    store = make_store(docs)
    coll = store.collection
    _run(store, ["--manifest", str(manifest)], monkeypatch)

    # parent_a was skipped (not deleted, not re-added).
    assert not any(parent_a in d for d in coll.delete_calls)
    # parent_b was processed (deleted + re-added).
    assert any(parent_b in d for d in coll.delete_calls)


def test_source_filter(make_store, monkeypatch):
    docs = [
        ("news/NVDA/a", "NVIDIA revenue grew.", {"ticker": "NVDA", "source": "yfinance_news"}),
        ("ir/AMD/b", "AMD launched a chip.", {"ticker": "AMD", "source": "ir"}),
    ]
    store = make_store(docs)
    coll = store.collection
    _run(store, ["--source", "ir"], monkeypatch)
    # Only the ir doc was touched.
    assert any("ir/AMD/b" in d for d in coll.delete_calls)
    assert not any("news/NVDA/a" in d for d in coll.delete_calls)


def test_ticker_filter(make_store, monkeypatch):
    docs = [
        ("news/NVDA/a", "NVIDIA revenue grew.", {"ticker": "NVDA", "source": "yfinance_news"}),
        ("news/AMD/b", "AMD launched a chip.", {"ticker": "AMD", "source": "yfinance_news"}),
    ]
    store = make_store(docs)
    coll = store.collection
    _run(store, ["--ticker", "AMD"], monkeypatch)
    assert any("news/AMD/b" in d for d in coll.delete_calls)
    assert not any("news/NVDA/a" in d for d in coll.delete_calls)


def test_backup_copies_chroma_dir(make_store, monkeypatch, tmp_path):
    docs = [("news/NVDA/a", "NVIDIA revenue grew.", {"ticker": "NVDA", "source": "yfinance_news"})]
    store = make_store(docs)
    _run(store, ["--backup"], monkeypatch)
    backups = list((tmp_path).glob("chroma.bak.*"))
    assert backups, "expected a chroma backup directory"


def test_limit_caps_groups(make_store, monkeypatch):
    docs = [
        ("news/NVDA/a", "NVIDIA revenue grew.", {"ticker": "NVDA", "source": "yfinance_news"}),
        ("news/AMD/b", "AMD launched a chip.", {"ticker": "AMD", "source": "yfinance_news"}),
        ("ir/MSFT/c", "Microsoft news.", {"ticker": "MSFT", "source": "ir"}),
    ]
    store = make_store(docs)
    coll = store.collection
    _run(store, ["--limit", "1"], monkeypatch)
    # Only one parent deleted.
    deleted = [d for grp in coll.delete_calls for d in grp]
    assert len(deleted) == 1


# ── Live re-embed (opt-in) ─────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_live_rechunk_smoke():
    """Exercise the real migration against the live chroma corpus on :8087.

    Skips unless the embedding endpoint is up; --limit 1 to keep it cheap.
    """
    import httpx
    try:
        r = httpx.get("http://127.0.0.1:8087/health", timeout=3)
        if r.status_code != 200:
            pytest.skip("embedding server not on :8087")
    except Exception:
        pytest.skip("embedding server not on :8087")
    code = rc.main(["--dry-run", "--limit", "1"])
    assert code == 0


# ── helpers ────────────────────────────────────────────────────────────

def coll_state(coll):
    res = coll.get()
    return [(i, t) for i, t in zip(res["ids"], res["documents"])]

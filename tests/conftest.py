"""Shared test fixtures and configuration."""

import pytest

from src.storage.sqlite_store import SQLiteStore
from src.storage.store import Store


class FakeCorpusChroma:
    """Metadata-only Chroma double for offline corpus explorer tests."""

    def __init__(self):
        self.records = []

    @staticmethod
    def _matches(metadata, where):
        if not where:
            return True
        if "$and" in where:
            return all(FakeCorpusChroma._matches(metadata, item) for item in where["$and"])
        for key, expected in where.items():
            actual = metadata.get(key)
            if isinstance(expected, dict):
                if "$gte" in expected and (actual is None or actual < expected["$gte"]):
                    return False
                if "$lte" in expected and (actual is None or actual > expected["$lte"]):
                    return False
            elif actual != expected:
                return False
        return True

    def _filtered(self, where=None):
        return [row for row in self.records if self._matches(row["metadata"], where)]

    def get_metadata(self, *, where=None, limit=None, offset=0):
        rows = self._filtered(where)[offset:]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def count(self):
        return len(self.records)

    def source_counts(self, *, limit=200, offset=0):
        counts = {}
        for row in self.records:
            source = row["metadata"].get("source")
            if source:
                counts[source] = counts.get(source, 0) + 1
        return [
            {"source": source, "count": count}
            for source, count in sorted(counts.items())
        ][offset:offset + limit]

    get_source_counts = source_counts

    def ticker_counts(self, *, limit=200, offset=0):
        counts = {}
        for row in self.records:
            ticker = row["metadata"].get("ticker")
            if ticker:
                item = counts.setdefault(ticker, {"ticker": ticker, "record_count": 0, "sources": set()})
                item["record_count"] += 1
                if row["metadata"].get("source"):
                    item["sources"].add(row["metadata"]["source"])
        rows = []
        for item in sorted(counts.values(), key=lambda value: value["ticker"]):
            rows.append({**item, "sources": sorted(item["sources"])})
        return rows[offset:offset + limit]

    get_ticker_counts = ticker_counts

    def search_document_families(self, *, query=None, source=None, ticker=None,
                                 date_from=None, date_to=None, limit=200, offset=0):
        rows = []
        for row in self._filtered():
            metadata = row["metadata"]
            if metadata.get("source") == "sec_filing":
                continue
            if source and metadata.get("source") != source:
                continue
            if ticker and metadata.get("ticker") != ticker.upper():
                continue
            date = metadata.get("date") or metadata.get("filing_date") or ""
            if date_from and date < date_from:
                continue
            if date_to and date > date_to:
                continue
            haystack = " ".join(str(metadata.get(key, "")) for key in (
                "source", "ticker", "date", "parent_id", "document_id",
            )).lower()
            if query and query.lower() not in haystack:
                continue
            rows.append(row)
        families = {}
        for row in rows:
            metadata = row["metadata"]
            parent_id = metadata.get("parent_id") or row["id"].split("#", 1)[0]
            family = families.setdefault(parent_id, {
                "id": parent_id, "metadata": dict(metadata), "chunk_count": 0,
            })
            family["chunk_count"] += 1
        return list(families.values())[offset:offset + limit]

    def list_filing_section_families(self, accession, *, limit=200, offset=0):
        rows = [row for row in self._filtered({"$and": [
            {"source": "sec_filing"}, {"accession": accession},
        ]})]
        families = {}
        for row in rows:
            metadata = row["metadata"]
            parent_id = metadata.get("parent_id") or row["id"].split("#", 1)[0]
            family = families.setdefault(parent_id, {
                "id": parent_id, "metadata": dict(metadata), "chunk_count": 0,
            })
            family["chunk_count"] += 1
        return list(families.values())[offset:offset + limit]

    get_filing_section_families = list_filing_section_families

    def get_section_chunks(self, parent_id, *, limit, offset=0):
        rows = self._filtered({"parent_id": parent_id})[offset:offset + limit]
        return [{"id": row["id"], "document": row.get("document"),
                 "metadata": dict(row["metadata"])} for row in rows]

    get_document_family = get_section_chunks

    def get_adjacent_sections(self, accession, section_index, *, before=1, after=1):
        rows = self._filtered({"$and": [
            {"source": "sec_filing"}, {"accession": accession},
        ]})
        return [{"id": row["id"], "document": row.get("document"),
                 "metadata": dict(row["metadata"])} for row in rows
                if section_index - before <= row["metadata"].get("section_index", -1) <= section_index + after]

    def count_filing_sections(self, accession=None):
        rows = self._filtered({"source": "sec_filing"})
        if accession is not None:
            rows = [row for row in rows if row["metadata"].get("accession") == accession]
        return len({row["metadata"].get("parent_id") for row in rows})

    def count_filing_section_chunks(self, parent_id):
        return len(self._filtered({"parent_id": parent_id}))

    def get_document(self, document_id):
        for row in self.records:
            if row["id"] == document_id:
                return {"id": row["id"], "document": row.get("document"),
                        "metadata": dict(row["metadata"])}
        return None


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="run @pytest.mark.live tests (network / live model on :8087)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    markexpr = config.getoption("-m", default="") or ""
    if "live" in markexpr and "not live" not in markexpr:
        return  # user explicitly selected live via -m
    skip_live = pytest.mark.skip(reason="needs --live (or -m live)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


# ── Offline safety (2.1.6.2) ──────────────────────────

@pytest.fixture(autouse=True)
def _no_fetch_on_miss_by_default(monkeypatch):
    """Keep the offline suite offline: the fetch-on-miss gate can never fire.

    Any test that exercises /query with a real config and an empty store would
    otherwise trigger a live yfinance fetch. Fetch-on-miss tests re-set
    FETCH_ON_MISS_MIN_CONFIDENCE to a real threshold explicitly.
    """
    monkeypatch.setattr(
        "src.middleware.app.FETCH_ON_MISS_MIN_CONFIDENCE", float("inf"), raising=False
    )


# ── Shared store fixtures (1.8.3) ──────────────────────

@pytest.fixture
def fresh_store(tmp_path):
    """Create a Store with temporary databases for testing.

    Uses the live embedding endpoint on :8087 — tests that exercise document
    storage/search via this fixture should be marked ``network``/``integration``.
    """
    db_path = tmp_path / "test.db"
    chroma_path = tmp_path / "test_chroma"
    store = Store(
        db_path=db_path,
        chroma_path=chroma_path,
        embedding_endpoint="http://127.0.0.1:8087/v1/embeddings",
    )
    yield store
    import shutil
    if chroma_path.exists():
        shutil.rmtree(chroma_path, ignore_errors=True)


@pytest.fixture
def seeded_store(fresh_store):
    """Store pre-seeded with structured facts and documents.

    Document seeding requires the live embedding endpoint (:8087).
    """
    store = fresh_store
    store.save_fundamental("NVDA", "total_revenue", 26.0, "usd", "2026-Q1")
    store.save_fundamental("NVDA", "net_income", 12.0, "usd", "2026-Q1")
    store.save_fundamental("AMD", "total_revenue", 5.8, "usd", "2026-Q1")
    store.save_document("test/nvda/10q", "NVIDIA reported strong datacenter growth...",
                        ticker="NVDA", source="sec", date="2026-03-15")
    store.save_document("test/amd/10q", "AMD launched new MI300X accelerators...",
                        ticker="AMD", source="sec", date="2026-03-15")
    return store


@pytest.fixture
def offline_store(tmp_path):
    """Real SQLite plus metadata-only fake Chroma; never starts model/network IO."""
    store = object.__new__(Store)
    store.sqlite = SQLiteStore(tmp_path / "offline.db")
    store.chroma = FakeCorpusChroma()
    return store

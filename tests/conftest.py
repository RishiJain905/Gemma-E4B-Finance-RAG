"""Shared test fixtures and configuration."""

import pytest

from src.storage.store import Store


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

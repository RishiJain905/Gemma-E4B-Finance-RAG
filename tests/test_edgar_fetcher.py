"""
tests/test_edgar_fetcher.py
Real/live pytest suite for SECEdgarFilingFetcher.

These tests hit the LIVE SEC EDGAR endpoints (company_tickers.json + the
data.sec.gov submissions API + the EDGAR Archives) and exercise the real
storage layer. They require live network access to SEC EDGAR. Any storage
path that touches embeddings uses the live llama-server embedding endpoint,
but the cases here only use the SQLite side of the Store.

Covers:
  - Live discover_filings("AAPL", ["10-K","10-Q"]) -> well-formed dicts
  - Live _resolve_cik("AAPL") == "0000320193"
  - Live download_filing_text(...) on the latest 10-K returns > 50 chars
  - Live register_discovered_filings("AAPL") registers then is idempotent
  - Pure-logic _derive_period checks (10-K and 10-Q) that need no network
"""

import re
from unittest.mock import MagicMock, patch

import pytest

from src.sec import SECEdgarFilingFetcher
from src.sec.edgar_fetcher import fetch_sec_company_tickers
from src.storage.store import Store


REQUIRED_KEYS = {
    "ticker",
    "filing_type",
    "filing_date",
    "period",
    "accession",
    "source_url",
    "cik",
}

ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")


# ── Fixtures ────────────────────────────────────────

@pytest.fixture(scope="module")
def fetcher():
    """A fetcher backed by the default store (read-only live discovery)."""
    return SECEdgarFilingFetcher(request_delay=0.2)


@pytest.fixture
def tmp_fetcher(tmp_path):
    """A fetcher with an isolated SQLite store for registration tests."""
    store = Store(db_path=tmp_path / "test.db")
    return SECEdgarFilingFetcher(store=store, request_delay=0.2)


# ── 1. Live discover_filings ────────────────────────

def test_discover_filings_live(fetcher):
    """discover_filings returns non-empty, well-formed filing dicts."""
    filings = fetcher.discover_filings("AAPL", ["10-K", "10-Q"])

    assert isinstance(filings, list)
    assert len(filings) > 0

    types = {f["filing_type"] for f in filings}
    assert "10-K" in types
    assert "10-Q" in types

    for f in filings:
        # All required keys present
        assert REQUIRED_KEYS.issubset(f.keys())
        assert f["ticker"] == "AAPL"
        assert f["filing_type"] in ("10-K", "10-Q")
        # Valid accession format: NNNNNNNNNN-NN-NNNNNN
        assert ACCESSION_RE.match(f["accession"]), f["accession"]
        # CIK is the zero-padded 10-digit Apple CIK
        assert f["cik"] == "0000320193"
        # Canonical EDGAR archive document URL
        assert f["source_url"].startswith(
            "https://www.sec.gov/Archives/edgar/data/320193/"
        )
        assert f["filing_date"]


def test_discover_filings_respects_count(fetcher):
    """The count argument truncates results per filing type."""
    filings = fetcher.discover_filings("AAPL", ["10-K"], count=2)
    assert len(filings) <= 2
    assert all(f["filing_type"] == "10-K" for f in filings)


# ── 2. Live CIK resolution ──────────────────────────

def test_resolve_cik_live(fetcher):
    """_resolve_cik maps AAPL to its zero-padded 10-digit CIK."""
    assert fetcher._resolve_cik("AAPL") == "0000320193"
    # Case-insensitive
    assert fetcher._resolve_cik("aapl") == "0000320193"


def test_resolve_cik_unknown(fetcher):
    """An unknown ticker resolves to None."""
    assert fetcher._resolve_cik("NOTAREALTICKER123") is None


# ── 3. Live download_filing_text ────────────────────

def test_download_filing_text_live(fetcher):
    """download_filing_text on the most recent 10-K returns substantial text."""
    filings = fetcher.discover_filings("AAPL", ["10-K"], count=1)
    assert filings, "expected at least one 10-K"

    text = fetcher.download_filing_text(filings[0])
    assert text is not None
    assert len(text) > 50
    # Sanity: stripped to plain text (no leftover tags)
    assert "<html" not in text.lower()


# ── 4. Live register (idempotent) ───────────────────

def test_register_discovered_filings_idempotent(tmp_fetcher):
    """register_discovered_filings registers rows, then is a no-op on rerun."""
    first = tmp_fetcher.register_discovered_filings("AAPL")
    assert first > 0

    # Rows are persisted in the filings table
    with tmp_fetcher.store.sqlite._connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM filings WHERE ticker = ?", ("AAPL",)
        ).fetchone()["n"]
    assert count == first

    # Second run discovers the same filings but registers nothing new
    second = tmp_fetcher.register_discovered_filings("AAPL")
    assert second == 0


# ── 5. Pure-logic _derive_period (no network) ───────

def test_derive_period_10k():
    """10-K filed in Q1 of year Y maps to fiscal year Y-1."""
    assert SECEdgarFilingFetcher._derive_period("2026-03-15", "10-K") == "2025"


def test_derive_period_10q():
    """10-Q filed in May (Q2) covers the prior quarter (Q1)."""
    assert SECEdgarFilingFetcher._derive_period("2026-05-10", "10-Q") == "2026-Q1"


def test_derive_period_10q_january():
    """10-Q filed in Jan-Mar (Q1) covers the previous year's Q4."""
    assert SECEdgarFilingFetcher._derive_period("2026-02-01", "10-Q") == "2025-Q4"


def test_derive_period_empty():
    """Empty filing date returns an empty period."""
    assert SECEdgarFilingFetcher._derive_period("", "10-K") == ""


@patch("src.sec.edgar_fetcher.requests.get")
def test_fetch_sec_company_tickers_parses_rows(mock_get):
    """SEC company_tickers.json rows are normalized for catalog reuse."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "0": {"ticker": "aapl", "cik_str": 320193, "title": "Apple Inc."},
        "1": {"ticker": "NVDA", "cik_str": "1045810", "title": "NVIDIA CORP"},
    }
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    rows = fetch_sec_company_tickers("Test User test@example.com")

    assert rows == [
        {"ticker": "AAPL", "cik": "0000320193", "title": "Apple Inc."},
        {"ticker": "NVDA", "cik": "0001045810", "title": "NVIDIA CORP"},
    ]
    mock_get.assert_called_once_with(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "Test User test@example.com"},
        timeout=30,
    )

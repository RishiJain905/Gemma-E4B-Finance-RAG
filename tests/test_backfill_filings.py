"""
tests/test_backfill_filings.py
Offline tests for the SEC substantive-filings backfill (src/sec/backfill.py),
its scheduler onboarding hook, and the two CLI wrappers.

Runs without ChromaDB, the model, or the network: a real SQLite ``filings``
table (tmp) with a mocked ChromaStore, a mocked EDGAR fetcher (discover), and a
mocked FilingProcessor whose ``_process_single_filing`` records the parse the
way the live path would. Covers idempotency, substantive-filing detection and
the scheduler onboarding trigger, reconcile gap detection + repair, per-filing
and per-form failure isolation, and coverage.yaml ticker sourcing.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from scripts import backfill_filings as BF
from scripts import reconcile_filings as RF
from src.sec import FilingProcessor, FilingScheduler
from src.sec.backfill import FilingBackfiller
from src.storage.store import Store
from src.universe.coverage import CoverageResolver


DEEP_WATCHLIST = {
    "SNDK", "PLTR", "NVDA", "MU", "INTC", "NBIS", "BB", "AMD",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL",
}
_NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    with patch("src.storage.store.ChromaStore") as chroma_cls:
        chroma_cls.return_value = MagicMock()
        yield Store(db_path=tmp_path / "finance.db", chroma_path=tmp_path / "chroma")


def _filing(ticker, form, accession, *, filing_date="2026-05-01", period="2026-Q1"):
    return {
        "ticker": ticker, "filing_type": form, "filing_date": filing_date,
        "period": period, "accession": accession,
        "source_url": f"https://www.sec.gov/Archives/{accession}.htm",
        "cik": "0000320193", "primary_document": f"{accession}.htm",
    }


def _make_backfiller(store, discovered_by_form=None, *, process_result=True,
                     process_side_effect=None):
    """FilingBackfiller wired to a mock fetcher + mock processor (no network)."""
    discovered_by_form = discovered_by_form or {}

    fetcher = MagicMock()

    def discover(ticker, filing_types, count):
        form = filing_types[0]
        return list(discovered_by_form.get((ticker, form), []))[:count]

    fetcher.discover_filings.side_effect = discover

    processor = MagicMock()

    def process(record):
        if process_side_effect is not None:
            return process_side_effect(record)
        # Mimic the live path: a successful process marks the filing parsed.
        store.sqlite.mark_filing_parsed(record["accession"])
        return process_result

    processor._process_single_filing.side_effect = process

    backfiller = FilingBackfiller(
        store, processor=processor, fetcher=fetcher,
        coverage=CoverageResolver(store),
    )
    return backfiller, fetcher, processor


# ── Idempotency ────────────────────────────────────────────

def test_backfill_is_idempotent(store):
    discovered = {
        ("NVDA", "10-K"): [_filing("NVDA", "10-K", "acc-k1",
                                    filing_date="2026-01-15", period="2025")],
        ("NVDA", "10-Q"): [_filing("NVDA", "10-Q", "acc-q1")],
        ("NVDA", "8-K"): [],
    }
    backfiller, _fetcher, processor = _make_backfiller(store, discovered)

    first = backfiller.backfill_ticker("NVDA")
    assert first["ingested"] == 2
    assert first["skipped"] == 0
    assert first["failed"] == 0

    second = backfiller.backfill_ticker("NVDA")
    assert second["ingested"] == 0
    assert second["skipped"] == 2

    # Processing ran only for the first pass; the re-run is a cheap no-op.
    assert processor._process_single_filing.call_count == 2


# ── Substantive detection + onboarding trigger ─────────────

def test_has_substantive_filing(store):
    backfiller, _f, _p = _make_backfiller(store)
    assert backfiller.has_substantive_filing("NVDA") is False

    store.register_filing("NVDA", "8-K", "2026-05-01", "", "acc-8k",
                          "https://www.sec.gov/x")
    assert backfiller.has_substantive_filing("NVDA") is False  # 8-K isn't periodic

    store.register_filing("NVDA", "10-Q", "2026-05-01", "2026-Q1", "acc-q",
                          "https://www.sec.gov/y")
    assert backfiller.has_substantive_filing("NVDA") is True


def test_scheduler_onboarding_triggers_only_for_gapped_deep_tickers(store):
    scheduler = FilingScheduler(
        store=store, processor=MagicMock(),
        coverage_resolver=CoverageResolver(store),
    )
    # Activate the gate (a populated registry would set this in production).
    scheduler.daily_index_discovery = MagicMock()

    fake = MagicMock()
    fake.has_substantive_filing.side_effect = lambda t: t == "NVDA"
    fake.onboard_ticker.return_value = {"ticker": "PLTR", "ingested": 3}

    with patch("src.sec.backfill.FilingBackfiller", return_value=fake), \
         patch.object(scheduler, "_tickers_for", return_value=["NVDA", "PLTR"]):
        result = scheduler._run_onboarding_backfill()

    assert result["triggered"] == 1
    fake.onboard_ticker.assert_called_once_with("PLTR")
    assert "PLTR" in result["tickers"]
    assert "NVDA" not in result["tickers"]


def test_scheduler_onboarding_inert_without_discovery(store):
    scheduler = FilingScheduler(store=store, processor=MagicMock())
    assert scheduler.daily_index_discovery is None
    assert scheduler._run_onboarding_backfill() == {"triggered": 0, "tickers": {}}


# ── Reconcile: gap detection + repair ──────────────────────

def test_reconcile_detects_gaps_and_repairs(store):
    # NVDA: a recent, indexed 10-Q -> not gapped.
    store.register_filing("NVDA", "10-Q", "2026-06-01", "2026-Q1", "acc-nvq",
                          "https://www.sec.gov/nvq")
    store.sqlite.mark_filing_parsed("acc-nvq", section_count=3)

    discovered = {
        ("PLTR", "10-K"): [_filing("PLTR", "10-K", "acc-pk",
                                   filing_date="2026-03-01", period="2025")],
        ("PLTR", "10-Q"): [_filing("PLTR", "10-Q", "acc-pq",
                                   filing_date="2026-06-10")],
    }
    backfiller, _f, _p = _make_backfiller(store, discovered)

    report = backfiller.reconcile(
        tickers=["NVDA", "PLTR"], repair=True, now=_NOW,
    )

    assert report["gaps"]["NVDA"]["gap"] is False
    assert report["gaps"]["PLTR"]["gap"] is True
    assert report["gaps"]["PLTR"]["reasons"] == ["no_recent_periodic", "no_indexed_text"]
    assert report["gapped_tickers"] == ["PLTR"]

    assert "PLTR" in report["repairs"]
    assert "NVDA" not in report["repairs"]
    assert report["repairs"]["PLTR"]["ingested"] == 2


def test_reconcile_report_only_does_not_repair(store):
    discovered = {("PLTR", "10-K"): [_filing("PLTR", "10-K", "acc-pk")]}
    backfiller, _f, processor = _make_backfiller(store, discovered)

    report = backfiller.reconcile(tickers=["PLTR"], now=_NOW)

    assert report["gaps"]["PLTR"]["gap"] is True
    assert report["repairs"] == {}
    processor._process_single_filing.assert_not_called()


def test_reconcile_recent_but_unindexed_is_gapped(store):
    # Recent periodic present but never indexed -> gap on no_indexed_text only.
    store.register_filing("MU", "10-Q", "2026-06-15", "2026-Q3", "acc-mu",
                          "https://www.sec.gov/mu")  # status stays unprocessed
    backfiller, _f, _p = _make_backfiller(store)

    report = backfiller.find_gaps(["MU"], now=_NOW)
    assert report["MU"]["gap"] is True
    assert report["MU"]["reasons"] == ["no_indexed_text"]


# ── Failure isolation ──────────────────────────────────────

def test_per_filing_failure_isolation(store):
    discovered = {
        ("NVDA", "10-K"): [
            _filing("NVDA", "10-K", "acc-bad", filing_date="2026-01-15", period="2025"),
            _filing("NVDA", "10-K", "acc-good", filing_date="2025-01-15", period="2024"),
        ],
    }

    def process(record):
        if record["accession"] == "acc-bad":
            raise RuntimeError("chroma write blew up")
        store.sqlite.mark_filing_parsed(record["accession"])
        return True

    backfiller, _f, _p = _make_backfiller(
        store, discovered, process_side_effect=process,
    )
    result = backfiller.backfill_ticker(
        "NVDA", filing_types=("10-K",), counts_per_type={"10-K": 2},
    )

    assert result["ingested"] == 1
    assert result["failed"] == 1
    assert store.get_filing("acc-good")["status"] == "parsed"


def test_process_incomplete_counts_as_failed(store):
    discovered = {("NVDA", "10-K"): [_filing("NVDA", "10-K", "acc-x")]}
    backfiller, _f, _p = _make_backfiller(
        store, discovered, process_result=False,
    )
    result = backfiller.backfill_ticker("NVDA", filing_types=("10-K",))
    assert result["ingested"] == 0
    assert result["failed"] == 1


def test_discovery_failure_isolated_per_form(store):
    fetcher = MagicMock()
    fetcher.discover_filings.side_effect = RuntimeError("EDGAR unreachable")
    backfiller = FilingBackfiller(
        store, processor=MagicMock(), fetcher=fetcher,
        coverage=CoverageResolver(store),
    )
    result = backfiller.backfill_ticker("NVDA", filing_types=("10-K", "10-Q"))
    assert result["failed"] == 2
    assert result["ingested"] == 0
    assert result["discovered"] == 0


def test_run_isolates_per_ticker(store):
    discovered = {("NVDA", "10-K"): [_filing("NVDA", "10-K", "acc-nv")]}
    backfiller, _f, _p = _make_backfiller(store, discovered)

    report = backfiller.run(["NVDA", "PLTR"], filing_types=("10-K",))
    assert report["tickers"]["NVDA"]["ingested"] == 1
    assert report["tickers"]["PLTR"]["ingested"] == 0
    assert report["totals"]["ingested"] == 1


# ── Dry run ────────────────────────────────────────────────

def test_dry_run_registers_and_processes_nothing(store):
    discovered = {
        ("NVDA", "10-K"): [_filing("NVDA", "10-K", "acc-k")],
        ("NVDA", "10-Q"): [], ("NVDA", "8-K"): [],
    }
    backfiller, _f, processor = _make_backfiller(store, discovered)

    result = backfiller.backfill_ticker("NVDA", dry_run=True)

    assert result["would_ingest"] == 1
    assert result["ingested"] == 0
    processor._process_single_filing.assert_not_called()
    assert store.get_filing("acc-k") is None


# ── Coverage sourcing ──────────────────────────────────────

def test_deep_tickers_come_from_coverage_policy(store):
    backfiller = FilingBackfiller(
        store, processor=MagicMock(), fetcher=MagicMock(),
        coverage=CoverageResolver(store),
    )
    assert set(backfiller.deep_tickers()) == DEEP_WATCHLIST


# ── Foreign private issuer (20-F) routing ──────────────────

def test_20f_routes_deep_in_backfill(store):
    """A 20-F (foreign issuer annual report) is registered + processed on the
    deep periodic path, not the broad event path."""
    discovered = {
        ("NBIS", "20-F"): [_filing("NBIS", "20-F", "acc-20f", period="2025")],
    }
    backfiller, _f, processor = _make_backfiller(store, discovered)

    result = backfiller.backfill_ticker(
        "NBIS", filing_types=("20-F",), counts_per_type={"20-F": 1},
    )

    assert result["ingested"] == 1
    record = processor._process_single_filing.call_args.args[0]
    assert record["discovery_scope"] == "deep"
    assert store.get_filing("acc-20f") is not None  # registered, not skipped


def test_20f_is_in_processor_index_forms():
    """The default FilingProcessor indexes 20-F full-text like a 10-K."""
    proc = FilingProcessor(store=MagicMock(), fetcher=MagicMock(), parser=MagicMock())
    assert "20-F" in proc.index_forms


# ── CLI wrappers ───────────────────────────────────────────

def test_backfill_cli_resolve_counts():
    assert BF._resolve_counts(None) == {"10-K": 2, "10-Q": 4, "8-K": 8}
    assert BF._resolve_counts(5) == {"10-K": 5, "10-Q": 5, "8-K": 5}


def test_backfill_cli_exit_zero_on_any_success(monkeypatch):
    fake = MagicMock()
    fake.deep_tickers.return_value = ["NVDA"]
    fake.run.return_value = {
        "dry_run": False,
        "tickers": {"NVDA": {
            "discovered": 1, "ingested": 1, "skipped": 0, "failed": 0,
            "would_ingest": 0, "filings": [],
        }},
        "totals": {"discovered": 1, "ingested": 1, "skipped": 0,
                   "failed": 0, "would_ingest": 0},
    }
    monkeypatch.setattr(BF, "Store", MagicMock())
    monkeypatch.setattr(BF, "FilingBackfiller", MagicMock(return_value=fake))

    assert BF.main(["--tickers", "NVDA"]) == 0


def test_backfill_cli_exit_one_when_every_ticker_failed(monkeypatch):
    fake = MagicMock()
    fake.run.return_value = {
        "dry_run": False,
        "tickers": {"NVDA": {
            "discovered": 1, "ingested": 0, "skipped": 0, "failed": 1,
            "would_ingest": 0, "filings": [],
        }},
        "totals": {"discovered": 1, "ingested": 0, "skipped": 0,
                   "failed": 1, "would_ingest": 0},
    }
    monkeypatch.setattr(BF, "Store", MagicMock())
    monkeypatch.setattr(BF, "FilingBackfiller", MagicMock(return_value=fake))

    assert BF.main(["--tickers", "NVDA"]) == 1


def test_reconcile_cli_repair_invokes_backfiller(monkeypatch):
    fake = MagicMock()
    fake.reconcile.return_value = {
        "recency_days": 120,
        "gaps": {"PLTR": {"ticker": "PLTR", "gap": True,
                          "reasons": ["no_recent_periodic"],
                          "latest_periodic": None, "filing_count": 0}},
        "gapped_tickers": ["PLTR"],
        "repairs": {"PLTR": {"ingested": 2, "skipped": 0, "failed": 0}},
    }
    monkeypatch.setattr(RF, "Store", MagicMock())
    monkeypatch.setattr(RF, "FilingBackfiller", MagicMock(return_value=fake))

    assert RF.main(["--tickers", "PLTR", "--repair"]) == 0
    _args, kwargs = fake.reconcile.call_args
    assert kwargs["repair"] is True

"""
scripts/backfill_filings.py
Backfill deep-watchlist substantive SEC filings (10-K/10-Q/8-K history).

The scheduler's daily-index cursor only reaches ~7 days back, so historical
periodic filings predate it and are never ingested. This operator script
discovers a bounded, most-recent-first slice per ticker via the per-company
EDGAR submissions endpoint and processes each filing through the same
``FilingProcessor`` path the scheduler uses (full-text section indexing when
``sec.index_filing_text`` is on). Re-runs are cheap no-ops: a filing already
stored AND parsed is skipped.

Usage:
    python scripts/backfill_filings.py --dry-run
    python scripts/backfill_filings.py                       # all deep tickers
    python scripts/backfill_filings.py --tickers NVDA PLTR
    python scripts/backfill_filings.py --filing-types 10-K,10-Q --count-per-type 4

Shared logic lives in ``src/sec/backfill.py`` (``FilingBackfiller``); this file
is a thin CLI wrapper. ``print`` is intentional — this is a ``scripts/`` tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sec.backfill import (  # noqa: E402
    DEFAULT_COUNTS_PER_TYPE,
    DEFAULT_FILING_TYPES,
    FilingBackfiller,
)
from src.storage.store import Store  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill substantive SEC filings for deep-watchlist tickers.",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to backfill (default: deep watchlist from configs/coverage.yaml).",
    )
    parser.add_argument(
        "--filing-types", default=",".join(DEFAULT_FILING_TYPES),
        help="Comma-separated filing types (default: 10-K,10-Q,8-K).",
    )
    parser.add_argument(
        "--count-per-type", type=int, default=None,
        help=(
            "Filings to fetch per type, most-recent-first. Default per type: "
            + ", ".join(f"{k}={v}" for k, v in DEFAULT_COUNTS_PER_TYPE.items())
            + ". A single value here overrides all types."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be ingested without registering or processing.",
    )
    return parser.parse_args(argv)


def _resolve_counts(count_per_type) -> dict:
    """A single --count-per-type overrides every type; otherwise per-type defaults."""
    if count_per_type is None:
        return dict(DEFAULT_COUNTS_PER_TYPE)
    return {form: count_per_type for form in DEFAULT_FILING_TYPES}


def _print_report(report: dict) -> None:
    """Print one line per filing plus a per-ticker summary table."""
    tickers = report["tickers"]
    for symbol in sorted(tickers):
        result = tickers[symbol]
        if result.get("error"):
            print(f"[{symbol}] TICKER FAILED: {result['error']}")
        for row in result.get("filings", []):
            print(
                f"[{row.get('ticker', symbol)}] "
                f"{str(row.get('filing_type') or '?'):<6} "
                f"{str(row.get('filing_date') or '----------'):<10} "
                f"{str(row.get('accession') or '-'):<22} "
                f"-> {row.get('status')}"
            )

    print()
    mode = "DRY RUN" if report.get("dry_run") else "BACKFILL"
    print(f"=== {mode} SUMMARY ===")
    header = f"{'ticker':<8} {'found':>6} {'ingest':>7} {'skip':>6} {'fail':>6}"
    if report.get("dry_run"):
        header += f" {'would':>6}"
    print(header)
    for symbol in sorted(tickers):
        result = tickers[symbol]
        line = (
            f"{symbol:<8} {int(result.get('discovered', 0)):>6} "
            f"{int(result.get('ingested', 0)):>7} "
            f"{int(result.get('skipped', 0)):>6} "
            f"{int(result.get('failed', 0)):>6}"
        )
        if report.get("dry_run"):
            line += f" {int(result.get('would_ingest', 0)):>6}"
        print(line)
    totals = report["totals"]
    print(
        f"{'TOTAL':<8} {totals['discovered']:>6} {totals['ingested']:>7} "
        f"{totals['skipped']:>6} {totals['failed']:>6}"
        + (f" {totals['would_ingest']:>6}" if report.get("dry_run") else "")
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    filing_types = tuple(
        part.strip().upper() for part in args.filing_types.split(",") if part.strip()
    )
    counts = _resolve_counts(args.count_per_type)

    store = Store()
    backfiller = FilingBackfiller(store)
    tickers = args.tickers or backfiller.deep_tickers()

    print(
        f"Backfilling {len(tickers)} ticker(s): {', '.join(tickers)}\n"
        f"Filing types: {', '.join(filing_types)}"
        + ("  (dry run)" if args.dry_run else "")
    )

    report = backfiller.run(
        tickers,
        filing_types=filing_types,
        counts_per_type=counts,
        dry_run=args.dry_run,
    )
    _print_report(report)

    # Exit non-zero only when EVERY ticker failed (none ingested or skipped
    # anything). A dry run is always informational -> exit 0.
    per_ticker = report["tickers"]
    if args.dry_run or not per_ticker:
        return 0
    any_success = any(
        int(r.get("ingested", 0)) or int(r.get("skipped", 0))
        for r in per_ticker.values()
    )
    return 0 if any_success else 1


if __name__ == "__main__":
    raise SystemExit(main())

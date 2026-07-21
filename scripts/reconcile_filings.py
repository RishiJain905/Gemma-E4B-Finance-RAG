"""
scripts/reconcile_filings.py
Periodic self-heal check for deep-watchlist SEC filing coverage.

For each deep ticker (configs/coverage.yaml sec_filing_text scope) this reports
whether it has (a) a 10-K or 10-Q filed within the recency window and (b) at
least one substantive filing whose text/chunks are indexed. Tickers missing
either are "gapped". With --repair, a bounded backfill (10-K + 10-Q) runs for
each gapped ticker through the same path scripts/backfill_filings.py uses.

Usage:
    python scripts/reconcile_filings.py
    python scripts/reconcile_filings.py --recency-days 120
    python scripts/reconcile_filings.py --repair
    python scripts/reconcile_filings.py --tickers NVDA PLTR --repair

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

from src.sec.backfill import FilingBackfiller, RECENCY_DAYS_DEFAULT  # noqa: E402
from src.storage.store import Store  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile deep-watchlist SEC filing coverage and optionally repair gaps.",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to check (default: deep watchlist from configs/coverage.yaml).",
    )
    parser.add_argument(
        "--recency-days", type=int, default=RECENCY_DAYS_DEFAULT,
        help=f"Max age of the newest 10-K/10-Q before a ticker is gapped (default {RECENCY_DAYS_DEFAULT}).",
    )
    parser.add_argument(
        "--repair", action="store_true",
        help="Run a bounded backfill (10-K + 10-Q) for each gapped ticker.",
    )
    return parser.parse_args(argv)


def _print_report(report: dict) -> None:
    gaps = report["gaps"]
    print(f"=== FILING RECONCILE (recency {report['recency_days']}d) ===")
    print(f"{'ticker':<8} {'status':<8} {'latest 10-K/Q':<14} {'reasons'}")
    for symbol in sorted(gaps):
        info = gaps[symbol]
        status = "GAP" if info["gap"] else "ok"
        latest = info.get("latest_periodic") or "-"
        reasons = ", ".join(info.get("reasons", [])) or "-"
        print(f"{symbol:<8} {status:<8} {str(latest):<14} {reasons}")

    gapped = report["gapped_tickers"]
    print()
    print(f"{len(gapped)} of {len(gaps)} tickers gapped: {', '.join(gapped) or '(none)'}")

    repairs = report.get("repairs") or {}
    if repairs:
        print("\n=== REPAIRS ===")
        for symbol in sorted(repairs):
            result = repairs[symbol]
            print(
                f"[{symbol}] ingested={int(result.get('ingested', 0))} "
                f"skipped={int(result.get('skipped', 0))} "
                f"failed={int(result.get('failed', 0))}"
            )


def main(argv=None) -> int:
    args = parse_args(argv)
    store = Store()
    backfiller = FilingBackfiller(store)

    report = backfiller.reconcile(
        args.tickers,
        recency_days=args.recency_days,
        repair=args.repair,
    )
    _print_report(report)

    # Without --repair this is a report (exit 0). With --repair, exit non-zero
    # if any gapped ticker still failed to ingest anything.
    if not args.repair:
        return 0
    repairs = report.get("repairs") or {}
    unresolved = any(
        not (int(r.get("ingested", 0)) or int(r.get("skipped", 0)))
        for r in repairs.values()
    )
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())

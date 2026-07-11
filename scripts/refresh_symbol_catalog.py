#!/usr/bin/env python3
"""scripts/refresh_symbol_catalog.py - Build the disk symbol catalog from SEC tickers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.sec.edgar_fetcher import (  # noqa: E402
    SECEdgarFilingFetcher,
    fetch_sec_company_tickers,
)


def _default_user_agent() -> str:
    return (
        os.environ.get("SEC_EDGAR_USER_AGENT")
        or SECEdgarFilingFetcher._load_config_user_agent()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the symbol catalog from SEC tickers.")
    parser.add_argument(
        "--output",
        default="data/symbol_catalog.json",
        help="Catalog output path.",
    )
    parser.add_argument(
        "--ttl-hours",
        type=int,
        default=168,
        help="Catalog TTL in hours.",
    )
    parser.add_argument(
        "--user-agent",
        default=_default_user_agent(),
        help="SEC EDGAR User-Agent.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)

    try:
        rows = fetch_sec_company_tickers(args.user_agent)
    except requests.exceptions.RequestException as exc:
        print(f"Failed to fetch SEC company tickers: {exc}", file=sys.stderr)
        return 1

    entries = [
        {"ticker": row["ticker"], "name": row["title"], "cik": row["cik"]}
        for row in rows
        if row.get("ticker") and row.get("title") and row.get("cik")
    ]
    catalog = {
        "generated_at": datetime.now(UTC).isoformat(),
        "ttl_hours": args.ttl_hours,
        "entries": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)} symbol catalog entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""scripts/migrate_phase2_3.py
Run the bounded, resumable Phase 2.3 identity and corpus metadata backfill.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.store import Store  # noqa: E402

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the migration CLI parser."""
    parser = argparse.ArgumentParser(
        description="Backfill Phase 2.3 metadata without re-embedding documents."
    )
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--chroma-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop cleanly after this many committed batches.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the resumable backfill and print a non-secret JSON summary."""
    args = build_parser().parse_args(argv)
    store = Store(db_path=args.db_path, chroma_path=args.chroma_path)
    result = store.migrate_phase2_3(
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


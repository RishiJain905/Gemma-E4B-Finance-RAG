"""scripts/rebuild_lexical_index.py
Bounded, resumable FTS5 rebuild and reconciliation without model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, help="SQLite database path")
    parser.add_argument("--chroma-path", type=Path, help="Chroma persistence path")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--restart", action="store_true", help="Discard saved rebuild progress")
    parser.add_argument(
        "--reconcile", action="store_true",
        help="Report lexical/Chroma identity and content drift instead of rebuilding",
    )
    parser.add_argument(
        "--repair", action="store_true",
        help="Repair reconciliation findings (implies --reconcile)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from src.storage.store import Store

    args = _parser().parse_args(argv)
    store = Store(db_path=args.db_path, chroma_path=args.chroma_path)
    if args.reconcile or args.repair:
        result = store.reconcile_lexical_index(
            repair=bool(args.repair), batch_size=args.batch_size,
        )
    else:
        result = store.rebuild_lexical_index(
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            restart=args.restart,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "degraded":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Emit resolved model serve settings as JSON for shell wrappers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.model_config import get_serve_settings  # noqa: E402


def main() -> int:
    try:
        settings = get_serve_settings()
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

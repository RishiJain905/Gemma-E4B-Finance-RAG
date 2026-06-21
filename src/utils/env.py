"""
src/utils/env.py
Zero-dependency .env loader.

Populates ``os.environ`` from a project-root ``.env`` file so credentials
(FRED_API_KEY, SEC_EDGAR_USER_AGENT) are available without requiring an
external dependency or a manually-exported shell environment.

Existing environment variables are never overwritten — a value already set
in the real environment takes precedence over the .env file.

Usage:
    from src.utils.env import load_env
    load_env()
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ENV_PATH = Path(__file__).parent.parent.parent / ".env"
_loaded = False


def load_env(path: Path = None, override: bool = False) -> dict:
    """Load key=value pairs from a .env file into ``os.environ``.

    Args:
        path: Path to the .env file (defaults to project-root ``.env``).
        override: If True, replace variables already set in the environment.

    Returns:
        A dict of the keys that were applied to ``os.environ``.
    """
    global _loaded
    env_path = Path(path) if path else _DEFAULT_ENV_PATH
    applied: dict[str, str] = {}

    if not env_path.exists():
        return applied

    try:
        with open(env_path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                # Strip surrounding quotes and trailing CR, keep inner content.
                value = value.strip().strip('"').strip("'").rstrip("\r")
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
                    applied[key] = value
    except Exception as e:  # noqa: BLE001 - never let env loading crash startup
        logger.warning("Failed to load .env from %s: %s", env_path, e)

    _loaded = True
    return applied

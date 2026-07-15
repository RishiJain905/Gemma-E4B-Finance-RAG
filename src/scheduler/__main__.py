"""src/scheduler/__main__.py
CLI shim for the explicit scheduler operations.

Supported operations are ``bootstrap``, ``daily``, ``repair``, ``retention``,
and ``status`` plus the legacy ``hourly``, ``weekly``, and ``all`` aliases.
"""

from src.utils.env import load_env

from . import main

if __name__ == "__main__":
    load_env()  # load .env credentials before any ingestion
    main()

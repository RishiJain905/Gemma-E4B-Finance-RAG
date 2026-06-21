"""CLI entry point: ``python -m src.scheduler <mode>``."""
from src.utils.env import load_env
from . import main

if __name__ == "__main__":
    load_env()  # load .env credentials before any ingestion
    main()

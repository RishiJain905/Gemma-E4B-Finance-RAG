#!/usr/bin/env python3
"""Validate the RAG system setup — configs, dependencies, endpoints.

Run:
    python scripts/validate_setup.py

Exits 1 if any hard errors are found; 0 if only warnings (e.g. model or
middleware not running).
"""

import sys
from pathlib import Path

errors: list[str] = []
warnings: list[str] = []


def check(condition, msg, is_error=False):
    if not condition:
        (errors if is_error else warnings).append(msg)


# 1. Project structure
PROJECT = Path(__file__).resolve().parent.parent
check((PROJECT / "src").is_dir(), "src/ directory missing", is_error=True)
check((PROJECT / "configs").is_dir(), "configs/ directory missing", is_error=True)
check((PROJECT / "data").is_dir(), "data/ directory missing")

# 2. Config files
for cfg in ["storage.yaml", "middleware.yaml", "watchlist.yaml",
            "fred.yaml", "gdelt.yaml", "ir.yaml", "model.yaml"]:
    check((PROJECT / "configs" / cfg).exists(), f"configs/{cfg} missing")

check((PROJECT / "configs" / "model.example.yaml").exists(),
      "configs/model.example.yaml missing")

# 3b. Model paths (for serve_model scripts)
try:
    sys.path.insert(0, str(PROJECT))
    from src.utils.model_config import get_serve_settings
    settings = get_serve_settings()
    check(Path(settings["main_model"]).exists(),
          f"Main model not found: {settings['main_model']} "
          "(set paths in configs/model.local.yaml)")
    check(Path(settings["server_exe"]).exists(),
          f"llama-server not found: {settings['server_exe']} "
          "(set paths.build_dir in configs/model.local.yaml)")
    print("OK   Model paths resolved from config")
except ValueError as e:
    warnings.append(
        f"Model paths not configured: {e}. "
        "Copy configs/model.example.yaml to configs/model.local.yaml."
    )
except Exception as e:  # noqa: BLE001
    warnings.append(f"Model config check failed: {e}")

# 3. Python dependencies
try:
    import yaml  # noqa: F401
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    import httpx  # noqa: F401
    import chromadb  # noqa: F401
    import yfinance  # noqa: F401
    print("OK   Core dependencies importable")
except ImportError as e:
    errors.append(f"Missing dependency: {e}")

# 4. Storage
try:
    sys.path.insert(0, str(PROJECT))
    from src.storage.store import Store
    store = Store()
    health = store.heartbeat()
    check(health.get("sqlite"), "SQLite health check failed", is_error=True)
    check(health.get("chroma"), "ChromaDB health check failed", is_error=True)
    print(f"OK   Storage (ChromaDB docs: {health.get('chroma_doc_count', 0)})")
except Exception as e:  # noqa: BLE001
    errors.append(f"Storage initialization failed: {e}")

# 5. Model endpoint
import httpx  # noqa: E402
try:
    r = httpx.get("http://127.0.0.1:8087/health", timeout=5)
    check(r.status_code == 200, f"Model health check: HTTP {r.status_code}")
    if r.status_code == 200:
        print("OK   Model endpoint reachable (:8087)")
except Exception as e:  # noqa: BLE001
    warnings.append(f"Model endpoint unreachable: {e}")

# 6. Middleware endpoint
try:
    r = httpx.get("http://127.0.0.1:8000/health", timeout=5)
    check(r.status_code == 200, f"Middleware health check: HTTP {r.status_code}")
    if r.status_code == 200:
        print("OK   Middleware endpoint reachable (:8000)")
except Exception as e:  # noqa: BLE001
    warnings.append(f"Middleware endpoint unreachable: {e}")

# Report
print(f"\n{'=' * 40}")
if errors:
    print(f"{len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
if warnings:
    print(f"{len(warnings)} warning(s):")
    for w in warnings:
        print(f"  - {w}")
if not errors and not warnings:
    print("All checks passed!")

sys.exit(1 if errors else 0)

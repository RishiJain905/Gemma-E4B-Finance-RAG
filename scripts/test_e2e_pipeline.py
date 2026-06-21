"""End-to-end pipeline smoke test: ingest -> store -> query -> answer.

Run:
    python scripts/test_e2e_pipeline.py [--force] [--no-middleware]

Steps:
    1. Run the daily ingestion via the UnifiedScheduler.
    2. Print the freshness report for NVDA.
    3. Run a hybrid search against the store.
    4. (Optional) Query the running FastAPI middleware on :8000.

Network/model failures are reported, not raised, so the script always
finishes with a readable summary.
"""

import argparse
import sys

from src.scheduler import UnifiedScheduler
from src.storage.store import Store

MIDDLEWARE_URL = "http://127.0.0.1:8000/query"


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E pipeline smoke test")
    parser.add_argument("--force", action="store_true",
                        help="Force ingestion regardless of cache freshness")
    parser.add_argument("--no-middleware", action="store_true",
                        help="Skip the live middleware query step")
    args = parser.parse_args()

    store = Store()
    sched = UnifiedScheduler(store=store)

    # 1. Run daily ingestion
    print("=== 1. Daily ingestion ===")
    try:
        results = sched.run_daily(force=args.force)
        for name, detail in results.items():
            print(f"  {name}: {detail.get('status')}")
    except Exception as e:  # noqa: BLE001
        print(f"  ingestion error: {e}")

    # 2. Freshness
    print("\n=== 2. Freshness (NVDA) ===")
    try:
        report = store.get_freshness_report("NVDA")
        print(f"  overall: {report['overall']}")
        for name, info in report["sources"].items():
            print(f"  {name}: {info['status']} (age={info.get('age_hours')}h)")
    except Exception as e:  # noqa: BLE001
        print(f"  freshness error: {e}")

    # 3. Search
    print("\n=== 3. Hybrid search ===")
    try:
        search_results = store.search("What is NVIDIA's revenue?")
        print(f"  found {len(search_results['documents'])} docs, "
              f"{len(search_results['facts'])} facts")
    except Exception as e:  # noqa: BLE001
        print(f"  search error: {e}")

    # 4. Query via middleware (optional)
    if not args.no_middleware:
        print("\n=== 4. Middleware query ===")
        try:
            import httpx
            resp = httpx.post(
                MIDDLEWARE_URL,
                json={"question": "What is NVIDIA's latest revenue?", "refresh": False},
                timeout=120,
            )
            answer = resp.json().get("answer", "")
            print(f"  HTTP {resp.status_code}: {answer[:200]}...")
        except Exception as e:  # noqa: BLE001
            print(f"  middleware not reachable: {e}")

    print("\nE2E pipeline run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

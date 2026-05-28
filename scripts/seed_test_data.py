#!/usr/bin/env python3
"""
scripts/seed_test_data.py
Phase 1.2 integration test + seed data loader.

Usage:
    python scripts/seed_test_data.py               # Run full test + seed
    python scripts/seed_test_data.py --seed-only   # Just seed data, skip tests
"""

import argparse
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.store import Store


SEED_FUNDAMENTALS = [
    ("NVDA", "revenue_q1_2026", 26.0, "usd", "2026-Q1", "quarterly", "earnings_call"),
    ("NVDA", "revenue_q2_2026", 29.8, "usd", "2026-Q2", "quarterly", "earnings_call"),
    ("NVDA", "datacenter_revenue_q1", 22.6, "usd", "2026-Q1", "quarterly", "earnings_call"),
    ("NVDA", "datacenter_revenue_q2", 26.1, "usd", "2026-Q2", "quarterly", "earnings_call"),
    ("NVDA", "gross_margin_q1", 0.737, "percent", "2026-Q1", "quarterly", "earnings_call"),
    ("NVDA", "gross_margin_q2", 0.745, "percent", "2026-Q2", "quarterly", "earnings_call"),
    ("NVDA", "pe_ratio_ttm", 38.5, "ratio", "2026-06", "ttm", "yfinance"),
    ("NVDA", "eps_ttm", 2.84, "usd", "2026-06", "ttm", "yfinance"),
    ("NVDA", "market_cap", 3200000000000, "usd", "2026-06", "point_in_time", "yfinance"),
    ("AMD", "revenue_q1_2026", 7.1, "usd", "2026-Q1", "quarterly", "earnings_call"),
    ("AMD", "revenue_q2_2026", 8.3, "usd", "2026-Q2", "quarterly", "earnings_call"),
    ("AMD", "datacenter_revenue_q1", 5.2, "usd", "2026-Q1", "quarterly", "earnings_call"),
    ("AMD", "datacenter_revenue_q2", 6.0, "usd", "2026-Q2", "quarterly", "earnings_call"),
    ("AMD", "gross_margin_q1", 0.512, "percent", "2026-Q1", "quarterly", "earnings_call"),
    ("AMD", "gross_margin_q2", 0.520, "percent", "2026-Q2", "quarterly", "earnings_call"),
    ("AMD", "pe_ratio_ttm", 42.1, "ratio", "2026-06", "ttm", "yfinance"),
    ("AMD", "eps_ttm", 1.12, "usd", "2026-06", "ttm", "yfinance"),
    ("AMD", "market_cap", 480000000000, "usd", "2026-06", "point_in_time", "yfinance"),
    ("META", "revenue_q1_2026", 42.5, "usd", "2026-Q1", "quarterly", "earnings_call"),
    ("META", "revenue_q2_2026", 47.1, "usd", "2026-Q2", "quarterly", "earnings_call"),
    ("META", "pe_ratio_ttm", 25.3, "ratio", "2026-06", "ttm", "yfinance"),
    ("META", "eps_ttm", 8.45, "usd", "2026-06", "ttm", "yfinance"),
    ("META", "market_cap", 1800000000000, "usd", "2026-06", "point_in_time", "yfinance"),
    ("CRWD", "revenue_q1_2026", 1.2, "usd", "2026-Q1", "quarterly", "earnings_call"),
    ("CRWD", "revenue_q2_2026", 1.35, "usd", "2026-Q2", "quarterly", "earnings_call"),
    ("CRWD", "pe_ratio_ttm", 78.4, "ratio", "2026-06", "ttm", "yfinance"),
    ("CRWD", "eps_ttm", 0.92, "usd", "2026-06", "ttm", "yfinance"),
    ("CRWD", "market_cap", 95000000000, "usd", "2026-06", "point_in_time", "yfinance"),
    ("AAPL", "revenue_q2_2026", 92.0, "usd", "2026-Q2", "quarterly", "earnings_call"),
    ("AAPL", "pe_ratio_ttm", 30.2, "ratio", "2026-06", "ttm", "yfinance"),
    ("AAPL", "eps_ttm", 6.58, "usd", "2026-06", "ttm", "yfinance"),
    ("AAPL", "market_cap", 3500000000000, "usd", "2026-06", "point_in_time", "yfinance"),
    ("MSFT", "revenue_q2_2026", 68.5, "usd", "2026-Q2", "quarterly", "earnings_call"),
    ("MSFT", "pe_ratio_ttm", 33.8, "ratio", "2026-06", "ttm", "yfinance"),
    ("MSFT", "eps_ttm", 11.92, "usd", "2026-06", "ttm", "yfinance"),
    ("MSFT", "market_cap", 3100000000000, "usd", "2026-06", "point_in_time", "yfinance"),
]

SEED_DOCUMENTS = [
    {
        "id": "seed/nvda-q2-earnings",
        "text": (
            "NVIDIA reported Q2 2026 revenue of $29.8 billion, exceeding analyst expectations of $28.5 billion. "
            "Data center revenue reached $26.1 billion, up 42% year-over-year, driven by continued demand for "
            "Blackwell GPUs used in AI training and inference workloads. The company guided Q3 revenue to "
            "$31.0-32.5 billion. Gross margins improved to 74.5%, up from 73.7% in Q1, reflecting favorable "
            "product mix toward higher-value data center products. CEO Jensen Huang stated that demand for "
            "AI computing continues to accelerate, with enterprise adoption broadening beyond cloud providers."
        ),
        "ticker": "NVDA",
        "source": "earnings_call",
        "date": "2026-08-20",
    },
    {
        "id": "seed/nvda-blackwell-analysis",
        "text": (
            "NVIDIA's Blackwell GPU architecture represents the biggest generational leap in the company's history. "
            "With 208 billion transistors and 4x the AI performance of Hopper, Blackwell is powering the next wave "
            "of AI model training. Major customers including Microsoft, Meta, and OpenAI have placed substantial orders. "
            "Supply chain constraints have been largely resolved, with TSMC's CoWoS packaging capacity increasing 50% "
            "year-over-year. The transition from Hopper to Blackwell is expected to drive an extended growth cycle "
            "through 2027 as enterprises upgrade their AI infrastructure."
        ),
        "ticker": "NVDA",
        "source": "analysis",
        "date": "2026-07-15",
    },
    {
        "id": "seed/amd-mi400-launch",
        "text": (
            "AMD announced the MI400 AI accelerator at Computex 2026, positioning it as a direct competitor to "
            "NVIDIA's Blackwell Ultra. The MI400 features 288GB of HBM4 memory and delivers 2.5 PFLOPS of FP8 "
            "performance. Early benchmark results show competitive performance on popular AI models. AMD has secured "
            "design wins with several major cloud providers. CEO Lisa Su emphasized AMD's open software strategy "
            "with ROCm 7.0, aiming to capture additional market share in the rapidly growing AI accelerator market."
        ),
        "ticker": "AMD",
        "source": "news",
        "date": "2026-06-03",
    },
    {
        "id": "seed/meta-ai-investment",
        "text": (
            "Meta Platforms increased its 2026 capital expenditure guidance to $65-70 billion, up from $55-60 billion, "
            "with the majority allocated to AI infrastructure including GPUs, data centers, and networking. "
            "CEO Mark Zuckerberg stated that AI investments are 'the highest priority' for the company, with "
            "applications spanning content recommendation, advertising optimization, and the development of "
            "general intelligence. Meta has deployed over 600,000 H100-equivalent GPUs and is rapidly expanding "
            "its AI research division."
        ),
        "ticker": "META",
        "source": "news",
        "date": "2026-07-30",
    },
    {
        "id": "seed/crowdstrike-q2",
        "text": (
            "CrowdStrike reported Q2 2026 revenue of $1.35 billion, representing 28% year-over-year growth. "
            "Annual recurring revenue (ARR) reached $5.2 billion. The company added 1,500 new subscription customers "
            "during the quarter, bringing total customers to over 35,000. CrowdStrike's Falcon platform continues to "
            "gain market share in the cybersecurity space, with particular strength in the public sector and financial "
            "services verticals. Net dollar retention remained above 120%, indicating strong upsell momentum."
        ),
        "ticker": "CRWD",
        "source": "earnings_call",
        "date": "2026-08-25",
    },
    {
        "id": "seed/fed-rate-decision",
        "text": (
            "The Federal Reserve kept the federal funds rate unchanged at 4.50% at its May 2026 meeting, "
            "citing persistent inflation pressures in the services sector. The dot plot indicated two potential "
            "rate cuts in the second half of 2026, contingent on inflation continuing its gradual decline. "
            "Fed Chair Jerome Powell emphasized the committee's data-dependent approach and noted that the "
            "labor market remains strong with unemployment at 3.8%. Market expectations for a September cut "
            "rose to 65% following the announcement."
        ),
        "ticker": None,
        "source": "fred",
        "date": "2026-05-07",
    },
]


def run_integration_tests(store: Store) -> bool:
    """Run integration tests against the storage layer."""
    passed = 0
    failed = 0

    def check(name: str, condition: bool) -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS {name}")
            passed += 1
        else:
            print(f"  FAIL {name}")
            failed += 1

    print("\n-- Integration Tests --\n")

    health = store.heartbeat()
    check("SQLite health", health.get("sqlite") is True)
    check("ChromaDB health", health.get("chroma") is True)

    store.save_fundamental("TEST", "unit_test", 42.0, period="2026-T1")
    fact = store.get_fundamental("TEST", "unit_test")
    check("Save & retrieve fundamental", bool(fact and fact["value"] == 42.0))
    check("Fact has correct ticker", bool(fact and fact["ticker"] == "TEST"))
    check("Fact has correct period", bool(fact and fact["period"] == "2026-T1"))

    store.save_fundamental("TEST", "unit_test", 99.0, period="2026-T1")
    fact = store.get_fundamental("TEST", "unit_test")
    check("Update fundamental", bool(fact and fact["value"] == 99.0))

    store.save_fundamental("TEST", "unit_test_2", 100.0, period="2026-T1")
    latest = store.get_fundamental("TEST", "unit_test_2")
    check("Get latest fundamental", bool(latest and latest["value"] == 100.0))

    facts = store.sqlite.search_facts(ticker="TEST")
    check("Search facts by ticker", len(facts) >= 2)

    doc_id = store.save_document(
        "test/integration-doc",
        "This is a test document about financial markets and investment strategies.",
        ticker="TEST",
        source="test",
        date="2026-01-01",
    )
    check("Save document to ChromaDB", doc_id == "test/integration-doc")
    check("Document count > 0", store.chroma.count() > 0)

    results = store.search("TEST financial markets investment")
    check("Semantic search returns results", len(results["documents"]) > 0)
    check("Search includes ticker detection", results.get("ticker") is not None)

    detected = store._detect_ticker("NVDA")
    check("Ticker detection: NVDA", detected == "NVDA")
    detected = store._detect_ticker("What is Apple's revenue?")
    check("Ticker detection: AAPL", detected == "AAPL")
    detected = store._detect_ticker("Tell me about the Fed rate decision")
    check("Ticker detection: None for macro query", detected is None)

    filed = store.register_filing(
        "NVDA",
        "10-Q",
        "2026-05-15",
        "2026-Q1",
        "test-accession-001",
        "https://sec.gov/...",
    )
    check("Register filing returns True for new", filed is True)

    duplicate = store.register_filing(
        "NVDA",
        "10-Q",
        "2026-05-15",
        "2026-Q1",
        "test-accession-001",
        "https://sec.gov/...",
    )
    check("Register filing returns False for duplicate", duplicate is False)

    store.process_filing(
        {
            "ticker": "TEST",
            "source_type": "sec",
            "filing_type": "10-K",
            "period": "2025-FY",
            "filing_date": "2025-12-31",
            "accession": "test-accession-002",
            "source_url": "https://sec.gov/test",
        },
        extracted_text="Annual report showing strong revenue growth across all segments.",
        extracted_facts=[
            {"metric": "annual_revenue", "value": 100.0, "unit": "usd", "period": "2025-FY"},
            {"metric": "annual_profit", "value": 25.0, "unit": "usd", "period": "2025-FY"},
        ],
    )
    check(
        "Process filing stores document",
        store.chroma.get_document("sec/TEST/10-K-2025-FY") is not None,
    )
    fact = store.get_fundamental("TEST", "annual_revenue")
    check("Process filing extracts facts", bool(fact and fact["value"] == 100.0))

    store.mark_cache_fresh("NVDA", "yfinance", 24)
    status = store.sqlite.get_cache_status("NVDA", "yfinance")
    check("Cache status exists", status is not None)
    check("Cache status is fresh", bool(status and status["status"] == "fresh"))

    real_results = store.search("NVIDIA data center performance")
    check("Real search returns documents", len(real_results["documents"]) > 0)
    check("Real search detects ticker", real_results["ticker"] is not None)

    # Cleanup test artifacts created by integration checks.
    store.chroma.delete_document("test/integration-doc")
    store.chroma.delete_document("sec/TEST/10-K-2025-FY")
    with store.sqlite._connect() as conn:
        conn.execute("DELETE FROM fundamentals WHERE ticker='TEST'")
        conn.execute("DELETE FROM filings WHERE accession IN ('test-accession-001', 'test-accession-002')")
        conn.commit()

    print(f"\n-- Results: {passed} passed, {failed} failed --\n")
    return failed == 0


def seed_data(store: Store) -> None:
    """Load seed fundamentals and documents into the database."""
    print("\n-- Seeding Data --\n")

    count = 0
    for row in SEED_FUNDAMENTALS:
        ticker, metric, value, unit, period, period_type, source_type = row
        store.save_fundamental(
            ticker=ticker,
            metric=metric,
            value=value,
            unit=unit,
            period=period,
            period_type=period_type,
            source_type=source_type,
        )
        count += 1
    print(f"  PASS Seeded fundamentals: {count}")

    doc_count = 0
    for doc in SEED_DOCUMENTS:
        store.save_document(
            document_id=doc["id"],
            text=doc["text"],
            ticker=doc["ticker"],
            source=doc["source"],
            date=doc["date"],
        )
        doc_count += 1
    print(f"  PASS Seeded documents: {doc_count}")

    seeded_tickers = ["NVDA", "AMD", "META", "CRWD", "AAPL", "MSFT"]
    for ticker in seeded_tickers:
        store.mark_cache_fresh(ticker, "seed", 168)
    print(f"  PASS Cache marked fresh for tickers: {len(seeded_tickers)}")

    count_after = len(store.get_fundamentals_batch("NVDA", ["revenue_q1_2026"]))
    print(f"  INFO NVDA fundamentals in DB: {count_after}")
    chroma_count = store.chroma.count()
    print(f"  INFO Documents in ChromaDB: {chroma_count}")
    print("\n-- Seeding Complete --\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed test data for Phase 1.2")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Skip tests and only seed data",
    )
    args = parser.parse_args()

    store = Store()

    if args.seed_only:
        seed_data(store)
        return

    success = run_integration_tests(store)
    if success:
        print("PASS All tests passed. Proceeding to seed data...")
        seed_data(store)
    else:
        print("FAIL Tests failed. Fix issues before seeding data.")
        sys.exit(1)


if __name__ == "__main__":
    main()

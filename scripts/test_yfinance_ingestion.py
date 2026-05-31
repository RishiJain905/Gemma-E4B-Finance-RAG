#!/usr/bin/env python3
"""
scripts/test_yfinance_ingestion.py
Phase 1.3 integration test for Yahoo Finance ingestion.

Usage:
    python scripts/test_yfinance_ingestion.py           # Full test suite
    python scripts/test_yfinance_ingestion.py --quick   # Skip ChromaDB news checks (hybrid search still embeds)
    python scripts/test_yfinance_ingestion.py --live    # Include live fetch tests (calls Yahoo)

Requires:
    - llama-server running on :8087 (news embedding and hybrid search query embeddings)
    - pip install yfinance
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.ingestion.yfinance_ingestor import YFinanceIngestor
from src.storage.store import Store


def test_basic_config(ingestor):
    """Test that the ingestor loads configuration correctly."""
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1

    print("\n── Config Tests ──\n")

    check("Watchlist loaded", len(ingestor.all_tickers) > 0)
    check("Core tickers defined", len(ingestor.core_tickers) >= 3)
    check("NVDA is in core", "NVDA" in ingestor.core_tickers)
    check("All tickers > core tickers", len(ingestor.all_tickers) > len(ingestor.core_tickers))
    check("Schedule has fundamentals TTL", ingestor.watchlist.get("schedule", {}).get("fundamentals") is not None)
    check("Schedule has news TTL", ingestor.watchlist.get("schedule", {}).get("news") is not None)

    print(f"\n── Config Results: {passed} passed, {failed} failed ──\n")
    return failed == 0


def test_ticker_fetch(ingestor, live_tests: bool = True):
    """Test that yfinance ticker fetches work."""
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1

    print("── Ticker Fetch Tests ──\n")

    t = ingestor._fetch_ticker("NVDA")
    check("Fetch NVDA returns Ticker object", t is not None)
    if t:
        check("NVDA info has regularMarketPrice", t.info.get("regularMarketPrice") is not None)
        check("NVDA has market cap", t.info.get("marketCap") is not None)
        check("NVDA has trailing PE", t.info.get("trailingPE") is not None)

    t = ingestor._fetch_ticker("ZZZXXXINVALID")
    check("Invalid ticker returns None", t is None)

    if live_tests:
        t = ingestor._fetch_ticker("AAPL")
        if t:
            news = t.news
            check("AAPL has news articles", news is not None and len(news) > 0)
            if news:
                first = news[0]
                fields = ingestor._extract_news_fields(first)
                check("News has title", bool(fields and fields["title"]))
                check("News has publisher", bool(fields and fields["publisher"]))
                check("News has link", bool(fields and fields["link"]))

    t = ingestor._fetch_ticker("SPY")
    if t:
        check("SPY (macro) fetches successfully", t.info.get("regularMarketPrice") is not None)

    print(f"\n── Ticker Fetch Results: {passed} passed, {failed} failed ──\n")
    return failed == 0


def test_fundamentals_ingestion(ingestor, store):
    """Test that fundamentals are fetched and stored correctly."""
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1

    print("── Fundamentals Ingestion Tests ──\n")

    store.mark_cache_stale("NVDA", "yfinance_fundamentals")

    t = ingestor._fetch_ticker("NVDA")
    check("Fetch NVDA for fundamentals", t is not None)
    if t:
        ingestor._ingest_ticker_fundamentals("NVDA", t)

        facts = store.get_fundamentals_batch("NVDA")
        check("Fundamentals stored for NVDA", len(facts) > 0)

        for metric in ["market_cap", "pe_ratio_ttm", "eps_ttm"]:
            fact = store.get_fundamental("NVDA", metric)
            check(f"  {metric} stored: {fact['value'] if fact else 'MISSING'}", fact is not None)

        status = store.sqlite.get_cache_status("NVDA", "yfinance_fundamentals")
        check("Cache marked fresh for NVDA fundamentals", status is not None and status["status"] == "fresh")

    store.mark_cache_stale("NVDA", "yfinance_fundamentals")
    ingestor._ingest_ticker_fundamentals("NVDA", t)
    status = store.sqlite.get_cache_status("NVDA", "yfinance_fundamentals")
    check("Re-ingestion updates cache fresh again", status is not None and status["status"] == "fresh")

    print(f"\n── Fundamentals Results: {passed} passed, {failed} failed ──\n")
    return failed == 0


def test_news_ingestion(ingestor, store, skip_embedding: bool = False):
    """Test that news is fetched and stored correctly."""
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1

    print("── News Ingestion Tests ──\n")

    store.mark_cache_stale("AAPL", "yfinance_news")

    t = ingestor._fetch_ticker("AAPL")
    check("Fetch AAPL for news", t is not None)
    if t:
        ingestor._ingest_ticker_news("AAPL", t)

        if not skip_embedding:
            news_docs = store.chroma.get_ticker_documents("AAPL", source="yfinance_news", limit=10)
            check(f"News stored in ChromaDB for AAPL ({len(news_docs)} found)", len(news_docs) > 0)
            if news_docs:
                check("News doc has correct ID prefix", news_docs[0]["id"].startswith("news/"))
                check("News doc has metadata", news_docs[0]["metadata"] is not None)

        status = store.sqlite.get_cache_status("AAPL", "yfinance_news")
        check("Cache marked fresh for AAPL news", status is not None and status["status"] == "fresh")

    if not skip_embedding and t:
        existing_count = store.chroma.count()
        ingestor._ingest_ticker_news("AAPL", t)
        count_after = store.chroma.count()
        check(f"News dedup: count unchanged ({existing_count} → {count_after})", count_after == existing_count)

    print(f"\n── News Results: {passed} passed, {failed} failed ──\n")
    return failed == 0


def test_cache_system(ingestor, store):
    """Test cache freshness, staleness, and reset."""
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1

    print("── Cache System Tests ──\n")

    report = ingestor.status_report()
    check("Status report returns dict", isinstance(report, dict))
    check("Status has fresh count", "fresh" in report)
    check("Status has stale count", "stale" in report)
    check("Status has details", len(report.get("details", [])) > 0)

    ingestor.reset_cache_all()
    report = ingestor.status_report()
    staleness = sum(
        1
        for d in report.get("details", [])
        if d.get("fundamentals") == "stale" or d.get("news") == "stale"
    )
    check("After reset, tickers are stale (not fresh)", staleness > 0 or report["fresh"] == 0)

    ingestor.ingest_stale_only()

    status = store.sqlite.get_cache_status("NVDA", "yfinance_fundamentals")
    check("After stale-only, NVDA fundamentals are fresh", status and status["status"] == "fresh")

    report = ingestor.status_report()
    check("Status shows most tickers fresh after full stale-only run", report["fresh"] > 0)

    print(f"\n── Cache System Results: {passed} passed, {failed} failed ──\n")
    return failed == 0


def test_hybrid_search_integration(store):
    """Test that the ingested data is retrievable through hybrid search."""
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1

    print("── Hybrid Search Integration Tests ──\n")

    results = store.search("NVDA")
    check("Hybrid search for NVDA returns results", len(results["documents"]) > 0 or len(results["facts"]) > 0)
    if results.get("ticker"):
        check("Search detected ticker", results["ticker"] == "NVDA")

    results = store.search("Apple earnings")
    if results.get("ticker"):
        check("Company name detection works (Apple → AAPL)", results["ticker"] == "AAPL")

    results = store.search("latest news about AI chips")
    check("News search returns any results", len(results["documents"]) > 0)

    results = store.search("S&P 500 market performance today")
    check("Macro search doesn't crash", isinstance(results, dict))

    facts = store.get_fundamentals_batch("NVDA")
    check("NVDA fundamentals batch is accessible", len(facts) > 0)

    print(f"\n── Hybrid Search Results: {passed} passed, {failed} failed ──\n")
    return failed == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1.3 Yahoo Finance ingestion tests")
    parser.add_argument("--quick", action="store_true", help="Skip embedding-dependent news/Chroma checks")
    parser.add_argument("--live", action="store_true", help="Run live fetch tests (hits Yahoo API)")
    args = parser.parse_args()

    ingestor = YFinanceIngestor()
    store = Store()

    results = []

    print("╔══════════════════════════════════════════════╗")
    print("║   Phase 1.3 — Yahoo Finance Ingestion Tests  ║")
    print("╚══════════════════════════════════════════════╝")

    results.append(test_basic_config(ingestor))
    results.append(test_ticker_fetch(ingestor, live_tests=args.live))
    results.append(test_fundamentals_ingestion(ingestor, store))
    results.append(test_news_ingestion(ingestor, store, skip_embedding=args.quick))
    results.append(test_cache_system(ingestor, store))
    results.append(test_hybrid_search_integration(store))

    total = len(results)
    passed_total = sum(1 for r in results if r)
    failed_total = total - passed_total

    print("╔══════════════════════════════════════════════╗")
    print(f"║   Final: {passed_total}/{total} test suites passed               ║")
    print("╚══════════════════════════════════════════════╝")

    return 0 if failed_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

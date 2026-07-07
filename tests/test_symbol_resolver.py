"""Offline tests for the SymbolResolver module."""

from __future__ import annotations

import builtins
import json


def _write_catalog(tmp_path):
    catalog_path = tmp_path / "symbol_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "generated_at": "2099-01-01T00:00:00+00:00",
                "ttl_hours": 168,
                "entries": [
                    {"ticker": "BB", "name": "BlackBerry Limited", "cik": "0001070235"},
                    {"ticker": "DAL", "name": "Delta Air Lines", "cik": "0000027904"},
                    {"ticker": "NVDA", "name": "NVIDIA Corp", "cik": "0001045810"},
                    {"ticker": "ARE", "name": "Alexandria Real Estate Equities", "cik": "0001035443"},
                    {"ticker": "TGT", "name": "Target Corp", "cik": "0000027419"},
                    {"ticker": "P", "name": "Everpure, Inc.", "cik": "0000099999"},
                    {"ticker": "E", "name": "Eni S.p.A.", "cik": "0000098888"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return catalog_path


class TestSymbolResolver:
    """Symbol resolution tests use only local maps and temporary catalogs."""

    def test_local_map_still_works(self, tmp_path):
        """Existing local company-name and ticker detection stay exact."""
        from src.middleware.symbol_resolver import SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))

        name_result = resolver.resolve("what is nvidia revenue")
        assert name_result.ticker == "NVDA", "Expected nvidia local map to resolve NVDA"
        assert name_result.confidence == 1.0, "Local map should be full confidence"
        assert name_result.source == "local_map", "Expected local map source for nvidia"

        ticker_result = resolver.resolve("What is NVDA revenue?")
        assert ticker_result.ticker == "NVDA", "Expected known ticker NVDA to resolve"
        assert ticker_result.confidence == 1.0, "Known ticker should be full confidence"
        assert ticker_result.source == "known_ticker", "Expected known ticker source"

    def test_catalog_exact_name(self, tmp_path):
        """A catalog company short name resolves without fuzzy matching."""
        from src.middleware.symbol_resolver import SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        result = resolver.resolve("BlackBerry")

        assert result.ticker == "BB", "Expected BlackBerry catalog entry to resolve BB"
        assert result.source == "catalog_exact", "Expected exact catalog source"
        assert result.resolved_name == "BlackBerry Limited", "Expected catalog title"
        assert result.confidence == 0.95, "Catalog exact confidence should be 0.95"

    def test_fuzzy_typo(self, tmp_path):
        """A typo above the threshold resolves through rapidfuzz."""
        from src.middleware.symbol_resolver import SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        result = resolver.resolve("blackbarry")

        assert result.ticker == "BB", "Expected blackbarry typo to resolve BB"
        assert result.confidence >= 0.86, "Expected fuzzy confidence above threshold"
        assert result.source == "fuzzy", "Expected fuzzy source for typo"

    def test_single_letter_tickers_never_match_prose(self, tmp_path):
        """'P/E' in a question must resolve BB, never single-letter tickers P/E."""
        from src.middleware.symbol_resolver import SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        result = resolver.resolve("What is BB's forward P/E?")

        assert result.ticker == "BB", (
            f"Expected BB from the ticker candidates, got {result.ticker}"
        )
        assert result.source == "catalog_exact", "Expected exact catalog source"

    def test_fuzzy_typo_inside_question(self, tmp_path):
        """A typo'd name embedded in a longer question still resolves."""
        from src.middleware.symbol_resolver import SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        result = resolver.resolve("what is blackbarry forward pe")

        assert result.ticker == "BB", "Expected embedded blackbarry typo to resolve BB"
        assert result.source == "fuzzy", "Expected fuzzy source for embedded typo"

    def test_below_threshold_returns_none(self, tmp_path):
        """Unrelated gibberish returns the shared no-match response."""
        from src.middleware.symbol_resolver import NO_MATCH, SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        result = resolver.resolve("qzxwvut jklmno")

        assert result == NO_MATCH, "Expected gibberish to return NO_MATCH"

    def test_ambiguous_prefers_known(self, tmp_path):
        """A known ticker wins before a similar catalog name is considered."""
        from src.middleware.symbol_resolver import SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        result = resolver.resolve("Compare NVDA with nvidi corp")

        assert result.ticker == "NVDA", "Expected known ticker to win deterministically"
        assert result.source == "known_ticker", "Expected known ticker source"
        assert result.confidence == 1.0, "Known ticker should beat fuzzy candidates"

    def test_short_query_no_fuzzy(self, tmp_path):
        """Very short token-only queries do not attempt fuzzy guesses."""
        from src.middleware.symbol_resolver import NO_MATCH, SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        result = resolver.resolve("ab cd")

        assert result == NO_MATCH, "Expected short query to skip fuzzy and miss"

    def test_missing_rapidfuzz_degrades(self, tmp_path, monkeypatch):
        """Import failures in optional rapidfuzz return no match without raising."""
        from src.middleware.symbol_resolver import NO_MATCH, SymbolResolver

        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "rapidfuzz":
                raise ImportError("rapidfuzz unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        result = resolver.resolve("blackbarry")

        assert result == NO_MATCH, "Expected missing rapidfuzz to degrade to NO_MATCH"

    def test_missing_catalog_file(self, tmp_path):
        """A missing catalog never breaks existing local resolution."""
        from src.middleware.symbol_resolver import SymbolResolver

        resolver = SymbolResolver(catalog_path=tmp_path / "missing.json")
        result = resolver.resolve("nvidia")

        assert result.ticker == "NVDA", "Expected local map to work without catalog"
        assert result.source == "local_map", "Expected local map source without catalog"

    def test_common_word_ticker_not_matched_in_prose(self, tmp_path):
        """Catalog symbols that are English words (ARE, IT, ALL) never match prose."""
        from src.middleware.symbol_resolver import NO_MATCH, SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        result = resolver.resolve("ARE stocks overvalued right now")

        assert result == NO_MATCH, "Expected uppercase English word ARE not to resolve"

    def test_ambiguous_alias_skipped_in_prose_but_exact_query_resolves(self, tmp_path):
        """'target' inside a question never matches TGT; the bare name still does."""
        from src.middleware.symbol_resolver import NO_MATCH, SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))

        prose = resolver.resolve("what is the analyst price target consensus")
        assert prose == NO_MATCH, "Expected 'price target' prose not to resolve TGT"

        exact = resolver.resolve("target")
        assert exact.ticker == "TGT", "Expected whole-query 'target' to resolve TGT"
        assert exact.source == "catalog_exact", "Expected exact catalog source"

    def test_negative_cache_short_circuits_repeated_misses(self, tmp_path, monkeypatch):
        """A repeated miss returns before catalog loading while the negative TTL is live."""
        from src.middleware.symbol_resolver import NO_MATCH, SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path), negative_ttl_s=60.0)
        first = resolver.resolve("qzxwvut jklmno")
        assert first == NO_MATCH, "Expected first miss to populate negative cache"

        def fail_if_called():
            raise AssertionError("Catalog should not load for negative-cached query")

        monkeypatch.setattr(resolver, "_ensure_catalog_loaded", fail_if_called)
        second = resolver.resolve("qzxwvut jklmno")

        assert second == NO_MATCH, "Expected second miss to use negative cache"

    def test_negative_cache_expires(self, tmp_path):
        """After the negative TTL elapses, the resolution chain runs again."""
        import time

        from src.middleware.symbol_resolver import NO_MATCH, SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path), negative_ttl_s=0.01)
        calls = []
        original_fuzzy = resolver._fuzzy

        def counting_fuzzy(text):
            calls.append(text)
            return original_fuzzy(text)

        resolver._fuzzy = counting_fuzzy

        assert resolver.resolve("qzxwvut jklmno") == NO_MATCH
        assert len(calls) == 1, "Expected first miss to run the full chain"
        resolver.resolve("qzxwvut jklmno")
        assert len(calls) == 1, "Expected cached miss to skip the chain"

        time.sleep(0.02)
        resolver.resolve("qzxwvut jklmno")
        assert len(calls) == 2, "Expected expired negative entry to re-run the chain"

    def test_malformed_catalog_degrades(self, tmp_path):
        """Corrupt catalog JSON never raises; local resolution still works."""
        from src.middleware.symbol_resolver import NO_MATCH, SymbolResolver

        bad_path = tmp_path / "symbol_catalog.json"
        bad_path.write_text("{not valid json", encoding="utf-8")

        resolver = SymbolResolver(catalog_path=bad_path)
        assert resolver.resolve("nvidia").ticker == "NVDA", (
            "Expected local map to survive a corrupt catalog"
        )
        assert resolver.resolve("BlackBerry") == NO_MATCH, (
            "Expected catalog lookups to miss cleanly with a corrupt catalog"
        )

    def test_resolve_never_touches_network(self, tmp_path, monkeypatch):
        """resolve() must work with all requests/urllib entry points disabled."""
        import requests

        from src.middleware.symbol_resolver import SymbolResolver

        def no_network(*_args, **_kwargs):
            raise AssertionError("resolve() must not perform network I/O")

        monkeypatch.setattr(requests, "get", no_network)
        monkeypatch.setattr(requests, "request", no_network)
        monkeypatch.setattr("urllib.request.urlopen", no_network)

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        assert resolver.resolve("nvidia").ticker == "NVDA"
        assert resolver.resolve("BlackBerry").ticker == "BB"
        assert resolver.resolve("blackbarry").ticker == "BB"
        assert resolver.resolve("qzxwvut jklmno").ticker is None

    def test_concurrent_first_load_is_consistent(self, tmp_path):
        """Parallel first-time resolutions never see a partially-loaded catalog."""
        import threading

        from src.middleware.symbol_resolver import SymbolResolver

        resolver = SymbolResolver(catalog_path=_write_catalog(tmp_path))
        results: list = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(resolver.resolve("BlackBerry").ticker)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == ["BB"] * 8, (
            f"Expected every concurrent resolve to return BB, got {results}"
        )

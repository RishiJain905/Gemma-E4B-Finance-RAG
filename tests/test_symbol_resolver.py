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

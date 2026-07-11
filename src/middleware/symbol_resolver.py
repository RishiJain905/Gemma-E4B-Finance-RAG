"""
src/middleware/symbol_resolver.py
Offline symbol resolution from local maps, a disk catalog, and optional fuzzy matching.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATALOG_PATH = _REPO_ROOT / "data" / "symbol_catalog.json"
_DEFAULT_FUZZY_THRESHOLD = 0.86
# Single-token company aliases that are also ordinary finance-prose words
# ("price target", "the gap between", ...). These only resolve when the whole
# query is exactly the name — never as a substring of a longer question.
_AMBIGUOUS_ALIASES = {
    "target",
    "gap",
    "first",
    "key",
    "core",
    "main",
    "best",
    "united",
    "national",
    "american",
    "general",
    "global",
    "shell",  # "shell company"
    "box",    # "box spread", "box office"
}

_LEGAL_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "llc",
    "plc",
}


@dataclass(frozen=True)
class Resolution:
    """Ticker resolution result with confidence and source metadata."""

    ticker: Optional[str]
    confidence: float
    resolved_name: Optional[str]
    source: str


NO_MATCH = Resolution(None, 0.0, None, "none")


class SymbolResolver:
    """Resolve company names and ticker-like text without network calls."""

    def __init__(
        self,
        catalog_path: Path | None = None,
        fuzzy_threshold: float = _DEFAULT_FUZZY_THRESHOLD,
        negative_ttl_s: float = 300.0,
    ) -> None:
        self.catalog_path = catalog_path or _DEFAULT_CATALOG_PATH
        self.fuzzy_threshold = fuzzy_threshold
        self.negative_ttl_s = negative_ttl_s
        self._negative_cache: dict[str, float] = {}
        self._catalog_lock = threading.Lock()
        self._catalog_loaded = False
        self._name_lookup: dict[str, tuple[str, str]] = {}
        self._ticker_lookup: dict[str, str] = {}
        self._catalog_names: list[str] = []
        self._catalog_name_to_entry: dict[str, tuple[str, str]] = {}

    def resolve(self, text: str) -> Resolution:
        """Resolve text to a ticker, or return NO_MATCH."""
        normalized_query = _normalize(text)
        if not normalized_query:
            return NO_MATCH

        now = time.monotonic()
        cached_until = self._negative_cache.get(normalized_query)
        if cached_until is not None:
            if cached_until > now:
                return NO_MATCH
            self._negative_cache.pop(normalized_query, None)

        for resolver in (self._local, self._catalog_exact, self._fuzzy):
            result = resolver(text)
            if result.ticker:
                return result

        self._cache_negative(normalized_query, now)
        return NO_MATCH

    def _local(self, text: str) -> Resolution:
        """Resolve using IntentParser's hardcoded map and known ticker set."""
        from .intent_parser import IntentParser

        normalized = text.lower()

        for company_name, ticker in IntentParser.COMPANY_TO_TICKER.items():
            if company_name in normalized:
                return Resolution(ticker, 1.0, company_name, "local_map")

        candidates = set(re.findall(r"\b[A-Z]{1,5}\b", text))
        for candidate in candidates:
            if candidate in IntentParser.KNOWN_TICKERS:
                return Resolution(candidate, 1.0, candidate, "known_ticker")

        return NO_MATCH

    def _catalog_exact(self, text: str) -> Resolution:
        """Resolve exact catalog names, aliases, and ticker symbols."""
        self._ensure_catalog_loaded()

        normalized_query = _normalize(text)
        if not normalized_query:
            return NO_MATCH

        ticker_match = self._ticker_from_text(text)
        if ticker_match:
            resolved_name = self._ticker_lookup[ticker_match]
            return Resolution(ticker_match, 0.95, resolved_name, "catalog_exact")

        entry = self._name_lookup.get(normalized_query)
        if not entry:
            entry = self._find_name_in_query(normalized_query)
        if not entry:
            return NO_MATCH

        ticker, resolved_name = entry
        return Resolution(ticker, 0.95, resolved_name, "catalog_exact")

    def _fuzzy(self, text: str) -> Resolution:
        """Resolve a likely company-name typo through rapidfuzz if available."""
        self._ensure_catalog_loaded()
        if not self._catalog_names or not self._passes_fuzzy_guards(text):
            return NO_MATCH

        try:
            from rapidfuzz import fuzz, process
        except Exception:
            return NO_MATCH

        normalized = _normalize(text)
        try:
            match = process.extractOne(
                normalized,
                self._catalog_names,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.fuzzy_threshold * 100,
            )
            # Whole-query matching misses a typo'd name embedded in a longer
            # question ("what is blackbarry forward pe"): token_set_ratio
            # needs an exact token overlap and the full-string ratio is
            # diluted by the other words. Second pass: match each substantial
            # token on plain fuzz.ratio, which (unlike token_set_ratio) does
            # not score 100 when one side is a token subset of the other
            # ("forward" vs "forward industries").
            if not match:
                for token in re.findall(r"[a-z]+", normalized):
                    if len(token) < 5:
                        continue
                    token_match = process.extractOne(
                        token,
                        self._catalog_names,
                        scorer=fuzz.ratio,
                        score_cutoff=self.fuzzy_threshold * 100,
                    )
                    if token_match and (not match or token_match[1] > match[1]):
                        match = token_match
        except Exception:
            return NO_MATCH

        if not match:
            return NO_MATCH

        name, score, _ = match
        ticker, resolved_name = self._catalog_name_to_entry[name]
        return Resolution(ticker, float(score) / 100.0, resolved_name, "fuzzy")

    def _ensure_catalog_loaded(self) -> None:
        if self._catalog_loaded:
            return
        with self._catalog_lock:
            if self._catalog_loaded:
                return
            try:
                self._load_catalog()
            finally:
                # Set last so concurrent resolvers never observe the flag
                # while the lookups are still being built.
                self._catalog_loaded = True

    def _load_catalog(self) -> None:
        try:
            raw = self.catalog_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except FileNotFoundError:
            logger.debug("Symbol catalog not found: %s", self.catalog_path)
            return
        except Exception as exc:
            logger.warning("Failed to load symbol catalog %s: %s", self.catalog_path, exc)
            return

        self._warn_if_catalog_expired(data)

        entries = data.get("entries", [])
        if not isinstance(entries, list):
            logger.warning("Symbol catalog %s has invalid entries", self.catalog_path)
            return

        # Build into locals and publish atomically at the end.
        name_lookup: dict[str, tuple[str, str]] = {}
        ticker_lookup: dict[str, str] = {}
        catalog_names: list[str] = []
        name_to_entry: dict[str, tuple[str, str]] = {}

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker", "")).upper().strip()
            name = str(entry.get("name", "")).strip()
            if not ticker or not name:
                continue

            ticker_lookup[ticker] = name
            for alias in _catalog_aliases(name):
                name_lookup.setdefault(alias, (ticker, name))
                # Ambiguous single-word aliases stay out of the fuzzy pool:
                # token_set_ratio scores 100 for any query merely containing
                # the word ("price target ..." → Target Corp).
                if alias in _AMBIGUOUS_ALIASES:
                    continue
                if alias not in name_to_entry:
                    catalog_names.append(alias)
                    name_to_entry[alias] = (ticker, name)

        self._name_lookup = name_lookup
        self._ticker_lookup = ticker_lookup
        self._catalog_names = catalog_names
        self._catalog_name_to_entry = name_to_entry

    def _warn_if_catalog_expired(self, data: dict) -> None:
        try:
            generated_at_raw = data.get("generated_at")
            ttl_hours = int(data.get("ttl_hours", 0))
            if not generated_at_raw or ttl_hours <= 0:
                return
            generated_at = datetime.fromisoformat(str(generated_at_raw))
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=UTC)
            expires_at = generated_at + timedelta(hours=ttl_hours)
        except Exception as exc:
            logger.warning("Symbol catalog %s has invalid TTL metadata: %s", self.catalog_path, exc)
            return

        if datetime.now(UTC) > expires_at:
            logger.warning("Symbol catalog %s is expired; using stale entries", self.catalog_path)

    def _ticker_from_text(self, text: str) -> Optional[str]:
        from .intent_parser import IntentParser

        # Catalog symbols include English words (ARE, ALL, IT, CAN, ...);
        # never treat those as tickers when they appear inside prose. Single
        # letters are also excluded — "P/E", "Q4" etc. collide with the many
        # one-letter symbols (P, E, F, T, ...); those companies still resolve
        # by name. Longest-first keeps multi-candidate queries deterministic.
        stopwords = IntentParser.COMMON_QUERY_WORDS
        candidates = sorted(set(re.findall(r"\b[A-Z]{2,5}\b", text)), key=lambda c: (-len(c), c))
        for candidate in candidates:
            if candidate in self._ticker_lookup and candidate not in stopwords:
                return candidate
        normalized = _normalize(text).upper()
        if (
            len(normalized) >= 2
            and normalized in self._ticker_lookup
            and normalized not in stopwords
        ):
            return normalized
        return None

    def _find_name_in_query(self, normalized_query: str) -> Optional[tuple[str, str]]:
        padded_query = f" {normalized_query} "
        matches = [
            (alias, entry)
            for alias, entry in self._name_lookup.items()
            if alias not in _AMBIGUOUS_ALIASES and f" {alias} " in padded_query
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: len(item[0]), reverse=True)
        return matches[0][1]

    def _passes_fuzzy_guards(self, text: str) -> bool:
        tokens = re.findall(r"[A-Za-z]+", text)
        if not tokens:
            return False
        if max(len(token) for token in tokens) < 4:
            return False
        if len(tokens) <= 2 and all(len(token) <= 3 for token in tokens):
            return False
        return True

    def _cache_negative(self, normalized_query: str, now: float) -> None:
        if self.negative_ttl_s > 0:
            self._negative_cache[normalized_query] = now + self.negative_ttl_s


_DEFAULT_RESOLVER: Optional[SymbolResolver] = None
_DEFAULT_RESOLVER_LOCK = threading.Lock()


def get_default_resolver() -> SymbolResolver:
    """Return a process-wide default resolver singleton."""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        with _DEFAULT_RESOLVER_LOCK:
            if _DEFAULT_RESOLVER is None:
                threshold = _DEFAULT_FUZZY_THRESHOLD
                raw_threshold = os.environ.get("RESOLVER_FUZZY_THRESHOLD")
                if raw_threshold:
                    try:
                        threshold = float(raw_threshold)
                    except ValueError:
                        logger.warning(
                            "Invalid RESOLVER_FUZZY_THRESHOLD=%r; using default",
                            raw_threshold,
                        )
                _DEFAULT_RESOLVER = SymbolResolver(fuzzy_threshold=threshold)
    return _DEFAULT_RESOLVER


def _normalize(text: str) -> str:
    lowered = text.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", normalized).strip()


def _catalog_aliases(name: str) -> set[str]:
    aliases = set()
    normalized = _normalize(name)
    if normalized:
        aliases.add(normalized)

    tokens = normalized.split()
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens = tokens[:-1]
        stripped = " ".join(tokens)
        if stripped:
            aliases.add(stripped)

    return aliases

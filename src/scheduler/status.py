"""src/scheduler/status.py
SQLite-only scheduler run summaries and coverage health calculations.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.scheduler.source_registry import SourceRegistry, SourceSpec

logger = logging.getLogger(__name__)


def _as_utc(value: Optional[str | datetime] = None) -> datetime:
    """Normalize an optional status clock value to an aware UTC datetime."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("as_of must be an ISO date or timestamp") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _percentage(numerator: int, denominator: int) -> float:
    """Return a bounded percentage while preserving an explicit denominator."""
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def _json_list(value: object) -> list[str]:
    """Decode a provider-independent JSON list from SQLite metadata."""
    if isinstance(value, list):
        return [str(item).upper() for item in value if str(item).strip()]
    if not isinstance(value, str) or not value:
        return []
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return []
    return _json_list(decoded)


def _active_securities(store) -> list[dict]:
    """Read the active security denominator without invoking a provider."""
    with store.sqlite._connect() as conn:
        rows = conn.execute(
            "SELECT security_id, ticker, sector FROM securities "
            "WHERE active=1 ORDER BY normalized_ticker"
        ).fetchall()
    return [dict(row) for row in rows]


def _breakdown(covered: int, denominator: int) -> dict[str, object]:
    """Build one denominator-explicit coverage breakdown."""
    return {
        "covered": int(covered),
        "denominator": int(denominator),
        "percentage": _percentage(int(covered), int(denominator)),
    }


def _recent_tickers(store, *, kind: str, cutoff: str) -> set[str]:
    """Return active ticker coverage for one bounded recent corpus capability."""
    tickers: set[str] = set()
    with store.sqlite._connect() as conn:
        if kind == "news":
            rows = conn.execute(
                "SELECT ci.tickers_json, cis.ticker "
                "FROM corpus_items ci "
                "LEFT JOIN corpus_item_securities cis "
                "ON cis.corpus_item_id=ci.corpus_item_id "
                "WHERE ci.item_type='news' AND ci.is_tombstone=0 "
                "AND ci.published_at IS NOT NULL AND ci.published_at >= ?",
                (cutoff,),
            ).fetchall()
        elif kind == "market":
            rows = conn.execute(
                "SELECT tickers_json FROM corpus_observations "
                "WHERE (source_category LIKE '%market%' OR source_name IN ('massive', 'yfinance')) "
                "AND COALESCE(observed_at, published_at, as_of_at, period_end) >= ?",
                (cutoff,),
            ).fetchall()
        elif kind == "sec":
            rows = conn.execute(
                "SELECT f.ticker FROM filings f "
                "JOIN securities s ON s.ticker=f.ticker AND s.active=1 "
                "WHERE f.filing_date IS NOT NULL AND f.filing_date >= ?",
                (cutoff[:10],),
            ).fetchall()
        else:
            raise ValueError(f"unknown coverage kind: {kind}")
    for row in rows:
        if kind == "sec":
            if row[0]:
                tickers.add(str(row[0]).upper())
            continue
        if kind == "news" and len(row) > 1 and row[1]:
            tickers.add(str(row[1]).upper())
        tickers.update(_json_list(row[0]))
    return {ticker for ticker in tickers if ticker}


def _source_aliases(spec: SourceSpec) -> tuple[str, ...]:
    """Return cache/cursor names used by the source's adapters."""
    aliases = {spec.name, spec.ttl_key, f"unified:{spec.name}"}
    adapter_alias = {
        "finnhub": "finnhub_news",
        "massive": "massive_market",
        "sec_filings": "sec_filings_discovery",
        "gdelt": "gdelt_news",
        "earnings_transcripts": "earnings_transcripts",
    }.get(spec.name)
    if adapter_alias:
        aliases.add(adapter_alias)
    return tuple(sorted(value for value in aliases if value))


def _timestamp(value: object) -> Optional[datetime]:
    """Parse one persisted provider timestamp without failing status output."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromtimestamp(float(text), tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _partition_is_due(
    cache: Optional[dict], cursor: Optional[dict], spec: SourceSpec, now: datetime,
) -> bool:
    """Return whether persisted success is outside its registry TTL."""
    if cache:
        next_due = _timestamp(cache.get("next_scheduled_update"))
        if next_due is not None:
            return next_due <= now
        updated = _timestamp(cache.get("last_updated"))
        if updated is not None:
            return updated + timedelta(hours=spec.ttl_hours) <= now
    if cursor:
        succeeded = _timestamp(cursor.get("last_successful_at"))
        if succeeded is not None:
            return succeeded + timedelta(hours=spec.ttl_hours) <= now
    return False


def _partition_health(store, coverage, spec: SourceSpec, now: datetime) -> dict[str, object]:
    """Classify expected source partitions using persisted freshness/cursors."""
    if spec.scope in {"global", "universe"}:
        partitions = ["__global__"]
    else:
        try:
            partitions = [str(value).upper() for value in coverage.tickers_for(spec.name)]
        except Exception:  # noqa: BLE001 - status remains useful for one bad policy
            partitions = []
        partitions = list(dict.fromkeys(partitions)) or ["__global__"]

    aliases = _source_aliases(spec)
    caches: dict[str, dict] = {}
    cursors: dict[str, dict] = {}
    with store.sqlite._connect() as conn:
        if aliases:
            placeholders = ",".join("?" for _ in aliases)
            cache_rows = conn.execute(
                "SELECT * FROM cache_meta WHERE source IN (" + placeholders + ")",
                list(aliases),
            ).fetchall()
            cursor_rows = conn.execute(
                "SELECT * FROM source_cursors WHERE source IN (" + placeholders + ")",
                list(aliases),
            ).fetchall()
            for row in cache_rows:
                key = str(row["ticker"] or "").upper()
                if key == "SCHEDULER" and spec.scope in {"global", "universe"}:
                    caches["__global__"] = dict(row)
                elif key and key != "SCHEDULER":
                    caches[key] = dict(row)
            for row in cursor_rows:
                cursors[str(row["partition_key"])] = dict(row)

    counts = {"never_fetched": 0, "stale": 0, "error": 0, "fresh": 0}
    for partition in partitions:
        cache = caches.get(partition)
        cursor = cursors.get(partition)
        due = _partition_is_due(cache, cursor, spec, now)
        circuit_active = bool(
            cursor
            and str(cursor.get("status")) == "circuit_open"
            and (
                _timestamp(cursor.get("cursor_value")) is None
                or _timestamp(cursor.get("cursor_value")) > now
            )
        )
        if spec.status == "disabled_missing_key":
            state = "error"
        elif cursor and str(cursor.get("status")) == "error":
            state = "error"
        elif circuit_active:
            state = "error"
        elif cache and cache.get("status") == "stale" and cache.get("error_message"):
            state = "error"
        elif due:
            state = "stale"
        elif cursor and cursor.get("last_successful_at"):
            state = "fresh"
        elif cache and cache.get("status") == "fresh":
            state = "fresh"
        elif cache or cursor:
            state = "stale"
        else:
            state = "never_fetched"
        counts[state] += 1

    denominator = len(partitions)
    result = {
        "source": spec.name,
        "scope": spec.scope,
        "denominator": denominator,
        **counts,
        "policy_revision": getattr(coverage, "revision", "unknown"),
    }
    for state in counts:
        result[f"{state}_percentage"] = _percentage(counts[state], denominator)
    return result


def _corpus_dates(store) -> dict[str, Optional[str]]:
    """Return corpus min/max dates from SQLite metadata only."""
    values: list[str] = []
    with store.sqlite._connect() as conn:
        queries = (
            "SELECT published_at, effective_at, as_of_at, observed_at, ingested_at "
            "FROM corpus_items WHERE is_tombstone=0",
            "SELECT published_at, as_of_at, observed_at, ingested_at, period_end "
            "FROM corpus_observations",
            "SELECT published_at, effective_at, announced_at, observed_at, ingested_at "
            "FROM corpus_events",
        )
        for query in queries:
            for row in conn.execute(query).fetchall():
                values.extend(str(value) for value in row if value)
    return {"oldest": min(values) if values else None, "newest": max(values) if values else None}


def _indexing_counts(store) -> dict[str, int]:
    """Read indexing backlog and error counts without touching Chroma."""
    with store.sqlite._connect() as conn:
        narrative = conn.execute(
            "SELECT "
            "SUM(CASE WHEN indexing_status='pending' THEN 1 ELSE 0 END) pending, "
            "SUM(CASE WHEN indexing_status='error' THEN 1 ELSE 0 END) error, "
            "COUNT(*) total FROM corpus_items WHERE is_tombstone=0"
        ).fetchone()
        filings = conn.execute(
            "SELECT "
            "SUM(CASE WHEN status='index_pending' THEN 1 ELSE 0 END) pending, "
            "SUM(CASE WHEN status='index_pending' AND index_error IS NOT NULL "
            "THEN 1 ELSE 0 END) error, COUNT(*) total FROM filings"
        ).fetchone()
    narrative_counts = {
        "pending": int(narrative["pending"] or 0),
        "error": int(narrative["error"] or 0),
        "total": int(narrative["total"] or 0),
    }
    filing_counts = {
        "pending": int(filings["pending"] or 0),
        "error": int(filings["error"] or 0),
        "total": int(filings["total"] or 0),
    }
    return {
        "pending": narrative_counts["pending"] + filing_counts["pending"],
        "error": narrative_counts["error"] + filing_counts["error"],
        "total": narrative_counts["total"] + filing_counts["total"],
        "narratives": narrative_counts,
        "filings": filing_counts,
    }


def build_coverage_health(
    store,
    coverage,
    registry: SourceRegistry,
    *,
    as_of: Optional[str | datetime] = None,
    recent_days: int = 7,
) -> dict[str, object]:
    """Calculate denominator-explicit coverage health with zero network calls."""
    now = _as_utc(as_of)
    active = _active_securities(store)
    active_tickers = {str(row["ticker"]).upper() for row in active if row.get("ticker")}
    denominator = len(active_tickers)
    cutoff = (now - timedelta(days=max(int(recent_days), 1))).isoformat().replace("+00:00", "Z")

    with store.sqlite._connect() as conn:
        membership_rows = conn.execute(
            "SELECT DISTINCT m.index_code, m.security_id "
            "FROM security_memberships m JOIN securities s ON s.security_id=m.security_id "
            "WHERE m.active=1 AND s.active=1"
        ).fetchall()
    by_index: dict[str, dict[str, object]] = {}
    for index_code in ("sp500", "nasdaq100"):
        covered = len({str(row[1]) for row in membership_rows if row[0] == index_code})
        by_index[index_code] = _breakdown(covered, denominator)

    scope_sets: dict[str, set[str]] = {"universe": set(active_tickers)}
    for scope, source_name in (("broad", "yfinance"), ("deep", "sec_companyfacts")):
        try:
            scope_sets[scope] = active_tickers & {
                str(ticker).upper() for ticker in coverage.tickers_for(source_name)
            }
        except Exception:  # noqa: BLE001
            scope_sets[scope] = set()
    scope_sets["sector"] = {
        str(row["ticker"]).upper() for row in active if row.get("sector")
    }
    by_scope = {scope: _breakdown(len(values), denominator) for scope, values in scope_sets.items()}

    sector_counts: dict[str, int] = {}
    for row in active:
        sector = str(row.get("sector") or "Unclassified")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    by_sector = {
        sector: _breakdown(count, denominator)
        for sector, count in sorted(sector_counts.items())
    }

    recent = {}
    for kind in ("news", "market", "sec"):
        covered = len(_recent_tickers(store, kind=kind, cutoff=cutoff) & active_tickers)
        recent[kind] = {**_breakdown(covered, denominator), "window_days": max(int(recent_days), 1)}

    partitions: dict[str, dict[str, object]] = {}
    for spec in registry.sources.values():
        health = _partition_health(store, coverage, spec, now)
        partitions[spec.name] = health
        # Keep adapter/cache aliases visible for operators who use the
        # logical freshness names (for example ``finnhub_news``).
        for alias in _source_aliases(spec):
            partitions.setdefault(alias, {**health, "source": alias})
    try:
        with store.sqlite._connect() as conn:
            row = conn.execute(
                "SELECT MAX(ended_at) FROM scheduler_runs "
                "WHERE status IN ('success', 'partial')"
            ).fetchone()
        last_successful = row[0] if row and row[0] else None
    except sqlite3.Error:
        last_successful = None

    return {
        "as_of": now.isoformat().replace("+00:00", "Z"),
        "policy_revision": str(getattr(coverage, "revision", "unknown")),
        "config_revision": str(registry.version),
        "active_securities": {
            "total": denominator,
            "by_index": by_index,
            "by_scope": by_scope,
            "by_sector": by_sector,
        },
        "recent_coverage": recent,
        "coverage": recent,
        "partitions": partitions,
        "indexing": _indexing_counts(store),
        "corpus_dates": _corpus_dates(store),
        "last_successful_refresh": last_successful,
    }


__all__ = ["build_coverage_health"]

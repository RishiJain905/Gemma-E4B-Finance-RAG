"""scripts/watch_scheduler.py
Read-only, full-screen terminal monitor for scheduler ingestion health.

Meant to be left open in a second terminal tab while
``python -m src.scheduler daily|weekly|all --force`` (or ``scripts/chat.py``)
runs in the first tab, so a run's progress and outcome are visible at a
glance. It only ever opens ``data/finance.db`` via a short-lived read-only
SQLite connection (``mode=ro`` + ``PRAGMA query_only``) and never writes to
or locks the database, so it is safe to run alongside a live ingestion.

Per-source "live" state during an active run is inferred from
``cache_meta`` (ticker ``SCHEDULER``, source ``unified:<name>``), which the
scheduler updates immediately after each source finishes
(``src/scheduler/__init__.py``). ``scheduler_run_sources`` -- the table with
exact item counts and ``error_class`` -- is only written once, in a batch,
when the whole run completes, so it powers the LAST COMPLETED RUN panel but
cannot show item counts while a run is still in progress.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# ── ANSI colors (safe on Windows Terminal once VT processing is enabled) ──

RESET = "\x1b[0m"
CLEAR_SCREEN = "\x1b[2J\x1b[H"


def _color(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}{RESET}"


def green(text: str) -> str:
    return _color("32", text)


def yellow(text: str) -> str:
    return _color("33", text)


def red(text: str) -> str:
    return _color("31", text)


def dim(text: str) -> str:
    return _color("2", text)


def bold(text: str) -> str:
    return _color("1", text)


def cyan(text: str) -> str:
    return _color("36", text)


# ── Data ────────────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    """One poll's worth of observed scheduler state."""

    available: bool
    now: datetime
    db_path: Path
    active_run: Optional[dict] = None
    last_run: Optional[dict] = None
    freshness: list = field(default_factory=list)
    queues: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a short-lived, read-only connection to the store.

    Raises on a missing file, a mid-migration/locked database, or any other
    connection failure -- callers treat that as "database not ready yet".
    """
    uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1")
    return conn


def _parse_dt(value: object) -> Optional[datetime]:
    """Parse an ISO or SQLite ``datetime('now')`` timestamp as aware UTC."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _json_list(value: object) -> list[str]:
    if not value:
        return []
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return []
    if isinstance(decoded, list):
        return [str(item) for item in decoded if str(item).strip()]
    return []


def _scheduler_cache_source(name: str) -> str:
    """Mirror ``UnifiedScheduler._scheduler_cache_source`` naming."""
    return f"unified:{name}"


# ── Section collectors (each isolated -- one failure never blanks the rest) ─

def _safe(errors: list, name: str, fn: Callable[[], object]) -> object:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a monitor must never crash on one bad query
        errors.append(f"{name}: {exc}")
        return None


def _classify_active_source(cache_row: Optional[dict], started_at, now: datetime) -> dict:
    """Best-effort per-source state for a run still in progress.

    ``cache_meta`` is the only table the scheduler updates incrementally, so
    a source is "pending" until its cache row changes; a row that was
    already fresh before the run started is shown as the scheduler's likely
    skip decision (TTL not yet due), not as "done by this run".
    """
    if not cache_row:
        return {"state": "pending", "detail": ""}
    status = cache_row.get("status")
    last_updated = _parse_dt(cache_row.get("last_updated"))
    next_due = _parse_dt(cache_row.get("next_scheduled_update"))
    if status == "fresh":
        if last_updated and started_at and last_updated >= started_at:
            return {"state": "success", "detail": "done"}
        if next_due and next_due > now:
            hours_left = (next_due - now).total_seconds() / 3600.0
            return {"state": "skipped", "detail": f"fresh, TTL {hours_left:.1f}h left"}
        return {"state": "pending", "detail": ""}
    if status == "stale":
        message = str(cache_row.get("error_message") or "").strip()
        return {"state": "error", "detail": message[:80] if message else "stale"}
    return {"state": "pending", "detail": ""}


def _collect_active_run(conn: sqlite3.Connection, now: datetime) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM scheduler_runs WHERE status='running' "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    run = dict(row)
    started_at = _parse_dt(run.get("started_at"))
    requested = _json_list(run.get("requested_sources_json"))
    sources = []
    for name in requested:
        cache_row = conn.execute(
            "SELECT * FROM cache_meta WHERE ticker='SCHEDULER' AND source=?",
            (_scheduler_cache_source(name),),
        ).fetchone()
        classified = _classify_active_source(
            dict(cache_row) if cache_row else None, started_at, now
        )
        sources.append({"name": name, **classified})
    run["sources"] = sources
    run["started_dt"] = started_at
    return run


def _collect_last_run(conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM scheduler_runs WHERE status != 'running' "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    run = dict(row)
    source_rows = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM scheduler_run_sources WHERE run_id=? ORDER BY source",
            (run["run_id"],),
        ).fetchall()
    ]
    success_rows = [r for r in source_rows if r.get("status") == "success"]
    skipped_rows = [r for r in source_rows if r.get("status") == "skipped"]
    error_rows = [r for r in source_rows if r.get("status") not in ("success", "skipped")]
    run["source_rows"] = source_rows
    run["success_count"] = len(success_rows)
    run["skipped_count"] = len(skipped_rows)
    run["error_count"] = len(error_rows)
    run["non_success_rows"] = skipped_rows + error_rows
    run["items_total"] = sum(int(r.get("items") or 0) for r in source_rows)
    return run


def _freshness_entry(name: str, cache_row: Optional[dict], now: datetime) -> dict:
    if not cache_row:
        return {"name": name, "state": "never", "age_hours": None, "ttl_hours": None}
    last_updated = _parse_dt(cache_row.get("last_updated"))
    next_due = _parse_dt(cache_row.get("next_scheduled_update"))
    age_hours = (now - last_updated).total_seconds() / 3600.0 if last_updated else None
    ttl_hours = None
    if last_updated and next_due:
        ttl_hours = round((next_due - last_updated).total_seconds() / 3600.0, 1)
    status = cache_row.get("status")
    if status == "stale" or (next_due is not None and next_due <= now):
        state = "stale"
    elif status == "fresh":
        state = "fresh"
    else:
        state = "never"
    return {
        "name": name,
        "state": state,
        "age_hours": age_hours,
        "ttl_hours": ttl_hours,
        "error_message": cache_row.get("error_message"),
    }


def _collect_freshness(conn: sqlite3.Connection, now: datetime) -> list:
    names: set[str] = set()
    for row in conn.execute(
        "SELECT DISTINCT source FROM cache_meta WHERE ticker='SCHEDULER'"
    ).fetchall():
        raw = str(row[0] or "")
        if raw:
            names.add(raw[len("unified:"):] if raw.startswith("unified:") else raw)
    for row in conn.execute("SELECT DISTINCT source FROM scheduler_run_sources").fetchall():
        if row[0]:
            names.add(str(row[0]))

    entries = []
    for name in names:
        cache_row = conn.execute(
            "SELECT * FROM cache_meta WHERE ticker='SCHEDULER' AND source=?",
            (_scheduler_cache_source(name),),
        ).fetchone()
        entries.append(_freshness_entry(name, dict(cache_row) if cache_row else None, now))

    rank = {"stale": 0, "never": 1, "fresh": 2}
    entries.sort(key=lambda e: (rank.get(e["state"], 1), -(e["age_hours"] or -1.0)))
    return entries


def _collect_queues(conn: sqlite3.Connection) -> dict:
    queues: dict = {}
    filings = conn.execute(
        "SELECT "
        "SUM(CASE WHEN status='parsed' THEN 1 ELSE 0 END) parsed, "
        "SUM(CASE WHEN status IN ('unprocessed', 'index_pending') THEN 1 ELSE 0 END) unprocessed "
        "FROM filings"
    ).fetchone()
    queues["filings_parsed"] = int(filings["parsed"] or 0) if filings else 0
    queues["filings_unprocessed"] = int(filings["unprocessed"] or 0) if filings else 0

    try:
        row = conn.execute("SELECT COUNT(*) FROM dead_letter").fetchone()
        queues["dead_letter"] = int(row[0]) if row else 0
    except sqlite3.Error:
        queues["dead_letter"] = 0

    row = conn.execute("SELECT COUNT(*) FROM corpus_items WHERE is_tombstone=0").fetchone()
    queues["corpus_items"] = int(row[0]) if row else 0

    lex = conn.execute(
        "SELECT row_count, indexed_revision FROM lexical_index_state WHERE id=1"
    ).fetchone()
    queues["fts_row_count"] = int(lex["row_count"]) if lex else 0
    queues["indexed_revision"] = int(lex["indexed_revision"]) if lex else 0

    rev = conn.execute("SELECT revision FROM store_revision WHERE id=1").fetchone()
    queues["store_revision"] = int(rev[0]) if rev else 0
    return queues


def collect_snapshot(db_path: Path) -> Snapshot:
    """Poll the store once and return a fully-isolated snapshot.

    Never raises: connection failure yields ``available=False``; any single
    section query failure is recorded in ``errors`` and that section is left
    empty rather than aborting the whole poll.
    """
    now = datetime.now(timezone.utc)
    snapshot = Snapshot(available=False, now=now, db_path=db_path)
    try:
        conn = connect_readonly(db_path)
    except Exception as exc:  # noqa: BLE001 - missing/locked/mid-migration DB
        snapshot.errors.append(f"connect: {exc}")
        return snapshot

    try:
        snapshot.available = True
        snapshot.active_run = _safe(
            snapshot.errors, "active_run", lambda: _collect_active_run(conn, now)
        )
        snapshot.last_run = _safe(
            snapshot.errors, "last_run", lambda: _collect_last_run(conn)
        )
        snapshot.freshness = _safe(
            snapshot.errors, "freshness", lambda: _collect_freshness(conn, now)
        ) or []
        snapshot.queues = _safe(
            snapshot.errors, "queues", lambda: _collect_queues(conn)
        ) or {}
    finally:
        conn.close()
    return snapshot


# ── Rendering (pure functions -- no I/O, so tests can call these directly) ──

def _format_duration(seconds: object) -> str:
    try:
        total = int(max(float(seconds), 0))
    except (TypeError, ValueError):
        return "?"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def render_header(snapshot: Snapshot) -> str:
    now_str = snapshot.now.strftime("%Y-%m-%d %H:%M:%S UTC")
    revision = (snapshot.queues or {}).get("store_revision")
    revision_str = str(revision) if revision is not None else "?"
    run_state = bold(green("ACTIVE")) if snapshot.active_run else dim("idle")
    return (
        f"{bold('Scheduler Watch')}   {now_str}\n"
        f"DB: {snapshot.db_path}   |   store revision: {revision_str}   |   run: {run_state}"
    )


def render_active_run(snapshot: Snapshot) -> str:
    run = snapshot.active_run
    if not run:
        return ""
    started = run.get("started_dt")
    elapsed = _format_duration((snapshot.now - started).total_seconds()) if started else "?"
    lines = [bold(cyan(f"ACTIVE RUN -- mode={run.get('mode')}  elapsed={elapsed}"))]
    for source in run.get("sources", []):
        name = source["name"]
        detail = source["detail"]
        state = source["state"]
        if state == "pending":
            lines.append(dim(f"  ...  {name}"))
        elif state == "success":
            lines.append(green(f"  ok   {name}  ({detail})"))
        elif state == "skipped":
            lines.append(yellow(f"  --   {name}  ({detail})"))
        else:
            lines.append(red(f"  !!   {name}  {detail}"))
    return "\n".join(lines)


def render_last_run(snapshot: Snapshot) -> str:
    run = snapshot.last_run
    if not run:
        return bold("LAST COMPLETED RUN") + "\n" + dim("  none recorded yet")
    if run["error_count"] == 0:
        verdict = green("ALL GOOD ✓")
    else:
        verdict = red(f"{run['error_count']} SOURCE ERRORS")
    lines = [f"{bold('LAST COMPLETED RUN')}  [{verdict}]"]
    lines.append(
        f"  mode={run.get('mode')}  ended={run.get('ended_at')}  "
        f"duration={_format_duration(run.get('duration_seconds') or 0)}  "
        f"{run['success_count']} success / {run['skipped_count']} skipped / "
        f"{run['error_count']} error  items={run['items_total']}"
    )
    for row in run["non_success_rows"]:
        label = row.get("error_class") or row.get("status") or "skipped"
        message = str(row.get("error_message") or "")[:80]
        line = f"    {row.get('source')}: {label}" + (f" -- {message}" if message else "")
        colorize = red if row.get("status") not in ("skipped",) else yellow
        lines.append(colorize(line))
    return "\n".join(lines)


def render_freshness(snapshot: Snapshot) -> str:
    lines = [bold("FRESHNESS")]
    if not snapshot.freshness:
        lines.append(dim("  no sources recorded yet"))
        return "\n".join(lines)
    for entry in snapshot.freshness:
        age = f"{entry['age_hours']:.1f}h" if entry["age_hours"] is not None else "--"
        ttl = f"{entry['ttl_hours']:.1f}h" if entry["ttl_hours"] is not None else "--"
        text = f"  {entry['name']:<28} age={age:<8} ttl={ttl:<8} state={entry['state']}"
        if entry["state"] == "stale":
            lines.append(red(text))
        elif entry["state"] == "never":
            lines.append(dim(text))
        else:
            lines.append(green(text))
    return "\n".join(lines)


def render_queues(snapshot: Snapshot, dead_letter_baseline: Optional[int] = None) -> str:
    queues = snapshot.queues or {}
    dead_letter = queues.get("dead_letter", 0)
    baseline = dead_letter_baseline if dead_letter_baseline is not None else dead_letter
    delta = dead_letter - baseline
    if delta > 0:
        dead_letter_text = red(f"dead_letter={dead_letter} (+{delta} new)")
    elif dead_letter:
        dead_letter_text = yellow(f"dead_letter={dead_letter}")
    else:
        dead_letter_text = dim(f"dead_letter={dead_letter}")

    indexed_revision = queues.get("indexed_revision", 0)
    store_revision = queues.get("store_revision", 0)
    fts_text = (
        f"fts={queues.get('fts_row_count', 0)} rows "
        f"@ rev {indexed_revision}/{store_revision}"
    )
    if indexed_revision != store_revision:
        fts_text = red(fts_text + " MISMATCH")
    else:
        fts_text = dim(fts_text)

    parts = [
        f"filings: {queues.get('filings_parsed', 0)} parsed / "
        f"{queues.get('filings_unprocessed', 0)} unprocessed",
        dead_letter_text,
        f"corpus_items={queues.get('corpus_items', 0)}",
        fts_text,
    ]
    return bold("QUEUES & STORES") + "\n  " + "   |   ".join(parts)


def render_footer(interval: float) -> str:
    return dim(f"Ctrl+C to exit   |   poll interval: {interval}s")


def render_frame(
    snapshot: Snapshot, *, interval: float, dead_letter_baseline: Optional[int] = None,
) -> str:
    """Render one full-screen frame from a snapshot. Pure -- no I/O."""
    if not snapshot.available:
        return "\n\n".join(
            [
                render_header(snapshot),
                dim("waiting for database..."),
                render_footer(interval),
            ]
        )
    sections = [render_header(snapshot)]
    active = render_active_run(snapshot)
    if active:
        sections.append(active)
    sections.append(render_last_run(snapshot))
    sections.append(render_freshness(snapshot))
    sections.append(render_queues(snapshot, dead_letter_baseline))
    sections.append(render_footer(interval))
    return "\n\n".join(sections)


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only, live scheduler-health monitor (data/finance.db)."
    )
    parser.add_argument(
        "--db", default="data/finance.db",
        help="Path to the SQLite store (default: data/finance.db)",
    )
    parser.add_argument(
        "--interval", type=float, default=3.0,
        help="Poll interval in seconds (default: 3)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Print one snapshot and exit (useful for scripting/CI)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    os.system("")  # enable ANSI/VT processing on legacy Windows consoles
    # Piped/redirected stdout on Windows falls back to cp1252, which cannot
    # encode the verdict glyphs (U+2713) and crashed --once in pipelines.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    db_path = Path(args.db)
    dead_letter_baseline: Optional[int] = None
    try:
        while True:
            snapshot = collect_snapshot(db_path)
            if dead_letter_baseline is None and snapshot.queues:
                dead_letter_baseline = snapshot.queues.get("dead_letter")
            frame = render_frame(
                snapshot, interval=args.interval, dead_letter_baseline=dead_letter_baseline,
            )
            if args.once:
                print(frame)
                return 0
            print(CLEAR_SCREEN + frame, flush=True)
            time.sleep(max(args.interval, 0.5))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

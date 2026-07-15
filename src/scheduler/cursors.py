"""src/scheduler/cursors.py
Transactional cursor checkpoints, overlap windows, and fair partition ordering.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterator, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

CURSOR_KINDS = frozenset({"date", "timestamp", "page_token", "daily_index", "none"})
_OVERLAP_PATTERN = re.compile(r"^(\d+)([dhm])$")
_T = TypeVar("_T")


class CursorStore(Protocol):
    """Minimal persistence contract used by the cursor coordinator."""

    def get_source_cursor_state(
        self, source: str, partition_key: str
    ) -> Optional[dict]: ...

    def set_source_cursors(self, updates: list[dict]) -> list[dict]: ...

    def list_source_cursor_states(self, source: str) -> list[dict]: ...


class _CursorTransaction:
    """Stage cursor changes and publish them in one SQLite transaction."""

    def __init__(self, manager: "CursorManager") -> None:
        self.manager = manager
        self.updates: list[dict] = []

    def advance(
        self,
        source: str,
        partition: str,
        cursor_value: Optional[str],
        *,
        kind: str,
        overlap: Optional[str] = None,
        run_id: Optional[str] = None,
        version: str = "1",
        status: str = "success",
    ) -> None:
        """Stage one successful logical partition checkpoint."""
        self.updates.append(
            self.manager._cursor_update(
                source,
                partition,
                cursor_value,
                kind=kind,
                overlap=overlap,
                run_id=run_id,
                version=version,
                status=status,
            )
        )


class CursorManager:
    """Coordinate committed source checkpoints independently of freshness state."""

    def __init__(self, store: CursorStore) -> None:
        self.store = store

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _cursor_update(
        self,
        source: str,
        partition: str,
        cursor_value: Optional[str],
        *,
        kind: str,
        overlap: Optional[str],
        run_id: Optional[str],
        version: str,
        status: str,
    ) -> dict:
        source = str(source or "").strip()
        partition = str(partition or "").strip()
        kind = str(kind or "").strip()
        if not source or not partition:
            raise ValueError("source and partition are required")
        if kind not in CURSOR_KINDS:
            raise ValueError(f"unsupported cursor kind: {kind}")
        if overlap is not None:
            self._parse_overlap(overlap)
        return {
            "source": source,
            "partition_key": partition,
            "cursor_value": cursor_value,
            "cursor_type": kind,
            "overlap_value": overlap,
            "last_successful_run_id": run_id,
            "version": str(version or "1"),
            "status": str(status or "success"),
            "updated_at": self._now().isoformat(),
        }

    @contextmanager
    def transaction(self) -> Iterator[_CursorTransaction]:
        """Commit all staged cursor advances together, or none on an exception."""
        transaction = _CursorTransaction(self)
        yield transaction
        if transaction.updates:
            self.store.set_source_cursors(transaction.updates)

    def advance(
        self,
        source: str,
        partition: str,
        cursor_value: Optional[str],
        *,
        kind: str,
        overlap: Optional[str] = None,
        run_id: Optional[str] = None,
        version: str = "1",
        status: str = "success",
    ) -> dict:
        """Commit one logical partition cursor after its records are durable."""
        update = self._cursor_update(
            source,
            partition,
            cursor_value,
            kind=kind,
            overlap=overlap,
            run_id=run_id,
            version=version,
            status=status,
        )
        return self.store.set_source_cursors([update])[0]

    def advance_after_commit(
        self,
        source: str,
        partition: str,
        cursor_value: Optional[str],
        *,
        kind: str,
        commit: Callable[[], _T],
        overlap: Optional[str] = None,
        run_id: Optional[str] = None,
        version: str = "1",
        status: str = "success",
    ) -> _T:
        """Run page persistence first and checkpoint only after it completes."""
        result = commit()
        self.advance(
            source,
            partition,
            cursor_value,
            kind=kind,
            overlap=overlap,
            run_id=run_id,
            version=version,
            status=status,
        )
        return result

    @staticmethod
    def _parse_overlap(overlap: str) -> timedelta:
        match = _OVERLAP_PATTERN.fullmatch(str(overlap or "").strip().lower())
        if not match:
            raise ValueError("overlap must use an integer suffix: d, h, or m")
        amount, unit = int(match.group(1)), match.group(2)
        if unit == "d":
            return timedelta(days=amount)
        if unit == "h":
            return timedelta(hours=amount)
        return timedelta(minutes=amount)

    @classmethod
    def overlap_start(
        cls,
        cursor_value: Optional[str],
        *,
        kind: str,
        overlap: Optional[str],
    ) -> Optional[str]:
        """Return a deliberate refetch boundary for date/time cursor kinds."""
        if cursor_value is None or not overlap or kind in {"page_token", "none"}:
            return cursor_value
        delta = cls._parse_overlap(overlap)
        if kind in {"date", "daily_index"}:
            parsed_date = date.fromisoformat(str(cursor_value))
            return (parsed_date - delta).isoformat()
        if kind == "timestamp":
            text = str(cursor_value)
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            shifted = parsed - delta
            value = shifted.isoformat()
            return value.replace("+00:00", "Z") if text.endswith("Z") else value
        raise ValueError(f"unsupported cursor kind: {kind}")

    def order_partitions(self, source: str, partitions: list[str]) -> list[str]:
        """Order never-fetched partitions first, then least-recently successful."""
        unique = {str(partition).strip() for partition in partitions if str(partition).strip()}
        states = {
            str(state["partition_key"]): state
            for state in self.store.list_source_cursor_states(source)
        }

        def order_key(partition: str) -> tuple[int, str, str]:
            state = states.get(partition)
            if state is None:
                return (0, "", partition)
            last_attempt = state.get("last_successful_at") or state.get("updated_at")
            return (1, str(last_attempt or ""), partition)

        return sorted(unique, key=order_key)

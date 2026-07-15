"""src/scheduler/budget.py
Per-run quota reservations that gate provider work before requests begin.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Optional

logger = logging.getLogger(__name__)


class BudgetExhaustedError(RuntimeError):
    """Raised before an HTTP call when no configured request capacity remains."""


class RunBudget:
    """Track and reserve bounded request/work capacity for one source run."""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        requests_per_day: int,
        requests_per_run: int,
        max_work_items: int,
        day_requests: int = 0,
        minute_requests: int = 0,
        minute_started: Optional[float] = None,
        provider_remaining: Optional[int] = None,
        provider_reset: Optional[str] = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        limits = {
            "requests_per_minute": requests_per_minute,
            "requests_per_day": requests_per_day,
            "requests_per_run": requests_per_run,
            "max_work_items": max_work_items,
        }
        if any(int(value) <= 0 for value in limits.values()):
            raise ValueError("budget limits must be positive integers")
        if int(day_requests) < 0 or int(minute_requests) < 0:
            raise ValueError("existing request usage must be non-negative")
        if provider_remaining is not None and int(provider_remaining) < 0:
            raise ValueError("provider remaining capacity must be non-negative")
        self.requests_per_minute = int(requests_per_minute)
        self.requests_per_day = int(requests_per_day)
        self.requests_per_run = int(requests_per_run)
        self.max_work_items = int(max_work_items)
        self.day_requests = int(day_requests)
        self._now_fn = now_fn
        self._minute_started = (
            float(now_fn()) if minute_started is None else float(minute_started)
        )
        self._minute_requests = int(minute_requests)
        self.attempted_requests = 0
        self.successful_requests = 0
        self.work_items_started = 0
        self.provider_remaining = (
            int(provider_remaining) if provider_remaining is not None else None
        )
        self.provider_reset = (
            str(provider_reset) if provider_reset is not None else None
        )
        self.retry_after: Optional[str] = None
        self.exhausted_reason: Optional[str] = None

    def _refresh_minute_window(self) -> None:
        now = float(self._now_fn())
        if now - self._minute_started >= 60.0:
            self._minute_started = now
            self._minute_requests = 0

    def _capacity_reason(self, requests: int, work_items: int) -> Optional[str]:
        self._refresh_minute_window()
        if self._minute_requests + requests > self.requests_per_minute:
            return "requests_per_minute"
        if self.day_requests + self.attempted_requests + requests > self.requests_per_day:
            return "requests_per_day"
        if self.attempted_requests + requests > self.requests_per_run:
            return "requests_per_run"
        if self.work_items_started + work_items > self.max_work_items:
            return "max_work_items_per_run"
        if self.provider_remaining is not None and requests > self.provider_remaining:
            return "provider_remaining"
        return None

    def can_reserve(self, *, requests: int = 1, work_items: int = 1) -> bool:
        """Return whether capacity exists without consuming it."""
        requests, work_items = self._validated_amounts(requests, work_items)
        return self._capacity_reason(requests, work_items) is None

    def reserve_work(self, *, work_items: int = 1) -> bool:
        """Reserve work selection while leaving request attempts to the HTTP gate."""
        work_items = int(work_items)
        if work_items <= 0:
            raise ValueError("work_items must be positive")
        reason = self._capacity_reason(1, work_items)
        self.exhausted_reason = reason
        if reason is not None:
            return False
        self.work_items_started += work_items
        return True

    def reserve(
        self,
        *,
        requests: int = 1,
        work_items: int = 1,
        force: bool = False,
    ) -> bool:
        """Reserve capacity before work starts; ``force`` never changes limits."""
        del force
        requests, work_items = self._validated_amounts(requests, work_items)
        reason = self._capacity_reason(requests, work_items)
        self.exhausted_reason = reason
        if reason is not None:
            return False
        self._minute_requests += requests
        self.attempted_requests += requests
        self.work_items_started += work_items
        if self.provider_remaining is not None:
            self.provider_remaining -= requests
        return True

    @staticmethod
    def _validated_amounts(requests: int, work_items: int) -> tuple[int, int]:
        requests = int(requests)
        work_items = int(work_items)
        if requests <= 0 or work_items < 0:
            raise ValueError("requests must be positive and work_items non-negative")
        return requests, work_items

    def record_success(self, requests: int = 1) -> None:
        """Record successful requests without changing reserved capacity."""
        requests = int(requests)
        if requests < 0:
            raise ValueError("successful requests must be non-negative")
        self.successful_requests = min(
            self.attempted_requests,
            self.successful_requests + requests,
        )

    def update_provider_limits(self, headers: Mapping[str, object]) -> None:
        """Apply common provider remaining/reset headers to future reservations."""
        normalized = {str(key).lower(): value for key, value in headers.items()}
        remaining = normalized.get("x-ratelimit-remaining")
        if remaining is None:
            remaining = normalized.get("ratelimit-remaining")
        if remaining is not None:
            try:
                self.provider_remaining = max(int(str(remaining)), 0)
            except ValueError:
                logger.debug("Ignoring invalid provider remaining header: %r", remaining)
        reset = normalized.get("x-ratelimit-reset")
        if reset is None:
            reset = normalized.get("ratelimit-reset")
        if reset is not None:
            self.provider_reset = str(reset)
        retry_after = normalized.get("retry-after")
        if retry_after is not None:
            self.retry_after = str(retry_after)

    def wrap_http_get(self, request: Callable[..., object]) -> Callable[..., object]:
        """Wrap an adapter HTTP callable with per-attempt quota reservations."""

        def budgeted_request(*args: object, **kwargs: object) -> object:
            if not self.reserve(requests=1, work_items=0):
                raise BudgetExhaustedError(
                    f"source request budget exhausted: {self.exhausted_reason}"
                )
            response = request(*args, **kwargs)
            headers = getattr(response, "headers", None)
            if isinstance(headers, Mapping):
                self.update_provider_limits(headers)
            status_code = int(getattr(response, "status_code", 200))
            if status_code < 400:
                self.record_success()
            return response

        return budgeted_request

    @property
    def minute_started(self) -> float:
        """Expose the rolling minute-window anchor for in-process reuse."""
        return self._minute_started

    @property
    def minute_requests(self) -> int:
        """Expose rolling minute attempts for the next budget object."""
        self._refresh_minute_window()
        return self._minute_requests

    def snapshot(self) -> dict:
        """Return current counters and remaining capacity for status reporting."""
        self._refresh_minute_window()
        return {
            "attempted_requests": self.attempted_requests,
            "successful_requests": self.successful_requests,
            "work_items_started": self.work_items_started,
            "provider_remaining": self.provider_remaining,
            "provider_reset": self.provider_reset,
            "retry_after": self.retry_after,
            "exhausted_reason": self.exhausted_reason,
            "remaining": {
                "minute": max(self.requests_per_minute - self._minute_requests, 0),
                "day": max(
                    self.requests_per_day - self.day_requests - self.attempted_requests,
                    0,
                ),
                "run": max(self.requests_per_run - self.attempted_requests, 0),
                "work_items": max(
                    self.max_work_items - self.work_items_started,
                    0,
                ),
            },
        }

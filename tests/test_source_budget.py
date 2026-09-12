"""tests/test_source_budget.py
Offline tests for per-run request and work-item reservation limits.
"""

import pytest

from src.scheduler.budget import BudgetExhaustedError, RunBudget


def _budget(**overrides) -> RunBudget:
    values = {
        "requests_per_minute": 3,
        "requests_per_day": 10,
        "requests_per_run": 5,
        "max_work_items": 4,
    }
    values.update(overrides)
    return RunBudget(**values)


def test_budget_reserves_capacity_before_requests_start():
    budget = _budget(requests_per_minute=2)

    assert budget.reserve(requests=2, work_items=1) is True
    assert budget.reserve(requests=1, work_items=1) is False
    assert budget.exhausted_reason == "requests_per_minute"
    assert budget.attempted_requests == 2
    assert budget.work_items_started == 1


def test_budget_enforces_day_and_run_limits_from_existing_usage():
    daily = _budget(requests_per_day=5, day_requests=4)
    per_run = _budget(requests_per_run=2)

    assert daily.reserve(requests=2) is False
    assert daily.exhausted_reason == "requests_per_day"
    assert per_run.reserve(requests=2) is True
    assert per_run.reserve(requests=1) is False
    assert per_run.exhausted_reason == "requests_per_run"


def test_budget_enforces_work_item_limit_independently():
    budget = _budget(max_work_items=2)

    assert budget.reserve(requests=1, work_items=2) is True
    assert budget.reserve(requests=1, work_items=1) is False
    assert budget.exhausted_reason == "max_work_items_per_run"


def test_provider_remaining_header_reduces_available_capacity():
    budget = _budget()
    budget.update_provider_limits(
        {
            "X-RateLimit-Remaining": "1",
            "X-RateLimit-Reset": "1784048400",
        }
    )

    assert budget.reserve(requests=1) is True
    assert budget.reserve(requests=1) is False
    assert budget.exhausted_reason == "provider_remaining"
    assert budget.provider_reset == "1784048400"


def test_successes_and_snapshot_are_tracked_separately_from_attempts():
    budget = _budget()
    assert budget.reserve(requests=2, work_items=1) is True
    budget.record_success(1)

    snapshot = budget.snapshot()

    assert snapshot["attempted_requests"] == 2
    assert snapshot["successful_requests"] == 1
    assert snapshot["work_items_started"] == 1
    assert snapshot["remaining"]["run"] == 3


def test_force_is_not_a_budget_bypass():
    budget = _budget(requests_per_day=1, day_requests=1)

    assert budget.reserve(requests=1, force=True) is False
    assert budget.attempted_requests == 0
    assert budget.exhausted_reason == "requests_per_day"


def test_budgeted_http_client_reserves_each_attempt_and_applies_headers():
    calls = []

    class Response:
        status_code = 200
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "123"}

    def request(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    budget = _budget()
    http_get = budget.wrap_http_get(request)

    assert http_get("https://example.test/first", timeout=1).status_code == 200
    with pytest.raises(BudgetExhaustedError, match="provider_remaining"):
        http_get("https://example.test/second", timeout=1)

    assert len(calls) == 1
    assert budget.attempted_requests == 1
    assert budget.successful_requests == 1
    assert budget.provider_reset == "123"


def test_budgeted_http_client_waits_for_next_minute_without_bypassing_quota():
    clock = {"now": 10.0}
    sleeps = []

    class Response:
        status_code = 200
        headers = {}

    budget = _budget(
        requests_per_minute=1,
        now_fn=lambda: clock["now"],
        minute_started=0.0,
    )

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    http_get = budget.wrap_http_get(
        lambda *_args, **_kwargs: Response(),
        wait_for_minute=True,
        max_wait_seconds=60,
        sleep_fn=sleep,
    )

    assert http_get("https://example.test/first").status_code == 200
    assert http_get("https://example.test/second").status_code == 200
    assert sleeps == [60.0]
    assert budget.attempted_requests == 2


def test_alpha_vantage_style_minute_budget_paces_instead_of_aborting():
    """A 5/min free-tier budget must wait for the window rather than skip the 6th call."""
    clock = {"now": 0.0}
    sleeps: list[float] = []

    class Response:
        status_code = 200
        headers = {}

    budget = _budget(
        requests_per_minute=5,
        requests_per_run=20,
        requests_per_day=25,
        now_fn=lambda: clock["now"],
        minute_started=0.0,
    )

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    http_get = budget.wrap_http_get(
        lambda *_args, **_kwargs: Response(),
        wait_for_minute=True,
        max_wait_seconds=60,
        sleep_fn=sleep,
    )

    for index in range(6):
        assert http_get(f"https://example.test/{index}").status_code == 200

    assert budget.attempted_requests == 6
    assert budget.exhausted_reason is None
    # Five paced intervals of 12s plus a residual wait into the next minute window.
    assert sleeps
    assert sum(sleeps) >= 60.0
    assert all(delay > 0 for delay in sleeps)

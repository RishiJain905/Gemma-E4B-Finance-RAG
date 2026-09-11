"""tests/test_provider_failure_policy.py
Offline contract tests for normalized provider retries and source circuits.
"""

from datetime import datetime, timezone
from email.utils import format_datetime

import pytest

from src.ingestion.errors import ErrorClass, ProviderError, parse_retry_after
from src.utils.resilience import ProviderRequestPolicy, provider_request_policy


class _Response:
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
        payload: object | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload if payload is not None else {"message": "provider error"}
        self.text = str(self._payload)

    def json(self) -> object:
        return self._payload


def test_retry_after_accepts_numeric_and_http_date_under_mocked_clock() -> None:
    now = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    reset = datetime(2026, 7, 14, 12, 2, 30, tzinfo=timezone.utc)

    numeric = parse_retry_after("90", now=now)
    dated = parse_retry_after(format_datetime(reset, usegmt=True), now=now)

    assert numeric.delay_seconds == 90.0
    assert numeric.reset_at == "2026-07-14T12:01:30Z"
    assert dated.delay_seconds == 150.0
    assert dated.reset_at == "2026-07-14T12:02:30Z"


def test_three_exhausted_429_attempts_open_circuit_before_later_partition() -> None:
    requested: list[str] = []
    sleeps: list[float] = []
    policy = ProviderRequestPolicy(
        source="vendor",
        max_attempts=3,
        base_delay=1,
        max_sleep=30,
        jitter_ratio=0,
        sleep_fn=sleeps.append,
        now_fn=lambda: datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
    )

    def request(partition: str) -> _Response:
        requested.append(partition)
        return _Response(429, headers={"Retry-After": "2"})

    with pytest.raises(ProviderError) as caught:
        policy.request(lambda: request("AAA"))

    with pytest.raises(ProviderError) as open_circuit:
        policy.request(lambda: request("BBB"))

    assert requested == ["AAA", "AAA", "AAA"]
    assert sleeps == [2.0, 2.0]
    assert caught.value.error_class is ErrorClass.RATE_LIMITED
    assert caught.value.attempts == 3
    assert len(caught.value.retry_timestamps) == 2
    assert caught.value.reset_at == "2026-07-14T12:00:02Z"
    assert open_circuit.value.circuit_open is True
    assert policy.remaining_work_skipped is True


@pytest.mark.parametrize(
    ("status_code", "expected_class"),
    [(401, ErrorClass.AUTHENTICATION), (403, ErrorClass.ENTITLEMENT)],
)
def test_authentication_and_entitlement_are_not_retried(
    status_code: int,
    expected_class: ErrorClass,
) -> None:
    calls = 0
    policy = ProviderRequestPolicy(
        source="vendor",
        max_attempts=3,
        sleep_fn=lambda _delay: None,
    )

    def request() -> _Response:
        nonlocal calls
        calls += 1
        return _Response(status_code)

    with pytest.raises(ProviderError) as caught:
        policy.request(request)

    assert calls == 1
    assert caught.value.error_class is expected_class
    assert caught.value.attempts == 1
    assert policy.circuit_opened_at is not None


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (
            402,
            "Premium Query Parameter: symbol is not available under your current subscription",
        ),
        (
            402,
            "Special Endpoint : this value set for 'symbol' is not available under your current subscription",
        ),
        (
            403,
            'Special Endpoint : this value set for "symbol" requires you to upgrade your plan',
        ),
    ],
)
def test_fmp_special_endpoint_402_is_item_not_entitlement(
    status_code: int,
    message: str,
) -> None:
    from src.ingestion.errors import error_class_for_http

    assert error_class_for_http(status_code, message) is ErrorClass.ITEM
    # Provider-wide plan blocks without symbol scoping remain entitlement.
    assert (
        error_class_for_http(403, "plan does not include this entitlement")
        is ErrorClass.ENTITLEMENT
    )


def test_fmp_item_skip_does_not_open_provider_circuit_mid_batch() -> None:
    """ITEM-class symbol blocks must not trip the HTTP circuit for later tickers."""
    requested: list[str] = []
    policy = ProviderRequestPolicy(
        source="fmp",
        max_attempts=1,
        sleep_fn=lambda _delay: None,
        now_fn=lambda: datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
    )

    def request(partition: str) -> _Response:
        requested.append(partition)
        if partition == "CRWD":
            return _Response(
                402,
                payload={
                    "Error Message": (
                        "Special Endpoint : this value set for 'symbol' is not "
                        "available under your current subscription"
                    )
                },
            )
        return _Response(200, payload=[{"symbol": partition}])

    with pytest.raises(ProviderError) as caught:
        policy.request(lambda: request("CRWD"))

    assert caught.value.error_class is ErrorClass.ITEM
    assert caught.value.circuit_open is False
    assert policy.circuit_opened_at is None
    assert policy.remaining_work_skipped is False

    # Later free-tier ticker still runs — circuit stayed closed.
    ok = policy.request(lambda: request("NVDA"))
    assert ok.status_code == 200
    assert requested == ["CRWD", "NVDA"]


def test_provider_error_message_redacts_credentials_and_payload_size() -> None:
    policy = ProviderRequestPolicy(source="vendor", max_attempts=1)
    response = _Response(
        401,
        payload={
            "message": "token=super-secret api_key=also-secret " + ("x" * 1_000)
        },
    )

    with pytest.raises(ProviderError) as caught:
        policy.request(lambda: response)

    message = caught.value.safe_message
    assert "super-secret" not in message
    assert "also-secret" not in message
    assert len(message) <= 300


def test_retry_after_sleep_is_capped_without_losing_provider_reset() -> None:
    now = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    responses = iter(
        [
            _Response(429, headers={"Retry-After": "600"}),
            _Response(200),
        ]
    )
    sleeps: list[float] = []
    policy = ProviderRequestPolicy(
        source="vendor",
        max_attempts=2,
        max_sleep=30,
        jitter_ratio=0,
        sleep_fn=sleeps.append,
        now_fn=lambda: now,
    )

    assert policy.request(lambda: next(responses)).status_code == 200
    assert sleeps == [30.0]
    assert policy.last_error is not None
    assert policy.last_error.reset_at == "2026-07-14T12:10:00Z"


def test_massive_minute_limit_is_rate_limited_without_immediate_retry() -> None:
    calls = 0
    policy = provider_request_policy(
        "massive",
        "vendor_market",
        sleep_fn=lambda _delay: None,
        now_fn=lambda: datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
    )

    def request() -> _Response:
        nonlocal calls
        calls += 1
        return _Response(
            429,
            payload={"error": "maximum requests per minute; wait or upgrade your subscription"},
        )

    with pytest.raises(ProviderError) as caught:
        policy.request(request)

    assert calls == 1
    assert caught.value.error_class is ErrorClass.RATE_LIMITED


@pytest.mark.parametrize("status_code", [403, 429])
def test_sec_throttling_is_rate_limited_once_with_cooldown(status_code: int) -> None:
    calls = 0
    policy = ProviderRequestPolicy(
        source="sec_filings",
        max_attempts=1,
        now_fn=lambda: datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
    )

    def request() -> _Response:
        nonlocal calls
        calls += 1
        return _Response(status_code, headers={"Retry-After": "60"})

    with pytest.raises(ProviderError) as caught:
        policy.request(request)

    assert calls == 1
    assert caught.value.error_class is ErrorClass.RATE_LIMITED
    assert caught.value.reset_at == "2026-07-14T12:01:00Z"
    assert policy.remaining_work_skipped is False

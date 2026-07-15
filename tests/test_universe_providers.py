"""tests/test_universe_providers.py
Offline contract tests for bounded universe provider adapters.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.universe.models import SnapshotValidationError
from src.universe.providers import (
    IVVHoldingsProvider,
    Nasdaq100Provider,
    SECCompanyTickersProvider,
    provider_from_config,
)


FIXTURES = Path(__file__).parent / "fixtures" / "universe"


def test_nasdaq_provider_parses_pinned_fixture() -> None:
    provider = Nasdaq100Provider(min_constituents=4)

    rows = provider.parse(FIXTURES / "nasdaq100.csv")

    assert [row.symbol for row in rows] == ["AAPL", "GOOG", "GOOGL", "MSFT"]
    assert all(row.index_code == "nasdaq100" for row in rows)
    assert rows[0].company_name == "Apple Inc."
    assert rows[0].sector == "Technology"


def test_ivv_provider_skips_cash_and_derivatives() -> None:
    provider = IVVHoldingsProvider(min_constituents=4)

    rows = provider.parse(FIXTURES / "ivv_holdings.csv")

    assert [row.symbol for row in rows] == ["AAPL", "BRK.B", "GOOGL", "MSFT"]
    assert all(row.index_code == "sp500" for row in rows)
    assert all(row.security_type == "common_stock" for row in rows)


def test_ivv_provider_scans_nine_metadata_rows_before_header() -> None:
    payload = "\n".join(
        [f"metadata row {index}" for index in range(9)]
        + [
            '"Ticker","Name","Sector","Asset Class","Exchange"',
            "AAPL,APPLE INC,Information Technology,Equity,NASDAQ",
        ]
    )

    rows = IVVHoldingsProvider(min_constituents=1).parse(payload)

    assert [row.symbol for row in rows] == ["AAPL"]


def test_nasdaq_provider_fetches_public_json_api_with_browser_headers() -> None:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": {
            "data": {
                "rows": [
                    {
                        "symbol": "AAPL",
                        "companyName": "Apple Inc.",
                        "sector": "Technology",
                    }
                ]
            }
        }
    }
    with patch("src.universe.providers.requests.get", return_value=response) as get:
        rows = Nasdaq100Provider(min_constituents=1).fetch()

    assert [row.symbol for row in rows] == ["AAPL"]
    assert get.call_args.kwargs["headers"]["Accept"] == "application/json"
    assert "Mozilla/5.0" in get.call_args.kwargs["headers"]["User-Agent"]


def test_provider_factory_uses_configured_url_timeout_and_minimum(tmp_path: Path) -> None:
    config = tmp_path / "universe.yaml"
    config.write_text(
        """providers:
  nasdaq100:
    url: https://api.example.test/nasdaq100
    min_constituents: 97
    timeout_seconds: 12
""",
        encoding="utf-8",
    )

    provider = provider_from_config("universe_nasdaq100", config_path=config)

    assert provider.source_url == "https://api.example.test/nasdaq100"
    assert provider.min_constituents == 97
    assert provider.timeout == 12


def test_sec_provider_parses_ticker_cik_and_exchange() -> None:
    provider = SECCompanyTickersProvider(min_records=5)

    rows = provider.parse(FIXTURES / "sec_company_tickers.json")

    assert len(rows) == 6
    apple = next(row for row in rows if row.symbol == "AAPL")
    assert apple.cik == "0000320193"
    assert apple.exchange == "Nasdaq"
    assert apple.index_code is None


@pytest.mark.parametrize(
    ("provider", "payload"),
    [
        (Nasdaq100Provider(min_constituents=1), "Company Name\nApple Inc.\n"),
        (
            IVVHoldingsProvider(min_constituents=1),
            "Ticker,Name,Sector,Asset Class\n,Apple Inc.,Technology,Equity\n",
        ),
        (SECCompanyTickersProvider(min_records=1), "{}"),
    ],
)
def test_providers_reject_malformed_or_empty_identifiers(provider, payload: str) -> None:
    with pytest.raises(SnapshotValidationError):
        provider.parse(payload)


def test_provider_rejects_duplicate_symbols() -> None:
    payload = (
        "Symbol,Company Name,Sector,Industry\n"
        "AAPL,Apple Inc.,Technology,Hardware\n"
        "AAPL,Apple Inc.,Technology,Hardware\n"
    )

    with pytest.raises(SnapshotValidationError, match="duplicate"):
        Nasdaq100Provider(min_constituents=1).parse(payload)


def test_provider_rejects_implausibly_small_snapshot() -> None:
    payload = (
        "Symbol,Company Name,Sector,Industry\n"
        "AAPL,Apple Inc.,Technology,Hardware\n"
    )

    with pytest.raises(SnapshotValidationError, match="at least 2"):
        Nasdaq100Provider(min_constituents=2).parse(payload)

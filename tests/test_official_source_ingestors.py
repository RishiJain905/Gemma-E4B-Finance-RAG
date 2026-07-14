"""Offline contract tests for Phase 2.3.2.3 official-source adapters."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.storage.store import Store


FIXTURES = Path(__file__).parent / "fixtures" / "providers" / "official"


class FakeResponse:
    """Small requests-compatible response for deterministic adapter tests."""

    def __init__(self, payload: object, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        if isinstance(payload, bytes):
            self.content = payload
            self.text = payload.decode("utf-8")
        elif isinstance(payload, str):
            self.content = payload.encode("utf-8")
            self.text = payload
        else:
            self.content = json.dumps(payload).encode("utf-8")
            self.text = json.dumps(payload)

    def json(self) -> object:
        if isinstance(self.payload, (str, bytes)):
            return json.loads(self.text)
        return self.payload


def _load_json(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _load_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _store(tmp_path: Path) -> tuple[Store, MagicMock]:
    with patch("src.storage.store.ChromaStore") as chroma_class:
        chroma = MagicMock()
        chroma_class.return_value = chroma
        store = Store(db_path=tmp_path / "official.db", chroma_path=tmp_path / "chroma")
    store.upsert_universe_snapshot(
        "sec",
        "2026-07-14T00:00:00Z",
        [
            {"symbol": "AAA", "company_name": "Alpha Health Corp", "sector": "Health Care"},
            {"symbol": "AH1", "company_name": "Ambiguous Health", "sector": "Health Care"},
            {"symbol": "AH2", "company_name": "Ambiguous Health", "sector": "Health Care"},
            {"symbol": "MOT", "company_name": "Alpha Motors", "sector": "Automotive"},
            {"symbol": "AM1", "company_name": "Ambiguous Motors", "sector": "Automotive"},
            {"symbol": "AM2", "company_name": "Ambiguous Motors", "sector": "Automotive"},
            {
                "symbol": "DEF",
                "company_name": "Alpha Defense Systems",
                "sector": "Aerospace & Defense",
            },
            {
                "symbol": "AD1",
                "company_name": "Ambiguous Defense",
                "sector": "Aerospace & Defense",
            },
            {
                "symbol": "AD2",
                "company_name": "Ambiguous Defense",
                "sector": "Aerospace & Defense",
            },
        ],
    )
    return store, chroma


def _coverage() -> MagicMock:
    coverage = MagicMock()
    coverage.tickers_for.return_value = ["AAA", "AH1", "AH2", "MOT", "AM1", "AM2", "DEF", "AD1", "AD2"]
    coverage.is_enabled.return_value = True
    return coverage


def test_catalog_entries_are_curated_and_have_the_required_provenance_shape() -> None:
    catalog = yaml.safe_load((Path("configs") / "official_sources.yaml").read_text(encoding="utf-8"))
    entries = catalog["entries"]
    agencies = {entry["agency"] for entry in entries}

    assert agencies == {
        "federal_reserve",
        "treasury",
        "bls",
        "bea",
        "eia",
        "ny_fed",
        "cftc",
        "openfda",
        "nhtsa",
        "usaspending",
    }
    required = {
        "name",
        "agency",
        "dataset_id",
        "unit",
        "frequency",
        "expected_cadence",
        "source_category",
        "sectors",
        "payload_type",
        "endpoint",
    }
    assert entries
    assert all(required <= set(entry) for entry in entries)
    assert all(entry["payload_type"] in {"structured", "narrative", "both"} for entry in entries)
    assert len({entry["name"] for entry in entries}) == len(entries)


def test_macro_adapters_parse_pinned_formats_and_use_the_correct_store_path(tmp_path: Path) -> None:
    from src.ingestion.official.bea import BEAIngestor
    from src.ingestion.official.bls import BLSIngestor
    from src.ingestion.official.cftc import CFTCIngestor
    from src.ingestion.official.eia import EIAIngestor
    from src.ingestion.official.federal_reserve import FederalReserveIngestor
    from src.ingestion.official.ny_fed import NYFedIngestor
    from src.ingestion.official.treasury import TreasuryIngestor

    store, chroma = _store(tmp_path)
    common = {
        "store": store,
        "coverage_resolver": _coverage(),
        "now_fn": lambda: "2026-07-15T00:00:00Z",
    }
    assert FederalReserveIngestor(**common).ingest(payload=_load_text("federal_reserve_rss.xml"))["stored"] == 5
    assert TreasuryIngestor(**common).ingest(payload=_load_text("treasury_daily.csv"))["stored"] == 3
    assert BLSIngestor(**common, api_key="bls-test").ingest(payload=_load_json("bls_release.json"))["stored"] == 5
    assert BEAIngestor(**common, api_key="bea-test").ingest(payload=_load_json("bea_release.json"))["stored"] == 4
    assert EIAIngestor(**common, api_key="eia-test").ingest(payload=_load_json("eia_release.json"))["stored"] == 4
    assert NYFedIngestor(**common).ingest(payload=_load_text("ny_fed_market.csv"))["stored"] == 3
    assert CFTCIngestor(**common).ingest(payload=_load_text("cftc_cot.csv"))["stored"] == 2

    assert store.sqlite.count_observations() == 19
    assert store.sqlite.count_events() == 1
    assert chroma.add_document.call_count == 6
    observation = store.sqlite.list_observations(source_name="treasury", limit=10)[0]
    assert observation["source_url"].startswith("https://home.treasury.gov")
    assert observation["vintage_at"] == "2026-07-14T12:00:00Z"
    assert observation["provider_record_id"] == "treasury-20260713"


def test_revised_observation_keeps_the_prior_vintage_in_sqlite(tmp_path: Path) -> None:
    from src.ingestion.official.treasury import TreasuryIngestor

    store, _chroma = _store(tmp_path)
    payload = (
        "Date,BC_10YEAR,REAL_10YEAR,BILL_4WEEK,vintage_at,source_id\n"
        "2026-07-13,4.27,1.83,4.35,2026-07-14T12:00:00Z,treasury-20260713\n"
        "2026-07-13,4.31,1.85,4.36,2026-07-15T12:00:00Z,treasury-20260713\n"
    )
    result = TreasuryIngestor(store=store).ingest(payload=payload)

    assert result["stored"] == 6
    with store.sqlite._connect() as conn:
        rows = conn.execute(
            "SELECT value_numeric, vintage_at FROM corpus_observations "
            "WHERE source_name=? AND metric_id=? ORDER BY vintage_at",
            ("treasury", "treasury_nominal_yield_10y"),
        ).fetchall()
    assert [(row["value_numeric"], row["vintage_at"]) for row in rows] == [
        (4.27, "2026-07-14T12:00:00Z"),
        (4.31, "2026-07-15T12:00:00Z"),
    ]


def test_sector_feeds_attach_only_exact_unambiguous_registry_identities(tmp_path: Path) -> None:
    from src.ingestion.official.nhtsa import NHTSAIngestor
    from src.ingestion.official.openfda import OpenFDAIngestor
    from src.ingestion.official.usaspending import USAspendingIngestor

    store, _chroma = _store(tmp_path)
    common = {"store": store, "coverage_resolver": _coverage()}
    OpenFDAIngestor(**common, api_key="openfda-test").ingest(payload=_load_json("openfda_events.json"))
    NHTSAIngestor(**common).ingest(payload=_load_json("nhtsa_events.json"))
    USAspendingIngestor(**common).ingest(payload=_load_json("usaspending_awards.json"))

    assert store.sqlite.count_events() == 7
    assert store.sqlite.get_event("event/openfda/FDA-REC-1")["security_ids"] == [store.get_security("AAA")["security_id"]]
    assert store.sqlite.get_event("event/nhtsa/NHTSA-REC-1")["security_ids"] == [store.get_security("MOT")["security_id"]]
    assert store.sqlite.get_event("event/usaspending/AWD-ALPHA-1")["security_ids"] == [store.get_security("DEF")["security_id"]]
    assert store.sqlite.get_event("event/openfda/FDA-SAF-1")["security_ids"] == []
    assert store.sqlite.get_event("event/nhtsa/NHTSA-INV-1")["security_ids"] == []
    assert store.sqlite.get_event("event/usaspending/AWD-AMBIG-1")["security_ids"] == []


def test_registry_uei_aliases_are_exact_and_do_not_use_substrings(tmp_path: Path) -> None:
    store, _chroma = _store(tmp_path)
    defense_id = store.get_security("DEF")["security_id"]

    assert store.register_security_alias(
        defense_id,
        "UEI-ALPHA",
        alias_type="recipient_uei",
        source="official_fixture",
    ) is True
    assert store.resolve_exact_security("UEI-ALPHA")["security_id"] == defense_id
    assert store.resolve_exact_security("UEI-ALPHA-EXTRA") is None


def test_missing_api_keys_disable_only_keyed_agencies_and_no_key_feeds_continue(tmp_path: Path) -> None:
    from src.ingestion.official.bea import BEAIngestor
    from src.ingestion.official.bls import BLSIngestor
    from src.ingestion.official.eia import EIAIngestor
    from src.ingestion.official.federal_reserve import FederalReserveIngestor
    from src.ingestion.official.treasury import TreasuryIngestor

    store, _chroma = _store(tmp_path)
    http_get = MagicMock(return_value=FakeResponse(_load_text("federal_reserve_rss.xml")))
    assert BLSIngestor(store=store, api_key="", http_get=http_get).ingest()["status"] == "disabled_missing_key"
    assert BEAIngestor(store=store, api_key="", http_get=http_get).ingest()["status"] == "disabled_missing_key"
    assert EIAIngestor(store=store, api_key="", http_get=http_get).ingest()["status"] == "disabled_missing_key"
    assert http_get.call_count == 0

    assert FederalReserveIngestor(store=store, http_get=http_get).ingest()["status"] == "ok"
    assert TreasuryIngestor(store=store, http_get=MagicMock(return_value=FakeResponse(_load_text("treasury_daily.csv")))).ingest()["status"] == "ok"


@pytest.mark.parametrize(
    ("module", "class_name", "fixture", "needs_key"),
    [
        ("federal_reserve", "FederalReserveIngestor", "federal_reserve_rss.xml", False),
        ("treasury", "TreasuryIngestor", "treasury_daily.csv", False),
        ("bls", "BLSIngestor", "bls_release.json", True),
        ("bea", "BEAIngestor", "bea_release.json", True),
        ("eia", "EIAIngestor", "eia_release.json", True),
        ("ny_fed", "NYFedIngestor", "ny_fed_market.csv", False),
        ("cftc", "CFTCIngestor", "cftc_cot.csv", False),
        ("openfda", "OpenFDAIngestor", "openfda_events.json", True),
        ("nhtsa", "NHTSAIngestor", "nhtsa_events.json", False),
        ("usaspending", "USAspendingIngestor", "usaspending_awards.json", False),
    ],
)
def test_replaying_each_official_fixture_is_idempotent(
    tmp_path: Path, module: str, class_name: str, fixture: str, needs_key: bool,
) -> None:
    import importlib

    module_object = importlib.import_module(f"src.ingestion.official.{module}")
    ingestor_class = getattr(module_object, class_name)
    store, chroma = _store(tmp_path)
    payload = _load_text(fixture) if fixture.endswith((".csv", ".xml")) else _load_json(fixture)
    kwargs = {
        "store": store,
        "coverage_resolver": _coverage(),
        "api_key": "fixture-key" if needs_key else None,
        "now_fn": lambda: "2026-07-15T00:00:00Z",
    }
    ingestor = ingestor_class(**kwargs)

    first = ingestor.ingest(payload=payload)
    counts_after_first = (store.sqlite.count_observations(), store.sqlite.count_events(), store.sqlite.count_corpus_items())
    second = ingestor.ingest(payload=payload)

    assert first["status"] in {"ok", "partial"}
    assert second["stored"] == 0
    assert second["duplicates"] >= first["stored"]
    assert (store.sqlite.count_observations(), store.sqlite.count_events(), store.sqlite.count_corpus_items()) == counts_after_first
    if class_name == "FederalReserveIngestor":
        assert chroma.add_document.call_count == first["stored"]


def test_csv_fixture_is_pinned_and_parseable_without_network() -> None:
    rows = list(csv.DictReader(io.StringIO(_load_text("treasury_daily.csv"))))
    assert rows[0]["source_id"] == "treasury-20260713"

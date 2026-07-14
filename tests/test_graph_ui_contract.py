"""tests/test_graph_ui_contract.py
Offline contract tests for the Phase 2.2.7.3 static graph interface.

Covers serving gating (404 when the observer is off), MIME types, traversal
rejection, pinned vendor hash/license integrity, required HTML landmarks and
controls, and static hygiene (no external URLs, no innerHTML-with-dynamic-data,
local font references).
"""

from __future__ import annotations

import hashlib
import re
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.middleware.app as appmod

STATIC_DIR = Path(appmod.__file__).parent / "static" / "graph"
INDEX = STATIC_DIR / "index.html"
CSS = STATIC_DIR / "graph.css"
JS = STATIC_DIR / "graph.js"
VENDOR = STATIC_DIR / "vendor"
LOOPBACK = ("127.0.0.1", 50000)


@pytest.fixture
def restore_config():
    """Set the module-level config without running the heavy lifespan."""
    original = appmod.config
    yield lambda enabled: setattr(
        appmod, "config", types.SimpleNamespace(enable_graph_observer=enabled)
    )
    appmod.config = original


def _client() -> TestClient:
    # Plain instantiation (no context manager) so lifespan does not overwrite
    # the config we set in the fixture or spin up Store/model clients.
    return TestClient(appmod.app, client=LOOPBACK)


# ── Serving gate ─────────────────────────────────────────────────────────────

def test_graph_index_404_when_disabled(restore_config):
    restore_config(False)
    assert _client().get("/graph").status_code == 404


def test_graph_static_404_when_disabled(restore_config):
    restore_config(False)
    assert _client().get("/graph/static/graph.css").status_code == 404


def test_graph_index_served_when_enabled(restore_config):
    restore_config(True)
    resp = _client().get("/graph")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "TRACE" in resp.text and 'id="graph-canvas"' in resp.text


@pytest.mark.parametrize(
    "asset, ctype",
    [
        ("graph.css", "text/css"),
        ("graph.js", "text/javascript"),
        ("vendor/cytoscape.min.js", "text/javascript"),
        ("fonts/CommitMono.woff2", "font/woff2"),
        ("fonts/RecursiveSans.woff2", "font/woff2"),
    ],
)
def test_static_assets_have_safe_mime(restore_config, asset, ctype):
    restore_config(True)
    resp = _client().get(f"/graph/static/{asset}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(ctype)


# ── Traversal / containment ──────────────────────────────────────────────────

def test_traversal_is_rejected(restore_config):
    restore_config(True)
    with pytest.raises(appmod.HTTPException) as exc:
        appmod._graph_static_file("../../config.py")
    assert exc.value.status_code == 404


def test_absolute_and_disallowed_extension_rejected(restore_config):
    restore_config(True)
    for bad in ("../app.py", "..\\..\\config.py", "graph.py", "vendor/SHA256SUMS.bak"):
        with pytest.raises(appmod.HTTPException):
            appmod._graph_static_file(bad)


def test_static_route_rejects_unknown_asset(restore_config):
    restore_config(True)
    assert _client().get("/graph/static/does-not-exist.js").status_code == 404


# ── Vendor hash + license integrity ──────────────────────────────────────────

def test_pinned_vendor_hashes_match_sha256sums():
    sums = (VENDOR / "SHA256SUMS").read_text(encoding="utf-8")
    entries = {}
    for line in sums.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        entries[name.lstrip("*").strip()] = digest
    assert entries, "SHA256SUMS must pin at least one asset"
    for name, expected in entries.items():
        target = STATIC_DIR / name
        assert target.is_file(), f"pinned asset missing: {name}"
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual == expected, f"hash drift for {name}"


def test_license_files_present():
    assert (VENDOR / "LICENSE-cytoscape.txt").is_file()
    assert (VENDOR / "LICENSE-fonts.txt").is_file()


# ── HTML landmarks, controls, aria ───────────────────────────────────────────

def test_index_has_landmarks_and_controls():
    html = INDEX.read_text(encoding="utf-8")
    required = [
        'role="banner"', "<main", 'role="complementary"', 'role="tablist"',
        'role="application"', 'aria-live="polite"', 'class="skip-link"',
        'id="query-picker"', 'id="toggle-follow"', 'id="scrubber"',
        'id="legend"', 'id="btn-fit"', 'id="btn-reset"', 'id="btn-pin"',
        'id="btn-collapse"', 'id="btn-export"', 'id="filter-status"',
        'id="filter-subquery"', 'id="filter-source"', 'id="filter-ticker"',
        'id="filter-kind"', 'id="filter-cited"', 'id="corpus-search"',
        'id="inspector"', "<table", 'id="conn-state"', 'id="error-panel"',
        'id="table-view"', 'id="node-list"',
    ]
    for token in required:
        assert token in html, f"index.html missing {token}"


def test_index_references_local_assets_only():
    html = INDEX.read_text(encoding="utf-8")
    assert "/graph/static/vendor/cytoscape.min.js" in html
    assert "/graph/static/graph.js" in html
    assert "/graph/static/graph.css" in html


# ── Static hygiene ───────────────────────────────────────────────────────────

def _strip_data_uris(text: str) -> str:
    # data: URIs are inline (never fetched) and may legitimately embed an XML
    # namespace like www.w3.org; drop them before the external-URL scan.
    return re.sub(r"data:[^\"')\s]+", "", text)


@pytest.mark.parametrize("path", [INDEX, CSS, JS])
def test_no_external_urls(path):
    text = _strip_data_uris(path.read_text(encoding="utf-8"))
    hits = re.findall(r"https?://[^\s\"')]+", text)
    assert not hits, f"external URL(s) in {path.name}: {hits}"


def test_graph_js_avoids_innerhtml_with_dynamic_data():
    js = JS.read_text(encoding="utf-8")
    assert ".innerHTML" not in js
    assert "insertAdjacentHTML" not in js


def test_fonts_referenced_locally():
    css = CSS.read_text(encoding="utf-8")
    assert "@font-face" in css
    assert "/graph/static/fonts/RecursiveSans.woff2" in css
    assert "/graph/static/fonts/CommitMono.woff2" in css
    assert "/graph/static/fonts/CommitMono-700.woff2" in css


def test_pure_state_namespace_is_defined_in_js():
    js = JS.read_text(encoding="utf-8")
    for fn in ("applyDelta", "selectVisible", "computeBeam", "nodeMatchesFilters", "stageColumnFor"):
        assert fn in js, f"pure state function {fn} missing from graph.js"


# ── 2.2.7.4: golden trace wire fixture + deep-link ───────────────────────────

FIXTURE = Path(__file__).parent / "fixtures" / "graph" / "query_trace_v1.json"

_LAYOUT_KEYS = frozenset({"position", "x", "y", "renderedPosition", "bbox", "pan", "zoom"})


def test_golden_query_trace_fixture_matches_wire_schema():
    """The golden fixture is a schema contract only — node kinds, edge relations,
    allowlisted metadata, bounded excerpts. It carries NO pixel/layout positions;
    layout is UI behavior while graph meaning is the contract."""
    import json

    from src.middleware.graph_observer import (
        EDGE_RELATIONS,
        NODE_KINDS,
        NODE_STATUSES,
        _DENIED_KEY,
    )

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == 1
    snapshot = fixture["snapshot"]
    assert snapshot["schema_version"] == 1
    assert snapshot["complete"] is True

    node_ids = set()
    for node in snapshot["nodes"]:
        assert node["kind"] in NODE_KINDS, node["kind"]
        assert node["status"] in NODE_STATUSES, node["status"]
        node_ids.add(node["id"])
        metadata = node.get("metadata") or {}
        for key in metadata:
            assert not _DENIED_KEY.search(key), f"denied metadata key: {key}"
        assert not (_LAYOUT_KEYS & set(metadata)), "fixture must not embed layout"
        assert not (_LAYOUT_KEYS & set(node)), "fixture node must not embed layout"
        if node["kind"] == "evidence":
            assert len(node.get("summary", "")) <= 1000

    relations_seen = set()
    for edge in snapshot["edges"]:
        assert edge["relation"] in EDGE_RELATIONS, edge["relation"]
        assert edge["source"] in node_ids, f"dangling source {edge['source']}"
        assert edge["target"] in node_ids, f"dangling target {edge['target']}"
        relations_seen.add(edge["relation"])
    # The trace threads evidence -> answer and evidence -> citation provenance.
    assert {"supports", "cited_by", "from_source"} <= relations_seen


def test_graph_js_supports_trace_deep_link_from_hash():
    js = JS.read_text(encoding="utf-8")
    assert "traceFromHash" in js
    assert "applyTraceHash" in js
    assert "location.hash" in js
    assert "hashchange" in js


# ── 2.2.7.x graph-UI defect fixes (guard against regression) ─────────────────

def test_graph_js_maps_legacy_intent_to_route_column():
    """Defect 1: the legacy intent/route stage node lands in the ROUTE column."""
    js = JS.read_text(encoding="utf-8")
    assert "intent: 1" in js


def test_graph_js_syncs_stage_rail_to_viewport():
    """Defect 3: the stage rail is projected through the Cytoscape viewport."""
    js = JS.read_text(encoding="utf-8")
    assert "syncStageRail" in js
    assert "pan zoom resize" in js
    assert "cy.zoom()" in js and "cy.pan()" in js


def test_graph_js_scrubber_is_raf_throttled_and_instant():
    """Defect 4: scrubbing coalesces per frame, positions instantly, no re-fit."""
    js = JS.read_text(encoding="utf-8")
    assert "scheduleScrubRender" in js
    assert "app.scrubbing" in js
    assert "ele.stop(true)" in js


def test_graph_js_folds_corpus_freshness_leaves():
    """Defect 5: freshness leaves fold into their ticker; leaf labels use LOD."""
    js = JS.read_text(encoding="utf-8")
    assert "foldCorpusForCanvas" in js
    assert "updateCorpusLabelLOD" in js
    assert "freshWorst" in js

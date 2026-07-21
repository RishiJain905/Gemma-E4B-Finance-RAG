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


# ── 2.3.5.1: source-aware live trace ─────────────────────────────────────────

MIXED_FIXTURE = Path(__file__).parent / "fixtures" / "graph" / "query_trace_mixed_v1.json"


def test_mixed_source_fixture_matches_wire_schema_and_resolves_to_ledger():
    """The 2.3.5.1 mixed fixture (SEC financing + company news + market
    observation + macro release) is schema_version 1 — Phase 2.2 readers still
    parse it — and every source/evidence/citation resolves to the ledger."""
    import json

    from src.middleware.graph_observer import (
        EDGE_RELATIONS,
        NODE_KINDS,
        NODE_STATUSES,
        _DENIED_KEY,
    )

    fixture = json.loads(MIXED_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == 1
    snapshot = fixture["snapshot"]
    assert snapshot["schema_version"] == 1 and snapshot["complete"] is True

    node_ids = set()
    evidence_ids = set()
    categories = set()
    for node in snapshot["nodes"]:
        assert node["kind"] in NODE_KINDS, node["kind"]
        assert node["status"] in NODE_STATUSES, node["status"]
        node_ids.add(node["id"])
        metadata = node.get("metadata") or {}
        for key in metadata:
            assert not _DENIED_KEY.search(key), f"denied metadata key: {key}"
        assert not (_LAYOUT_KEYS & set(metadata)), "fixture must not embed layout"
        if node["kind"] == "evidence":
            assert len(node.get("summary", "")) <= 1000
            evidence_ids.add(metadata.get("evidence_id"))
            categories.add(metadata.get("source_category"))

    # Four distinct source categories, no provider-specific node kinds.
    assert categories == {"sec", "company_news", "market_data", "central_bank"}

    for edge in snapshot["edges"]:
        assert edge["relation"] in EDGE_RELATIONS, edge["relation"]
        assert edge["source"] in node_ids, f"dangling source {edge['source']}"
        assert edge["target"] in node_ids, f"dangling target {edge['target']}"
        # Every source/citation link resolves to a real evidence node.
        if edge["relation"] in ("from_source", "cited_by"):
            assert edge["source"].rsplit(":", 1)[-1] in evidence_ids

    # provider/publisher distinction is preserved on the wire.
    news = next(n for n in snapshot["nodes"]
                if n["kind"] == "evidence" and n["metadata"].get("source_category") == "company_news")
    assert news["metadata"]["provider"] == "finnhub"
    assert news["metadata"]["publisher"] == "Reuters"


def test_graph_js_renders_source_aware_metadata_and_subtypes():
    """The inspector flattens nested date-semantics, and evidence subtypes ride
    on metadata-driven data attributes (never new node kinds)."""
    js = JS.read_text(encoding="utf-8")
    assert "formatMetaValue" in js
    assert "isPlainObject" in js
    # Authority tier + primary/corroborating role drive style, not node kinds.
    assert "atier" in js and "erole" in js
    assert 'atier="primary"' in js or "atier=\"direct_sec\"" in js
    assert 'erole="corroborating"' in js


def test_graph_css_handles_richer_provenance_values():
    css = CSS.read_text(encoding="utf-8")
    assert "overflow-wrap: anywhere" in css


# ── 2.3.5.2: aggregation-first, faceted Corpus Explorer ──────────────────────

def test_corpus_explorer_has_three_surfaces_and_facet_rail():
    """Facet rail, inventory/results pane, and canvas are distinct surfaces; the
    inventory list is a keyboard-navigable listbox usable without Cytoscape."""
    html = INDEX.read_text(encoding="utf-8")
    for token in (
        'id="facet-rail"', 'aria-label="Corpus facets"', 'id="facet-groups"',
        'id="inventory-pane"', 'id="inventory-list"', 'role="listbox"',
        'id="inventory-empty"', 'id="inventory-count"', 'id="btn-inventory-more"',
        'id="corpus-presets"', 'id="btn-facets-clear"', 'id="btn-facets-toggle"',
        'id="work-area"',
    ):
        assert token in html, f"index.html missing {token}"


def test_corpus_presets_are_url_shortcuts_in_html_and_js():
    """All seven deterministic presets exist as saved entry points and expansions."""
    html = INDEX.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    presets = [
        "index-coverage", "financing-events", "latest-news", "macro-policy",
        "stale-sources", "indexing-backlog", "ticker-research",
    ]
    for preset in presets:
        assert f'value="{preset}"' in html, f"preset option {preset} missing"
        assert f'"{preset}"' in js, f"preset {preset} not expanded in CORPUS_PRESETS"
    assert "CORPUS_PRESETS" in js and "applyPreset" in js


def test_corpus_ia_pure_state_helpers_are_defined():
    """The URL<->facet-state layer is pure and testable, like GraphState."""
    js = JS.read_text(encoding="utf-8")
    for fn in (
        "CorpusIA", "parseCorpusHash", "corpusHash", "toggleFacetValue",
        "applyPreset", "activeFacetCount", "accountingFilters",
        "emptyCorpusState",
    ):
        assert fn in js, f"CorpusIA helper {fn} missing from graph.js"


def test_corpus_filter_state_lives_in_url_not_browser_storage():
    """Filter state is in the URL hash for refresh/back/forward/deep-link; no
    cookies, analytics, or localStorage (spec Step 1)."""
    js = JS.read_text(encoding="utf-8")
    assert "corpusHash" in js and "parseCorpusHash" in js
    assert "window.location.hash" in js
    assert "history.replaceState" in js
    assert "hashchange" in js
    for banned in ("localStorage", "sessionStorage", "document.cookie"):
        assert banned not in js, f"corpus state must not use {banned}"


def test_corpus_facet_counts_come_from_bounded_aggregate_endpoint():
    """Counts are shown before expansion, projected from the authoritative bounded
    aggregates endpoint — never inferred from a rendered corpus."""
    js = JS.read_text(encoding="utf-8")
    assert "corpus/aggregates" in js
    assert "refreshAggregates" in js and "renderFacetRail" in js
    assert "authorityCounts" in js  # authority tier derived from source-category buckets


def test_corpus_inventory_is_keyboard_navigable():
    """The inventory list is fully keyboard-navigable (spec testing list)."""
    js = JS.read_text(encoding="utf-8")
    assert "wireInventoryKeyboard" in js
    for key in ("ArrowDown", "ArrowUp", "Home", "End"):
        assert key in js, f"inventory keyboard missing {key}"
    assert 'role="option"' in js or 'setAttribute("role", "option")' in js


def test_corpus_expansion_is_bounded_and_drills_one_level():
    """Aggregation-first: the landing renders aggregate groups, and expansion adds
    one bounded level at a time under the visible-node cap."""
    js = JS.read_text(encoding="utf-8")
    assert "renderAggregatesInventory" in js and "drillAggregate" in js
    assert "CORPUS_DRILL_ORDER" in js
    assert "app.cap" in js  # per-page loads still respect the visible element cap


def test_corpus_preserves_freshness_fold_and_label_lod():
    """Phase 2.2 folding + label level-of-detail remain intact (spec Step 5)."""
    js = JS.read_text(encoding="utf-8")
    assert "foldCorpusForCanvas" in js and "freshWorst" in js
    assert "updateCorpusLabelLOD" in js
    # The inspector still surfaces the folded freshness rollup by status counts.
    assert "freshAgg" in js


def test_corpus_status_uses_text_and_glyph_not_colour_alone():
    """Status is text/icon/shape in addition to colour (2.2.7.3 rule)."""
    js = JS.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    assert "STATUS_GLYPH" in js
    for badge in (".st-badge", ".st-stale", ".st-error", ".st-indexing"):
        assert badge in css, f"status badge style {badge} missing"
    # Authority tier renders as a labelled pip, colour reinforcing the word.
    assert ".auth-primary" in css


def test_corpus_zero_empty_error_and_indexing_states_direct_action():
    """Empty/zero/error/stale/indexing states are present and actionable."""
    js = JS.read_text(encoding="utf-8")
    html = INDEX.read_text(encoding="utf-8")
    assert "No items match these facets" in js
    assert "still indexing" in js
    assert "INDEXING_LABELS" in js and "FRESH_LABELS" in js
    assert "Pick a source group or preset to explore the corpus." in html


def test_corpus_narrow_layout_uses_drawers_at_1280():
    """At ≤1280px the facet rail is an independent drawer and the result list
    stays usable with the canvas collapsed (spec Step 5)."""
    css = CSS.read_text(encoding="utf-8")
    assert "@media (max-width: 1280px)" in css
    assert '[data-mode="corpus"] .facet-rail' in css
    assert "translateX(-100%)" in css
    assert '[data-facets="open"]' in css
    # Work area reflows to a column so the inventory pane stays full-width.
    assert '[data-mode="corpus"] .work-area' in css


# ── 2.3.5.3: bounded projection, UI performance, and accessibility ───────────


def test_corpus_view_requests_abort_when_filters_change():
    """Obsolete corpus fetches are cancelled on a filter change so a slow older
    page can never overwrite the current facets (spec Step 4)."""
    js = JS.read_text(encoding="utf-8")
    assert "AbortController" in js
    assert "abortObsoleteCorpusRequests" in js
    # The choke point aborts before issuing the new view's requests.
    assert "corpusSignal" in js
    assert "isAbortError" in js


def test_inspector_is_closable_by_keyboard():
    """The inspector closes with Escape from anywhere outside a text field
    (spec Step 6 keyboard operation)."""
    js = JS.read_text(encoding="utf-8")
    assert 'e.key !== "Escape"' in js or 'e.key === "Escape"' in js
    assert "clearInspector" in js


def test_reduced_motion_disables_looping_graph_effects():
    """Reduced-motion stops the looping edge-flow animation entirely rather than
    spinning an idle rAF loop (spec Step 6)."""
    js = JS.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    assert "if (reduceMotion) return;" in js
    assert "@media (prefers-reduced-motion: reduce)" in css


# ── 2.3.5.4: collapsible corpus panels, canvas scoping, hub clustering ───────

def test_corpus_panels_have_collapse_controls():
    """Facets/Inventory are collapsible from both the toolbar and the panel heads;
    the error panel title now carries an id so showError() can address it."""
    html = INDEX.read_text(encoding="utf-8")
    for token in (
        'id="btn-facets-toggle"', 'id="btn-inventory-toggle"',
        'id="btn-facets-collapse"', 'id="btn-inventory-collapse"',
        'id="error-title"',
    ):
        assert token in html, f"index.html missing {token}"
    # The wide-screen kill-switch that made the facets button unreachable is gone.
    assert "#btn-facets-toggle { display: none !important; }" not in CSS.read_text(encoding="utf-8")


def test_corpus_panel_collapse_state_is_attribute_driven_no_storage():
    """Collapse state lives on <html> data-attributes and in memory only — never
    in browser storage — and both panels re-fit the canvas after a toggle."""
    js = JS.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    assert "setFacetsOpen" in js and "setInventoryOpen" in js
    assert "data-inventory" in js and "data-inventory" in css
    assert "refitCanvasSoon" in js and "cy.resize()" in js
    for banned in ("localStorage", "sessionStorage", "document.cookie"):
        assert banned not in js, f"collapse state must not use {banned}"


def test_corpus_canvas_scopes_to_selection_and_restores_overview():
    """Facet/search selection scopes the canvas subgraph; clearing restores the
    overview snapshot. Live traces never repaint the corpus canvas (mode guard)."""
    js = JS.read_text(encoding="utf-8")
    assert "scopeCanvasToSelection" in js and "restoreOverviewCanvas" in js
    assert "overviewEdges" in js
    assert 'if (app.mode !== "live") return;' in js


def test_corpus_edgeless_results_group_into_hub_miniature_graphs():
    """Edgeless result sets synthesize per-ticker/source canvas-only hubs; hubs are
    LOD-exempt and their clicks fit the cluster instead of selecting a node."""
    js = JS.read_text(encoding="utf-8")
    assert "clusterCorpusHubs" in js
    assert "fitCluster" in js
    assert 'startsWith("hub:")' in js
    assert 'node[hub]' in js

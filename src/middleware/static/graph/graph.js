/* src/middleware/static/graph/graph.js
   Trace Telemetry — live retrieval graph interface (vanilla, no framework).

   Layout:
     1. GraphState  — pure, DOM-free, testable state functions (window.GraphState)
     2. Cytoscape stylesheet + shape/status language
     3. Live Trace controller
     4. Corpus Explorer controller
     5. Resilient SSE client
     6. Inspector, node list, table sync, accessibility
     7. Bootstrap

   Everything reads the real 2.2.7.1/2.2.7.2 wire contracts. No innerHTML with
   dynamic data; no external requests; same-origin API/SSE only. */

"use strict";

/* ══ 1. Pure state (testable) ══════════════════════════════════════════════ */
(function (root) {
  const STAGE_COLUMNS = ["compile", "route", "retrieve", "evidence", "generate", "validate"];
  const BEAM_RELATIONS = new Set([
    "supports", "from_source", "cited_by", "validated_as", "returned",
    "retrieved", "expanded_from", "corrected_by",
  ]);

  function createTraceState(queryId) {
    return { queryId: queryId || null, nodes: new Map(), edges: new Map(), complete: false };
  }

  // Apply one delta to a single trace's state. Idempotent and order-safe:
  // stale/duplicate upserts (by sequence) are dropped. Returns {changed,...}.
  function applyDelta(state, delta) {
    const op = delta && delta.operation;
    if (op === "upsert_node" && delta.node) {
      const n = delta.node;
      const prev = state.nodes.get(n.id);
      if (prev && Number(n.updated_sequence) <= Number(prev.updated_sequence)) {
        return { changed: false, stale: true };
      }
      state.nodes.set(n.id, n);
      return { changed: true };
    }
    if (op === "upsert_edge" && delta.edge) {
      const e = delta.edge;
      const prev = state.edges.get(e.id);
      if (prev && Number(e.sequence) < Number(prev.sequence)) {
        return { changed: false, stale: true };
      }
      state.edges.set(e.id, e);
      return { changed: true };
    }
    if (op === "remove") {
      const id = delta.summary && delta.summary.id;
      const had = state.nodes.delete(id) || state.edges.delete(id);
      return { changed: had };
    }
    if (op === "trace_complete") {
      state.complete = true;
      return { changed: true };
    }
    if (op === "trace_evicted") {
      state.nodes.clear();
      state.edges.clear();
      return { changed: true, evicted: true };
    }
    return { changed: false };
  }

  // Route a delta from the shared multi-trace stream to its trace state.
  function ingest(store, delta) {
    const op = delta && delta.operation;
    if (op === "reset_required") return { reset: true };
    const qid = delta && delta.query_id;
    if (!qid || qid === "*") return { changed: false };
    let st = store.get(qid);
    if (!st) { st = createTraceState(qid); store.set(qid, st); }
    const result = applyDelta(st, delta);
    if (result.evicted) store.delete(qid);
    return Object.assign({ queryId: qid }, result);
  }

  function citedContext(nodes, edges) {
    const ids = new Set();
    const evidenceIds = new Set();
    for (const n of nodes) {
      if (n.kind === "citation") {
        const eid = n.metadata && n.metadata.evidence_id;
        if (eid) evidenceIds.add(String(eid));
      }
    }
    for (const e of edges) {
      if (e.relation === "cited_by" || e.relation === "validated_as" || e.relation === "supports") {
        ids.add(e.source);
        ids.add(e.target);
      }
    }
    return { ids, evidenceIds };
  }

  // Predicate: does one node satisfy the active filter set? Structural nodes
  // that lack a filtered dimension stay visible so the pipeline stays legible.
  function nodeMatchesFilters(node, filters, cited) {
    const m = node.metadata || {};
    if (filters.status && node.status !== filters.status) return false;
    if (filters.subquery && m.subquery_id != null && String(m.subquery_id) !== filters.subquery) return false;
    if (filters.source) {
      const src = m.source_type != null ? m.source_type : m.source;
      if (src != null && String(src) !== filters.source) return false;
    }
    if (filters.ticker && m.ticker != null && String(m.ticker) !== filters.ticker) return false;
    if (filters.kind && node.kind === "evidence" && String(m.kind || "") !== filters.kind) return false;
    if (filters.citedOnly && node.kind === "evidence") {
      const eid = m.evidence_id != null ? String(m.evidence_id) : null;
      if (!cited.ids.has(node.id) && !(eid && cited.evidenceIds.has(eid))) return false;
    }
    if (filters.maxSequence != null && Number(node.created_sequence) > filters.maxSequence) return false;
    return true;
  }

  // Filter + cap: keep at most `cap` newest nodes, then keep only edges whose
  // endpoints both survive. Never silently drops — returns hidden counts.
  function selectVisible(state, filters, cap) {
    const allNodes = Array.from(state.nodes.values());
    const allEdges = Array.from(state.edges.values());
    const cited = citedContext(allNodes, allEdges);
    let nodes = allNodes.filter((n) => nodeMatchesFilters(n, filters, cited));
    const hiddenByFilter = allNodes.length - nodes.length;
    let hiddenByCap = 0;
    if (cap && nodes.length > cap) {
      nodes = nodes
        .slice()
        .sort((a, b) => Number(b.updated_sequence) - Number(a.updated_sequence))
        .slice(0, cap);
      hiddenByCap = allNodes.filter((n) => nodeMatchesFilters(n, filters, cited)).length - nodes.length;
    }
    const visible = new Set(nodes.map((n) => n.id));
    let edges = allEdges.filter((e) => visible.has(e.source) && visible.has(e.target));
    if (filters.maxSequence != null) {
      edges = edges.filter((e) => Number(e.sequence) <= filters.maxSequence);
    }
    return { nodes, edges, hiddenByFilter, hiddenByCap, total: allNodes.length };
  }

  function stageColumnFor(node) {
    switch (node.kind) {
      case "query":
      case "plan":
        return 0;
      case "subquery":
        return 1;
      case "tool":
        return 2;
      case "evidence":
      case "source":
        return 3;
      case "answer":
        return 4;
      case "citation":
        return 5;
      case "stage": {
        const name = String(node.label || "").toLowerCase().trim();
        const map = {
          compile: 0, route: 1, intent: 1, retrieve: 2, grade: 2, correct: 2,
          pack: 3, generate: 4, validate: 5, "query error": 5,
        };
        if (name in map) return map[name];
        // Tolerate composed labels ("route · intent"): first known token wins,
        // so the legacy intent/route node always lands in the ROUTE column.
        for (const part of name.split(/[^a-z]+/)) {
          if (part && part in map) return map[part];
        }
        return 3;
      }
      default:
        return 3;
    }
  }

  // Provenance beam: BFS from a selected node over provenance relations.
  function computeBeam(nodes, edges, selectedId) {
    const adj = new Map();
    for (const e of edges) {
      if (!BEAM_RELATIONS.has(e.relation)) continue;
      if (!adj.has(e.source)) adj.set(e.source, []);
      if (!adj.has(e.target)) adj.set(e.target, []);
      adj.get(e.source).push(e.target);
      adj.get(e.target).push(e.source);
    }
    const seen = new Set([selectedId]);
    const queue = [selectedId];
    while (queue.length) {
      const cur = queue.shift();
      for (const nb of adj.get(cur) || []) {
        if (!seen.has(nb)) { seen.add(nb); queue.push(nb); }
      }
    }
    const beamEdges = new Set();
    for (const e of edges) {
      if (seen.has(e.source) && seen.has(e.target)) beamEdges.add(e.id);
    }
    return { nodes: seen, edges: beamEdges };
  }

  const BEAM_KINDS = new Set(["answer", "citation", "evidence"]);

  root.GraphState = {
    STAGE_COLUMNS,
    createTraceState,
    applyDelta,
    ingest,
    citedContext,
    nodeMatchesFilters,
    selectVisible,
    stageColumnFor,
    computeBeam,
    beamTriggers: (kind) => BEAM_KINDS.has(kind),
  };
})(typeof window !== "undefined" ? window : globalThis);

/* ══ 1b. Corpus Explorer IA — pure, DOM-free, testable (2.3.5.2) ═══════════ */
/* Aggregation-first navigation: all filter state lives in the URL hash and
   nowhere else (no cookies, no analytics, no browser storage), so a
   reload/back/forward/deep-link is a pure function of the hash. Facets are
   combinable; provider names are present but never the first navigation level. */
(function (root) {
  // Combinable facet dimensions. Those with a truthy `dim` get authoritative
  // "counts before expansion" from GET /corpus/aggregates (group_by=dim); the
  // rest are combinable controls whose universe-wide counts are the deeper
  // index work deferred to 2.3.5.3 (see design-direction-2.3.5.md).
  const CORPUS_FACET_GROUPS = [
    { key: "coverage_tier", label: "Coverage tier", dim: null,
      options: ["broad", "deep", "sector", "global"] },
    { key: "index", label: "Index", dim: null,
      options: ["sp500", "nasdaq100", "overlap", "off_index"] },
    { key: "sector", label: "Sector / industry", dim: null, options: [] },
    { key: "ticker", label: "Ticker", dim: null, options: [] },
    { key: "source_category", label: "Source category", dim: "source_category" },
    { key: "source", label: "Source", dim: "source" },
    { key: "item_type", label: "Item type", dim: "item_type" },
    { key: "event_type", label: "Event type", dim: null, options: [] },
    { key: "form", label: "Form / exhibit", dim: null, options: [] },
    { key: "indexing_state", label: "Indexing state", dim: "indexing_state",
      options: ["indexed", "pending", "failed", "not_applicable"] },
    { key: "freshness", label: "Freshness", dim: null,
      options: ["fresh", "stale", "error", "never"] },
    { key: "authority", label: "Authority tier", dim: null,
      options: ["primary", "structured", "licensed", "analysis", "discovery"] },
    { key: "date", label: "Date range", dim: "year", type: "date" },
  ];
  const CORPUS_FACET_KEYS = CORPUS_FACET_GROUPS
    .filter((group) => group.key !== "date")
    .map((group) => group.key);

  // Drill order for the aggregation-first hierarchy (spec Step-0 diagram):
  // selecting an aggregate group descends one authoritative level at a time.
  // Every level here is backed by real data now (index options + the
  // accounting-backed source_category/item_type/year counts); sector stays a
  // combinable rail facet whose universe-wide counts are 2.3.5.3 index work.
  const CORPUS_DRILL_ORDER = ["index", "source_category", "item_type", "year"];

  // Deterministic presets are URL shortcuts over normal facets, not separate API
  // behavior. Each returns the facet/view changes it layers onto current state.
  const CORPUS_PRESETS = {
    "index-coverage": { groupBy: "index", view: "aggregates" },
    "financing-events": {
      facets: {
        item_type: ["filing", "filing_exhibit"],
        event_type: ["debt_raise", "equity_raise", "convertible_offering", "shelf_registration"],
      }, view: "results",
    },
    "latest-news": { facets: { source_category: ["company_news"] }, view: "results" },
    "macro-policy": {
      facets: { source_category: ["central_bank", "treasury", "economic_agency"] },
      view: "results",
    },
    "stale-sources": { facets: { freshness: ["stale", "error"] }, view: "results" },
    "indexing-backlog": { facets: { indexing_state: ["pending", "failed"] }, view: "results" },
    "ticker-research": { view: "results", groupBy: "source_category" },
  };

  function emptyCorpusState() {
    const facets = {};
    for (const key of CORPUS_FACET_KEYS) facets[key] = [];
    return {
      facets, q: "", preset: "", view: "aggregates",
      groupBy: "source_category", published_from: "", published_to: "",
    };
  }

  // Parse the corpus hash into state. Unknown keys are ignored; combined facets
  // round-trip exactly, which is what makes a reload preserve every filter.
  function parseCorpusHash(hash) {
    const state = emptyCorpusState();
    const raw = String(hash || "").replace(/^#/, "");
    const params = new URLSearchParams(raw);
    for (const key of CORPUS_FACET_KEYS) {
      const value = params.get(key);
      state.facets[key] = value ? value.split(",").map((v) => v.trim()).filter(Boolean) : [];
    }
    state.q = params.get("q") || "";
    state.preset = params.get("preset") || "";
    state.view = params.get("view") === "results" ? "results" : "aggregates";
    state.groupBy = params.get("groupby") || "source_category";
    state.published_from = params.get("published_from") || "";
    state.published_to = params.get("published_to") || "";
    return state;
  }

  // Serialize state back to a stable hash string. Only non-empty values appear,
  // so equivalent states produce identical hashes (clean back/forward history).
  function corpusHash(state) {
    const params = new URLSearchParams();
    params.set("mode", "corpus");
    for (const key of CORPUS_FACET_KEYS) {
      const vals = (state.facets && state.facets[key]) || [];
      if (vals.length) params.set(key, vals.join(","));
    }
    if (state.q) params.set("q", state.q);
    if (state.preset) params.set("preset", state.preset);
    if (state.view === "results") params.set("view", "results");
    if (state.groupBy && state.groupBy !== "source_category") params.set("groupby", state.groupBy);
    if (state.published_from) params.set("published_from", state.published_from);
    if (state.published_to) params.set("published_to", state.published_to);
    return "#" + params.toString();
  }

  function toggleFacetValue(state, key, value) {
    const next = emptyCorpusState();
    Object.assign(next, JSON.parse(JSON.stringify(state)));
    if (!CORPUS_FACET_KEYS.includes(key)) return next;
    const set = new Set(next.facets[key] || []);
    if (set.has(value)) set.delete(value); else set.add(value);
    next.facets[key] = Array.from(set);
    next.preset = "";  // a manual facet edit leaves preset mode
    return next;
  }

  // Apply a preset over the current state. Presets replace only the dimensions
  // they name, so a chosen ticker survives ticker-research and users can layer.
  function applyPreset(state, presetId) {
    const preset = CORPUS_PRESETS[presetId];
    const next = emptyCorpusState();
    Object.assign(next, JSON.parse(JSON.stringify(state)));
    next.preset = presetId;
    if (!preset) return next;
    if (preset.facets) {
      for (const key of Object.keys(preset.facets)) next.facets[key] = preset.facets[key].slice();
    }
    if (preset.view) next.view = preset.view;
    if (preset.groupBy) next.groupBy = preset.groupBy;
    return next;
  }

  function activeFacetCount(state) {
    let total = 0;
    for (const key of CORPUS_FACET_KEYS) total += (state.facets[key] || []).length;
    if (state.published_from || state.published_to) total += 1;
    return total;
  }

  // The subset of active facets the aggregates/accounting endpoint understands
  // as single-value narrowing filters (2.3.5.2 authoritative dimensions).
  function accountingFilters(state) {
    const filters = {};
    for (const key of ["source_category", "source", "item_type", "indexing_state"]) {
      const vals = state.facets[key] || [];
      if (vals.length === 1) filters[key] = vals[0];
    }
    return filters;
  }

  root.CorpusIA = {
    CORPUS_FACET_GROUPS,
    CORPUS_FACET_KEYS,
    CORPUS_DRILL_ORDER,
    CORPUS_PRESETS,
    emptyCorpusState,
    parseCorpusHash,
    corpusHash,
    toggleFacetValue,
    applyPreset,
    activeFacetCount,
    accountingFilters,
  };
})(typeof window !== "undefined" ? window : globalThis);

/* ══ Everything below needs the DOM; skip under a bare test harness. ═══════ */
if (typeof document !== "undefined" && document.getElementById("graph-canvas")) {
  (function () {
    const GS = window.GraphState;
    const $ = (id) => document.getElementById(id);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const API = "/graph/api";
    const STATUS_CLASS = {
      active: "s-active", complete: "s-complete", partial: "s-partial",
      pending: "s-pending", fallback: "s-fallback", dropped: "s-dropped",
      error: "s-error", supported: "s-supported", missing: "s-missing",
      conflict: "s-conflict",
    };
    const KIND_GLYPH = {
      query: "▭", plan: "◈", subquery: "◇", stage: "▢", tool: "⬡",
      evidence: "◆", source: "◯", answer: "▣", citation: "⌂",
      ticker: "$", metric: "#", fact: "◆", filing: "▤", section: "▥",
      document_family: "▤", freshness: "◔", scheduler_source: "⟳",
    };

    /* ── shared app state ── */
    const app = {
      mode: "live",
      traces: new Map(),          // queryId -> trace state
      activeId: null,
      pinnedTrace: false,         // user picked a specific trace
      follow: true,
      filters: { status: "", subquery: "", source: "", ticker: "", kind: "", citedOnly: false, maxSequence: null },
      selectedId: null,
      pinnedNode: null,
      scrubberMax: 0,
      cap: 5000,
      corpus: {
        revision: null, cursor: null, lastQuery: null,
        nodes: new Map(), edges: new Map(), freshAgg: new Map(),
        state: null, aggregates: new Map(), rows: [], selectedRowId: null,
      },
      dragging: false,
      scrubbing: false,           // slider drag in progress -> instant, no fit
      scrubRaf: 0,                // coalesces scrub-driven re-renders
      pending: [],                // batched deltas
      rafHandle: 0,
      knownTraceIds: new Set(),
    };

    /* ══ 2. Cytoscape stylesheet ══════════════════════════════════════════ */
    const C = getComputedStyle(document.documentElement);
    const tok = (n) => C.getPropertyValue(n).trim();
    const COL = {
      canvas: tok("--canvas"), surface: tok("--surface"), surface2: tok("--surface-2"),
      hairline: tok("--hairline"), ink: tok("--ink"), inkDim: tok("--ink-dim"),
      flow: tok("--flow"), pending: tok("--pending"), supported: tok("--supported"),
      conflict: tok("--conflict"), steel: tok("--steel"),
    };
    const statusColor = {
      active: COL.flow, complete: COL.supported, partial: COL.pending,
      pending: COL.pending, fallback: COL.pending, dropped: COL.steel,
      error: COL.conflict,
    };

    function cyStyle() {
      const nodeColor = (ele) => statusColor[ele.data("status")] || COL.steel;
      return [
        {
          selector: "node",
          style: {
            "background-color": COL.surface2,
            "border-width": 1.5,
            "border-color": nodeColor,
            "label": "data(label)",
            "color": COL.ink,
            "font-family": "Commit Mono, monospace",
            "font-size": 11,
            "text-wrap": "ellipsis",
            "text-max-width": 120,
            "text-valign": "bottom",
            "text-margin-y": 4,
            "width": 34, "height": 34,
            "transition-property": "border-color, background-color, opacity",
            "transition-duration": reduceMotion ? "0ms" : "180ms",
          },
        },
        { selector: 'node[kind="query"]', style: { shape: "round-rectangle", width: 120, height: 34, "text-valign": "center", "text-margin-y": 0, "background-color": COL.surface } },
        { selector: 'node[kind="plan"]', style: { shape: "round-rectangle", width: 64, height: 30 } },
        { selector: 'node[kind="subquery"]', style: { shape: "round-rectangle", width: 78, height: 30 } },
        { selector: 'node[kind="stage"]', style: { shape: "round-rectangle", width: 80, height: 28, "background-color": COL.surface } },
        { selector: 'node[kind="tool"]', style: { shape: "hexagon", width: 40, height: 38 } },
        { selector: 'node[kind="evidence"]', style: { shape: "diamond", width: 34, height: 34 } },
        { selector: 'node[kind="evidence"][emd="document"]', style: { shape: "cut-rectangle", width: 34, height: 40 } },
        // Source-aware subtypes (2.3.5.1): drawn from metadata, never a new node
        // kind. Primary/authoritative evidence reads bolder; corroborating (a
        // secondary source on the same event) is lighter and dashed.
        { selector: 'node[kind="evidence"][atier="primary"], node[kind="evidence"][atier="direct_sec"]', style: { "border-width": 3, "background-color": COL.surface } },
        { selector: 'node[kind="evidence"][erole="corroborating"]', style: { "border-style": "dashed", "background-opacity": 0.55 } },
        { selector: 'node[kind="source"]', style: { shape: "ellipse", "background-opacity": 0, "border-width": 2.4, width: 30, height: 30 } },
        { selector: 'node[kind="answer"]', style: { shape: "round-rectangle", "border-style": "double", "border-width": 4, width: 96, height: 34, "text-valign": "center", "text-margin-y": 0 } },
        { selector: 'node[kind="citation"]', style: { shape: "tag", width: 40, height: 30 } },
        { selector: 'node[status="error"]', style: { "border-style": "dashed" } },
        { selector: 'node[status="fallback"]', style: { "border-style": "dashed" } },
        { selector: 'node[status="dropped"]', style: { "border-style": "dotted", opacity: 0.7 } },
        { selector: 'node[status="active"]', style: { "border-width": 2.4, "underlay-color": COL.flow, "underlay-opacity": reduceMotion ? 0 : 0.35, "underlay-padding": 8 } },
        { selector: "node:selected", style: { "border-width": 3, "border-color": COL.ink } },
        {
          selector: "edge",
          style: {
            "width": 1.5,
            "line-color": COL.steel,
            "target-arrow-color": COL.steel,
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.8,
            "curve-style": "bezier",
            "opacity": 0.75,
            "transition-property": "line-color, opacity",
            "transition-duration": reduceMotion ? "0ms" : "180ms",
          },
        },
        { selector: "edge.inflight", style: { "line-color": COL.flow, "target-arrow-color": COL.flow, "line-style": "dashed", "opacity": 1 } },
        { selector: ".beam", style: { "line-color": COL.flow, "target-arrow-color": COL.flow, "border-color": COL.supported, opacity: 1, "z-index": 20 } },
        { selector: "node.beam", style: { "border-color": COL.supported, "border-width": 3 } },
        { selector: ".dimmed", style: { opacity: 0.28 } },
        {
          selector: "node[kind = 'ticker'], node[kind='metric'], node[kind='fact'], node[kind='filing'], node[kind='section'], node[kind='document_family'], node[kind='freshness'], node[kind='scheduler_source']",
          style: { "background-color": COL.surface2, "border-color": COL.flow },
        },
        // Corpus: a ticker whose folded freshness leaves carry a worst status
        // rings in that status colour (must follow the kind rule to win).
        { selector: 'node[freshWorst="complete"]', style: { "border-color": COL.supported, "border-width": 2.4 } },
        { selector: 'node[freshWorst="pending"]', style: { "border-color": COL.pending, "border-width": 2.4 } },
        { selector: 'node[freshWorst="dropped"]', style: { "border-color": COL.steel, "border-width": 2.4 } },
        { selector: 'node[freshWorst="error"]', style: { "border-color": COL.conflict, "border-width": 2.4, "border-style": "dashed" } },
      ];
    }

    let cy = null;

    /* ══ 3./4. Rendering ══════════════════════════════════════════════════ */
    function nodeToEle(n) {
      const m = n.metadata || {};
      const emd = m.kind || "";
      // Source-aware subtypes ride on data attributes (never new node kinds), so
      // authority tier and primary/corroborating role can drive style/legend.
      return {
        group: "nodes",
        data: {
          id: n.id, label: displayLabel(n), kind: n.kind,
          status: n.status, emd, raw: n,
          atier: m.authority_tier || "", erole: m.evidence_role || "",
          scat: m.source_category || "",
        },
      };
    }
    function edgeToEle(e) {
      return { group: "edges", data: { id: e.id, source: e.source, target: e.target, relation: e.relation, raw: e } };
    }
    function displayLabel(n) {
      if (n.kind === "evidence") {
        const m = n.metadata || {};
        return m.evidence_id || m.ticker || n.label || "evidence";
      }
      return n.label || n.kind;
    }
    function isPlainObject(v) {
      return v != null && typeof v === "object" && !Array.isArray(v);
    }
    // Render allowlisted metadata values without innerHTML. Nested date-semantics
    // maps flatten to "key: value · …" so timestamps stay legible in the inspector.
    function formatMetaValue(val) {
      if (Array.isArray(val)) return val.join(", ");
      if (isPlainObject(val)) {
        return Object.keys(val)
          .filter((k) => val[k] != null && val[k] !== "")
          .map((k) => `${k.replace(/_/g, " ")}: ${val[k]}`)
          .join(" · ");
      }
      return String(val);
    }

    function activeTrace() {
      return app.activeId ? app.traces.get(app.activeId) : null;
    }

    // Coalesce scrubber `input` bursts into one render per animation frame and
    // mark the interaction so renders stay instant and viewport-stable.
    function scheduleScrubRender() {
      app.scrubbing = true;
      if (app.scrubRaf) return;
      app.scrubRaf = requestAnimationFrame(() => {
        app.scrubRaf = 0;
        renderLive();
      });
    }

    function renderLive() {
      if (!cy) return;
      const trace = activeTrace();
      $("canvas-empty").hidden = !!(trace && trace.nodes.size);
      if (!trace) { cy.elements().remove(); syncSecondaryViews([], []); return; }
      app.filters.maxSequence = atLive() ? null : app.scrubberValue;
      const vis = GS.selectVisible(trace, app.filters, app.cap);
      diffRender(vis.nodes, vis.edges);
      layoutLive(vis.nodes);
      applyInflight();
      updateHiddenCount(vis);
      syncSecondaryViews(vis.nodes, vis.edges);
      refreshFilterOptions(trace);
      if (app.selectedId) applyBeam();
    }

    // Reconcile Cytoscape elements against the target node/edge set.
    function diffRender(nodes, edges) {
      const want = new Set(nodes.map((n) => n.id).concat(edges.map((e) => e.id)));
      cy.batch(() => {
        cy.elements().forEach((ele) => { if (!want.has(ele.id())) ele.remove(); });
        for (const n of nodes) {
          const ex = cy.getElementById(n.id);
          if (ex.nonempty()) {
            ex.data("status", n.status);
            ex.data("label", displayLabel(n));
            ex.data("raw", n);
          } else {
            const ele = cy.add(nodeToEle(n));
            // Seed a distinct starting position (per stage column) so freshly
            // added endpoints never coincide — avoids Cytoscape overlap warnings.
            const col = GS.stageColumnFor(n);
            const w = cy.width() || 900;
            ele.position({ x: (w / 6) * col + w / 12, y: 60 + (Number(n.created_sequence) % 12) * 42 });
            // Fade-in belongs to live streaming only; scrubbing positions
            // instantly so dragging the slider never triggers entrance motion.
            // Only the starting opacity is set here — layoutLive owns the ramp
            // to 1, so its stop(true) can never strand a node invisible.
            if (!reduceMotion && !app.scrubbing) ele.style("opacity", 0);
          }
        }
        for (const e of edges) {
          if (cy.getElementById(e.id).empty()) cy.add(edgeToEle(e));
        }
      });
    }

    function layoutLive(nodes) {
      if (app.dragging || !nodes.length) return;
      const w = cy.width() || 900;
      const h = cy.height() || 600;
      const colW = w / 6;
      const buckets = [[], [], [], [], [], []];
      for (const n of nodes) buckets[GS.stageColumnFor(n)].push(n);
      for (const b of buckets) b.sort((a, c) => Number(a.created_sequence) - Number(c.created_sequence));
      const positions = {};
      const usableH = Math.max(120, h - 80);
      const minGap = 46;
      buckets.forEach((bucket, col) => {
        const laneLeft = colW * col;
        const count = bucket.length;
        // Stagger a tall bucket into 2+ sub-columns within its lane so a long
        // evidence chain never forces fit() to zoom the whole trace out. A small
        // bucket keeps the reference look: one centered column.
        const maxRows = Math.max(1, Math.floor(usableH / minGap));
        const subCols = Math.max(1, Math.ceil(count / maxRows));
        const rows = Math.max(1, Math.ceil(count / subCols));
        const gap = Math.min(70, usableH / rows);
        bucket.forEach((n, i) => {
          const sub = Math.floor(i / rows);
          const row = i % rows;
          const x = subCols === 1
            ? laneLeft + colW / 2
            : laneLeft + (colW * (sub + 1)) / (subCols + 1);
          positions[n.id] = { x, y: 60 + gap * row + gap / 2 };
        });
      });
      const instant = reduceMotion || app.scrubbing;
      cy.nodes().forEach((ele) => {
        const p = positions[ele.id()];
        if (!p) return;
        ele.stop(true);  // stop AND clear queued motion so rapid renders never stack animations
        if (instant) { ele.style("opacity", 1); ele.position(p); }
        else ele.animate({ position: p, style: { opacity: 1 } }, { duration: 200, easing: "ease-out" });
      });
      markHotLanes(buckets);
      // Never re-fit while scrubbing — the viewport must stay put under the slider.
      if (app.follow && !app.scrubbing) {
        requestAnimationFrame(() => { cy.fit(cy.elements(), 48); syncStageRail(); });
      } else {
        syncStageRail();
      }
    }

    function markHotLanes(buckets) {
      const rail = $("stage-rail");
      Array.from(rail.children).forEach((lane, col) => {
        const hot = buckets[col] && buckets[col].some((n) => n.status === "active");
        lane.classList.toggle("hot", !!hot);
      });
    }

    // Keep the DOM stage-rail lanes glued to the Cytoscape viewport: project the
    // fixed model-space lane geometry (colW = container/6, matching layoutLive)
    // through the current pan/zoom so lanes and node columns stay aligned after
    // auto-fit and any manual zoom/pan. Cheap: a direct transform per lane, no
    // layout thrash. Labels are positioned (not scaled), so text never stretches.
    function syncStageRail() {
      if (!cy || app.mode !== "live") return;
      const rail = $("stage-rail");
      const lanes = rail.children;
      if (!lanes.length) return;
      const z = cy.zoom();
      const pan = cy.pan();
      const laneW = (cy.width() || 900) / 6;
      for (let col = 0; col < lanes.length; col++) {
        const lane = lanes[col];
        lane.style.transform = `translateX(${col * laneW * z + pan.x}px)`;
        lane.style.width = `${laneW * z}px`;
      }
    }

    let dashOffset = 0;
    function applyInflight() {
      if (!cy) return;
      cy.edges().removeClass("inflight");
      const active = new Set(cy.nodes('[status="active"]').map((n) => n.id()));
      cy.edges().forEach((e) => {
        if (active.has(e.source().id()) || active.has(e.target().id())) e.addClass("inflight");
      });
    }
    function tickDash() {
      // Reduced-motion stops the looping edge-flow animation entirely rather
      // than spinning an idle rAF loop (2.3.5.3 Step 6).
      if (reduceMotion) return;
      if (cy) {
        dashOffset = (dashOffset - 0.8) % 24;
        cy.edges(".inflight").style("line-dash-offset", dashOffset);
      }
      requestAnimationFrame(tickDash);
    }

    /* ── beam ── */
    function applyBeam() {
      if (!cy) return;
      cy.elements().removeClass("beam dimmed");
      const node = app.selectedId ? cy.getElementById(app.selectedId) : cy.collection();
      if (node.empty()) return;
      const raw = node.data("raw");
      if (!raw || !GS.beamTriggers(raw.kind)) return;
      const trace = activeTrace();
      if (!trace) return;
      const beam = GS.computeBeam(
        Array.from(trace.nodes.values()), Array.from(trace.edges.values()), app.selectedId
      );
      cy.batch(() => {
        cy.elements().forEach((ele) => {
          const inBeam = beam.nodes.has(ele.id()) || beam.edges.has(ele.id());
          ele.addClass(inBeam ? "beam" : "dimmed");
        });
      });
    }
    function clearBeam() { if (cy) cy.elements().removeClass("beam dimmed"); }

    /* ══ Corpus Explorer ══════════════════════════════════════════════════ */
    // Freshness leaf status -> node status colour; worst wins per ticker.
    const FRESH_RANK = { complete: 0, pending: 1, dropped: 2, error: 3 };
    function freshStatus(raw) {
      const v = String(raw || "").toLowerCase();
      if (v.includes("error") || v.includes("fail")) return "error";
      if (v.includes("never") || v.includes("miss")) return "dropped";
      if (v.includes("stale") || v.includes("expired") || v.includes("due") || v.includes("pending")) return "pending";
      if (v.includes("fresh") || v.includes("ok") || v.includes("current")) return "complete";
      return "pending";  // unknown reads as "needs attention", never a false green
    }

    // Fold the ~150 freshness leaves out of the CANVAS view: aggregate each into
    // its ticker (worst-status ring + per-status counts) so the overview reads
    // as sources + tickers instead of a hairball. The full node set still flows
    // to the list/table/inspector — folding is purely a canvas concern.
    function foldCorpusForCanvas(rawNodes, rawEdges) {
      const byId = new Map(rawNodes.map((n) => [n.id, n]));
      const freshToTicker = new Map();
      for (const e of rawEdges) {
        if (e.relation === "freshness_for") freshToTicker.set(e.source, e.target);
      }
      const agg = new Map();       // tickerId -> { counts, worst }
      const folded = new Set();    // freshness node ids removed from canvas
      for (const n of rawNodes) {
        if (n.kind !== "freshness") continue;
        const tickerId = freshToTicker.get(n.id);
        if (!tickerId || !byId.has(tickerId)) continue;  // orphan freshness stays visible
        folded.add(n.id);
        const status = freshStatus((n.metadata && n.metadata.status) || "");
        const entry = agg.get(tickerId) || { counts: {}, worst: "complete" };
        entry.counts[status] = (entry.counts[status] || 0) + 1;
        if (FRESH_RANK[status] > FRESH_RANK[entry.worst]) entry.worst = status;
        agg.set(tickerId, entry);
      }
      app.corpus.freshAgg = agg;

      const nodes = [];
      for (const n of rawNodes) {
        if (folded.has(n.id)) continue;
        const data = { id: n.id, label: n.label || n.kind, kind: n.kind, status: "complete", emd: "", raw: n };
        const summary = agg.get(n.id);
        if (n.kind === "ticker" && summary) {
          data.freshWorst = summary.worst;
          data.status = summary.worst;
        }
        nodes.push({ group: "nodes", data });
      }
      const ids = new Set(nodes.map((n) => n.data.id));
      const edges = Array.from(rawEdges)
        .filter((e) => !folded.has(e.source) && !folded.has(e.target) && ids.has(e.source) && ids.has(e.target))
        .map((e) => ({ group: "edges", data: { id: e.id, source: e.source, target: e.target, relation: e.relation, raw: e } }));
      return { nodes, edges, foldedCount: folded.size };
    }

    // Label level-of-detail: hubs (sources, tickers) always labelled; leaf labels
    // fade in only past a zoom threshold, keeping the overview free of label soup.
    let corpusLeafLabelsShown = true;
    function updateCorpusLabelLOD(force) {
      if (!cy || app.mode !== "corpus") return;
      const show = cy.zoom() >= 0.6;
      if (!force && show === corpusLeafLabelsShown) return;
      corpusLeafLabelsShown = show;
      cy.batch(() => {
        cy.nodes().forEach((ele) => {
          const kind = ele.data("kind");
          if (kind === "source" || kind === "ticker") return;
          ele.style("text-opacity", show ? 1 : 0);
        });
      });
    }

    function renderCorpus() {
      if (!cy) return;
      const rawNodes = Array.from(app.corpus.nodes.values());
      const rawEdges = Array.from(app.corpus.edges.values());
      const { nodes, edges } = foldCorpusForCanvas(rawNodes, rawEdges);
      $("canvas-empty").hidden = nodes.length > 0;
      if (!nodes.length) $("canvas-empty").textContent = "Search the corpus or pick a source group to expand.";
      cy.elements().remove();
      cy.add(nodes);
      cy.add(edges);
      if (nodes.length) {
        const layout = cy.layout({
          name: "cose", animate: !reduceMotion, animationDuration: 400, fit: true,
          padding: 60, nodeRepulsion: 14000, idealEdgeLength: 140, edgeElasticity: 100,
          gravity: 0.22, componentSpacing: 140, nodeOverlap: 20, randomize: false,
        });
        layout.run();
        corpusLeafLabelsShown = true;   // force LOD to re-evaluate against post-fit zoom
        updateCorpusLabelLOD(true);
      }
      // List/table/inspector see the complete corpus — freshness stays inspectable.
      syncSecondaryViews(rawNodes, rawEdges);
      $("hidden-count").textContent = app.corpus.cursor ? "more pages available" : "";
      $("btn-corpus-more").hidden = !app.corpus.cursor;
    }

    async function corpusOverview() {
      try {
        const data = await getJSON(`${API}/corpus/overview`, corpusSignal());
        loadCorpusPage(data, true);
      } catch (err) { showError("Corpus overview failed", err); }
    }
    async function corpusSearch(reset) {
      const q = $("corpus-search").value.trim();
      app.corpus.lastQuery = q;
      let url = `${API}/corpus/search?q=${encodeURIComponent(q)}`;
      if (reset) { app.corpus.cursor = null; }
      if (app.corpus.cursor && !reset) url += `&cursor=${encodeURIComponent(app.corpus.cursor)}`;
      try {
        const data = await getJSON(url);
        loadCorpusPage(data, reset);
      } catch (err) { showError("Corpus search failed", err); }
    }
    async function corpusExpand(nodeId) {
      const url = `${API}/corpus/nodes/${encodeURIComponent(nodeId)}/neighbors`;
      try {
        const data = await getJSON(url);
        loadCorpusPage(data, false);
        announce(`Expanded ${data.nodes.length} neighbors.`);
      } catch (err) {
        if (err.status === 409) { announce("Corpus changed — reloading overview."); corpusOverview(); }
        else showError("Expand failed", err);
      }
    }
    function loadCorpusPage(data, reset) {
      if (reset) { app.corpus.nodes.clear(); app.corpus.edges.clear(); }
      const revChanged = app.corpus.revision != null && app.corpus.revision !== data.corpus_revision;
      app.corpus.revision = data.corpus_revision;
      app.corpus.cursor = data.next_cursor || null;
      const cap = app.cap;
      for (const n of data.nodes || []) {
        if (app.corpus.nodes.size >= cap) break;
        app.corpus.nodes.set(n.id, n);
      }
      for (const e of data.edges || []) app.corpus.edges.set(e.id, e);
      setCorpusRevision(data.corpus_revision, revChanged);
      if (data.truncated) announce("Result truncated to the visible element cap.");
      renderCorpus();
    }
    function setCorpusRevision(rev, changed) {
      $("corpus-revision").textContent = rev == null ? "—" : String(rev);
      if (changed) announce(`Corpus revision changed to ${rev}. Reload to refresh.`);
    }

    /* ══ 4b. Aggregation-first IA: facets, inventory, URL state (2.3.5.2) ══ */
    const CI = window.CorpusIA;
    // Authoritative aggregate dimensions we fetch for facet counts (bounded).
    const AGG_DIMS = ["source_category", "source", "item_type", "indexing_state", "year"];
    const FRESH_LABELS = { fresh: "fresh ●", stale: "stale ⌛", error: "error ✕", never: "never ◌" };
    const INDEXING_LABELS = {
      indexed: "indexed ●", pending: "pending ⟳", failed: "failed ✕",
      not_applicable: "n/a",
    };
    const STATUS_GLYPH = {
      fresh: "●", stale: "⌛", error: "✕", indexing: "⟳", backlog: "⏳",
      complete: "●", pending: "◌", dropped: "⌀",
    };

    function corpusState() {
      if (!app.corpus.state) app.corpus.state = CI.emptyCorpusState();
      return app.corpus.state;
    }

    // Every corpus state change routes through the hash so refresh/back/forward
    // and deep links stay a pure function of the URL (no cookies, no storage).
    function navigateCorpus(next) {
      app.corpus.state = next;
      const hash = CI.corpusHash(next);
      if (window.location.hash === hash) applyCorpusState(next);
      else window.location.hash = hash;   // hashchange handler re-applies
    }
    function onCorpusHashChange() {
      if (app.mode !== "corpus") return;
      const state = CI.parseCorpusHash(window.location.hash);
      app.corpus.state = state;
      applyCorpusState(state);
    }

    async function applyCorpusState(state) {
      // A filter/preset/search/drill change starts a new view: cancel the prior
      // one so an out-of-order response never clobbers the current facets.
      abortObsoleteCorpusRequests();
      $("corpus-presets").value = state.preset || "";
      $("corpus-search").value = state.q || "";
      updateFacetToggleBadge(state);
      await refreshAggregates(state);
      renderFacetRail(state);
      if (state.view === "results") renderResults(state);
      else renderAggregatesInventory(state);
    }

    function updateFacetToggleBadge(state) {
      const count = CI.activeFacetCount(state);
      const btn = $("btn-facets-toggle");
      btn.textContent = count ? `Facets (${count})` : "Facets";
    }

    // Fetch one bounded aggregate page per dimension, narrowed by the single-value
    // facets the accounting endpoint understands. Revision-cached; fully fail-soft.
    async function refreshAggregates(state) {
      const filters = CI.accountingFilters(state);
      const query = new URLSearchParams(filters);
      await Promise.all(AGG_DIMS.map(async (dim) => {
        const params = new URLSearchParams(query);
        params.set("group_by", dim);
        // A dimension never narrows by its own facet (that would zero its siblings).
        params.delete(dim);
        try {
          const data = await getJSON(`${API}/corpus/aggregates?${params.toString()}`, corpusSignal());
          const agg = (data && data.aggregates) || { buckets: [] };
          app.corpus.aggregates.set(dim, agg.buckets || []);
          if (data && data.corpus_revision != null) {
            setCorpusRevision(data.corpus_revision, app.corpus.revision != null && app.corpus.revision !== data.corpus_revision);
            app.corpus.revision = data.corpus_revision;
          }
        } catch (err) { if (isAbortError(err)) return; app.corpus.aggregates.set(dim, []); }
      }));
    }

    function authorityCounts() {
      const totals = {};
      for (const bucket of app.corpus.aggregates.get("source_category") || []) {
        const tier = bucket.authority_tier || "discovery";
        totals[tier] = (totals[tier] || 0) + (bucket.count || 0);
      }
      return totals;
    }
    function overviewNodeValues() {
      return (app.corpus.overviewNodes || app.corpus.nodes).values();
    }
    function overviewFreshMap() {
      return app.corpus.overviewFresh || app.corpus.freshAgg;
    }
    function freshnessCounts() {
      const totals = {};
      for (const entry of overviewFreshMap().values()) {
        for (const status of Object.keys(entry.counts || {})) {
          const key = status === "complete" ? "fresh" : status === "dropped" ? "never" : status;
          totals[key] = (totals[key] || 0) + entry.counts[status];
        }
      }
      return totals;
    }

    // Build the persistent facet rail. Dimension-backed groups show authoritative
    // counts before expansion; fixed-vocabulary groups render as combinable
    // toggles even where universe-wide counts are 2.3.5.3 work.
    function renderFacetRail(state) {
      const host = $("facet-groups");
      host.replaceChildren();
      const authority = authorityCounts();
      const freshness = freshnessCounts();
      for (const group of CI.CORPUS_FACET_GROUPS) {
        const section = document.createElement("section");
        section.className = "facet-group";
        const open = group.key === "source_category" || group.key === "item_type"
          || (state.facets[group.key] && state.facets[group.key].length);
        section.dataset.open = String(!!open);

        const head = document.createElement("button");
        head.type = "button";
        head.className = "facet-group-head";
        const label = document.createElement("span");
        label.className = "field-label";
        label.textContent = group.label;
        const active = document.createElement("span");
        active.className = "fg-active mono";
        const activeVals = group.key === "date"
          ? ((state.published_from || state.published_to) ? 1 : 0)
          : (state.facets[group.key] || []).length;
        active.textContent = activeVals ? String(activeVals) : "";
        const chev = document.createElement("span");
        chev.className = "chev"; chev.textContent = "▾"; chev.setAttribute("aria-hidden", "true");
        head.append(label, active, chev);
        head.setAttribute("aria-expanded", String(!!open));
        head.addEventListener("click", () => {
          const nowOpen = section.dataset.open !== "true";
          section.dataset.open = String(nowOpen);
          head.setAttribute("aria-expanded", String(nowOpen));
        });
        section.appendChild(head);

        if (group.key === "date") {
          section.appendChild(renderDateFacet(state));
          host.appendChild(section);
          continue;
        }

        let options = [];
        if (group.key === "authority") {
          options = group.options.map((v) => ({ key: v, label: v, count: authority[v] }));
        } else if (group.key === "freshness") {
          options = group.options.map((v) => ({ key: v, label: FRESH_LABELS[v] || v, count: freshness[v] }));
        } else if (group.key === "indexing_state") {
          const counts = mapBuckets("indexing_state");
          options = group.options.map((v) => ({ key: v, label: INDEXING_LABELS[v] || v, count: counts[v] }));
        } else if (group.dim) {
          options = (app.corpus.aggregates.get(group.dim) || []).map((b) => ({
            key: b.key, label: b.label || b.key, count: b.count,
            authority: b.authority_tier, global: b.global_source,
          }));
        } else if (group.key === "ticker") {
          options = tickerOptions();
        } else {
          options = (group.options || []).map((v) => ({ key: v, label: v }));
        }

        const list = document.createElement("ul");
        list.className = "facet-options";
        if (!options.length) {
          const note = document.createElement("li");
          note.className = "facet-empty";
          note.textContent = group.dim
            ? "No values yet."
            : "Combinable filter — counts arrive with 2.3.5.3 index work.";
          list.appendChild(note);
        }
        const selected = new Set(state.facets[group.key] || []);
        for (const opt of options) {
          list.appendChild(renderFacetOption(group.key, opt, selected.has(String(opt.key))));
        }
        section.appendChild(list);
        host.appendChild(section);
      }
    }

    function mapBuckets(dim) {
      const out = {};
      for (const b of app.corpus.aggregates.get(dim) || []) out[b.key] = b.count;
      return out;
    }
    function tickerOptions() {
      const out = [];
      for (const n of overviewNodeValues()) {
        if (n.kind === "ticker") out.push({ key: n.label, label: n.label });
      }
      return out.sort((a, b) => a.label.localeCompare(b.label)).slice(0, 100);
    }

    function renderFacetOption(groupKey, opt, isSelected) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "facet-option";
      btn.setAttribute("aria-pressed", String(!!isSelected));
      const mark = document.createElement("span");
      mark.className = "fo-mark"; mark.textContent = "✓"; mark.setAttribute("aria-hidden", "true");
      const label = document.createElement("span");
      label.className = "fo-label"; label.textContent = opt.label;
      btn.append(mark, label);
      if (opt.authority) {
        const pip = document.createElement("span");
        pip.className = "auth-pip auth-" + opt.authority;
        pip.textContent = opt.global ? opt.authority + " ·global" : opt.authority;
        btn.appendChild(pip);
      }
      if (opt.count != null) {
        const count = document.createElement("span");
        count.className = "fo-count"; count.textContent = fmtCount(opt.count);
        btn.appendChild(count);
      }
      btn.addEventListener("click", () => navigateCorpus(CI.toggleFacetValue(corpusState(), groupKey, String(opt.key))));
      li.appendChild(btn);
      return li;
    }

    function renderDateFacet(state) {
      const wrap = document.createElement("div");
      wrap.className = "facet-date";
      for (const [key, text] of [["published_from", "From"], ["published_to", "To"]]) {
        const row = document.createElement("label");
        const span = document.createElement("span");
        span.className = "field-label"; span.textContent = text;
        const input = document.createElement("input");
        input.type = "date"; input.value = state[key] || "";
        input.setAttribute("aria-label", `Published ${text.toLowerCase()}`);
        input.addEventListener("change", () => {
          const next = CI.emptyCorpusState();
          Object.assign(next, JSON.parse(JSON.stringify(corpusState())));
          next[key] = input.value; next.preset = "";
          navigateCorpus(next);
        });
        row.append(span, input);
        wrap.appendChild(row);
      }
      // Authoritative year buckets shown as read-only counts beside the range.
      const years = app.corpus.aggregates.get("year") || [];
      if (years.length) {
        const list = document.createElement("ul");
        list.className = "facet-options";
        for (const b of years.slice(0, 12)) {
          const li = document.createElement("li");
          const item = document.createElement("span");
          item.className = "facet-option";
          const label = document.createElement("span");
          label.className = "fo-label"; label.textContent = b.key;
          const count = document.createElement("span");
          count.className = "fo-count"; count.textContent = fmtCount(b.count);
          item.append(label, count);
          li.appendChild(item);
          list.appendChild(li);
        }
        wrap.appendChild(list);
      }
      return wrap;
    }

    function fmtCount(n) {
      const v = Number(n) || 0;
      return v >= 1000 ? v.toLocaleString("en-US") : String(v);
    }

    /* ── Inventory / results pane (the complete accessible surface) ──────── */
    function setInventory(rows, opts) {
      app.corpus.rows = rows;
      const list = $("inventory-list");
      list.replaceChildren();
      const empty = $("inventory-empty");
      empty.hidden = rows.length > 0;
      if (!rows.length) empty.textContent = (opts && opts.emptyText) || "No items match these facets. Clear a filter or pick a preset.";
      rows.forEach((row, i) => list.appendChild(renderInvRow(row, i)));
      $("inventory-count").textContent = (opts && opts.countText) || `${rows.length} shown`;
      $("btn-inventory-more").hidden = !(opts && opts.more);
      // First row is the roving-focus entry point for keyboard users.
      const first = list.querySelector(".inv-row");
      if (first) first.tabIndex = 0;
    }

    function renderInvRow(row, index) {
      const li = document.createElement("li");
      li.className = "inv-row" + (row.aggregate ? " is-aggregate" : "");
      li.setAttribute("role", "option");
      li.dataset.rowId = String(row.id != null ? row.id : index);
      li.tabIndex = -1;
      li.setAttribute("aria-selected", row.id === app.corpus.selectedRowId ? "true" : "false");
      const glyph = document.createElement("span");
      glyph.className = "inv-glyph"; glyph.setAttribute("aria-hidden", "true");
      glyph.textContent = KIND_GLYPH[row.kind] || (row.aggregate ? "▦" : "•");
      const title = document.createElement("span");
      title.className = "inv-title"; title.textContent = row.title;
      const meta = document.createElement("span");
      meta.className = "inv-meta"; meta.textContent = row.meta || "";
      const sub = document.createElement("span");
      sub.className = "inv-sub"; sub.textContent = row.sub || "";
      const badges = document.createElement("span");
      badges.className = "inv-badges";
      for (const b of row.badges || []) {
        const el = document.createElement("span");
        el.className = "inv-badge" + (b.status ? " st-badge st-" + b.status : "");
        el.textContent = b.text;
        badges.appendChild(el);
      }
      li.append(glyph, title, meta, sub, badges);
      li.addEventListener("click", () => activateInvRow(row));
      return li;
    }

    function activateInvRow(row) {
      app.corpus.selectedRowId = row.id;
      highlightInvRows(row.id);
      if (row.aggregate) { drillAggregate(row); return; }
      if (row.node) { selectCorpusItem(row.node); }
    }
    function highlightInvRows(id) {
      document.querySelectorAll(".inv-row").forEach((r) =>
        r.setAttribute("aria-selected", r.dataset.rowId === String(id) ? "true" : "false"));
    }

    // Aggregation-first landing: render the current drill dimension's buckets as
    // aggregate group rows. Never renders the corpus — one bounded page of counts.
    function renderAggregatesInventory(state) {
      const dim = state.groupBy || "source_category";
      let rows;
      if (dim === "ticker") rows = tickerInventoryRows();
      else if (dim === "index" || dim === "sector") rows = optionAggRows(dim);
      else rows = aggregateRows(dim, app.corpus.aggregates.get(dim) || []);
      setInventory(rows, {
        countText: `${rows.length} groups · ${dim.replace(/_/g, " ")}`,
        emptyText: "No aggregates yet. Corpus may be empty or still indexing.",
      });
    }

    // Fixed-vocabulary drill levels (index/sector) render as aggregate group rows
    // without universe-wide counts (2.3.5.3 index work); they still drill and set
    // the matching combinable facet.
    function optionAggRows(dim) {
      const group = CI.CORPUS_FACET_GROUPS.find((g) => g.key === dim);
      const labels = {
        sp500: "S&P 500", nasdaq100: "Nasdaq-100",
        overlap: "Index overlap", off_index: "Off-index (deep)",
      };
      return ((group && group.options) || []).map((key) => ({
        id: `agg:${dim}:${key}`, aggregate: true, dim, key, kind: "source",
        title: labels[key] || key, meta: "group",
        sub: dim.replace(/_/g, " "), badges: [],
      }));
    }

    function aggregateRows(dim, buckets) {
      return buckets.map((b) => {
        const badges = [];
        if (b.authority_tier) badges.push({ text: b.authority_tier });
        if (b.global_source) badges.push({ text: "global" });
        return {
          id: `agg:${dim}:${b.key}`, aggregate: true, dim, key: b.key, kind: "source",
          title: b.label || b.key,
          meta: `${fmtCount(b.count)} items`,
          sub: dim === "source_category" ? "source category" : dim.replace(/_/g, " "),
          badges,
        };
      });
    }

    // Per-security badges (SEC 14 · news 86 · market 252 · official 3) derive from
    // the bounded overview ticker projection, not hidden per-item DOM nodes.
    function tickerInventoryRows() {
      const OFFICIAL = new Set(["fred", "federal_reserve", "treasury", "bls", "bea", "eia"]);
      const rows = [];
      const fresh_map = overviewFreshMap();
      for (const n of overviewNodeValues()) {
        if (n.kind !== "ticker") continue;
        const m = n.metadata || {};
        const counts = m.source_counts || {};
        const badges = [];
        let sec = 0, news = 0, market = 0, official = 0;
        for (const s of Object.keys(counts)) {
          const key = s.toLowerCase();
          if (key.includes("sec")) sec += counts[s];
          else if (key.includes("news") || key.includes("finnhub") || key.includes("gdelt")) news += counts[s];
          else if (OFFICIAL.has(key)) official += counts[s];
          else market += counts[s];
        }
        if (sec) badges.push({ text: `SEC ${sec}` });
        if (news) badges.push({ text: `news ${news}` });
        if (market) badges.push({ text: `market ${market}` });
        if (official) badges.push({ text: `official ${official}` });
        const fresh = fresh_map.get(n.id);
        if (fresh && fresh.worst && fresh.worst !== "complete") {
          const map = { pending: "stale", dropped: "backlog", error: "error" };
          badges.push({ text: (map[fresh.worst] || fresh.worst), status: map[fresh.worst] || "stale" });
        }
        rows.push({
          id: `tick:${n.id}`, node: n, kind: "ticker", title: n.label,
          meta: `${fmtCount(m.record_count || 0)} items`,
          sub: (m.company_name || "").slice(0, 40),
          badges,
        });
      }
      return rows.sort((a, b) => a.title.localeCompare(b.title));
    }

    // Drill one authoritative level: fix the chosen aggregate as a facet and
    // advance groupBy to the next level. Leaf level switches to item results.
    function drillAggregate(row) {
      const next = CI.emptyCorpusState();
      Object.assign(next, JSON.parse(JSON.stringify(corpusState())));
      const facetKey = row.dim;
      if (CI.CORPUS_FACET_KEYS.includes(facetKey)) {
        next.facets[facetKey] = [String(row.key)];
      } else if (row.dim === "year" && /^\d{4}$/.test(String(row.key))) {
        next.published_from = `${row.key}-01-01`;
        next.published_to = `${row.key}-12-31`;
      }
      const order = CI.CORPUS_DRILL_ORDER;
      const idx = order.indexOf(row.dim);
      const nextDim = idx >= 0 && idx < order.length - 1 ? order[idx + 1] : null;
      if (nextDim) { next.groupBy = nextDim; next.view = "aggregates"; }
      else next.view = "results";
      next.preset = "";
      navigateCorpus(next);
      announce(`Expanded ${row.title}.`);
    }

    // Results view: item rows from the bounded search projection. The same nodes
    // populate the canvas (selected/searched only, folded/LOD preserved).
    async function renderResults(state) {
      const q = state.q || "";
      const ticker = (state.facets.ticker || [])[0] || "";
      let url = `${API}/corpus/search?q=${encodeURIComponent(q)}`;
      if (ticker) url += `&ticker=${encodeURIComponent(ticker)}`;
      const cats = state.facets.source_category || [];
      // Category facet maps to concrete node kinds so the item list narrows.
      const kinds = catKinds(cats);
      if (kinds.length) url += `&kinds=${encodeURIComponent(kinds.join(","))}`;
      app.corpus.lastResultsUrl = url;
      try {
        const data = await getJSON(url, corpusSignal());
        loadCorpusPage(data, true);
        app.corpus.resultsCursor = data.next_cursor || null;
        const rows = resultRows(data.nodes || [], state);
        setInventory(rows, {
          countText: `${rows.length} items` + (data.truncated ? " · truncated to cap" : ""),
          more: !!data.next_cursor,
          emptyText: "No items match these facets. Clear a filter or pick a preset.",
        });
      } catch (err) { showError("Corpus search failed", err); }
    }
    async function loadMoreResults() {
      if (!app.corpus.resultsCursor || !app.corpus.lastResultsUrl) return;
      const url = `${app.corpus.lastResultsUrl}&cursor=${encodeURIComponent(app.corpus.resultsCursor)}`;
      try {
        const data = await getJSON(url, corpusSignal());
        loadCorpusPage(data, false);
        app.corpus.resultsCursor = data.next_cursor || null;
        app.corpus.rows = app.corpus.rows.concat(resultRows(data.nodes || [], corpusState()));
        setInventory(app.corpus.rows, {
          countText: `${app.corpus.rows.length} items`, more: !!data.next_cursor,
        });
      } catch (err) { showError("Load next page failed", err); }
    }
    function submitCorpusSearch() {
      const next = CI.emptyCorpusState();
      Object.assign(next, JSON.parse(JSON.stringify(corpusState())));
      next.q = $("corpus-search").value.trim();
      next.view = "results"; next.preset = "";
      navigateCorpus(next);
    }
    function findInvRow(rowId) {
      return app.corpus.rows.find((r, i) => String(r.id != null ? r.id : i) === rowId) || null;
    }
    function wireInventoryKeyboard() {
      const list = $("inventory-list");
      list.addEventListener("keydown", (e) => {
        const rows = Array.from(list.querySelectorAll(".inv-row"));
        if (!rows.length) return;
        let idx = rows.findIndex((r) => r === document.activeElement);
        if (idx < 0) idx = rows.findIndex((r) => r.getAttribute("aria-selected") === "true");
        if (e.key === "ArrowDown") { e.preventDefault(); idx = Math.min(rows.length - 1, idx + 1); }
        else if (e.key === "ArrowUp") { e.preventDefault(); idx = Math.max(0, Math.max(idx, 0) - 1); }
        else if (e.key === "Home") { e.preventDefault(); idx = 0; }
        else if (e.key === "End") { e.preventDefault(); idx = rows.length - 1; }
        else if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          const cur = rows[Math.max(0, idx)];
          if (cur) { const row = findInvRow(cur.dataset.rowId); if (row) activateInvRow(row); }
          return;
        } else return;
        const row = rows[idx];
        if (row) {
          rows.forEach((r) => { r.tabIndex = -1; });
          row.tabIndex = 0; row.focus();
        }
      });
    }
    function catKinds(cats) {
      const map = {
        sec: ["filing", "section"], company_news: ["document_family"],
        market_data: ["metric", "fact"], issuer: ["document_family"],
      };
      const out = new Set();
      for (const c of cats) for (const k of map[c] || []) out.add(k);
      return Array.from(out);
    }
    function resultRows(nodes, state) {
      const rows = [];
      for (const n of nodes) {
        const m = n.metadata || {};
        const status = m.status || m.indexing_status || "";
        const badges = [];
        if (m.authority_tier) badges.push({ text: m.authority_tier });
        const st = corpusRowStatus(m);
        if (st) badges.push({ text: STATUS_GLYPH[st.key] + " " + st.key, status: st.key });
        rows.push({
          id: `item:${n.id}`, node: n, kind: n.kind, title: displayLabel(n),
          meta: [m.form || m.item_type || n.kind, dateOf(m)].filter(Boolean).join(" · "),
          sub: [m.ticker, m.source_category || m.source].filter(Boolean).join(" · "),
          badges,
        });
      }
      return rows;
    }
    function corpusRowStatus(m) {
      const s = String(m.status || m.indexing_status || "").toLowerCase();
      if (s.includes("error") || s.includes("fail")) return { key: "error" };
      if (s.includes("pending") || s.includes("index")) return { key: "indexing" };
      if (s.includes("stale")) return { key: "stale" };
      return null;
    }
    function dateOf(m) {
      return m.filing_date || m.date || m.published_at || m.as_of || m.last_updated || "";
    }

    function selectCorpusItem(node) {
      if (!app.corpus.nodes.has(node.id)) { app.corpus.nodes.set(node.id, node); renderCorpus(); }
      selectNode(node.id, false);
    }

    /* ══ 5. Resilient SSE client ══════════════════════════════════════════ */
    const stream = {
      es: null, lastEventId: null, backoff: 1000, timer: 0, countdown: 0,
    };
    function connect() {
      closeStream();
      const qs = stream.lastEventId != null ? `?last_sequence=${encodeURIComponent(stream.lastEventId)}` : "";
      let es;
      try { es = new EventSource(`${API}/events${qs}`); }
      catch (err) { scheduleReconnect(); return; }
      stream.es = es;
      const ops = ["upsert_node", "upsert_edge", "remove", "trace_complete", "trace_evicted", "reset_required"];
      ops.forEach((op) => es.addEventListener(op, onStreamEvent));
      es.onopen = () => { stream.backoff = 1000; setConnection("live", "live"); };
      es.onerror = () => { if (es.readyState === EventSource.CLOSED || es.readyState === EventSource.CONNECTING) scheduleReconnect(); };
    }
    function onStreamEvent(evt) {
      if (evt.lastEventId) stream.lastEventId = evt.lastEventId;
      let delta;
      try { delta = JSON.parse(evt.data); } catch (e) { return; }
      if (delta.operation === "reset_required") { reloadSnapshot(); return; }
      app.pending.push(delta);
      scheduleFlush();
    }
    function scheduleFlush() {
      if (app.rafHandle) return;
      app.rafHandle = requestAnimationFrame(() => {
        app.rafHandle = 0;
        const batch = app.pending;
        app.pending = [];
        let touchedActive = false;
        for (const delta of batch) {
          const res = GS.ingest(app.traces, delta);
          if (res.queryId && !app.knownTraceIds.has(res.queryId)) { app.knownTraceIds.add(res.queryId); refreshTraceList(); }
          if (res.queryId === app.activeId) touchedActive = true;
          if (app.follow && !app.pinnedTrace) maybeSelectNewest(delta);
        }
        if (touchedActive || !app.activeId) {
          recomputeScrubber();
          if (app.mode === "live") renderLive();
        }
      });
    }
    function scheduleReconnect() {
      closeStream();
      const delay = stream.backoff;
      stream.backoff = Math.min(stream.backoff * 2, 30000);
      stream.countdown = Math.round(delay / 1000);
      setConnection("reconnecting", `reconnecting… ${stream.countdown}s`);
      clearInterval(stream.timer);
      stream.timer = setInterval(() => {
        stream.countdown -= 1;
        if (stream.countdown <= 0) { clearInterval(stream.timer); return; }
        setConnection("reconnecting", `reconnecting… ${stream.countdown}s`);
      }, 1000);
      setTimeout(() => { clearInterval(stream.timer); connect(); }, delay);
    }
    function closeStream() {
      if (stream.es) { try { stream.es.close(); } catch (e) { /* ignore */ } stream.es = null; }
    }
    function setConnection(state, label) {
      document.documentElement.setAttribute("data-connection", state);
      $("conn-label").textContent = label;
    }

    /* ══ trace list + selection ═══════════════════════════════════════════ */
    async function refreshTraceList() {
      try {
        const list = await getJSON(`${API}/traces?limit=50`);
        const picker = $("query-picker");
        const prev = picker.value;
        picker.replaceChildren();
        for (const t of list) {
          const opt = document.createElement("option");
          opt.value = t.query_id;
          const preview = (t.question_preview || t.query_id).slice(0, 40);
          opt.textContent = `${t.complete ? "✓" : "◌"} ${preview}`;
          picker.appendChild(opt);
        }
        if (app.follow && !app.pinnedTrace && list.length) {
          selectTrace(list[0].query_id);
          picker.value = list[0].query_id;
        } else if (prev) {
          picker.value = prev;
        }
      } catch (err) { /* trace list is best-effort */ }
    }
    function maybeSelectNewest(delta) {
      if (!app.activeId && delta.query_id && delta.query_id !== "*") selectTrace(delta.query_id);
    }
    async function selectTrace(queryId) {
      if (!queryId) return;
      app.activeId = queryId;
      try {
        const snap = await getJSON(`${API}/traces/${encodeURIComponent(queryId)}`);
        const st = GS.createTraceState(queryId);
        for (const n of snap.nodes || []) st.nodes.set(n.id, n);
        for (const e of snap.edges || []) st.edges.set(e.id, e);
        st.complete = snap.complete;
        app.traces.set(queryId, st);
      } catch (err) { /* keep any streamed state */ }
      app.selectedId = null;
      recomputeScrubber();
      renderLive();
      clearInspector();
    }
    async function reloadSnapshot() {
      announce("Stream reset — reloading snapshot.");
      if (app.activeId) await selectTrace(app.activeId);
      else await refreshTraceList();
    }

    /* ══ scrubber ═════════════════════════════════════════════════════════ */
    function recomputeScrubber() {
      const trace = activeTrace();
      let max = 0;
      if (trace) {
        for (const n of trace.nodes.values()) max = Math.max(max, Number(n.updated_sequence) || 0);
        for (const e of trace.edges.values()) max = Math.max(max, Number(e.sequence) || 0);
      }
      app.scrubberMax = max;
      const sc = $("scrubber");
      sc.max = String(max);
      if (app.atLiveSticky !== false) { sc.value = String(max); app.scrubberValue = max; }
    }
    function atLive() { return Number($("scrubber").value) >= app.scrubberMax; }

    /* ══ 6. Inspector, node list, table ═══════════════════════════════════ */
    function selectNode(id, fromCanvas) {
      app.selectedId = id;
      const raw = lookupNode(id);
      if (!raw) { clearInspector(); return; }
      fillInspector(raw);
      if (app.mode === "live") applyBeam();
      highlightRows(id);
      if (cy && !fromCanvas) {
        const ele = cy.getElementById(id);
        if (ele.nonempty()) { cy.elements().unselect(); ele.select(); if (app.mode === "corpus") cy.center(ele); }
      }
      $("btn-pin").disabled = false;
      $("btn-collapse").disabled = !(raw.kind === "stage");
    }
    function lookupNode(id) {
      if (app.mode === "corpus") return app.corpus.nodes.get(id);
      const trace = activeTrace();
      return trace ? trace.nodes.get(id) : null;
    }
    function fillInspector(node) {
      $("inspector-empty").hidden = true;
      $("inspector-body").hidden = false;
      $("inspector-close").hidden = false;
      const kindEl = $("insp-kind");
      kindEl.textContent = node.kind;
      const status = node.status || (node.metadata && node.metadata.status) || "complete";
      const statusEl = $("insp-status");
      statusEl.className = "chip status-chip mono " + (STATUS_CLASS[status] || "");
      statusEl.textContent = status;
      $("insp-label").textContent = node.label || node.kind;

      const dl = $("insp-meta");
      dl.replaceChildren();
      const meta = node.metadata || {};
      for (const key of Object.keys(meta)) {
        const val = meta[key];
        if (val == null || val === "" || (Array.isArray(val) && !val.length)) continue;
        if (isPlainObject(val) && !Object.keys(val).length) continue;
        const dt = document.createElement("dt");
        dt.textContent = key.replace(/_/g, " ");
        const dd = document.createElement("dd");
        if ((key === "source_url" || key === "url") && typeof val === "string") {
          const a = document.createElement("a");
          a.href = val; a.textContent = val; a.rel = "noreferrer noopener"; a.target = "_blank";
          dd.appendChild(a);
        } else {
          dd.textContent = formatMetaValue(val);
        }
        dl.appendChild(dt); dl.appendChild(dd);
      }

      // Corpus tickers surface their folded freshness rollup (counts by status)
      // so the data folded off the canvas stays visible where you inspect it.
      if (app.mode === "corpus" && node.kind === "ticker" && app.corpus.freshAgg) {
        const summary = app.corpus.freshAgg.get(node.id);
        if (summary) {
          const total = Object.keys(summary.counts).reduce((sum, s) => sum + summary.counts[s], 0);
          const parts = Object.keys(summary.counts).sort().map((s) => `${summary.counts[s]} ${s}`);
          const dt = document.createElement("dt");
          dt.textContent = "freshness";
          const dd = document.createElement("dd");
          dd.textContent = `${total} tracked · ${parts.join(", ")}`;
          dl.appendChild(dt); dl.appendChild(dd);
        }
      }

      const excerpt = node.excerpt || (node.kind === "evidence" ? node.summary : "");
      $("insp-excerpt-wrap").hidden = !excerpt;
      $("insp-excerpt").textContent = excerpt || "";

      const provWrap = $("insp-provenance-wrap");
      const provList = $("insp-provenance");
      provList.replaceChildren();
      if (app.mode === "live" && GS.beamTriggers(node.kind)) {
        const trace = activeTrace();
        if (trace) {
          const beam = GS.computeBeam(Array.from(trace.nodes.values()), Array.from(trace.edges.values()), node.id);
          const labels = [];
          for (const nid of beam.nodes) {
            const bn = trace.nodes.get(nid);
            if (bn && nid !== node.id) labels.push(bn);
          }
          labels.sort((a, b) => GS.stageColumnFor(a) - GS.stageColumnFor(b));
          for (const bn of labels) {
            const li = document.createElement("li");
            li.textContent = `${bn.kind} · ${displayLabel(bn)}`;
            provList.appendChild(li);
          }
          provWrap.hidden = labels.length === 0;
        }
      } else {
        provWrap.hidden = true;
      }

      if (app.mode === "corpus" && node.id && !node.excerpt) enrichCorpusNode(node.id);
    }
    async function enrichCorpusNode(id) {
      try {
        const data = await getJSON(`${API}/corpus/nodes/${encodeURIComponent(id)}`);
        const detail = (data.nodes || [])[0];
        if (detail && detail.excerpt) { $("insp-excerpt-wrap").hidden = false; $("insp-excerpt").textContent = detail.excerpt; }
      } catch (err) { /* excerpt is optional */ }
    }
    function clearInspector() {
      app.selectedId = null;
      $("inspector-empty").hidden = false;
      $("inspector-body").hidden = true;
      $("inspector-close").hidden = true;
      $("btn-pin").disabled = true;
      $("btn-collapse").disabled = true;
      clearBeam();
      highlightRows(null);
    }

    /* node list + table (synchronized list view) */
    function syncSecondaryViews(nodes, edges) {
      const list = $("node-list");
      list.replaceChildren();
      const sorted = nodes.slice().sort((a, b) => (a.kind || "").localeCompare(b.kind || ""));
      for (const n of sorted) {
        const li = document.createElement("li");
        li.className = "node-row";
        li.setAttribute("role", "option");
        li.tabIndex = -1;
        li.dataset.id = n.id;
        li.setAttribute("aria-selected", n.id === app.selectedId ? "true" : "false");
        const glyph = document.createElement("span");
        glyph.className = "glyph"; glyph.textContent = KIND_GLYPH[n.kind] || "•"; glyph.setAttribute("aria-hidden", "true");
        const label = document.createElement("span");
        label.className = "n-label"; label.textContent = displayLabel(n);
        const st = document.createElement("span");
        st.className = "n-status " + (STATUS_CLASS[n.status] || ""); st.setAttribute("aria-hidden", "true");
        li.append(glyph, label, st);
        li.addEventListener("click", () => selectNode(n.id, false));
        list.appendChild(li);
      }
      $("node-count").textContent = `${nodes.length}`;

      const body = $("table-body");
      body.replaceChildren();
      for (const n of sorted) {
        const tr = document.createElement("tr");
        tr.dataset.id = n.id;
        tr.append(td(n.kind), td(displayLabel(n)), td(n.status || ""), td(nodeDetail(n)));
        tr.addEventListener("click", () => selectNode(n.id, false));
        body.appendChild(tr);
      }
      if (app.selectedId) highlightRows(app.selectedId);
    }
    function td(text) { const c = document.createElement("td"); c.className = "mono"; c.textContent = text; return c; }
    function nodeDetail(n) {
      const m = n.metadata || {};
      const parts = [];
      if (m.ticker) parts.push(m.ticker);
      if (m.metric) parts.push(m.metric);
      if (m.score != null) parts.push(`score ${m.score}`);
      if (m.elapsed_ms != null) parts.push(`${m.elapsed_ms}ms`);
      if (m.count != null) parts.push(`${m.count}`);
      // Source-aware summary: category + item/event type + role, so the table
      // reads as "who said it, what kind, primary or corroborating".
      if (m.source_category) parts.push(m.source_category);
      else if (m.source_type) parts.push(m.source_type);
      if (m.item_type) parts.push(m.item_type);
      if (m.event_type) parts.push(m.event_type);
      if (m.evidence_role) parts.push(m.evidence_role);
      return parts.join(" · ");
    }
    function highlightRows(id) {
      document.querySelectorAll(".node-row").forEach((r) => r.setAttribute("aria-selected", r.dataset.id === id ? "true" : "false"));
      document.querySelectorAll("#table-body tr").forEach((r) => r.classList.toggle("sel", r.dataset.id === id));
    }

    function updateHiddenCount(vis) {
      const hidden = vis.hiddenByFilter + vis.hiddenByCap;
      const parts = [];
      if (vis.hiddenByCap) parts.push(`${vis.hiddenByCap} over cap`);
      if (vis.hiddenByFilter) parts.push(`${vis.hiddenByFilter} filtered`);
      $("hidden-count").textContent = hidden ? `${hidden} hidden (${parts.join(", ")})` : "";
    }

    function refreshFilterOptions(trace) {
      const dims = { subquery: new Set(), source: new Set(), ticker: new Set(), kind: new Set() };
      for (const n of trace.nodes.values()) {
        const m = n.metadata || {};
        if (m.subquery_id) dims.subquery.add(String(m.subquery_id));
        const src = m.source_type != null ? m.source_type : m.source;
        if (src) dims.source.add(String(src));
        if (m.ticker) dims.ticker.add(String(m.ticker));
        if (n.kind === "evidence" && m.kind) dims.kind.add(String(m.kind));
      }
      fillOptions("filter-subquery", dims.subquery, app.filters.subquery);
      fillOptions("filter-source", dims.source, app.filters.source);
      fillOptions("filter-ticker", dims.ticker, app.filters.ticker);
      fillOptions("filter-kind", dims.kind, app.filters.kind);
    }
    function fillOptions(id, values, current) {
      const sel = $(id);
      const wanted = ["", ...Array.from(values).sort()];
      const existing = Array.from(sel.options).map((o) => o.value);
      if (existing.length === wanted.length && existing.every((v, i) => v === wanted[i])) return;
      sel.replaceChildren();
      for (const v of wanted) {
        const opt = document.createElement("option");
        opt.value = v; opt.textContent = v || "all";
        sel.appendChild(opt);
      }
      sel.value = current;
    }

    /* ══ HTTP helper ══════════════════════════════════════════════════════ */
    async function getJSON(url, signal) {
      const resp = await fetch(url, { headers: { Accept: "application/json" }, signal });
      if (!resp.ok) { const err = new Error(`HTTP ${resp.status}`); err.status = resp.status; throw err; }
      return resp.json();
    }

    // Cancel in-flight corpus view requests when the filter set changes so a
    // slow older page can never overwrite the current one (2.3.5.3 Step 4).
    function abortObsoleteCorpusRequests() {
      if (app.corpusAbort) app.corpusAbort.abort();
      app.corpusAbort = new AbortController();
      return app.corpusAbort.signal;
    }
    function corpusSignal() {
      return app.corpusAbort ? app.corpusAbort.signal : undefined;
    }
    function isAbortError(err) {
      return err && (err.name === "AbortError" || err.code === 20);
    }

    /* ══ error panel + read-only fallback ═════════════════════════════════ */
    function showError(title, err) {
      if (isAbortError(err)) return;   // superseded by a newer request; not an error
      $("error-panel").hidden = false;
      $("error-title").textContent = title;
      $("error-detail").textContent = err && err.message ? err.message : String(err);
      $("table-view").hidden = false;
      $("btn-view-toggle").setAttribute("aria-pressed", "true");
      announce(`${title}. A read-only table is available.`);
    }
    function hideError() { $("error-panel").hidden = true; }

    function announce(msg) { $("live-status").textContent = msg; }

    /* ══ mode switching ═══════════════════════════════════════════════════ */
    function setMode(mode) {
      app.mode = mode;
      document.documentElement.setAttribute("data-mode", mode);
      $("tab-live").setAttribute("aria-selected", String(mode === "live"));
      $("tab-corpus").setAttribute("aria-selected", String(mode === "corpus"));
      $("live-controls").hidden = mode !== "live";
      $("live-filters").hidden = mode !== "live";
      $("corpus-controls").hidden = mode !== "corpus";
      $("query-picker-field").hidden = mode !== "live";
      $("corpus-revision-field").hidden = mode !== "corpus";
      $("facet-rail").hidden = mode !== "corpus";
      $("inventory-pane").hidden = mode !== "corpus";
      clearInspector();
      if (mode === "corpus") {
        enterCorpus();
      } else {
        document.documentElement.removeAttribute("data-facets");
        renderLive();
        syncStageRail();
      }
    }

    // Enter Corpus Explorer: the bounded overview seeds the canvas, freshness
    // fold, and ticker projection; then the URL hash drives the facet rail and
    // aggregation-first inventory. mode=corpus is stamped without a history entry.
    async function enterCorpus() {
      if (!app.corpus.nodes.size) await corpusOverview();
      else renderCorpus();
      // Snapshot the bounded overview projection so the facet rail's freshness
      // and ticker derivations survive a later search replacing the canvas nodes.
      app.corpus.overviewNodes = new Map(app.corpus.nodes);
      app.corpus.overviewFresh = new Map(app.corpus.freshAgg);
      const state = CI.parseCorpusHash(window.location.hash);
      app.corpus.state = state;
      const hash = CI.corpusHash(state);
      if (window.location.hash !== hash) history.replaceState(null, "", hash);
      applyCorpusState(state);
    }

    /* ══ 7. Bootstrap + events ════════════════════════════════════════════ */
    function buildStageRail() {
      const rail = $("stage-rail");
      rail.replaceChildren();
      for (const name of GS.STAGE_COLUMNS) {
        const lane = document.createElement("div");
        lane.className = "lane";
        const label = document.createElement("span");
        label.className = "lane-label";
        label.textContent = name;
        lane.appendChild(label);
        rail.appendChild(lane);
      }
      syncStageRail();
    }
    function buildLegend() {
      const legend = $("legend");
      const items = [
        ["--flow", "active / flow"], ["--supported", "supported ✓"],
        ["--pending", "pending ◌"], ["--conflict", "error / conflict ✕"],
        ["--steel", "dropped ⌀"],
      ];
      for (const [token, label] of items) {
        const span = document.createElement("span");
        span.className = "legend-item";
        const sw = document.createElement("span");
        sw.className = "legend-swatch";
        sw.style.background = tok(token);
        span.append(sw, document.createTextNode(label));
        legend.appendChild(span);
      }
    }

    function initCytoscape() {
      try {
        cy = cytoscape({
          container: $("graph-canvas"),
          style: cyStyle(),
          minZoom: 0.2, maxZoom: 3,
          boxSelectionEnabled: false,
        });
        cy.on("tap", "node", (evt) => selectNode(evt.target.id(), true));
        cy.on("tap", (evt) => { if (evt.target === cy) clearInspector(); });
        cy.on("dbltap", "node", (evt) => { if (app.mode === "corpus") corpusExpand(evt.target.id()); });
        cy.on("grab", () => { app.dragging = true; });
        cy.on("free", () => { app.dragging = false; });
        // Reproject the stage rail whenever the viewport moves so lanes track
        // node columns through auto-fit, manual zoom, pan, and container resize.
        cy.on("pan zoom resize", syncStageRail);
        cy.on("zoom", () => updateCorpusLabelLOD());
        return true;
      } catch (err) {
        showError("Graph renderer failed to start", err);
        return false;
      }
    }

    function wireControls() {
      $("tab-live").addEventListener("click", () => setMode("live"));
      $("tab-corpus").addEventListener("click", () => setMode("corpus"));

      $("toggle-follow").addEventListener("click", () => {
        app.follow = !app.follow;
        const btn = $("toggle-follow");
        btn.setAttribute("aria-checked", String(app.follow));
        btn.dataset.on = String(app.follow);
        if (app.follow) { app.pinnedTrace = false; app.atLiveSticky = true; refreshTraceList(); }
      });

      $("query-picker").addEventListener("change", (e) => {
        app.pinnedTrace = true; app.follow = false;
        $("toggle-follow").setAttribute("aria-checked", "false");
        $("toggle-follow").dataset.on = "false";
        selectTrace(e.target.value);
      });

      $("btn-fit").addEventListener("click", () => { if (cy) cy.fit(cy.elements(), 48); });
      $("btn-reset").addEventListener("click", () => {
        app.filters = { status: "", subquery: "", source: "", ticker: "", kind: "", citedOnly: false, maxSequence: null };
        ["filter-status", "filter-subquery", "filter-source", "filter-ticker", "filter-kind"].forEach((id) => { $(id).value = ""; });
        $("filter-cited").checked = false;
        app.atLiveSticky = true;
        clearInspector();
        recomputeScrubber();
        if (cy) cy.reset();
        renderLive();
      });
      $("btn-pin").addEventListener("click", () => {
        const btn = $("btn-pin");
        if (!app.selectedId || !cy) return;
        const ele = cy.getElementById(app.selectedId);
        if (app.pinnedNode === app.selectedId) { app.pinnedNode = null; ele.unlock(); btn.setAttribute("aria-pressed", "false"); }
        else { app.pinnedNode = app.selectedId; ele.lock(); btn.setAttribute("aria-pressed", "true"); }
      });
      $("btn-collapse").addEventListener("click", () => {
        if (!app.selectedId || !cy) return;
        const node = lookupNode(app.selectedId);
        if (!node || node.kind !== "stage") return;
        const col = GS.stageColumnFor(node);
        const trace = activeTrace();
        if (!trace) return;
        const key = `col-${col}`;
        app.collapsed = app.collapsed || new Set();
        if (app.collapsed.has(key)) app.collapsed.delete(key); else app.collapsed.add(key);
        cy.nodes().forEach((ele) => {
          const raw = ele.data("raw");
          if (raw && raw.kind !== "stage" && GS.stageColumnFor(raw) === col) {
            ele.style("display", app.collapsed.has(key) ? "none" : "element");
          }
        });
      });
      $("btn-export").addEventListener("click", exportSnapshot);

      $("btn-view-toggle").addEventListener("click", () => {
        const t = $("table-view");
        t.hidden = !t.hidden;
        $("btn-view-toggle").setAttribute("aria-pressed", String(!t.hidden));
      });

      $("filter-status").addEventListener("change", (e) => { app.filters.status = e.target.value; renderLive(); });
      $("filter-subquery").addEventListener("change", (e) => { app.filters.subquery = e.target.value; renderLive(); });
      $("filter-source").addEventListener("change", (e) => { app.filters.source = e.target.value; renderLive(); });
      $("filter-ticker").addEventListener("change", (e) => { app.filters.ticker = e.target.value; renderLive(); });
      $("filter-kind").addEventListener("change", (e) => { app.filters.kind = e.target.value; renderLive(); });
      $("filter-cited").addEventListener("change", (e) => { app.filters.citedOnly = e.target.checked; renderLive(); });
      $("btn-clear-filters").addEventListener("click", () => $("btn-reset").click());

      $("scrubber").addEventListener("input", (e) => {
        app.scrubberValue = Number(e.target.value);
        app.atLiveSticky = app.scrubberValue >= app.scrubberMax;
        $("scrubber-value").textContent = app.atLiveSticky ? "live" : `seq ${app.scrubberValue}`;
        $("scrubber").setAttribute("aria-valuetext", app.atLiveSticky ? "live" : `sequence ${app.scrubberValue}`);
        scheduleScrubRender();
      });
      // Drag end (or keyboard commit): leave scrub mode so live updates animate
      // and follow-fit resumes. atLiveSticky already tracks whether we're at head.
      $("scrubber").addEventListener("change", () => {
        app.scrubbing = false;
        renderLive();
      });

      $("btn-retry").addEventListener("click", () => { hideError(); if (!cy) initCytoscape(); reloadSnapshot(); });
      $("inspector-close").addEventListener("click", clearInspector);
      // Escape closes the inspector from anywhere except a text field, keeping
      // the panel fully keyboard-operable (2.3.5.3 Step 6).
      document.addEventListener("keydown", (e) => {
        if (e.key !== "Escape") return;
        const tag = (document.activeElement && document.activeElement.tagName) || "";
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        if (!$("inspector-body").hidden) { clearInspector(); e.preventDefault(); }
      });

      $("corpus-search").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); submitCorpusSearch(); } });
      $("btn-corpus-search").addEventListener("click", submitCorpusSearch);
      $("btn-corpus-overview").addEventListener("click", () => navigateCorpus(CI.emptyCorpusState()));
      $("btn-corpus-more").addEventListener("click", () => {
        if (app.corpus.lastQuery != null) corpusSearch(false);
      });

      // Aggregation-first controls (2.3.5.2): presets, facet clear, drawer, paging.
      $("corpus-presets").addEventListener("change", (e) => {
        if (e.target.value) navigateCorpus(CI.applyPreset(corpusState(), e.target.value));
      });
      $("btn-facets-clear").addEventListener("click", () => navigateCorpus(CI.emptyCorpusState()));
      $("btn-facets-toggle").addEventListener("click", () => {
        const open = document.documentElement.getAttribute("data-facets") === "open";
        document.documentElement.setAttribute("data-facets", open ? "closed" : "open");
        $("btn-facets-toggle").setAttribute("aria-expanded", String(!open));
      });
      $("btn-inventory-more").addEventListener("click", loadMoreResults);
      wireInventoryKeyboard();

      wireKeyboard();
    }

    function wireKeyboard() {
      const list = $("node-list");
      list.addEventListener("keydown", (e) => {
        const rows = Array.from(list.querySelectorAll(".node-row"));
        if (!rows.length) return;
        let idx = rows.findIndex((r) => r.dataset.id === app.selectedId);
        if (e.key === "ArrowDown") { e.preventDefault(); idx = Math.min(rows.length - 1, idx + 1); }
        else if (e.key === "ArrowUp") { e.preventDefault(); idx = Math.max(0, idx - 1); }
        else if (e.key === "Home") { e.preventDefault(); idx = 0; }
        else if (e.key === "End") { e.preventDefault(); idx = rows.length - 1; }
        else if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          const cur = rows[Math.max(0, idx)];
          if (cur) { if (app.mode === "corpus") corpusExpand(cur.dataset.id); else selectNode(cur.dataset.id, false); }
          return;
        } else return;
        const row = rows[idx];
        if (row) { selectNode(row.dataset.id, false); row.focus(); }
      });
    }

    function exportSnapshot() {
      let payload;
      if (app.mode === "corpus") {
        payload = {
          mode: "corpus", corpus_revision: app.corpus.revision,
          nodes: Array.from(app.corpus.nodes.values()),
          edges: Array.from(app.corpus.edges.values()),
        };
      } else {
        const trace = activeTrace();
        payload = {
          mode: "live", query_id: app.activeId,
          nodes: trace ? Array.from(trace.nodes.values()) : [],
          edges: trace ? Array.from(trace.edges.values()) : [],
        };
      }
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `trace-snapshot-${app.mode}-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      announce("Redacted snapshot exported.");
    }

    // Deep-link support for the chat client's `/graph trace` command (2.2.7.4):
    // a `#trace=<query_id>` fragment focuses that trace instead of following the
    // newest. Same-origin, read-only; the id is only ever passed to selectTrace.
    function traceFromHash() {
      const m = /(?:^|[#&])trace=([^&]+)/.exec(window.location.hash || "");
      if (!m) return null;
      try { return decodeURIComponent(m[1]); } catch (_e) { return m[1]; }
    }
    async function applyTraceHash() {
      const queryId = traceFromHash();
      if (!queryId) return;
      app.pinnedTrace = true;               // stop follow-live from overriding it
      await selectTrace(queryId);
      const picker = $("query-picker");
      if (picker) picker.value = queryId;
    }

    async function boot() {
      buildStageRail();
      buildLegend();
      app.atLiveSticky = true;
      const okConfig = await preflight();
      if (!okConfig) return;
      const rendererReady = initCytoscape();
      wireControls();
      if (rendererReady) { requestAnimationFrame(tickDash); syncStageRail(); }
      await refreshTraceList();
      await applyTraceHash();
      connect();
      window.addEventListener("resize", () => { if (app.mode === "live") { renderLive(); syncStageRail(); } });
      window.addEventListener("hashchange", () => {
        if (app.mode === "corpus") onCorpusHashChange();
        else applyTraceHash();
      });
    }

    // Confirm the observer is on before wiring the live stream. If it is off
    // the API 404s; we surface a clear message instead of a blank canvas.
    async function preflight() {
      try {
        await getJSON(`${API}/health`);
        return true;
      } catch (err) {
        if (err.status === 404) {
          showError("Graph observer is disabled", new Error("Set ENABLE_GRAPH_OBSERVER=1 and restart the middleware."));
          $("canvas-empty").hidden = true;
          return false;
        }
        return true; // transient error — proceed and let the stream retry
      }
    }

    // Expose a few controller hooks for the implementation-time smoke test.
    window.__graph = { app, selectNode, setMode, renderLive, connect };

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
    else boot();
  })();
}

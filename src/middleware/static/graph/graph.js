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
          compile: 0, route: 1, retrieve: 2, grade: 2, correct: 2,
          pack: 3, generate: 4, validate: 5, "query error": 5,
        };
        return name in map ? map[name] : 3;
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
      corpus: { revision: null, cursor: null, lastQuery: null, nodes: new Map(), edges: new Map() },
      dragging: false,
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
      ];
    }

    let cy = null;

    /* ══ 3./4. Rendering ══════════════════════════════════════════════════ */
    function nodeToEle(n) {
      const emd = (n.metadata && n.metadata.kind) || "";
      return {
        group: "nodes",
        data: {
          id: n.id, label: displayLabel(n), kind: n.kind,
          status: n.status, emd, raw: n,
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

    function activeTrace() {
      return app.activeId ? app.traces.get(app.activeId) : null;
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
            if (!reduceMotion) { ele.style("opacity", 0); ele.animate({ style: { opacity: 1 } }, { duration: 200 }); }
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
      buckets.forEach((bucket, col) => {
        const gap = Math.min(70, (h - 80) / Math.max(1, bucket.length));
        bucket.forEach((n, i) => {
          positions[n.id] = { x: colW * col + colW / 2, y: 60 + gap * i + gap / 2 };
        });
      });
      cy.nodes().forEach((ele) => {
        const p = positions[ele.id()];
        if (!p) return;
        if (reduceMotion) ele.position(p);
        else ele.animate({ position: p }, { duration: 200, easing: "ease-out" });
      });
      markHotLanes(buckets);
      if (app.follow) requestAnimationFrame(() => cy.fit(cy.elements(), 48));
    }

    function markHotLanes(buckets) {
      const rail = $("stage-rail");
      Array.from(rail.children).forEach((lane, col) => {
        const hot = buckets[col] && buckets[col].some((n) => n.status === "active");
        lane.classList.toggle("hot", !!hot);
      });
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
      if (cy && !reduceMotion) {
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
    function corpusEle(container) {
      const nodes = Array.from(app.corpus.nodes.values()).map((n) => ({
        group: "nodes",
        data: { id: n.id, label: n.label || n.kind, kind: n.kind, status: "complete", emd: "", raw: n },
      }));
      const ids = new Set(nodes.map((n) => n.data.id));
      const edges = Array.from(app.corpus.edges.values())
        .filter((e) => ids.has(e.source) && ids.has(e.target))
        .map((e) => ({ group: "edges", data: { id: e.id, source: e.source, target: e.target, relation: e.relation, raw: e } }));
      return { nodes, edges, hidden: container };
    }
    function renderCorpus() {
      if (!cy) return;
      const { nodes, edges } = corpusEle();
      $("canvas-empty").hidden = nodes.length > 0;
      if (!nodes.length) $("canvas-empty").textContent = "Search the corpus or pick a source group to expand.";
      cy.elements().remove();
      cy.add(nodes);
      cy.add(edges);
      if (nodes.length) {
        const layout = cy.layout({ name: "cose", animate: !reduceMotion, animationDuration: 300, fit: true, padding: 40, nodeRepulsion: 6000 });
        layout.run();
      }
      syncSecondaryViews(nodes.map((n) => n.data.raw), edges.map((e) => e.data.raw));
      $("hidden-count").textContent = app.corpus.cursor ? "more pages available" : "";
      $("btn-corpus-more").hidden = !app.corpus.cursor;
    }

    async function corpusOverview() {
      try {
        const data = await getJSON(`${API}/corpus/overview`);
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
        const dt = document.createElement("dt");
        dt.textContent = key.replace(/_/g, " ");
        const dd = document.createElement("dd");
        if ((key === "source_url" || key === "url") && typeof val === "string") {
          const a = document.createElement("a");
          a.href = val; a.textContent = val; a.rel = "noreferrer noopener"; a.target = "_blank";
          dd.appendChild(a);
        } else {
          dd.textContent = Array.isArray(val) ? val.join(", ") : String(val);
        }
        dl.appendChild(dt); dl.appendChild(dd);
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
      if (m.source_type) parts.push(m.source_type);
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
    async function getJSON(url) {
      const resp = await fetch(url, { headers: { Accept: "application/json" } });
      if (!resp.ok) { const err = new Error(`HTTP ${resp.status}`); err.status = resp.status; throw err; }
      return resp.json();
    }

    /* ══ error panel + read-only fallback ═════════════════════════════════ */
    function showError(title, err) {
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
      $("scrubber-bar").querySelectorAll("input,.field-label").forEach(() => {});
      clearInspector();
      if (mode === "corpus") {
        if (!app.corpus.nodes.size) corpusOverview();
        else renderCorpus();
      } else {
        renderLive();
      }
    }

    /* ══ 7. Bootstrap + events ════════════════════════════════════════════ */
    function buildStageRail() {
      const rail = $("stage-rail");
      rail.style.gridTemplateColumns = "repeat(6, 1fr)";
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
        renderLive();
      });

      $("btn-retry").addEventListener("click", () => { hideError(); if (!cy) initCytoscape(); reloadSnapshot(); });
      $("inspector-close").addEventListener("click", clearInspector);

      $("corpus-search").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); corpusSearch(true); } });
      $("btn-corpus-search").addEventListener("click", () => corpusSearch(true));
      $("btn-corpus-overview").addEventListener("click", () => corpusOverview());
      $("btn-corpus-more").addEventListener("click", () => {
        if (app.corpus.lastQuery != null) corpusSearch(false);
      });

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
      if (rendererReady) requestAnimationFrame(tickDash);
      await refreshTraceList();
      await applyTraceHash();
      connect();
      window.addEventListener("resize", () => { if (app.mode === "live") renderLive(); });
      window.addEventListener("hashchange", () => { applyTraceHash(); });
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

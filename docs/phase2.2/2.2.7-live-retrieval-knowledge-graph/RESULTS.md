# 2.2.7 Live Retrieval Knowledge Graph — Results

**Scope:** 2.2.7.1–2.2.7.4 (query-trace event model + loopback API, Store-backed
corpus projection/explorer, Cytoscape single-page UI, chat integration + security
+ evaluation). Offline sign-off; no GPU/model required.

**Path note:** the 2.2.7.4 spec names
`docs/plans/phase2.2/2.2.7-live-retrieval-knowledge-graph/RESULTS.md`. This repo's
convention (`docs/phase2.2/README-2.2.md`) keeps each feature's `RESULTS.md`
inside its feature folder under `docs/phase2.2/`, so this file lives here.

## Ship decision

**Ship disabled by default.** `enable_graph_observer: false`. When enabled, the
UI and API are served **loopback-only** with a strict same-origin CSP; there is
**no remote-exposure path and no bypass flag** in Phase 2.2. Rationale: the
observer opens a new view onto questions, evidence excerpts, source URLs, and
errors, so it is only safe as a strictly local, opt-in tool. Remote exposure
would require a separate design (authentication, TLS, proxy trust, retention) and
is intentionally out of scope. Rollback is flipping the flag back to `false`;
nothing is persisted.

## Completeness & truthfulness (offline)

Seeded/offline query contexts assert the trace is complete and truthful
(`tests/test_graph_observer.py`, `tests/test_graph_ui_contract.py`):

- Every executed subquery / stage / tool / evidence / source / citation appears
  exactly once; no node claims an unexecuted stage or unused evidence.
- `supports` / `cited_by` edges resolve to the emitted evidence ledger; every
  edge endpoint resolves to a node (no dangling edges); node ids are unique.
- Legacy, adaptive (fast/standard/complex), degraded (model-unavailable), and
  error paths all reach a valid **terminal** graph state (`complete = true`; error
  paths carry an error-status node).
- Two concurrent queries never share nodes/edges (ids are `query_id`-prefixed).
- **Observer disabled** yields no deltas and a **byte-compatible** `/query`
  response — the only difference when enabled is the optional `graph_trace_id`.
- A golden wire fixture (`tests/fixtures/graph/query_trace_v1.json`) pins schema
  compatibility (node kinds, edge relations, allowlisted metadata, bounded
  excerpts) and carries **no pixel/layout positions** — layout is UI behavior,
  graph meaning is the contract.

## Overhead & non-interference (offline)

- **Publish p95 < 2 ms** over 600 publishes across 50 traces
  (`test_normal_publish_p95_below_two_ms`); `publish` is synchronous and adds no
  awaited browser work to the query path.
- **Non-blocking fan-out:** a full/slow SSE subscriber is marked for reset and
  never blocks a publish; the published trace is unaffected.
- **Fail-soft:** an observer callback that raises disables itself and discards
  future deltas rather than affecting the query.
- **Limits hold under stress:** trace-count, element, TTL, excerpt, and
  question-preview limits are enforced by the `TraceHub` (and clamped again by
  `MiddlewareConfig`); the explorer pages/expansions are hard element-capped.

## Limits, redaction & security (offline)

- **Loopback-only, no bypass:** every `/graph*` route (UI, static bundle, API)
  returns 404 to a non-loopback client and while the observer is disabled
  (`test_all_graph_surfaces_reject_non_loopback_clients`, disabled-404 tests).
- **Strict same-origin CSP + hardening headers** on graph responses
  (`default-src 'none'`; `script-src 'self'`; `connect-src 'self'`;
  `frame-ancestors 'none'`; `nosniff`; `no-referrer`; same-origin COOP/CORP).
  `style-src` includes `'unsafe-inline'` **only** because Cytoscape injects one
  fixed `position: relative` `<style>` at runtime; inline style cannot execute
  code and `script-src` stays strict `'self'` (the XSS-critical directive). No
  third-party origin ever appears.
- **`Cache-Control: no-store`** on trace/corpus detail responses; the SSE stream
  keeps `no-cache`.
- **Redaction by construction:** questions stored as a bounded preview + SHA-256
  digest; evidence bodies bounded; secret keys and local paths scrubbed
  recursively; source links kept only when `http`/`https`; node ids never expose
  local paths. Canary secret/path strings do not survive in any snapshot / list /
  evidence / health output (`test_no_canary_secret_or_local_path_survives_...`).
- **No third-party surface:** no cookies, localStorage, service worker, analytics,
  or external font/script requests; the UI never writes dynamic data via
  `innerHTML`.

## Browser acceptance (implementation-time smoke)

Middleware started with `ENABLE_GRAPH_OBSERVER=1` on a spare port (model down —
degraded path still produces a trace); driven with Playwright:

- The UI loads under the strict CSP with **zero console errors and zero CSP
  violations**; Cytoscape renders (canvas present) and the SSE stream connects
  ("live").
- A `POST /query` produced a trace (`graph_trace_id` returned, `complete = true`)
  that appeared in the trace picker.
- `/graph#trace=<id>` deep-links the specific trace: it is selected and pinned
  (follow-live does not override it) with its nodes/edges rendered — this backs
  the chat client's `/graph trace` command.
- Security headers verified on the wire: strict CSP, `nosniff`, `no-referrer`,
  `no-store` (HTML/detail), non-loopback client → 404.

A later optional **live** smoke (real generation) may run only when a GPU is
explicitly authorized; it is not required for this offline sign-off.

## Decisions

- **`window.__graph` debug hook kept, deliberately.** `graph.js` exposes a small
  read-only controller hook (`{app, selectNode, setMode, renderLive, connect}`)
  used by the implementation-time smoke to assert the UI booted under CSP. It
  exposes nothing beyond already-rendered, already-redacted state; the entire page
  is loopback-only and observer-gated. Retained as a low-cost, verifiable smoke
  seam.
- **`graph.js` trace deep-link added.** A same-origin `#trace=<query_id>` fragment
  focuses (and pins) that trace on boot and on `hashchange`; this is what
  `/graph trace` targets. Re-smoked after the CSP change — clean.
- **RESULTS.md location.** Written in the feature folder per repo convention (see
  path note above) rather than under `docs/plans/`.

## Gate

`powershell -ExecutionPolicy Bypass -File scripts\verify.ps1` → **VERIFY: PASS**
(offline, `-m "not live"`). Live-model promotion remains PENDING (GPU evaluation
not authorized), consistent with the rest of Phase 2.2.

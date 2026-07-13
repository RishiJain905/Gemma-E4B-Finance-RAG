# Design direction — 2.2.7.3 "TraceAlchemy Telemetry" graph UI

Authored by the orchestrator (Fable). The implementer follows this exactly; where the
spec (2.2.7.3 Step 2) and this doc overlap, they agree — this doc makes the spec's
direction executable. The subject is a **financial retrieval observatory**: an
instrument you leave open on a side monitor while chatting with a local finance RAG.
It should feel like a trading-floor telemetry panel crossed with an air-traffic
flight-progress board — calm, dense, truthful — never like an "AI dashboard".

## Tokens

Palette (dark-only; this is an instrument, not a website — no light theme):

| token | hex | use |
|---|---|---|
| `--canvas` | `#0B0D10` | page/graph background (cold graphite) |
| `--surface` | `#14171C` | panels, inspector, header |
| `--surface-2` | `#1B1F26` | hover rows, chips, scrubber track |
| `--hairline` | `#262B33` | 1px borders, grid rules |
| `--ink` | `#EDE6D8` | primary text (warm ivory — the warmth against cold graphite is the identity) |
| `--ink-dim` | `#98937F`* | secondary text, labels (*warm gray, keep warm not blue) |
| `--flow` | `#4FD9EA` | retrieval flow, active stage, links, connection-live |
| `--pending` | `#E5B44C` | pending/borderline/partial |
| `--supported` | `#72C98F` | supported evidence, validated citations, PASS states |
| `--conflict` | `#EF6F5C` | errors, conflicts, dropped-with-error |
| `--steel` | `#5A6470` | dropped/inactive/evicted, disabled controls |

Rules: status is always color **plus** shape/line-style/icon (spec requirement).
Never purple, never gradients on surfaces, no glass blur. One restrained glow only:
active nodes may carry a soft `--flow` outer glow while `status=active`.

Canvas texture: 24px hairline grid at ~4% opacity plus a static noise tile
(tiny inline data-URI PNG, ~2% opacity). It should read as graph paper for
telemetry, visible only when you look for it.

## Typography

Two families, both bundled (already pinned in vendor/):

- **Recursive** (variable; axes incl. `MONO`, `CASL`, `wght`) — all UI text.
  - UI/controls/labels: `font-variation-settings: "MONO" 0, "CASL" 0`, wght 440–520.
  - Panel headers & mode tabs: wght 620, small-caps-style tracked labels
    (uppercase, `letter-spacing: 0.08em`, 11px).
  - The one display moment: the header wordmark "TRACE TELEMETRY" set in Recursive
    with `"CASL" 1` (casual axis) at wght 700 — a single flash of personality.
- **Commit Mono** (400/700) — every identifier, number, ticker, score, timestamp,
  sequence, accession, query-id. If it's data, it's mono. This split (warm humanist
  UI vs. strict mono data) is the typographic system.

Scale: 11 / 12.5 / 14 / 16 / 22px. No text below 11px.

## Layout

```
┌──────────────────────────────────────────────────────────────┐
│ header: wordmark · mode tabs [Live Trace|Corpus] · conn dot   │
│         query picker (mono ids) · corpus revision · follow ◉  │
├───────────────────────────────────────────────┬──────────────┤
│  graph canvas (Cytoscape)                     │ inspector    │
│  — Live Trace: 6 stage columns behind graph   │  (360px)     │
│    COMPILE ROUTE RETRIEVE EVIDENCE GEN VALID  │ label, kind  │
│    (hairline rules + tracked column labels)   │ status chip  │
│  — Corpus: free cose layout, no columns       │ metadata kv  │
│                                               │ excerpt      │
│                                               │ provenance   │
├───────────────────────────────────────────────┴──────────────┤
│ timeline scrubber (seq 0 ─────●───── live) · legend · counts │
└──────────────────────────────────────────────────────────────┘
```

- **Signature element — the stage rail + provenance beam.** In Live Trace mode the
  canvas carries six faint vertical lanes labeled like flight-progress strips; the
  trace grows left→right through them. Selecting an answer/citation/evidence node
  triggers the *provenance beam*: the full supporting path (evidence → source →
  answer edges) lights `--flow`→`--supported` while everything off-path dims to
  `--steel` at 35% opacity. The beam is the product thesis — "show me why" — as one
  interaction. Spend the boldness here; keep everything else quiet.
- Node shapes per spec Step 2 (query=round-rect, tool=hexagon, fact=diamond,
  doc/section=page glyph, source=ring, answer=double border). Edges: thin (1.5px),
  directional arrows, `--steel` at rest; cyan animated dash *only* while in flight.
- Table fallback / synchronized list view: same data, `<table>` under the canvas,
  toggled; also shown automatically if Cytoscape/SSE init fails, with a plain error
  panel (what failed + how to retry).

## Motion

- State transitions 160–220ms ease-out; nothing loops except (a) the connection
  dot's 2s breathing while connected, (b) active-node glow pulse, (c) in-flight
  edge dash. All three stop when their state resolves.
- New-node entrance: fade + 4px rise, no bounce, no scale-from-zero.
- `prefers-reduced-motion`: kill pulse/dash/entrance — statuses change by opacity
  step only. Layout animations off; `fit` jumps are instant.

## Copy

Sentence case, plain verbs, no mascot voice. Controls: "Follow live", "Fit view",
"Pin node", "Collapse stage", "Export snapshot", "Reload snapshot". Statuses are
words + glyphs: "supported ✓", "pending ◌", "dropped ⌀", "conflict ✕". Connection
states: "live", "reconnecting…", "disconnected — retrying in 4s". Empty states
direct action: "No traces yet. Ask a question in chat and it appears here." /
"Search the corpus or pick a source group to expand." Errors say what happened and
what to do, never apologize.

## Quality floor (non-negotiable)

Keyboard reachable everything (roving focus in the node list mirrors canvas
selection), visible `:focus-visible` rings in `--flow`, WCAG 2.1 AA contrast
(ivory on graphite passes; check `--ink-dim` ≥ 4.5:1 on `--surface`, adjust up if
needed), 24px min targets, `aria-live="polite"` for connection/query status,
responsive down to a 1280px laptop (inspector collapses to a bottom sheet).

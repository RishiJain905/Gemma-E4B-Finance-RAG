# Design direction — 2.3.5.2 Corpus Explorer information architecture

Authored by the implementer (opus-2352) under orchestrator direction. This record
is the design contract for the aggregation-first, faceted Corpus Explorer. It
**extends** the 2.2.7.3 "TraceAlchemy Telemetry" direction rather than replacing
it: every token, typeface, motion rule, and security posture from
[`design-direction-2.2.7.3.md`](../../../phase2.2/2.2.7-live-retrieval-knowledge-graph/assets/design-direction-2.2.7.3.md)
still governs. Corpus Explorer is the instrument's *second mode*, not a second
product.

## The problem, restated

The Phase 2.2 explorer opens from sources/tickers/facts/filings and draws that
inventory directly. At Phase 2.3 scale (≥600 securities, ≥100k corpus items) that
is a hairball. The landing view must **explain coverage and freshness without
rendering the corpus**. Phase 2.2 already proved the fold pattern for freshness
leaves (worst-status ring + per-status counts, full inventory retained in
list/inspector); 2.3.5.2 generalizes that fold to every dense dimension.

## Design thesis: a coverage ledger, not a map

The Live Trace signature is the stage rail + provenance beam — "show me why" as
one interaction. The Corpus signature is its calm sibling: **the coverage
ledger** — a departure-board reading of what the corpus holds, by authority and
freshness, before a single document node is drawn. Where Live Trace grows a trace
left→right through six lanes, Corpus opens as a bank of aggregate rows with count
bars, authority glyphs, and freshness rings. Both read as the same telemetry
panel; the boldness stays spent on the beam, and Corpus stays quiet and dense.

## Three surfaces (spec Step 2)

The corpus mode reorganizes the stage-region into three role-distinct surfaces,
plus the shared inspector:

```
┌──────────────────────────────────────────────────────────────────────┐
│ toolbar: presets ▾ · search · Overview · [aggregates | results] · view│
├───────────────┬───────────────────────────────────┬──────────────────┤
│ facet rail    │ inventory / results pane          │ canvas (Cytoscape)│
│ (persistent)  │ (paged rows; complete a11y surface)│ selected only    │
│  coverage tier│  ┌ SEC / EDGAR    14,203  primary ┐│  aggregate nodes  │
│  index        │  │ company news   86,110  licensed││  + selected result│
│  sector       │  │ market data   252,004  struct. ││  + immediate rels │
│  source cat.  │  └ …one bounded page at a time    ┘│                   │
│  item / event │                                    │                   │
│  form/exhibit │  rows: title·ticker·type·source·   │                   │
│  date range   │        date·status  (keyboard)     │                   │
│  freshness    │                                    │                   │
│  authority    │                                    │                   │
├───────────────┴───────────────────────────────────┴──────────────────┤
│ inspector (shared aside): full safe metadata · provenance origins ·   │
│ coverage/membership · freshness rollup · indexing state · excerpt     │
└──────────────────────────────────────────────────────────────────────┘
```

- **Facet rail** — persistent, grouped, collapsible. Every facet from spec Step 1:
  index/coverage tier, sector/industry, ticker, source category + specific source,
  item type + event type, form/exhibit, published/effective/as-of date range,
  freshness/indexing/error state, authority tier. Facets are **combinable**;
  counts appear **before** expansion; provider names are present but never the
  first navigation level. Rendered in the 2.2.7.3 field-label small-caps idiom.
- **Inventory / results pane** — the complete accessible navigation surface. Paged
  rows carrying title/ticker/type/source/date/status, fully keyboard-navigable
  (roving `tabindex`, Arrow/Home/End/Enter), and **useful with Cytoscape
  unavailable** — it is the same DOM used as the table fallback. Selecting a row
  focuses/adds it to the canvas and fills the inspector.
- **Canvas** — only the selected aggregate groups, selected results, and their
  immediate relationships. Never the full result set. Freshness folding and label
  level-of-detail from Phase 2.2 are preserved exactly. Expanding a group adds one
  bounded page and updates its aggregate count; it never auto-expands descendants.
- **Inspector** — unchanged role, richer corpus payload: complete safe metadata,
  provenance origins, coverage/membership, freshness rollup, indexing state, one
  bounded excerpt.

## Navigation hierarchy (projection, not storage)

```
market universe
├── index (S&P 500 · Nasdaq-100 · overlap · off-index deep)
│   └── sector → security → source category → item/event type → year/month → item
└── global sources
    └── central bank · treasury · economic agency · regulator
        └── item/event type → year/month → item
```

A user may enter at any facet; the hierarchy is a default drill path, never a
required traversal. Global sources (macro/policy authorities) are a parallel
branch so authoritative non-issuer evidence never hides under a ticker.

## Density (spec Step 3)

- **Aggregate count nodes** and **time buckets** for dense branches — never one
  DOM node per hidden item. A source-category node reads `SEC · 14,203`; a year
  bucket reads `2024 · 3,110`.
- **Per-security badges** derived from aggregate reads, not hidden DOM: e.g.
  `SEC 14 · news 86 · market 252 · official 3`, newest published date, and
  stale / error / indexing-backlog indicators.
- **Status is always text/icon/shape in addition to color** (2.2.7.3 rule):
  fresh `●`, pending `◌`, stale `⌛`, error `✕`, indexing `⟳`, backlog `⏳`.

## Data sources — project first, one new bounded endpoint

Per the phase bound "prefer projecting from existing endpoints; 2.3.5.3 does the
deeper API/index work," 2.3.5.2 adds exactly **one** bounded, read-only,
loopback-only endpoint and otherwise reuses the Phase 2.2 corpus API.

- **`GET /graph/api/corpus/aggregates`** (new) — wraps the existing authoritative
  `Store.get_corpus_accounting(group_by, …)`, which already returns bounded
  `COUNT`/`SUM` rows straight from SQLite metadata (never Chroma bodies) for
  `source_category`, `source`, `item_type`, `security`, `year`, `month`,
  `indexing_state`. The endpoint validates/normalizes combinable facet filters,
  attaches the canonical authority tier for `source_category` groupings (from
  `evidence_taxonomy`), and returns revision-keyed, cursor-paged buckets. This is
  the honest "counts before expansion" engine for the facet rail, the
  aggregate-count nodes, and the per-security badges.
- **Reused**: `/corpus/overview` (sources/tickers/freshness/scheduler),
  `/corpus/search` (label/metadata search, cursor-paged), `/corpus/nodes/{id}` and
  `/neighbors` (bounded expansion), `/corpus/filings/{accession}/sections`,
  `/corpus/refresh-status` (freshness/indexing rollup).

### Explicit 2.3.5.3 boundary

`index` and `sector` are rendered as first-class, combinable facet controls that
participate fully in URL state and the drill hierarchy. Their **authoritative
universe-wide counts** require joining the security registry to corpus items —
that is the "deeper API/index work" the phase plan assigns to 2.3.5.3. In 2.3.5.2
these two facets filter and label from the data already on loaded rows and the
registry option lists; every other listed facet has authoritative counts now.
This boundary is deliberate and documented, not an omission.

## URL state, deep links, presets (spec Steps 1 & 4)

All corpus filter state lives in the **URL hash** — refresh/back/forward/deep-link
work; **no cookies, no analytics, no localStorage**. The hash is a flat
`key=value&…` encoding under a `corpus` marker, e.g.
`#mode=corpus&source_category=sec&item_type=filing&year=2024`. Combined facets
survive a reload by construction (the controller is a pure function of the hash).

Deterministic presets are **URL shortcuts over normal facets**, not separate API
behavior:

| preset | expands to |
|---|---|
| `index-coverage` | landing aggregates grouped by index/coverage tier |
| `financing-events` | `item_type=filing,filing_exhibit` + SEC financing event types |
| `latest-news` | `source_category=company_news` sorted newest |
| `macro-policy` | `source_category=central_bank,treasury,economic_agency` |
| `stale-sources` | `freshness=stale,error` |
| `indexing-backlog` | `indexing_state=pending,failed` |
| `ticker-research` | one-ticker view: `ticker=<T>` across all categories |

## Preserved Phase 2.2 behaviors (spec Step 5, non-negotiable)

Freshness leaves stay folded into worst-status rings + per-status counts on the
canvas while remaining fully present in the list/table/inspector; hub labels
(sources, tickers, aggregate groups) stay visible at overview zoom while leaf
labels obey the existing zoom LOD; provenance beam, reduced-motion behavior, table
fallback, strict same-origin CSP, and loopback-only model are untouched. No
frontend framework, no build chain — vanilla JS/CSS as today.

## Responsive (spec Step 5)

Above 1280px: three columns (facet rail | inventory | canvas) with the shared
inspector. At ≤1280px the facet rail and inspector become independently
accessible drawers/bottom sheets; the inventory/results pane stays usable with the
canvas collapsed, because it is plain DOM. Verified at a 1280px laptop.

## Quality floor (inherited from 2.2.7.3)

Keyboard-reachable everything (roving focus in the inventory mirrors canvas
selection), visible `:focus-visible` rings in `--flow`, WCAG 2.1 AA contrast,
24px min targets, `aria-live` for count/status changes, empty/zero/error/stale/
indexing states that direct action rather than apologize.

## Copy

Sentence case, plain verbs, no mascot voice — matching 2.2.7.3. Facet groups are
nouns the reader controls ("Source category", "Freshness", "Authority"). Empty
states direct action: "No items match these facets. Clear a filter or pick a
preset." Counts read as data (mono): `14,203 items`. Presets are verbs of intent
("Financing events", "Latest news", "Stale sources").
</content>
</invoke>

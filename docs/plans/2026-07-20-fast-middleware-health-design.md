# Fast Middleware Health Design

## Problem

`python scripts/chat.py` starts middleware and polls `GET /health`. Two detailed
health operations scale poorly: `ChromaCollection.count()` can hydrate the large
local HNSW segment, and the full scheduler status report currently takes about
10 seconds against the populated database. The chat launcher repeatedly invokes
this detailed path and can terminate an otherwise valid middleware process.

## Decision

Keep the existing `/health` response contract, including
`storage.chroma_doc_count`, without calling Chroma's public count operation.
The persistent FTS index already maintains a document row count and a corpus
revision transactionally alongside Chroma writes. Health will use that count
only when the SQLite indexed revision matches the revision published in Chroma
and no lexical rebuild is in progress.

If the revisions do not match, health will return `null` for the document count.
It will not load the vector index merely to produce diagnostic metadata.
SQLite and Chroma responsiveness remain independent health signals.

Add an optional `details=false` mode to `/health`. It preserves storage, model,
and capability checks while skipping the scheduler status report and freshness
expansion. The chat client's automatic startup and capability probes use this
mode. The interactive `/health` command and existing callers continue to get the
full response by default.

## Alternatives Considered

1. Remove the document count from `/health`. This is the smallest change but
   unnecessarily changes the response consumed by the chat client and setup
   validator.
2. Read Chroma's internal SQLite tables directly. This is fast and exact but
   couples application code to an undocumented Chroma schema.
3. Increase the launcher timeout. This preserves the expensive behavior and
   only delays failure on larger corpora.

## Testing

Add regression coverage proving that heartbeat returns the revision-consistent
metadata count without calling `ChromaStore.count()`, and that a revision
mismatch returns an unavailable count without falling back to the expensive
operation. Also prove lightweight health skips scheduler expansion and every
automatic chat startup probe requests lightweight health. Run scoped
verification gates, a live one-command startup check, then the full gate.

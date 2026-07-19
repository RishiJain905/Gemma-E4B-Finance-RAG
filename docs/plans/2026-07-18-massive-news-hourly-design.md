# Massive News Hourly Design

## Goal

Replace unreliable hourly GDELT fetching with the already-entitled Massive news endpoint while preserving daily Massive market/corporate-action ingestion and showing each capability as a separate scheduler row.

## Source registry and scheduling

Add `massive_news` as an enabled hourly source with its own `unified:massive_news` freshness key, one-hour TTL, timestamp cursor, and two-hour overlap. Keep `massive` daily-only. Mark `gdelt` configured-disabled rather than removing it, so its 33 existing indexed documents remain queryable and its disabled state is explicit.

`massive_news` and `massive` use separate scheduler/freshness identities but share persisted provider quota accounting under the `massive` quota bucket. Hourly news is capped at one market-wide page of at most 1,000 articles and uses the existing 15-second Massive pacing. The existing top-of-hour hourly schedule is naturally separated from the daily run that begins at :15; shared accounting provides the additional quota guard.

## News ingestion

Extend the existing `MassiveIngestor.ingest_news` path rather than creating a second adapter. On first use it reads a bounded initial window. Later runs start from the durable `massive_news/US` timestamp cursor minus two hours, request results oldest-first, rely on canonical narrative identity for deduplication, and advance the cursor only after every selected record stores successfully. A storage or indexing failure leaves the cursor unchanged and reports partial/error so the next hourly run retries the overlap.

## Watcher behavior

`scripts/watch_scheduler.py` remains database-read-only and additionally reads only the `enabled` flags from `configs/sources.yaml`. Configured-disabled sources render as `state=disabled` instead of stale. After its first live run, `massive_news` appears as its own freshness row, independently from daily `massive`.

## Verification and rollout

Use offline regression tests for registry selection, shared quota accounting, cursor advancement/failure behavior, hourly routing, and watcher rendering. Require the full `scripts\verify.ps1` gate. Then run the live hourly scheduler, confirm `massive_news` is successful/fresh and GDELT is disabled, inspect the watcher output, confirm the Federal backfill is progressing or complete, and push local `Rishi-Ghost` to `origin/Rishi-Ghost`.

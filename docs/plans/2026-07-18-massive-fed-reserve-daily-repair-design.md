# Massive and Federal Reserve Daily Repair Design

## Goal

Make the daily scheduler complete Massive and Federal Reserve successfully without weakening ambiguity checks or freshness semantics.

## Massive

The registry currently contains two active `CBOE` rows. Massive receives a valid `CBOE` bar, but ordinary resolution correctly rejects the ambiguous ticker. Add an explicit `prefer_cik` option to the existing resolver, defaulting to `False`, and use it only for Massive records. When direct ticker candidates are ambiguous, the option may select a candidate only when exactly one has a CIK. All existing callers retain their current behavior.

## Federal Reserve

The live RSS response begins with a UTF-8 BOM, while `requests.Response.text` can expose the BOM as mojibake because the response omits a charset. Decode response bytes as `utf-8-sig` in the official text helper, with the current text value as a fallback. Point each catalog entry at its actual Federal Reserve feed and let the Federal Reserve parser match entry selectors against the item category and title. Accept `dc:date` in addition to RSS `pubDate` for the statistical RDF feed.

## Historical Federal Reserve releases

Keep recurring daily work bounded to the newest 50 statistical releases. Reuse the existing explicit `bootstrap --source federal_reserve` operation for the older archive: it starts immediately after those 50 records, persists batches of 50, checkpoints the last committed official `rdf:about` identity, and safely resumes after interruption. Once the archive is complete, later Federal bootstrap calls are no-ops. A failed record stops the batch without advancing its cursor.

## Safety and verification

No recurring scheduler cadence, freshness rule, or GDELT behavior is changed. Regression tests must prove ordinary ambiguous resolution remains unresolved, Massive opts into the unique-CIK preference, live-shaped Federal Reserve payloads parse through the HTTP path, and the explicit Federal bootstrap resumes without skipping failed work. Run the scoped tests, the full repository verification gate, then a forced daily scheduler run. Run the hourly scheduler separately before starting the one-time Federal bootstrap.

The Federal Data Download feed contains more than one thousand historical announcements. Bound its newest-first statistical entry to 50 items so the daily run includes every current-year item in the live feed without turning into a historical embedding backfill.

The user subsequently expanded scope to include the pre-existing live Chroma integration failure. Its root cause is test lifecycle wiring: `TestClient` initialized a default store/retriever pair before the fixture replaced only the store, so `/search` queried a different collection. The repair is test-only: inject the temporary store, a normal `MiddlewareConfig`, and a reset retriever without running application lifespan. Production middleware and Chroma behavior remain unchanged.

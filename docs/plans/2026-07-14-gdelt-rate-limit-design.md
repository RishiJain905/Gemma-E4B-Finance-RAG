# GDELT Rate-Limit Fail-Fast Design

## Problem

After GDELT returns HTTP 429 for all three attempts, the HTTP layer returns
`None`. The search layer treats that as an empty successful response, tries a
less-specific fallback query, and then continues through aliases and remaining
tickers. The scheduler therefore issues more requests to a clearly throttled
service and can incorrectly mark GDELT fresh with zero stored articles.

## Design

Introduce a small `GDELTRateLimitError` raised only when HTTP 429 retries are
exhausted. Do not catch that exception in the per-query soft-failure handler.
It must escape the GDELT ingestor so all fallback queries, ticker aliases,
remaining tickers, and GKG enrichment stop for the current GDELT run.

The existing scheduler source boundary will catch the exception, mark GDELT
stale, add its best-effort dead-letter entry, and continue with subsequent data
sources. Other HTTP and parsing failures retain their current fail-soft behavior.

If the final 429 includes a `Retry-After` header, include it in the exception
message for an actionable scheduler error. This change does not add persistent
cooldown state or alter refresh command semantics.

## Verification

Add offline regression tests proving that exhausted 429s raise the typed error,
that no bare-query or alias request follows the first exhausted search, and that
the scheduler continues to the next source while recording GDELT as an error.
Run the scoped GDELT and scheduler tests, then the repository verification gate.

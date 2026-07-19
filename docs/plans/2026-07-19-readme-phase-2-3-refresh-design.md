# Phase 2.3 README Refresh Design

## Goal

Make the public README describe the repository as it operates after Phase 2.3,
including the current source registry, scheduler cadences, interactive chat
workflow, operational monitor, and licensing.

## Scope

- Replace the legacy six-source framing with the current registry-driven
  ingestion model without turning the README into a duplicate of the full
  configuration reference.
- Correct scheduler guidance: daily registry refreshes, hourly Massive News,
  weekly SEC/transcript/CFTC work, disabled GDELT ingestion, and truthful
  status reporting.
- Explain that Phase 2.3 capabilities are implemented and enabled in the
  committed configuration, while sources that require missing credentials
  disable independently.
- Expand the quick start with the optional Phase 2.3 provider keys, one-time
  bootstrap guidance, ongoing refresh commands, and the fact that `--force`
  does not bypass quotas or circuit cooldowns.
- Promote the most useful `scripts/chat.py` behavior from `scripts/CHAT.md`:
  streaming, multiline questions, bounded client-owned history, grounding and
  verbosity controls, evaluation commands, graph deep links, and scheduler
  refresh commands.
- Document the read-only scheduler watcher and its separate `massive` and
  `massive_news` rows, with GDELT shown as disabled.
- Refresh the project tree and replace the clone placeholder with the public
  repository URL.
- Add the standard MIT license text at the repository root and link it from the
  README.

## Documentation boundaries

The README will provide the supported happy path and operational overview.
Detailed flag tables, schemas, command metadata, and migration internals remain
in `docs/CONFIGURATION.md`, `docs/ARCHITECTURE.md`, `docs/API.md`, and
`scripts/CHAT.md`.

The GDELT sentiment endpoint remains documented because previously ingested
GDELT data is preserved and queryable; only scheduled GDELT ingestion is
disabled. Massive market data and Massive News remain separate logical sources
because they use different cadences and freshness rows while sharing the same
provider quota.

## Verification

- Check all relative Markdown links resolve.
- Check the README no longer contains the known stale scheduler/default-off
  claims or clone placeholder.
- Run the repository's full `scripts\verify.ps1` gate before committing the
  implementation.

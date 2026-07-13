# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Picking the right models for workflows and subagents

Rankings, higher = better. Cost reflects what I actually pay (OpenAI has really generous limits), not list price. Intelligence is how hard a problem you can handle the model unsupervised. Taste covers UI/UX, code quality, API design, and copy.

### Claude models

| model | cost | intelligence | taste |
|-------|------|--------------|-------|
| sonnet-5 | 5 | 5 | 7 |
| opus-4.8 | 4 | 7 | 8 |
| fable-5 | 2 | 9 | 9 |

### Claude subagent presets (model + effort)

The Agent tool has no per-spawn effort parameter — effort is pinned in the agent definition. Five presets live in `.claude/agents/` (spawn via `subagent_type: "<preset>"`); Fable is spawned plainly with `model: 'fable'` and inherits the session's effort:

| preset | model | effort | use for |
|--------|-------|--------|---------|
| `sonnet-low` | sonnet-5 | low | Trivial mechanical side tasks with zero design decisions: file sweeps, renames, doc/config tweaks, simple test fixes. |
| `sonnet-xhigh` | sonnet-5 | xhigh | Default for **compact** well-specified work: single-file/single-feature implementation, test authoring, user-facing UI/copy at taste 7. Also the budget fallback when usage is tight. |
| `sonnet-max` | sonnet-5 | max | Niche only: debugging with a known repro, or deep-but-mechanical work confined to one file/domain. Not a rung on the escalation ladder — at max effort on multi-file volume it burns more tokens than opus-xhigh finishing in one pass. |
| `opus-xhigh` | opus-4.8 | xhigh | **Default implementer.** Multi-file features, cross-cutting integration, API design, plan/implementation reviews, subtle debugging. First choice once a task spans files/subsystems, regardless of how clear the spec is. |
| `opus-max` | opus-4.8 | max | Heaviest delegation: architectural refactors, root-cause hunts that survived an opus-xhigh attempt, high-risk changes to shared pipelines. Last stop before Fable does it personally. |
| *(fable, no preset)* | fable-5 | inherits session | Open-ended design and judgment calls the orchestrator would otherwise keep; rare — usually the orchestrator IS Fable. |

Routing by complexity — ask three questions: *does the task span more than one file/subsystem?*, *what breaks if it's slightly wrong?*, and *does the task's true complexity sit within this preset's ceiling at its pinned effort?*
- Route by ceiling, not ladder position: estimate the task's true complexity first and assign the preset whose ceiling comfortably covers it. If a task genuinely demands `opus-max` (architectural refactor, gnarly root-cause, high-risk shared-pipeline change) or Fable (open-ended design, novel judgment), start there on the first pass — never assign a first pass to a preset you expect to fail just because the ladder starts lower.
- Spec explicit, scope compact, failure caught by the gate → `sonnet-low`/`sonnet-xhigh`.
- Scope grows to multi-file — even with a crystal-clear spec → `opus-xhigh` directly. Don't ladder through sonnet-max; fewer smart tokens beat more cheap ones (max-effort Sonnet thinking plus extra gate iterations usually out-burns Opus finishing in one pass).
- Ambiguity, judgment, taste ≥ 8, or cross-subsystem blast radius → `opus-xhigh`; add high-risk on top → `opus-max`.
- Escalation is one-way and immediate: the same miss or failure twice on a preset → next tier (sonnet-xhigh → opus-xhigh → opus-max → Fable inline), never a retry at the same tier. sonnet-max sits off-ladder as a special-purpose tool, not an escalation step. Escalation corrects misjudged routing; it is not a substitute for honest first-pass ceiling matching.
- These presets do not replace Codex routing: bulk/mechanical clear-spec diffs still go to a GPT-5.6 Codex model first; the presets cover work needing Claude judgment/taste, reviews, and the Codex-down fallback.
- Presets fix only the floor (model, effort, verify-gate discipline); all task-specific steering — scope, approach, constraints, what a prior attempt got wrong, report format — goes in the spawn `prompt`, which layers on top of the preset's system prompt. Steer there; don't create new agent files for one-off specializations.

### Codex models

| model | cost | intelligence | taste | effort |
|-------|------|--------------|-------|--------|
| gpt-5.6-luna | 10 | 7 | 7 | max |
| gpt-5.6-terra | 8 | 8 | 7 | max |
| gpt-5.6-sol | 8 | 8 | 8 | high |
| gpt-5.6-sol | 6 | 9 | 8 | max |

How to apply:
- These are defaults, not limits. You have standing permission to override them: if a cheaper model's output doesn't meet the bar, rerun or redo the work with a smarter model without asking. Judge the output, not the price tag. Escalating costs less than shipping mediocre work.
- Cost is a tie-breaker only; when axes conflict for anything that ships, intelligence > taste > cost.
- Bulk/mechanical work (clear-spec implementation, data analysis, migrations): use a GPT-5.6 Codex model; the specific model and effort are defined in the delegation section.
- Anything user-facing (UI, copy, API design) needs taste ≥ 7.
- When the factor comes down to taste and design always utilize Claude models first with codex models only being used a fallback in that specific scenario. 
- Reviews of plans/implementations: fable-5 or opus-4.8, optionally a GPT-5.6 Codex model as an extra independent perspective.
- Never use Haiku.
- Mechanics: GPT-5.6 Codex models are accessed from Claude Code through the Codex plugin. For implementation, debugging, investigation, data analysis, or other delegated work, use `/codex:rescue --model <model> --effort <effort> <task>`. Add `--background` for longer-running work, then use `/codex:status` and `/codex:result` to monitor it and retrieve the result. For reviews, use `/codex:review` or `/codex:adversarial-review`.
- Claude models (sonnet-5, opus-4.8, fable-5) run via the Agent tool — prefer the effort presets above (`subagent_type: "sonnet-xhigh"` etc.) over a bare `model:` parameter, since a bare spawn cannot set effort.

Using GPT-5.6 inside workflows and subagents:
- The Agent/Workflow `model` parameter only accepts Claude models. To delegate work to a GPT-5.6 Codex model, use the Codex plugin's bundled `codex:codex-rescue` subagent rather than creating a custom Claude wrapper. If a wrapper is absolutely required, spawn a Claude wrapper with `model: 'sonnet', effort: 'low'` and instruct it to write a self-contained Codex prompt. Prefer the plugin whenever possible.
- For implementation or investigation, choose the model and effort up front from the task-fit guidance below, then invoke `/codex:rescue --model <model> --effort <effort> --background <self-contained task>`.
- Use `/codex:status` to check progress and `/codex:result` to retrieve the completed response.
- For an independent code review, run `/codex:review --background`.
- For a review focused on challenging design decisions, assumptions, or specific risk areas, run `/codex:adversarial-review --background <focus>`.
- Claude may also delegate naturally by being instructed to ask Codex to complete a task.

Understanding which Codex model + effort to use:
- Select the model before spawning based on the task's expected complexity and required capability ceiling. Do not start with Luna and move upward only after failure.
- **Spec clarity is not the discriminator.** Nearly every task here arrives as a detailed spec doc, so "is the spec clear?" separates nothing — a detailed spec tells you *what* to build, it does not make a hard problem easy. Route on the intrinsic difficulty of the work and on how a mistake would surface, not on how well it is written up. A clear spec is what makes Luna *possible*; it is not what makes Sol unnecessary.
- **Luna/max:** Luna at max effort is a genuinely capable tier, not a budget tier — it benchmarks above Sol/low and roughly level with Sol/medium, at a fraction of the cost. It is the right call for the large majority of delegated work: bulk and mechanical changes, migrations, data analysis, routine investigation, test authoring, and implementation where the spec fixes both the *what* and most of the *how*, so the job is mainly faithful translation into code — **including multi-file work**, provided a mistake fails *loudly* (a test or the verify gate catches it). Send the task here unless you can name a trigger below.
- **Sol/high — when a named trigger fires:** For work that stays hard *after* the spec is perfectly clear. Choose Sol/high when at least one holds, and state which one at dispatch: (a) **correctness rests on invariants a passing test won't prove** — concurrency, cache coherency and invalidation, ordering or streaming/tool-call interleaving, retry and idempotency, auth and security boundaries; (b) **the spec settles the *what* but leaves a real design decision open** — schema, API surface, algorithm choice — where the wrong pick becomes lock-in; (c) **it changes a shared pipeline where errors degrade quality silently instead of failing** — retrieval fusion and ranking, prompt assembly, embeddings, scheduler isolation; (d) debugging with no clear repro or working hypothesis; (e) a Luna/max attempt at the same task already missed the bar.
- Do **not** escalate to Sol/high merely because a task feels important, touches several files, is production code, or leaves you unsure — none of those are triggers. Multi-file is a *volume* signal, not a *difficulty* signal, and Luna/max handles volume cheaply. The question is never "is this task big?" but "would a subtly wrong answer here pass the gate and ship?" Guessing low costs one re-dispatch on the rare miss; habitually guessing high costs extra on every task.
- **Exceptional — Terra/max:** Reserve for genuinely hard, long-horizon, tool-heavy coding or analysis where sustained technical execution is the main constraint and design taste is not. Luna and Sol are generally more cost-efficient, so choose Terra/max only when the task specifically benefits from its higher coding-agent ceiling.
- **Exceptional — Sol/max:** Reserve for genuinely hard, high-stakes, or deeply ambiguous work requiring the highest broad reasoning and judgment ceiling, such as novel debugging, consequential architecture, or difficult cross-system changes. Do not use it when Sol/high can confidently cover the task.
- Luna/max is the normal choice; Sol/high is the justified exception. Terra/max and Sol/max are intentional choices for tasks assessed as exceptionally difficult before delegation, not fallback steps in an escalation ladder.

Fallback handling (Codex usage limits / zero credits):
- **Detection is the wrapper's job — every Codex dispatch must be verified, not assumed.** Immediately after dispatching, the wrapper checks the job result for the limit signatures: (a) an explicit "You've hit your usage limit… try again at HH:MM" error; (b) the instant-fail pattern — `task_complete` within seconds of submission with `last_agent_message: null`, usually alongside a `token_count`/`rate_limits` event showing `has_credits: false` or `balance: "0"` (visible in the newest `~/.codex/sessions/**/rollout-*.jsonl`). A dispatch that produced no repo changes and no agent message did NOT run — treat it as a limit failure, never as success.
- **The wrapper never performs the fallback itself.** On detecting a limit failure it must NOT spawn subagents, NOT retry Codex, and NOT wait for the reset. It reports straight back to the orchestrating (main) session with: the failure signature it matched, the quoted reset time if present, and the untouched task spec. Then it stops.
- **The orchestrator owns the reroute.** On receiving that report, the main session spawns the Claude subagent itself using the preset table above (multi-file/cross-cutting → `opus-xhigh`; compact well-specified → `sonnet-xhigh`; architectural/high-risk → `opus-max`), passing the same task spec. This keeps model routing, budget awareness, and gate discipline in one place.
- **While credits are known-exhausted, skip Codex entirely** for subsequent tasks and route directly to Claude presets until a later dispatch (or the quoted reset time passing) proves Codex is back.
- The orchestrator should still babysit every dispatch with a working-tree watcher: zero writes within ~8 minutes of a dispatch means inspect the newest Codex rollout file for the instant-fail signature rather than waiting longer.

When Using Plan mode:
- Inherited / current model the user is using will be the model that is used to create the plan for the task at hand. This will likely be Fable 5 or Opus 4.8
- Once Fable 5 or Opus 4.8 has thought of a plan, spawn a subagent who will use `model: 'sonnet 5'` and the thinking effort will be based on complexity of task. This sonnet 5 model will create a HTML file using the frontend design skill. This HTML file should outline the entire plan and be presented to me (user).
- Instead of the typical MD file that is shown as the plan outline before the user (me) clicks proceed to implement, this HTML file will replace it. Make sure the Artifact HTML created is opened for the user when you are ready to show the plan and HTML file. 
- The objective is to visualize the plan prior to implementation so that its easier to optimize the plan before any code is written. 
- All subagents launched in Plan Mode will use `'model: 'sonnet 5'`. Effort level can be your choice based on complexity of task given to the model. This includes `Explore` Agents. The only Exception is the `plan` Agent who can use the `Model: 'Opus 4.8'` as the plan-agent default when specs are detailed and exploration ran first; `Model: 'Fable 5'` for open-ended or high ambigutiy design.

Built-in agents:
-Built-in agent types (`Explore`, `general-purpose`) are always spawned with an explicit `model:` — default `model: "sonnet"` — and never on Fable 5. Their definitions otherwise resolve their own model (Explore was observed defaulting to Opus 4.8), and an inheriting built-in in a Fable session would burn Fable tokens on survey work. Built-ins are for cheap search/survey only; anything needing more intelligence routes through the presets above or stays inline with the orchestrator.

### Loops: which primitive to trigger

A loop = repeated work cycles until a stop condition. The deterministic stop condition for all code loops in this repo is the verify gate — `scripts\verify.ps1` / `scripts/verify.sh`, final line `VERIFY: PASS|FAIL` — governed by the project skill `verify-rag-change`. Use that skill before claiming any code change done, in or out of a loop.

Route by task size; never a bigger loop than the task needs:

- **Short** (one file / obvious fix, ~≤3 turns): plain turn-based work. No /goal, no subagents, no background jobs. Run the gate once before reporting done; read only the verdict + failures.
- **Medium** (multi-file feature/bugfix with a checkable done-state): best run as `/goal <task>. Done when scripts\verify.ps1 prints VERIFY: PASS. Stop after 4 tries.` Iterate on the scoped gate (`-TestPath tests\test_x.py`) while fixing; the full gate is the exit check. Bulk/mechanical diffs → gpt-5.6 via `/codex:rescue` (table above); close with `/codex:review --background` for a fresh-context review. If a medium task arrives as a plain prompt, still enforce the gate, and put the ready-to-paste /goal one-liner in the final summary so the next run can be hands-off.
- **Long** (multi-phase, hours, or waiting on external systems): plan mode first (rules above), then each phase runs as its own medium /goal loop with its own PASS exit — never one giant loop. Watching external state (CI, PR reviews) → `/loop` with the interval matched to how fast the target changes (~4m for active CI; ≥20m for idle watching — avoid ~5m, it's the worst cache breakpoint), or interval-free `/loop` so Claude self-paces. Recurring repo upkeep (ingestion/scheduler health) → `/schedule` routine, not a live session.
- **Every size**: deterministic steps go in scripts, not reasoning; the same failure surviving two fix attempts means stop patching — change approach or escalate the model; pilot one slice before any fan-out; audit burn with `/usage` and `/goal` (no args).

## What this is

Hybrid finance RAG system: six data sources (SEC EDGAR, Yahoo Finance, FRED, GDELT, earnings transcripts, IR pages) are ingested into dual stores — SQLite (`data/finance.db`, structured facts/filings/freshness) and ChromaDB (`data/chroma`, document embeddings) — and served through a FastAPI middleware (`:8000`) that does intent parsing → hybrid retrieval → prompt augmentation → a locally-served fine-tuned Gemma model ("TraceAlchemy") on `llama-server` (`:8087`, chat + embeddings from the same server). An optional, **local read-only** live retrieval-graph observer (Phase 2.2.7, disabled by default, loopback-only) visualizes each query's pipeline and the corpus at `/graph`.

## Commands

```bash
pytest tests/ -v -m "not live"        # standard run (offline, deterministic)
pytest tests/ -v                      # full suite
pytest tests/test_retriever.py -v     # single file
pytest tests/test_retriever.py::TestClass::test_name -v   # single test
pytest tests/ --live                  # opt-in live tests (need llama-server on :8087 + network)
pytest tests/ -v --cov=src --cov-report=term-missing      # coverage
python scripts/validate_setup.py      # environment sanity check
python -m src.scheduler daily --force # run ingestion (modes: daily/hourly/weekly/all/status)
python scripts/chat.py                # interactive client (auto-starts middleware)
uvicorn src.middleware.app:app --port 8000                # middleware directly
ruff check .                          # lint
```

Windows stack scripts: `scripts/serve_model.ps1 start` (llama-server), `scripts/start_stack.ps1` / `stop_stack.ps1` (middleware). `.sh` equivalents exist for Unix.

`--strict-markers` is enabled; registered markers are `live`, `slow`, `integration`, `regression`, `e2e`, `network` (see `pytest.ini`). Network/integration tests skip cleanly when `:8087` or upstream services are unavailable — keep new tests offline-safe by mocking external services.

## Architecture (big picture)

`docs/ARCHITECTURE.md` is the authoritative reference (components, schema, query flow, failure modes). Key structural points that span multiple files:

- **Storage facade** — `src/storage/store.py` (`Store`) unifies `SQLiteStore` and `ChromaStore`; middleware and ingestors go through it. Freshness/TTL state lives in the `cache_meta` table; scheduler cadence uses the synthetic ticker `"SCHEDULER"`.
- **Query pipeline** — `src/middleware/app.py` orchestrates `IntentParser` → `Retriever` (strategy chosen from intent: facts_only / hybrid / documents_only / comparison / macro / broad) → `PromptAugmenter` → model call. If the model is down, `/query` returns a **degraded** answer from raw retrieval instead of failing.
- **Retrieval (Phase 2.1.2)** — optional hybrid pipeline: ChromaDB vector search + BM25 (`src/middleware/lexical_index.py`) → RRF fusion → optional cross-encoder/LLM re-rank (`src/middleware/reranker.py`). Each stage is toggleable in `configs/middleware.yaml` or via env (`ENABLE_LEXICAL`, `ENABLE_RERANKER`, `RERANKER_BACKEND`). Every stage fails soft — a query never errors because a retrieval stage failed.
- **Ingestion** — `UnifiedScheduler` (`src/scheduler/__init__.py`) drives all sources with TTL gating, staggered execution, and per-source failure isolation (log + mark stale + dead-letter, continue). Resilience primitives (retry/backoff, `CircuitBreaker`, `DeadLetterQueue`) live in `src/utils/resilience.py`.
- **Embeddings** — `TraceAlchemyEmbeddingFunction` in `chroma_store.py` POSTs to llama-server's `/v1/embeddings` (requires `--embeddings --pooling mean` on the server). Documents >1000 chars are chunked with 150-char overlap and stored as `"{id}#{i}"` with parent metadata.
- **Live retrieval graph (Phase 2.2.7, optional)** — a read-only *observability* side channel (not GraphRAG; never changes retrieval or the answer). One extra observer subscribes the existing `QueryEvent` emitter (`src/middleware/stream_events.py`) and `graph_observer.py` projects events into a bounded, redacted `TraceHub`; `corpus_graph.py` projects the Store inventory for the explorer; `graph_api.py` + the static UI (`src/middleware/static/graph/`, Cytoscape) serve `/graph`. Disabled by default (`enable_graph_observer`); a single HTTP middleware in `app.py` gates every `/graph*` route to loopback and stamps a strict same-origin CSP. When on, `/query` responses add only an optional `graph_trace_id`.

## Configuration

- Runtime config is YAML under `configs/` (documented in `docs/CONFIGURATION.md`); secrets go in `.env` (`FRED_API_KEY`, `SEC_EDGAR_USER_AGENT`).
- Machine-specific model paths go in `configs/model.local.yaml` (gitignored; copy from `configs/model.example.yaml`) or env vars `MAIN_MODEL_PATH` / `LLAMA_BUILD_DIR` / `DRAFT_MODEL_PATH`. Path resolution is in `src/utils/model_config.py` / `scripts/resolve_model_paths.py`.
- The live graph observer is off by default; enable it locally with `ENABLE_GRAPH_OBSERVER=1` (see `graph_*`/`corpus_*` keys in `docs/CONFIGURATION.md`). It is loopback-only with no bypass flag — never expose `/graph*` remotely or behind a proxy.
- Never commit `.env`, `configs/model.local.yaml`, or anything under `data/`.

## Conventions

- **Branching**: the active development branch is `Rishi-Ghost` — branch from and PR into it, not `main`.
- Module docstrings start with the file path and a one-line purpose; annotate public signatures.
- Lazy local imports inside functions in hot paths (e.g. the middleware pipeline) are intentional — don't hoist them.
- `logger = logging.getLogger(__name__)` in library code; `print` only in `scripts/`.
- Per-source/per-item failures are swallowed and recorded, not propagated — preserve this isolation pattern in scheduler/refresh paths.
- Graph observer output is redacted **by construction** (allowlisted node/edge metadata, question digest + bounded preview, bounded excerpts, secret/local-path scrubbing, http(s)-only links) and must never block a query (non-blocking fan-out, fail-soft). Graph tests are offline: `tests/test_graph_observer.py`, `tests/test_graph_api.py`, `tests/test_graph_ui_contract.py`, chat coverage in `tests/test_chat_client.py`, golden fixture `tests/fixtures/graph/query_trace_v1.json`.

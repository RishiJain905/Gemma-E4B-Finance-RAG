# How to Loop — Gemma-E4B-Finance-RAG

A loop is Claude repeating cycles of work (gather context → act → check → repeat) until a stop condition is met. This guide covers the loop system built for this repo: what was installed, how to start each kind of loop from a prompt, and how to keep token burn low on a 5x Max plan. The orchestrator-facing summary of all of this lives in `CLAUDE.md` under "Loops: which primitive to trigger" — Claude reads that every session; this file is the human manual.

---

## The 30-second version

| Loop type | You hand off | Trigger | Stops when | Command |
|---|---|---|---|---|
| **Turn-based** | the check | your prompt | Claude judges it's done | (none — every prompt) |
| **Goal-based** | the stop condition | your prompt | condition true OR turn cap hit | `/goal` |
| **Time-based** | the trigger | a clock | you cancel it / work completes | `/loop`, `/schedule` |
| **Proactive** | the prompt itself | event/schedule, no human | each task exits on its goal | `/schedule` + `/goal` + skills |

The single most useful command in this repo:

```
/goal <describe the change>. Done when scripts\verify.ps1 prints VERIFY: PASS. Stop after 4 tries.
```

That's a medium-task loop with a deterministic exit. Everything else is a variation.

---

## What was set up (and why)

**1. The verify gate — `scripts/verify.ps1` and `scripts/verify.sh`** (new)
Runs `ruff check .` + the offline pytest suite (`-m "not live"` — no network, no llama-server needed). The **final line is machine-readable**: `VERIFY: PASS` (exit 0) or `VERIFY: FAIL (ruff, pytest)` (exit 1).

Why it matters: `/goal` works by having an evaluator model check your stop condition after each turn. A fuzzy condition ("the feature works well") makes the evaluator guess and the loop end too early or run too long. "The script prints `VERIFY: PASS`" is a string match — the loop can't lie to itself, and Claude doesn't spend tokens reasoning about whether it's done. This is the article's "use scripts for deterministic work" advice applied to the exit check itself.

Scoped mode for fast inner-loop iteration (runs one test file instead of the whole suite):

```
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 -TestPath tests\test_retriever.py
bash scripts/verify.sh tests/test_retriever.py
```

Note: the gate's very first run caught real drift. `.venv` was missing `ruff`, `pytest-asyncio`, and `rapidfuzz` — and the missing `rapidfuzz` wasn't just failing a test, it was silently disabling fuzzy symbol resolution at runtime (the resolver swallows the ImportError by design). All three are installed now, and `ruff`/`pytest-asyncio` were added to `requirements.txt`. That's the gate doing its job on day one: 5 "test failures" that were actually one environment problem.

**2. Project skill — `.claude/skills/verify-rag-change/SKILL.md`** (new)
Auto-discovered by Claude Code because it lives in the repo. It tells any Claude session (yours, a /goal loop, a scheduled routine) when to run the gate, how to read it, and the two-strikes rule: the same failure surviving two fix attempts means stop patching and re-diagnose or escalate the model — the anti-doom-loop circuit breaker.

**3. CLAUDE.md routing** (updated)
The "Loops: which primitive to trigger" subsection tells the orchestrator which loop fits short / medium / long tasks, so it behaves correctly inside a loop and recommends the right primitive when you didn't use one.

**4. Nothing needed downloading for the primitives.** `/goal`, `/loop`, and `/schedule` are built into Claude Code (`/goal` needs ≥ v2.1.139; you're on **2.1.202** ✓). And your superpowers plugin already ships the two loop-critical discipline skills: `verification-before-completion` (evidence before completion claims — the generic version of our gate) and `systematic-debugging` (what the two-strikes rule escalates into). They're installed and auto-apply; no action needed.

---

## Starting each loop

### Short tasks — no loop machinery at all

*One file, obvious fix, done in a few turns.* Just prompt normally:

> Fix the off-by-one in the chunk overlap logic in `src/storage/chroma_store.py`.

CLAUDE.md + the `verify-rag-change` skill make Claude run the gate once before reporting done. **Do not** wrap short tasks in `/goal` — you'd pay for an evaluator check per turn and gain nothing. This is the cheapest loop; keep it cheap by being specific (file, symptom, expected behavior) so Claude doesn't burn turns exploring.

### Medium tasks — `/goal` with the gate as the exit

*Multi-file feature or bugfix where "done" is checkable.* Template:

```
/goal <task>. Done when scripts\verify.ps1 prints VERIFY: PASS. Stop after 4 tries.
```

Real examples for this repo:

```
/goal Add a --json flag to "python -m src.scheduler status" that emits per-source freshness as machine-readable JSON, with tests. Done when scripts\verify.ps1 prints VERIFY: PASS. Stop after 4 tries.

/goal Fix whatever is breaking hybrid retrieval when ENABLE_LEXICAL=1 and the BM25 index is empty. Done when scripts\verify.ps1 -TestPath tests\test_retriever.py prints VERIFY: PASS and then the full scripts\verify.ps1 prints VERIFY: PASS. Stop after 5 tries.

/goal Raise test coverage of src/middleware/lexical_index.py to 90%. Done when "pytest tests/ -m 'not live' --cov=src/middleware/lexical_index.py --cov-report=term" reports >=90% AND scripts\verify.ps1 prints VERIFY: PASS. Stop after 5 tries.
```

Rules of thumb:
- **Always set a turn cap** ("stop after 4–5 tries"). It's your budget ceiling; without it a stuck loop burns tokens until you notice.
- **Chain conditions with AND** when you need both a specific metric and the gate (coverage example above).
- Inside the loop, CLAUDE.md tells Claude to iterate on the **scoped** gate and only run the full gate as the exit check, and to hand bulk/mechanical diffs to gpt-5.5 via `/codex:rescue` (effectively free on your plan) while the session model does routing + judgment.
- **Close out with a fresh-context review**: after the loop exits, run `/codex:review --background`. A reviewer that didn't watch the loop reason is unbiased, and on gpt-5.5 it costs you ~nothing.

### Long tasks — plan mode first, then phased /goal loops

*Multi-phase work, hours of effort, or anything waiting on external systems.* Never run one giant loop — an early wrong turn compounds for hours, and one huge context is expensive. Instead:

1. **Plan mode** (your existing CLAUDE.md workflow: Fable/Opus plans, Sonnet renders the HTML plan artifact). The plan's job for looping: split work into phases where **each phase leaves the gate green**.
2. **Run each phase as its own medium `/goal` loop** with its own `VERIFY: PASS` exit and turn cap. Fresh-ish context per phase, and a failed phase can't silently corrupt the next.
3. **External waits use `/loop`, not a spinning session.** Two flavors:

```
/loop 4m Check PR #7: address new review comments, fix failing CI, push. Stop when the PR is merged or closed.

/loop Watch the background ingestion run (python -m src.scheduler daily --force). When it finishes, summarize per-source results and any dead-letter entries, then stop. Pace your own check-ins.
```

The second form (no interval) is **dynamic /loop** — Claude picks its own wake-up times, which is the smarter default when the wait time is unpredictable.

### Recurring work — `/schedule`

*Same task, new inputs, on a clock. Runs as a routine, not a live session (research preview; cloud-side, so it survives your machine being off).* For this repo the obvious one is ingestion health:

```
/schedule every weekday at 8am: run "python -m src.scheduler status", summarize which sources are stale beyond their TTL and why, and if the dead-letter queue is non-empty diagnose the most recent entry.
```

### Proactive combo — the full composition

When the pieces above are trusted individually, compose them (pilot on one run before letting it recur):

```
/schedule nightly at 2am: run the offline test suite and ruff via scripts\verify.ps1. /goal: don't stop until every new failure is either fixed (gate prints VERIFY: PASS) or written up in docs/reports/ with a minimal repro. Stop after 6 tries per night.
```

---

## Token-efficiency rules (the 5x Max survival guide)

1. **Right-size the primitive.** Short → turn-based. Medium → /goal. Long → phases + /loop for waits. Recurring → /schedule. The routing in CLAUDE.md enforces this; overriding it upward is almost always waste.
2. **Deterministic exit criteria.** Every reasoning token spent deciding "am I done?" is waste — the gate's string match costs zero. Same for any repeated procedure: if a loop does a step the same way twice, ask Claude to script it.
3. **Read verdicts, not logs.** The gate prints failures + one verdict line. The skill instructs Claude to read only those. When you check a loop's progress yourself, do the same.
4. **Turn caps are budget caps.** 4–5 tries for medium loops. If it didn't converge in 5, the problem is diagnosis, not persistence — that's what the two-strikes escalation is for.
5. **Match /loop intervals to reality, and mind the cache.** The prompt cache lives ~5 minutes: a `/loop 4m` keeps context cached between iterations (cheap); `/loop 5m` re-reads everything every time (worst case); anything slower should jump straight to 20–30m. Only poll fast when the watched thing actually changes fast (active CI ≈ 4m). For unpredictable waits, use interval-free `/loop` and let Claude pace itself.
6. **Route models inside loops.** Judgment/stop decisions on the session model; bulk implementation to gpt-5.5 (`/codex:rescue`); reviews to `/codex:review --background`. Your cost table already encodes this — loops just apply it repeatedly, so the savings compound.
7. **Pilot before fan-out.** Any recurring or multi-agent construct: run it once on one slice, check `/usage`, then let it recur.
8. **Audit.** `/goal` with no arguments shows turns + tokens so far; `/usage` breaks down burn by skills/subagents/MCPs; `/workflows` shows per-agent usage if you ever use dynamic workflows.
9. **When a loop produces a bad result, fix the system, not just the result.** Add the missed check to `verify.ps1` or a rule to the skill — every future loop inherits the fix for free. That's the compounding-returns move from the article.

## Monitoring & stopping

- **/goal loop:** exits on PASS or the turn cap; interrupt anytime with Esc. `/goal` (no args) = progress + token report.
- **/loop:** runs on your machine; stops on its stop condition, or interrupt with Esc / tell Claude to stop the loop. It dies if the machine sleeps — that's what `/schedule` is for.
- **/schedule:** manage routines with `/schedule` (list/cancel), or ask Claude to list and delete them.

## New skills — summary for you

| Skill | Where | Status | What it does |
|---|---|---|---|
| `verify-rag-change` | `.claude/skills/verify-rag-change/` (this repo) | **new — created** | Repo-specific gate discipline: run `verify.ps1`/`.sh`, read verdict only, two-strikes escalation, gate = /goal stop condition |
| `verification-before-completion` | superpowers plugin | already installed | Generic "evidence before completion claims" discipline — backs the project skill |
| `systematic-debugging` | superpowers plugin | already installed | Where the two-strikes rule sends Claude instead of blind re-patching |

CLAUDE.md now references `verify-rag-change` explicitly, so every session (and every loop) picks it up. The `/goal`, `/loop`, `/schedule` primitives are built-in — nothing was (or needed to be) downloaded for them.

## References

- [Claude Code docs: /goal](https://code.claude.com/docs/en/goal) (also see the loop, schedule, and dynamic-workflows pages)
- The ClaudeDevs "Getting started with loops" article (saved locally as `loops.txt`)

---

# COPY-PASTE BLOCK — "Picking the right models for workflows and subagents" (full updated section for CLAUDE.md)

> **Note:** `CLAUDE.md` in this repo has already been updated with exactly this content. This copy exists so you can re-paste it after a revert, or port the whole workflow to another project's CLAUDE.md.

```markdown
## Picking the right models for workflows and subagents

Rankings, higher = better. Cost reflects what I actually pay (OpenAI has really generous limits), not list price. Intelligence is how hard a problem you can handle the model unsupervised. Taste covers UI/UX, code quality, API design, and copy.

| model | cost | intelligence | taste |
|-------|------|--------------|-------|
| gpt-5.5 | 9 | 8 | 5 |
| sonnet-5 | 5 | 5 | 7 |
| opus-4.8 | 4 | 7 | 8 |
| fable-5 | 2 | 9 | 9 |

How to apply:
- These are defaults, not limits. You have standing permission to override them: if a cheaper model's output doesn't meet the bar, rerun or redo the work with a smarter model without asking. Judge the output, not the price tag. Escalating costs less than shipping mediocre work.
- Cost is a tie-breaker only; when axes conflict for anything that ships, intelligence > taste > cost.
- Bulk/mechanical work (clear-spec implementation, data analysis, migrations): gpt-5.5 — it's effectively free.
- Anything user-facing (UI, copy, API design) needs taste ≥ 7.
- Reviews of plans/implementations: fable-5 or opus-4.8, optionally gpt-5.5 as an extra independent perspective.
- Never use Haiku.
- Mechanics: gpt-5.5 is accessed from Claude Code through the Codex plugin. For implementation, debugging, investigation, data analysis, or other delegated work, use `/codex:rescue --model gpt-5.5 --effort high <task>`. Add `--background` for longer-running work, then use `/codex:status` and `/codex:result` to monitor it and retrieve the result. For reviews, use `/codex:review` or `/codex:adversarial-review`; these use the model selected by the Codex configuration, so gpt-5.5 should be configured as the default model in `~/.codex/config.toml` or the repository's `.codex/config.toml`.
- Claude models (sonnet-5, opus-4.8, fable-5) run via the Agent/Workflow model parameter.

Using gpt-5.5 inside workflows and subagents:
- The Agent/Workflow `model` parameter only accepts Claude models. To delegate work to gpt-5.5, use the Codex plugin's bundled `codex:codex-rescue` subagent rather than creating a custom Claude wrapper. If a wrapper is needed and is a must only then spawn a Claude wrapper agent with `model: 'sonnet', effort: 'low'` whose prompt instructs it to write a self-contained codex prompt. Plugin is priority and first target as it is setup with this intent and workflow in mind. 
- For implementation or investigation, invoke `/codex:rescue --model gpt-5.5 --effort high --background <self-contained task>`.
- Use `/codex:status` to check progress and `/codex:result` to retrieve the completed response.
- For an independent code review, run `/codex:review --background`.
- For a review focused on challenging design decisions, assumptions, or specific risk areas, run `/codex:adversarial-review --background <focus>`.
- Claude may also delegate naturally by being instructed to ask Codex to complete a task.

When Using Plan mode:
- Inherited / current model the user is using will be the model that is used to create the plan for the task at hand. This will likely be Fable 5 or Opus 4.8
- Once Fable 5 or Opus 4.8 has thought of a plan, spawn a subagent who will use `model: 'sonnet 5'` and the thinking effort will be based on complexity of task. This sonnet 5 model will create a HTML file using the frontend design skill. This HTML file should outline the entire plan and be presented to me (user).
- Instead of the typical MD file that is shown as the plan outline before the user (me) clicks proceed to implement, this HTML file will replace it. Make sure the Artifact HTML created is opened for the user when you are ready to show the plan and HTML file. 
- The objective is to visualize the plan prior to implementation so that its easier to optimize the plan before any code is written. 
- All subagents launched in Plan Mode will use `'model: 'sonnet 5'`. Effort level can be your choice based on complexity of task given to the model. This includes `Explore` Agents. The only Exception is the `plan` Agent who can use the `Model: 'Opus 4.8'` as the plan-agent default when specs are detailed and exploration ran first; `Model: 'Fable 5'` for open-ended or high ambigutiy design.

### Loops: which primitive to trigger

A loop = repeated work cycles until a stop condition (full guide + starter prompts: `how-to-loop.md`). The deterministic stop condition for all code loops in this repo is the verify gate — `scripts\verify.ps1` / `scripts/verify.sh`, final line `VERIFY: PASS|FAIL` — governed by the project skill `verify-rag-change`. Use that skill before claiming any code change done, in or out of a loop.

Route by task size; never a bigger loop than the task needs:

- **Short** (one file / obvious fix, ~≤3 turns): plain turn-based work. No /goal, no subagents, no background jobs. Run the gate once before reporting done; read only the verdict + failures.
- **Medium** (multi-file feature/bugfix with a checkable done-state): best run as `/goal <task>. Done when scripts\verify.ps1 prints VERIFY: PASS. Stop after 4 tries.` Iterate on the scoped gate (`-TestPath tests\test_x.py`) while fixing; the full gate is the exit check. Bulk/mechanical diffs → gpt-5.5 via `/codex:rescue` (table above); close with `/codex:review --background` for a fresh-context review. If a medium task arrives as a plain prompt, still enforce the gate, and put the ready-to-paste /goal one-liner in the final summary so the next run can be hands-off.
- **Long** (multi-phase, hours, or waiting on external systems): plan mode first (rules above), then each phase runs as its own medium /goal loop with its own PASS exit — never one giant loop. Watching external state (CI, PR reviews) → `/loop` with the interval matched to how fast the target changes (~4m for active CI; ≥20m for idle watching — avoid ~5m, it's the worst cache breakpoint), or interval-free `/loop` so Claude self-paces. Recurring repo upkeep (ingestion/scheduler health) → `/schedule` routine, not a live session.
- **Every size**: deterministic steps go in scripts, not reasoning; the same failure surviving two fix attempts means stop patching — change approach or escalate the model; pilot one slice before any fan-out; audit burn with `/usage` and `/goal` (no args).
```

# How to Loop — Universal (any project)

A loop is Claude repeating cycles of work (gather context → act → check → repeat) until a stop condition is met. This is the universal edition of `how-to-loop.md`: nothing here assumes a specific repo, language, or test framework, so the copy-paste block at the bottom can be plugged into **any** project's CLAUDE.md. A copy of this guide lives at `~/.claude/how-to-loop-global.md` so it's reachable from every project.

---

## The 30-second version

| Loop type | You hand off | Trigger | Stops when | Command |
|---|---|---|---|---|
| **Turn-based** | the check | your prompt | Claude judges it's done | (none — every prompt) |
| **Goal-based** | the stop condition | your prompt | condition true OR turn cap hit | `/goal` |
| **Time-based** | the trigger | a clock | you cancel it / work completes | `/loop`, `/schedule` |
| **Proactive** | the prompt itself | event/schedule, no human | each task exits on its goal | `/schedule` + `/goal` + skills |

The single most useful command in any repo:

```
/goal <describe the change>. Done when the repo verify gate (scripts\verify.ps1 or scripts/verify.sh) prints VERIFY: PASS. Stop after 4 tries.
```

That's a medium-task loop with a deterministic exit. Everything else is a variation.

---

## What the universal setup is (and why)

**1. The verify-gate convention** — instead of one repo's script, a contract every repo follows:

> One command (by convention `scripts/verify.ps1` + `scripts/verify.sh`; `make verify` / `npm run verify` also count) runs the repo's lint + offline tests, and its **final line is machine-readable**: `VERIFY: PASS` (exit 0) or `VERIFY: FAIL (<stages>)` (exit 1). Offline and deterministic — external services mocked. Optionally takes a test path as an argument for fast scoped runs.

Why it matters: `/goal` works by having an evaluator model check your stop condition after each turn. A fuzzy condition ("the feature works well") makes the evaluator guess and the loop end too early or run too long. "The gate prints `VERIFY: PASS`" is a string match — the loop can't lie to itself, and Claude doesn't spend tokens reasoning about whether it's done. Because the convention is identical everywhere, the same /goal sentence works in every repo — muscle memory instead of per-project prompts.

**2. Global skill — `~/.claude/skills/verify-loop-gate/SKILL.md`** (new)
Personal skills apply to every project on this machine. This one teaches any Claude session the gate contract, and — the key universal move — **to create the gate scripts in any repo that doesn't have them yet**, built from that repo's own lint/test commands. It also carries the discipline: read only the verdict + failures, and the two-strikes rule (same failure surviving two fix attempts → stop patching, re-diagnose or escalate the model). Project-level verification skills (like `verify-rag-change` in the Gemma RAG repo) take precedence when present.

**3. CLAUDE.md routing** — the copy-paste block at the bottom. Two ways to deploy it:
- **Per project** (recommended): paste into each repo's CLAUDE.md as you start looping there. You keep control, and repos with different needs can diverge.
- **Once, globally**: paste into `~/.claude/CLAUDE.md` and it applies to every project on the machine automatically. Cheaper to maintain, but it loads into *every* session everywhere — including non-code work where it's dead weight.

**4. Nothing needs downloading for the primitives.** `/goal`, `/loop`, and `/schedule` are built into Claude Code (`/goal` needs ≥ v2.1.139; you're on 2.1.202 ✓). The superpowers plugin is installed at user scope, so its two loop-critical discipline skills — `verification-before-completion` (evidence before completion claims) and `systematic-debugging` (what two-strikes escalates into) — are **already global** in every project. No action needed.

A note from the pilot: the Gemma RAG repo was the first to get a gate, and its very first run caught 5 "test failures" that were actually one environment problem (missing venv packages, one of which was silently disabling a runtime feature). Expect the first gate run in any repo to surface similar drift — that's the gate paying for itself, not a problem with the gate.

---

## Starting each loop

### Short tasks — no loop machinery at all

*One file, obvious fix, done in a few turns.* Just prompt normally:

> Fix the off-by-one in the pagination cursor in `src/api/list_orders.ts`.

The routing block + `verify-loop-gate` skill make Claude run the gate once before reporting done. **Do not** wrap short tasks in `/goal` — you'd pay for an evaluator check per turn and gain nothing. This is the cheapest loop; keep it cheap by being specific (file, symptom, expected behavior) so Claude doesn't burn turns exploring.

### Medium tasks — `/goal` with the gate as the exit

*Multi-file feature or bugfix where "done" is checkable.* Template:

```
/goal <task>. Done when the repo verify gate prints VERIFY: PASS. Stop after 4 tries.
```

Generic examples that work in any codebase:

```
/goal Add rate limiting to the /api/upload endpoint, with tests. Done when the repo verify gate prints VERIFY: PASS. Stop after 4 tries.

/goal Fix the date-parsing bug from issue #42 and add a regression test. Done when the new test passes AND the verify gate prints VERIFY: PASS. Stop after 5 tries.

/goal Raise test coverage of src/auth/ to 90%. Done when the coverage report shows >=90% for src/auth/ AND the verify gate prints VERIFY: PASS. Stop after 5 tries.
```

Rules of thumb:
- **Always set a turn cap** ("stop after 4–5 tries"). It's your budget ceiling; without it a stuck loop burns tokens until you notice.
- **Chain conditions with AND** when you need both a specific metric and the gate (coverage example above).
- Inside the loop, the routing block tells Claude to iterate on the **scoped** gate (narrowest test file/target) and only run the full gate as the exit check, and to hand bulk/mechanical diffs to gpt-5.5 via `/codex:rescue` (effectively free on your plan) while the session model does routing + judgment.
- **Close out with a fresh-context review**: after the loop exits, run `/codex:review --background`. A reviewer that didn't watch the loop reason is unbiased, and on gpt-5.5 it costs you ~nothing.

### Long tasks — plan mode first, then phased /goal loops

*Multi-phase work, hours of effort, or anything waiting on external systems.* Never run one giant loop — an early wrong turn compounds for hours, and one huge context is expensive. Instead:

1. **Plan mode** (your existing workflow: Fable/Opus plans, Sonnet renders the HTML plan artifact). The plan's job for looping: split work into phases where **each phase leaves the gate green**.
2. **Run each phase as its own medium `/goal` loop** with its own `VERIFY: PASS` exit and turn cap. Fresh-ish context per phase, and a failed phase can't silently corrupt the next.
3. **External waits use `/loop`, not a spinning session.** Two flavors:

```
/loop 4m Check PR #7: address new review comments, fix failing CI, push. Stop when the PR is merged or closed.

/loop Watch the data migration running in the background. When it finishes, validate the row counts against the source, summarize, then stop. Pace your own check-ins.
```

The second form (no interval) is **dynamic /loop** — Claude picks its own wake-up times, which is the smarter default when the wait time is unpredictable.

### Recurring work — `/schedule`

*Same task, new inputs, on a clock. Runs as a routine, not a live session (research preview; cloud-side, so it survives your machine being off).* Generic examples:

```
/schedule every weekday at 8am: triage new issues on the repo — label them, attempt a quick reproduction, close duplicates, and summarize anything that needs my attention.

/schedule every Monday at 9am: check for dependency updates, apply the safe minor/patch bumps on a branch, run the verify gate, and open a PR if it prints VERIFY: PASS.
```

### Proactive combo — the full composition

When the pieces above are trusted individually, compose them (pilot on one run before letting it recur):

```
/schedule nightly at 2am: run the repo verify gate. /goal: don't stop until every new failure is either fixed (gate prints VERIFY: PASS) or written up in docs/reports/ with a minimal repro. Stop after 6 tries per night.
```

---

## Token-efficiency rules (the 5x Max survival guide)

1. **Right-size the primitive.** Short → turn-based. Medium → /goal. Long → phases + /loop for waits. Recurring → /schedule. The routing block enforces this; overriding it upward is almost always waste.
2. **Deterministic exit criteria.** Every reasoning token spent deciding "am I done?" is waste — the gate's string match costs zero. Same for any repeated procedure: if a loop does a step the same way twice, ask Claude to script it.
3. **Read verdicts, not logs.** The gate prints failures + one verdict line. The skill instructs Claude to read only those. When you check a loop's progress yourself, do the same.
4. **Turn caps are budget caps.** 4–5 tries for medium loops. If it didn't converge in 5, the problem is diagnosis, not persistence — that's what the two-strikes escalation is for.
5. **Match /loop intervals to reality, and mind the cache.** The prompt cache lives ~5 minutes: a `/loop 4m` keeps context cached between iterations (cheap); `/loop 5m` re-reads everything every time (worst case); anything slower should jump straight to 20–30m. Only poll fast when the watched thing actually changes fast (active CI ≈ 4m). For unpredictable waits, use interval-free `/loop` and let Claude pace itself.
6. **Route models inside loops.** Judgment/stop decisions on the session model; bulk implementation to gpt-5.5 (`/codex:rescue`); reviews to `/codex:review --background`. Your cost table already encodes this — loops just apply it repeatedly, so the savings compound.
7. **Pilot before fan-out.** Any recurring or multi-agent construct: run it once on one slice, check `/usage`, then let it recur.
8. **Audit.** `/goal` with no arguments shows turns + tokens so far; `/usage` breaks down burn by skills/subagents/MCPs; `/workflows` shows per-agent usage if you ever use dynamic workflows.
9. **When a loop produces a bad result, fix the system, not just the result.** Add the missed check to the repo's gate script or a rule to the skill — every future loop in every repo inherits the fix for free. That's the compounding-returns move.

## Monitoring & stopping

- **/goal loop:** exits on PASS or the turn cap; interrupt anytime with Esc. `/goal` (no args) = progress + token report.
- **/loop:** runs on your machine; stops on its stop condition, or interrupt with Esc / tell Claude to stop the loop. It dies if the machine sleeps — that's what `/schedule` is for.
- **/schedule:** manage routines with `/schedule` (list/cancel), or ask Claude to list and delete them.

## Skills — summary for you

| Skill | Where | Scope | What it does |
|---|---|---|---|
| `verify-loop-gate` | `~/.claude/skills/verify-loop-gate/` | **global — new** | Gate contract + discipline for every repo; creates `scripts/verify.ps1|sh` in repos that lack one; gate = /goal stop condition; two-strikes escalation |
| `verification-before-completion` | superpowers plugin | global (user-scope plugin) | Generic "evidence before completion claims" discipline — backs the gate skill |
| `systematic-debugging` | superpowers plugin | global (user-scope plugin) | Where the two-strikes rule sends Claude instead of blind re-patching |
| `verify-rag-change` | Gemma RAG repo `.claude/skills/` | that repo only | Project override of `verify-loop-gate` — repo-specific gate details; project skills always win over the global one |

The copy-paste block below references `verify-loop-gate`, so any CLAUDE.md you paste it into picks the global skill up. `/goal`, `/loop`, `/schedule` are built-in — nothing to download for them.

## References

- [Claude Code docs: /goal](https://code.claude.com/docs/en/goal) (also see the loop, schedule, and dynamic-workflows pages)
- The ClaudeDevs "Getting started with loops" article (saved in the Gemma RAG repo as `loops.txt`)
- Repo-specific edition of this guide: `how-to-loop.md` in the Gemma RAG repo

---

# COPY-PASTE BLOCK — "Picking the right models for workflows and subagents" (universal, for any CLAUDE.md)

> **Note:** This is the repo-agnostic version — paste it into any project's CLAUDE.md (or once into `~/.claude/CLAUDE.md` to apply everywhere). It references only the global `verify-loop-gate` skill and the gate *convention*, never a specific repo's files. The Gemma RAG repo keeps its own specialized version — don't paste this over it.

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

A loop = repeated work cycles until a stop condition (full guide + starter prompts: `~/.claude/how-to-loop-global.md`). The deterministic stop condition for all code loops is the repo's verify gate — by convention `scripts/verify.ps1` / `scripts/verify.sh` (a repo-native `make verify` / `npm run verify` also counts), final line `VERIFY: PASS|FAIL` — governed by the global skill `verify-loop-gate`. Use that skill before claiming any code change done; if the repo has no gate yet, that skill says to create one from the repo's own lint + offline-test commands.

Route by task size; never a bigger loop than the task needs:

- **Short** (one file / obvious fix, ~≤3 turns): plain turn-based work. No /goal, no subagents, no background jobs. Run the gate once before reporting done; read only the verdict + failures.
- **Medium** (multi-file feature/bugfix with a checkable done-state): best run as `/goal <task>. Done when the repo verify gate prints VERIFY: PASS. Stop after 4 tries.` Iterate on the narrowest scoped check (one test file/target) while fixing; the full gate is the exit check. Bulk/mechanical diffs → gpt-5.5 via `/codex:rescue` (table above); close with `/codex:review --background` for a fresh-context review. If a medium task arrives as a plain prompt, still enforce the gate, and put the ready-to-paste /goal one-liner in the final summary so the next run can be hands-off.
- **Long** (multi-phase, hours, or waiting on external systems): plan mode first (rules above), then each phase runs as its own medium /goal loop with its own PASS exit — never one giant loop. Watching external state (CI, PR reviews) → `/loop` with the interval matched to how fast the target changes (~4m for active CI; ≥20m for idle watching — avoid ~5m, it's the worst cache breakpoint), or interval-free `/loop` so Claude self-paces. Recurring repo upkeep (nightly checks, dependency bumps, issue triage) → `/schedule` routine, not a live session.
- **Every size**: deterministic steps go in scripts, not reasoning; the same failure surviving two fix attempts means stop patching — change approach or escalate the model; pilot one slice before any fan-out; audit burn with `/usage` and `/goal` (no args).
```

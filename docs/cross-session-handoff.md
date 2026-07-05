# Coordinating Two Claude Code Sessions — The "Watch & Wait" Handoff

This guide explains how to make **one Claude Code session wait for another session to finish a goal**, and then automatically begin its own goal once the first one is done.

Claude Code has **no built-in "ping another session" feature**. Sessions are isolated — one session has no address or handle that another session can reach. But sessions **share the same disk**, and that's enough to build a clean, reliable handoff.

The idea in one line: **the worker writes a "I'm done" file; the waiter watches for that file and starts when it appears.**

---

## The two roles

| Role | What it does | Which session |
|------|--------------|---------------|
| **Worker** | Runs its goal, then writes a completion marker file as its final step | The session currently doing work |
| **Waiter** | Watches for the marker file, then starts its own goal | The idle session waiting to begin |

You (the human) start **both** sessions, give each its instructions, and walk away. The handoff happens automatically.

---

## The shared marker file

Pick one path both sessions agree on. Default:

```
.claude/handoff/worker.done
```

The worker writes this file *only when its goal is actually finished and verified*. Its existence = "the worker is done." The waiter treats the appearance of this file as its starting gun.

The file should contain a short summary so the waiter starts informed, not blind:

```json
{
  "status": "complete",
  "goal": "<what the worker was asked to do>",
  "summary": "<2-3 sentences: what was built/changed>",
  "commit": "<git SHA of the final commit, if relevant>",
  "at": "<timestamp>"
}
```

---

## Step-by-step setup

### Step 1 — Agree on the marker path

For a single handoff, use `.claude/handoff/worker.done`.
For multiple independent goals, give each a unique slug: `.claude/handoff/<goal-slug>.done`.

### Step 2 — Start the Worker session

Open a Claude Code session and give it:

1. **Its goal** (what it should accomplish).
2. **This final instruction**, appended to the goal:

   > When you have finished the goal and verified it (tests pass, changes committed if requested), write a completion marker to `.claude/handoff/worker.done` as a JSON object with `status`, `goal`, `summary`, `commit` (if applicable), and `at` (timestamp). This is your last action — do not do anything after writing it.

The worker must **actively** signal. There is no "session ended" event another session can observe, so the marker file *is* the coordination primitive. If the worker forgets to write it, the waiter waits forever — so make this instruction explicit and non-negotiable.

### Step 3 — Start the Waiter session (this one)

In the waiter session, arm a **background watcher** that exits the moment the marker appears:

```bash
until [ -f .claude/handoff/worker.done ]; do sleep 5; done
```

Run it with `run_in_background: true`. This gives you **exactly one notification** when the file appears — not a stream of events, not polling noise. Just "he's done."

When the watcher fires:

1. Read `.claude/handoff/worker.done` to learn what the worker produced.
2. Optionally inspect the worker's work (`git log`, `git diff`, read changed files) to ground yourself in what actually landed.
3. Begin your own goal.

That's the whole loop.

---

## Why this shape (the reasoning)

- **`run_in_background` + `until` loop = one clean notification.** This is deliberately *not* a `Monitor` (which streams many events). For "tell me when he's done," you want a single completion signal, and `until … done` exits on its own the moment the condition is true.
- **Polling every 5 seconds is cheap.** It touches the filesystem once per tick. Drop to 2s if you want it snappier; raise to 15s if you don't care about latency. Avoid sub-second polling — it buys you nothing and spams disk.
- **The marker carries the summary.** Without it, the waiter starts blind and has to re-derive what the worker did. With it, the waiter starts *informed* and can immediately build on the worker's output.
- **The marker is authoritative.** The waiter doesn't try to guess "is the worker done?" from process lists or logs. The worker says it's done by writing the file. Simple, unambiguous, no false positives.

---

## Making it robust

### The worker might fail or crash

A worker that crashes never writes `.done`, so the waiter waits forever. Fix this by having the worker write a **different** file on failure:

```
.claude/handoff/worker.done   → success
.claude/handoff/worker.failed → failure (with error details)
```

Then the waiter watches for **either**:

```bash
until [ -f .claude/handoff/worker.done ] || [ -f .claude/handoff/worker.failed ]; do sleep 5; done
```

When it fires, the waiter checks which file exists and decides: proceed (on `.done`) or report the failure to you (on `.failed`).

### You want the handoff to be verifiable

Have the worker include the **commit SHA** in the marker and only write `.done` after tests pass. Then the waiter can run `git log -p <sha>` or `git diff <prev>..<sha>` to see exactly what changed before starting — no guesswork about whether the worker actually finished.

### You want a deadline

If you're worried the worker could hang silently, give the waiter a timeout instead of an open-ended `until`:

```bash
# Wait up to 30 minutes (1800s), checking every 5s
end=$((SECONDS + 1800))
until [ -f .claude/handoff/worker.done ] || [ $SECONDS -ge $end ]; do sleep 5; done
```

If the loop exits because of the deadline (and the file still isn't there), the waiter should tell you the worker didn't finish in time.

### You want to chain more than two sessions

Each handoff is just a marker file. Session A writes `goal-a.done`. Session B watches for `goal-a.done`, runs, then writes `goal-b.done`. Session C watches for `goal-b.done`. You can chain N sessions this way with zero extra machinery — just a different filename per hop.

---

## A fully-worked example

**You say to the Worker session:**
> Refactor `src/billing/` to use the new `Invoice` dataclass. Run the test suite; only when it passes, write `.claude/handoff/billing-refactor.done` with the commit SHA and a one-line summary, then stop.

**You say to the Waiter session (this one):**
> Your goal is to add a `refund()` method to the new `Invoice` class. But don't start yet — the Worker is refactoring `src/billing/` first. Watch `.claude/handoff/billing-refactor.done` and begin only when it appears.

**The Waiter (me) does this:**
```bash
until [ -f .claude/handoff/billing-refactor.done ]; do sleep 5; done
```
(background)

**When it fires**, I read the marker, pull up the worker's final commit, find the new `Invoice` class, and start adding `refund()`.

You did nothing in between. The handoff was automatic.

---

## Quick reference

| Need | Do this |
|------|---------|
| Signal "I'm done" | Worker writes `.claude/handoff/<goal>.done` as its final step |
| Wait for the signal | Waiter runs `until [ -f …/.done ]; do sleep 5; done` in the background |
| Start informed | Waiter reads the marker file before beginning its goal |
| Handle worker failure | Worker also writes `.failed`; waiter watches for either |
| Add a deadline | Waiter's loop also exits on a `SECONDS` cap |
| Chain N sessions | One `.done` file per hop; each session watches the previous one's file |

---

## What this is *not*

- It's **not** a real inter-session message bus. There's no addressing, no two-way chat, no "send a message to session X." It's a one-way "done" flag plus a polling watcher.
- It's **not** durable across machine restarts in any special way — it's just a file on disk, which is fine for local, same-machine sessions.
- It does **not** let an idle session wake up on its own. The waiter must be a *running* session with the background watcher armed. A fully closed session can't react to anything; you'd need a hook or cron-style trigger for that, which is a different and more limited pattern.

If you need richer coordination (two-way messages, queues, multiple workers), the next step up is a **custom MCP server** with shared state — but that's a build project, not a built-in feature. For the common "wait then go" case above, the file marker is all you need.
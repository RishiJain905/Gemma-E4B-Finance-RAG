# Implementation Execution Workflow

## Overview

A lightweight execution workflow for the Gemma-E4B-Finance-RAG multi-agent stack. Designed to keep agents on a clear path without over-engineering. The approved implementation plan is the contract — follow it, verify against it, and flag divergence.

---

## 1. Role & Operating Mode

The orchestrator is the **context manager and integration owner** for the entire task.

The orchestrator is responsible for:
- Maintaining global context across all workstreams
- Preserving the approved implementation plan
- Delegating specialist work to subagents
- Identifying dependencies, overlap, and execution order
- Owning integration and merge strategy
- Ensuring no approved requirements are lost across handoffs

The orchestrator **does not** default to implementing specialist work directly. Delegate to the appropriate subagent.

### Who Does What

| Task | Owner |
|------|-------|
| Backend / API / data pipeline | backend specialist |
| Frontend / UI / visualization | frontend specialist |
| Tests (creation or expansion) | whoever owns the code |
| Integration and merging | orchestrator only |
| Cross-stream conflict resolution | orchestrator only |

---

## 2. Plan Intake

- Read the approved implementation plan and any referenced specs before touching code.
- If an approved plan already exists, execute against it — do not write a replacement.
- If the plan conflicts with the current repo state, flag it briefly and continue.
- Before implementation begins, identify:
  - Shared files and overlap between workstreams
  - Integration and conflict zones
  - Ordering or migration dependencies
  - Verification strategy

---

## 3. Development

### Delegation Rules

- Delegate **all implementation work** to subagents when a matching role exists.
- Delegation is mandatory — not optional based on task size or convenience.
- The orchestrator may step in directly only when:
  - The work is integration-owned by the orchestrator
  - A blocking issue prevents safe delegation
  - No suitable specialist role exists
  - Subagent output requires intervention to unblock progress

### Development Loop

1. **Delegate** — assign task to the appropriate specialist subagent
2. **Implement** — subagent completes the work
3. **Verify** — run relevant tests to confirm the change works
   - Bug fix: write a test that reproduces the bug first, confirm it fails, implement fix, confirm it passes
   - Feature: implement, then run existing tests to verify behavior
   - If no tests exist for the area, note that in the report
4. **Report** — subagent reports completion status and any blockers

### Parallel-First Rule

Run independent tasks in parallel. Do not serialize work that has no dependencies between it.

---

## 4. Integration

- The orchestrator **owns all merging** into the target branch.
- Subagents complete isolated work; the orchestrator handles merge ordering and conflict resolution.
- Choose the safest integration approach per situation:
  - `merge` — for straightforward combining of completed work
  - `rebase` — for linear history when appropriate
  - `cherry-pick` — for selective porting of specific commits
- Resolve conflicts centrally. Do not let subagents merge their own streams.
- Preserve all approved functionality from every stream during integration.

---

## 5. Verification & Cleanup

### Verification

- Run the full relevant test suite after integration.
- Report:
  - What was done
  - Tests that passed / tests that failed
  - Any remaining blockers

### Cleanup

- Delete merged local and remote source branches when appropriate.
- Do not claim completion without passing test results on the integrated branch.

---

## Red Flags — Don't Do This

- **Don't rewrite the approved plan** — execute it
- **Don't skip verification** before claiming done
- **Don't delegate integration** — orchestrator only
- **Don't implement outside the approved scope** — flag scope creep instead

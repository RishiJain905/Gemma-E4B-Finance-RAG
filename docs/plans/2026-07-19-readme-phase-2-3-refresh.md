# Phase 2.3 README Refresh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish an accurate post-Phase 2.3 README and a root MIT license.

**Architecture:** Keep the README as the concise entry point, deriving scheduler and source claims from `configs/sources.yaml` and chat behavior from `scripts/CHAT.md`. Link to the detailed configuration, architecture, API, chat, and Phase 2.3 result documents instead of duplicating their full reference material.

**Tech Stack:** Markdown, YAML configuration references, PowerShell verification gate, Git/GitHub CLI.

---

### Task 1: Refresh the public overview and setup path

**Files:**
- Modify: `README.md`

**Step 1:** Replace the six-source framing and clone placeholder with the public repository URL and registry-driven Phase 2.3 overview.

**Step 2:** Expand environment setup with optional Phase 2.3 provider keys while preserving the required FRED and SEC guidance.

**Step 3:** Update initial ingestion guidance to distinguish one-time bootstrap from incremental daily/hourly/weekly operation.

### Task 2: Refresh chat and scheduler operations

**Files:**
- Modify: `README.md`

**Step 1:** Add the high-value commands and behavior documented in `scripts/CHAT.md`: streaming controls, multiline questions, bounded local history, evaluation commands, graph deep links, and scheduler refresh commands.

**Step 2:** Document `scripts/watch_scheduler.py` as a read-only live monitor, including separate Massive market/news freshness rows and disabled GDELT status.

**Step 3:** Correct Phase 2.3 rollout wording, source configuration ownership, and project structure entries.

### Task 3: Add the project license

**Files:**
- Create: `LICENSE`
- Modify: `README.md`

**Step 1:** Add the standard MIT license text with the repository owner's copyright.

**Step 2:** Link the README license section to the root file and preserve the existing third-party vendor licenses.

### Task 4: Verify and publish

**Files:**
- Verify: `README.md`
- Verify: `LICENSE`

**Step 1:** Check Markdown links and known stale-claim patterns.

**Step 2:** Run `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1` and require `VERIFY: PASS`.

**Step 3:** Commit and push `Rishi-Ghost`.

**Step 4:** Open a pull request from `Rishi-Ghost` to `main`, merge it, and confirm `origin/main` contains the merged commit.

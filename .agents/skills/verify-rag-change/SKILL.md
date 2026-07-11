---
name: verify-rag-change
description: Use after changing any code in this repo — before claiming work is done, committing, ending a /goal iteration, or reporting a loop result — and when defining what "done" means for a loop stop condition.
---

# Verify a change (the gate)

One command decides done-ness. Never claim complete from a clean edit alone.

- Windows: `powershell -ExecutionPolicy Bypass -File scripts\verify.ps1`
- Unix / Git Bash: `bash scripts/verify.sh`
- Scoped (inner loop, while iterating): `scripts\verify.ps1 -TestPath tests\test_retriever.py` or `bash scripts/verify.sh tests/test_retriever.py`. A scoped PASS is progress, not done — only the full gate ends the task.

It runs ruff + the offline pytest suite (`-m "not live"`): no network, no llama-server needed. The final line is the verdict — `VERIFY: PASS` (exit 0) or `VERIFY: FAIL (ruff, pytest)` (exit 1).

## Rules

1. Read the verdict line and failure blocks only. Don't re-run tools the gate already ran; for one test's full traceback use `pytest tests/test_x.py::test_y -v`.
2. FAIL → fix → rerun the gate. No partially verified hand-backs.
3. Same failure surviving two fix attempts: stop patching. Re-diagnose (superpowers:systematic-debugging) or escalate the implementation model per AGENTS.md.
4. In `/goal` loops this gate IS the stop condition: "Done when scripts\verify.ps1 prints VERIFY: PASS."
5. New behavior needs a test in the offline suite before a PASS means anything. Mock external services; a test that needs the network or :8087 must carry the `live` marker or it breaks the gate for everyone.

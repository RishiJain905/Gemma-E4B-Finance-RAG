"""
eval/score.py — Stage 2 of the eval harness (2.1.1.2, 2.2.1.2): scoring CLI + report.

Loads a run artifact (the latest by default, or ``--run <path>``), computes every
metric via ``metrics.score_all``, and emits:

  - ``eval/runs/<ts>.summary.json`` — the machine-readable metrics block.
  - a console table — overall + per-category scores.
  - ``eval/runs/<ts>.report.md``    — a readable diff vs. the previous run (--report).

Promote the latest summary to a committed, policy-specific baseline with
``--set-baseline --policy strict|graded`` (2.2.1.2 Step 5). Baselines live at
``eval/baselines/<policy>.json`` — replacing the single ``eval/baseline.json``
so a strict-mode run is never compared against a graded-mode bar or vice versa.

Usage:
    python eval/score.py                  # score the latest run, with LLM-judge
    python eval/score.py --no-judge       # skip the (slow) LLM-judge
    python eval/score.py --run eval/runs/<ts>.jsonl
    python eval/score.py --report         # also write a markdown diff vs prev run
    python eval/score.py --set-baseline --policy graded   # promote to eval/baselines/graded.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
BASELINES_DIR = EVAL_DIR / "baselines"

_REPO_ROOT = EVAL_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval import metrics as M  # noqa: E402
from src.middleware import evidence_trace  # noqa: E402

# Display order + which direction is "better" for each metric.
_DISPLAY = [
    ("intent_accuracy", "higher"),
    ("ticker_accuracy", "higher"),
    ("retrieval_hit_rate", "higher"),
    ("keyword_coverage", "higher"),
    ("refusal_rate", "lower"),
    ("grounded_faithfulness", "higher"),
    ("policy_compliance", "higher"),
    ("answer_relevance", "higher"),
]


# ── IO ─────────────────────────────────────────────────────────────────

def load_run(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Run {path} is empty")
    return rows


def latest_run(runs_dir: Path = RUNS_DIR) -> Optional[Path]:
    runs = sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def latest_summary(runs_dir: Path = RUNS_DIR) -> Optional[Path]:
    sums = sorted(runs_dir.glob("*.summary.json"), key=lambda p: p.stat().st_mtime)
    return sums[-1] if sums else None


def summary_path_for(run_path: Path) -> Path:
    return run_path.with_suffix(".summary.json")  # <ts>.jsonl -> <ts>.summary.json


def write_summary(summary: dict, run_path: Path) -> Path:
    out = summary_path_for(run_path)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def baseline_path(policy: str, baselines_dir: Optional[Path] = None) -> Path:
    """Path to the committed baseline for one answer policy.

    Reads the module-level ``BASELINES_DIR`` at call time (not as a bound
    default) so tests can monkeypatch it.
    """
    return (baselines_dir or BASELINES_DIR) / f"{policy}.json"


# ── Display ────────────────────────────────────────────────────────────

def _metric_value(summary: dict, name: str):
    if name in M.SCALAR_METRICS:
        return summary.get(name)
    block = summary.get(name)
    if isinstance(block, dict):
        return block.get("score")
    return None


def _fmt(v) -> str:
    if v is None:
        return "  n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def console_table(summary: dict) -> str:
    """Render the overall metrics table (string)."""
    ts = summary.get("generated_at", "unknown")
    policy = summary.get("answer_policy", "unknown")
    lines = [f"=== Eval summary ({ts}) - policy={policy} ===", ""]
    for name, direction in _DISPLAY:
        val = _metric_value(summary, name)
        suffix = "   (lower is better)" if direction == "lower" else ""
        lines.append(f"{name:<22} {_fmt(val)}{suffix}")
    lines.append("")
    lines.append(f"cases: {summary.get('n_cases', 0)}  "
                 f"model_available: {summary.get('n_model_available', 0)}  "
                 f"errors: {summary.get('n_errors', 0)}  "
                 f"trace_errors: {summary.get('n_trace_errors', 0)}")

    # Per-category table.
    per = summary.get("per_category") or {}
    if per:
        lines.append("")
        lines.append("=== Per category ===")
        header = f"{'category':<14}{'n':>3}{'intent':>8}{'ticker':>8}{'retr':>8}{'kw':>8}{'refuse':>8}"
        lines.append(header)
        for cat in sorted(per):
            c = per[cat]
            lines.append(
                f"{cat:<14}{c.get('n', 0):>3}"
                f"{_fmt(c.get('intent_accuracy')):>8}"
                f"{_fmt(c.get('ticker_accuracy')):>8}"
                f"{_fmt(c.get('retrieval_hit_rate')):>8}"
                f"{_fmt(c.get('keyword_coverage')):>8}"
                f"{_fmt(c.get('refusal_rate')):>8}"
            )
    return "\n".join(lines)


def report_md(summary: dict, prev: Optional[dict]) -> str:
    """A readable markdown report, with a diff vs. the previous run."""
    lines = [f"# Eval report ({summary.get('generated_at', 'unknown')})", ""]
    lines.append(f"Run: `{summary.get('source_run', 'n/a')}`  "
                 f"Policy: {summary.get('answer_policy', 'unknown')}  "
                 f"Cases: {summary.get('n_cases', 0)}  "
                 f"Model available: {summary.get('n_model_available', 0)}/{summary.get('n_cases', 0)}")
    lines.append("")
    lines.append("| metric | current | previous | delta |")
    lines.append("|---|---|---|---|")
    for name, direction in _DISPLAY:
        cur = _metric_value(summary, name)
        prv = _metric_value(prev, name) if prev else None
        if cur is None and prv is None:
            delta = ""
        elif cur is None or prv is None:
            delta = "—"
        else:
            d = round(cur - prv, 4)
            # For lower-is-better, a positive delta is a regression.
            arrow = "" if d == 0 else ("⚠️" if (direction == "higher" and d < 0) or
                                       (direction == "lower" and d > 0) else "✓")
            delta = f"{d:+.2f} {arrow}"
        lines.append(f"| {name} | {_fmt(cur)} | {_fmt(prv)} | {delta} |")
    lines.append("")
    if prev:
        lines.append(f"_Previous run: `{prev.get('source_run', 'n/a')}`_")
    return "\n".join(lines)


# ── Baseline ───────────────────────────────────────────────────────────

# Default regression tolerances (the bar the gate enforces).
DEFAULT_TOLERANCES = {
    "intent_accuracy": 0.05,
    "ticker_accuracy": 0.05,
    "retrieval_hit_rate": 0.05,
    "keyword_coverage": 0.05,
    "refusal_rate": 0.05,
    "grounded_faithfulness": 0.05,
    "policy_compliance": 0.05,
    "answer_relevance": 0.05,
}


def _eligible_counts(summary: dict) -> dict:
    """Eligible-case counts for each judge metric (2.2.1.2 Step 5) — the
    gate compares these run over run so eligibility can't silently narrow."""
    out = {}
    for name in M.JUDGE_METRICS:
        block = summary.get(name)
        if isinstance(block, dict):
            out[name] = block.get("n_eligible")
    return out


def set_baseline(summary: dict, tolerances: Optional[dict] = None,
                  path: Optional[Path] = None, policy: Optional[str] = None) -> Path:
    """Write a committed, policy-specific baseline from a summary block.

    ``path`` overrides the destination (tests use a tmp path); otherwise the
    destination is derived from ``policy`` (or the summary's own
    ``answer_policy``) as ``eval/baselines/<policy>.json``.
    """
    resolved_policy = policy or summary.get("answer_policy") or "graded"
    out_path = path or baseline_path(resolved_policy)
    base = dict(summary)
    base["answer_policy"] = resolved_policy
    base["trace_schema_version"] = summary.get(
        "trace_schema_version", evidence_trace.SCHEMA_VERSION)
    base["score_schema_version"] = summary.get("score_schema_version", M.SCORE_SCHEMA_VERSION)
    base["eligible_counts"] = _eligible_counts(summary)
    base["tolerances"] = tolerances or DEFAULT_TOLERANCES
    base["is_baseline"] = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Score a run artifact.")
    p.add_argument("--run", type=Path, default=None,
                   help="Run artifact to score (default: latest in eval/runs).")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the LLM-judge (grounded_faithfulness/policy_compliance/"
                        "answer_relevance -> null).")
    p.add_argument("--report", action="store_true",
                   help="Also write a markdown diff vs the previous run.")
    p.add_argument("--set-baseline", action="store_true",
                   help="Promote the scored summary to eval/baselines/<policy>.json and exit.")
    p.add_argument("--policy", choices=["strict", "graded"], default=None,
                   help="Baseline policy to promote to with --set-baseline "
                        "(default: infer from the run's answer_policy).")
    p.add_argument("--model-endpoint", default=M.DEFAULT_MODEL_ENDPOINT,
                   help="LLM-judge endpoint (default :8087).")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="LLM-judge per-call timeout (seconds).")
    args = p.parse_args(argv)

    run_path = args.run or latest_run()
    if run_path is None or not run_path.exists():
        print("No run artifact found. Run `python eval/run_eval.py` first.", file=sys.stderr)
        return 2

    rows = load_run(run_path)
    summary = M.score_all(rows, endpoint=args.model_endpoint, timeout=args.timeout,
                          run_judge=not args.no_judge)
    summary["source_run"] = str(run_path)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary["trace_schema_version"] = evidence_trace.SCHEMA_VERSION

    out = write_summary(summary, run_path)
    print(console_table(summary))
    print(f"\nWrote summary -> {out}")

    if args.report:
        prev_path = latest_summary()
        prev = None
        # Don't diff against the run's own summary (resolve to handle relative
        # vs absolute path forms).
        out_resolved = out.resolve()
        if prev_path and prev_path.resolve() != out_resolved:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
        rep = run_path.with_suffix(".report.md")
        rep.write_text(report_md(summary, prev), encoding="utf-8")
        print(f"Wrote report  -> {rep}")

    if args.set_baseline:
        base = set_baseline(summary, policy=args.policy)
        print(f"Baseline set  -> {base}  (commit this file)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

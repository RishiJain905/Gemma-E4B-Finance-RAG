"""
eval/gate.py — Stage 3 of the eval harness (2.1.1.2): the regression gate.

Loads the latest summary (or ``--summary <path>``) and the committed baseline
(``eval/baseline.json``) and exits non-zero if any metric regresses beyond its
tolerance. Run this before/after each Phase 2.1 feature and (optionally) in CI:

    python eval/run_eval.py && python eval/score.py && python eval/gate.py

A metric regresses when:
  - higher-is-better (intent/ticker accuracy, retrieval hit-rate, keyword
    coverage, faithfulness, answer_relevance): current < baseline - tolerance
  - lower-is-better  (refusal_rate): current > baseline + tolerance

Metrics that are ``null`` on either side (e.g. the LLM-judge was skipped) are
skipped, not failed — the gate never fails on missing data, only on real drops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
BASELINE = EVAL_DIR / "baseline.json"

# Which direction is "better" for each gated metric.
DIRECTIONS = {
    "intent_accuracy": "higher",
    "ticker_accuracy": "higher",
    "retrieval_hit_rate": "higher",
    "keyword_coverage": "higher",
    "refusal_rate": "lower",
    "faithfulness": "higher",
    "answer_relevance": "higher",
}

DEFAULT_TOLERANCES = {
    "intent_accuracy": 0.05,
    "ticker_accuracy": 0.05,
    "retrieval_hit_rate": 0.05,
    "keyword_coverage": 0.05,
    "refusal_rate": 0.05,
    "faithfulness": 0.05,
    "answer_relevance": 0.05,
}


def _metric_value(summary: dict, name: str):
    """Extract a scalar metric value (judge metrics pull .score)."""
    if name in ("faithfulness", "answer_relevance"):
        block = summary.get(name)
        if isinstance(block, dict):
            return block.get("score")
        return block
    return summary.get(name)


def compare_to_baseline(current: dict, baseline: dict,
                        tolerances: Optional[dict] = None) -> dict:
    """Compare a current summary to the baseline.

    Returns ``{passed, regressions, skipped, improvements}`` where each entry in
    ``regressions`` / ``improvements`` is a dict describing the metric.
    """
    tols = tolerances or baseline.get("tolerances") or DEFAULT_TOLERANCES
    regressions: list[dict] = []
    skipped: list[dict] = []
    improvements: list[dict] = []

    for metric, direction in DIRECTIONS.items():
        cur = _metric_value(current, metric)
        base = _metric_value(baseline, metric)
        if cur is None and base is None:
            skipped.append({"metric": metric, "current": cur, "baseline": base})
            continue
        if base is not None and cur is None:
            # Was scored in the baseline but not now (e.g. model/judge down) —
            # treat as a regression so a model-down run can't silently pass.
            regressions.append({"metric": metric, "baseline": base,
                                "current": None, "delta": None,
                                "tolerance": None, "direction": direction,
                                "reason": "metric unavailable (was scored in baseline)"})
            continue
        if base is None and cur is not None:
            # New capability in current run — neither a regression nor an improvement.
            skipped.append({"metric": metric, "current": cur, "baseline": base})
            continue
        tol = tols.get(metric, DEFAULT_TOLERANCES.get(metric, 0.05))
        delta = round(cur - base, 4)
        if direction == "higher":
            regressed = cur < base - tol
            improved = cur > base + tol
        else:  # lower is better
            regressed = cur > base + tol
            improved = cur < base - tol
        entry = {"metric": metric, "baseline": base, "current": cur,
                  "delta": delta, "tolerance": tol, "direction": direction}
        if regressed:
            regressions.append(entry)
        elif improved:
            improvements.append(entry)

    return {"passed": len(regressions) == 0, "regressions": regressions,
            "skipped": skipped, "improvements": improvements}


def _fmt_num(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def render_report(result: dict) -> str:
    """Human-readable pass/fail report."""
    lines = ["=== Regression gate ==="]
    if result["passed"]:
        lines.append("RESULT: PASS - no metric regressed beyond tolerance.")
    else:
        lines.append(f"RESULT: FAIL - {len(result['regressions'])} metric(s) regressed.")
    lines.append("")

    if result["regressions"]:
        lines.append("Regressions:")
        lines.append(f"  {'metric':<22}{'baseline':>10}{'current':>10}{'delta':>10}{'tol':>8}")
        for r in result["regressions"]:
            delta = "n/a" if r.get("delta") is None else f"{r['delta']:+.2f}"
            tol = "n/a" if r.get("tolerance") is None else f"{r['tolerance']:.2f}"
            reason = f"  [{r['reason']}]" if r.get("reason") else ""
            lines.append(f"  {r['metric']:<22}{_fmt_num(r['baseline']):>10}"
                         f"{_fmt_num(r['current']):>10}{delta:>10}{tol:>8}{reason}")
        lines.append("")

    if result["improvements"]:
        lines.append("Improvements:")
        for r in result["improvements"]:
            lines.append(f"  {r['metric']:<22}{_fmt_num(r['baseline']):>10}"
                         f"{_fmt_num(r['current']):>10}{r['delta']:+.2f}")

    if result["skipped"]:
        names = ", ".join(s["metric"] for s in result["skipped"])
        lines.append(f"Skipped (null on one side): {names}")

    return "\n".join(lines)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_summary(runs_dir: Path = RUNS_DIR) -> Optional[Path]:
    sums = sorted(runs_dir.glob("*.summary.json"), key=lambda p: p.stat().st_mtime)
    return sums[-1] if sums else None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Regression gate vs eval/baseline.json.")
    p.add_argument("--summary", type=Path, default=None,
                   help="Summary to gate (default: latest in eval/runs).")
    p.add_argument("--baseline", type=Path, default=BASELINE,
                   help="Baseline file (default: eval/baseline.json).")
    args = p.parse_args(argv)

    summary_path = args.summary or latest_summary()
    if summary_path is None or not summary_path.exists():
        print("No summary found. Run `python eval/score.py` first.", file=sys.stderr)
        return 2
    if not args.baseline.exists():
        print(f"Baseline not found: {args.baseline}. "
              "Run `python eval/score.py --set-baseline` first.", file=sys.stderr)
        return 2

    current = _load_json(summary_path)
    baseline = _load_json(args.baseline)
    result = compare_to_baseline(current, baseline)

    print(render_report(result))
    print(f"\nsummary:  {summary_path}")
    print(f"baseline: {args.baseline}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""
eval/gate.py — Stage 3 of the eval harness (2.1.1.2, 2.2.1.2): the regression gate.

Loads the latest summary (or ``--summary <path>``) and a policy-specific
committed baseline (``eval/baselines/<policy>.json``, ``--policy strict|graded``)
and exits non-zero if any metric regresses beyond its tolerance. Run this
before/after each Phase 2 feature and (optionally) in CI:

    python eval/run_eval.py && python eval/score.py --policy graded && python eval/gate.py --policy graded

A metric regresses when:
  - higher-is-better (intent/ticker accuracy, retrieval hit-rate, keyword
    coverage, grounded_faithfulness, policy_compliance, answer_relevance):
    current < baseline - tolerance
  - lower-is-better  (refusal_rate): current > baseline + tolerance

Metrics that are ``null`` on either side (e.g. the LLM-judge was skipped) are
skipped, not failed — the gate never fails on missing data, only on real drops.

2.2.1.2 adds hard pre-metric checks (before any tolerance comparison is even
attempted) so a measurement-fidelity regression can never be masked by
comparable-looking numbers:

  - the current run's answer policy doesn't match the requested/baseline policy;
  - the evidence-trace or score schema version differs from the baseline's;
  - the dataset digest differs from the baseline's (different golden inputs);
  - any answerable model row in the current run lacks a complete evidence trace;
  - a judge metric's eligible-case denominator dropped versus the baseline.

Any of these fails the gate immediately, before regressions are compared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
BASELINES_DIR = EVAL_DIR / "baselines"

_REPO_ROOT = EVAL_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.middleware import evidence_trace  # noqa: E402
from eval import metrics as M  # noqa: E402

# Which direction is "better" for each gated metric.
DIRECTIONS = {
    "intent_accuracy": "higher",
    "ticker_accuracy": "higher",
    "retrieval_hit_rate": "higher",
    "keyword_coverage": "higher",
    "refusal_rate": "lower",
    "grounded_faithfulness": "higher",
    "policy_compliance": "higher",
    "answer_relevance": "higher",
}

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

# Phase 2.2 conversational-query acceptance thresholds (2.2.1.3 Step 5).
# Absolute floors/ceilings, not baseline-relative tolerances: each ``(direction,
# threshold)`` activates only when its metric block is present in the current
# summary (i.e. the run scored Phase 2.2 fixtures). "min" = score must be >=
# threshold; "max" = rate must be <= threshold.
PHASE22_THRESHOLDS = {
    "entity_carryover_accuracy": ("min", 0.90),
    "metric_carryover_accuracy": ("min", 0.90),
    "timeframe_carryover_accuracy": ("min", 0.90),
    "verbose_paraphrase_parity": ("min", 0.90),
    "compound_subquestion_coverage": ("min", 0.85),
    "stale_disclosure_rate": ("min", 1.00),
    "unanswerable_numeric_hallucination_rate": ("max", 0.00),
    "cross_session_leakage_rate": ("max", 0.00),
}


# Phase 2.2.3.4 adaptive-mode promotion gates (Step 5). The cross-config
# quality/latency deltas (plan/router accuracy, complex-query correctness,
# multi-turn Recall@10, single-turn nDCG@10, simple-query p95 latency) are
# computed from TWO run summaries (legacy vs adaptive) in RESULTS.md — a gate
# over a single summary can't see them. Documented here as the promotion
# contract; the machine-checkable subset that IS single-run is the budget
# safety invariant enforced by :func:`adaptive_safety_checks`.
PROMOTION_GATES = {
    "plan_router_accuracy_min": 0.90,
    "deterministic_route_accuracy_min": 0.95,
    "write_tool_routes_max": 0,
    "complex_correctness_improvement_pp_min": 10.0,
    "multiturn_recall_improvement_pp_min": 10.0,
    "singleturn_ndcg_max_regression": 0.02,
    "simple_p95_latency_max_regression_pct": 10.0,
    "max_subqueries": 3,
    "max_retrieval_rounds": 2,
    "max_planning_calls": 1,
    "max_rerank_calls": 1,
}

EVIDENCE_SUFFICIENCY_GATES = {
    "abstention_f1_min": 0.85,
    "unsupported_number_relative_reduction_min": 0.30,
    "unnecessary_retry_rate_max": 0.15,
    "retrieval_rounds_above_two_max": 0,
    "single_turn_correctness_max_regression": 0.02,
}

# Phase 2.2.4.3 citation-provenance / numeric-validation promotion gates
# (Step 5). ``citation_support_rate`` and ``accepted_absent_citations`` are
# single-run absolutes; the unsupported-number relative reduction is comparative
# (baseline vs current). The false-positive-on-non-claims and validator-p95
# gates are live-run measurements recorded in RESULTS.md, not single-summary.
CITATION_VALIDATION_GATES = {
    "citation_support_rate_min": 0.95,
    "accepted_absent_citations_max": 0,
    "numeric_unsupported_relative_reduction_min": 0.30,
}


def citation_validation_checks(current: dict, baseline: dict) -> list[str]:
    """Enforce the 2.2.4.3 citation/numeric absolute + comparative gates.

    Inert (returns ``[]``) unless the current summary carries a
    ``citation_validation`` block, so pre-2.2.4.3 and placeholder-baseline runs
    are unaffected.
    """
    block = current.get("citation_validation")
    if not isinstance(block, dict):
        return []
    failures: list[str] = []

    accepted_absent = block.get("accepted_absent_citations", 0)
    if accepted_absent:
        failures.append(
            f"{accepted_absent} citation(s) to ids absent from model-visible "
            "evidence were accepted (must be 0)")

    support = block.get("citation_support_rate")
    if support is not None and support < CITATION_VALIDATION_GATES["citation_support_rate_min"]:
        failures.append(
            f"citation_support_rate {support:.2f} < required minimum "
            f"{CITATION_VALIDATION_GATES['citation_support_rate_min']:.2f}")

    baseline_block = baseline.get("citation_validation") or {}
    current_unsupported = block.get("numeric_unsupported_rate")
    baseline_unsupported = baseline_block.get("numeric_unsupported_rate")
    if baseline_unsupported not in (None, 0) and current_unsupported is not None:
        reduction = (baseline_unsupported - current_unsupported) / baseline_unsupported
        if reduction < CITATION_VALIDATION_GATES["numeric_unsupported_relative_reduction_min"]:
            failures.append(
                f"unsupported-number relative reduction {reduction:.2f} < required "
                f"{CITATION_VALIDATION_GATES['numeric_unsupported_relative_reduction_min']:.2f}")
    return failures


def evidence_sufficiency_checks(current: dict, baseline: dict) -> list[str]:
    """Enforce the 2.2.4.1 absolute, safety, and comparative promotion gates."""
    block = current.get("evidence_sufficiency")
    if not isinstance(block, dict):
        return []
    failures: list[str] = []
    abstention = block.get("abstention_f1")
    if abstention is not None and abstention < EVIDENCE_SUFFICIENCY_GATES["abstention_f1_min"]:
        failures.append(
            f"abstention_f1 {abstention:.2f} < required minimum "
            f"{EVIDENCE_SUFFICIENCY_GATES['abstention_f1_min']:.2f}")
    retry_rate = block.get("unnecessary_retry_rate")
    if retry_rate is not None and retry_rate > EVIDENCE_SUFFICIENCY_GATES["unnecessary_retry_rate_max"]:
        failures.append(
            f"unnecessary_retry_rate {retry_rate:.2f} > allowed maximum "
            f"{EVIDENCE_SUFFICIENCY_GATES['unnecessary_retry_rate_max']:.2f}")
    above_two = (block.get("retrieval_rounds") or {}).get("above_two", 0)
    if above_two > EVIDENCE_SUFFICIENCY_GATES["retrieval_rounds_above_two_max"]:
        failures.append(
            f"retrieval rounds exceeded two in {above_two} request(s)")

    baseline_block = baseline.get("evidence_sufficiency") or {}
    current_unsupported = block.get("unsupported_number_rate")
    baseline_unsupported = baseline_block.get("unsupported_number_rate")
    if baseline_unsupported not in (None, 0) and current_unsupported is not None:
        reduction = (baseline_unsupported - current_unsupported) / baseline_unsupported
        if reduction < EVIDENCE_SUFFICIENCY_GATES["unsupported_number_relative_reduction_min"]:
            failures.append(
                f"unsupported-number relative reduction {reduction:.2f} < required "
                f"{EVIDENCE_SUFFICIENCY_GATES['unsupported_number_relative_reduction_min']:.2f}")

    current_correctness = _metric_value(current, "answer_relevance")
    baseline_correctness = _metric_value(baseline, "answer_relevance")
    if current_correctness is None or baseline_correctness is None:
        current_correctness = current.get("keyword_coverage")
        baseline_correctness = baseline.get("keyword_coverage")
    if (
        current_correctness is not None and baseline_correctness is not None
        and current_correctness < baseline_correctness
        - EVIDENCE_SUFFICIENCY_GATES["single_turn_correctness_max_regression"]
    ):
        failures.append(
            f"single-turn correctness regressed by "
            f"{baseline_correctness - current_correctness:.2f} (maximum 0.02)")
    return failures


def adaptive_safety_checks(current: dict) -> list[str]:
    """Single-run adaptive safety invariants (2.2.3.4 Step 5). Empty = clear.

    Activated only when the summary carries an ``adaptive`` block (a labeled /
    adaptive run). Enforces the two invariants observable within one run:
      - zero write-tool routes (the deterministic router must never route to a
        write tool);
      - no request exceeded a hard budget cap (subqueries/rounds/planning/rerank).
    The comparative promotion gates live in RESULTS.md over two summaries.
    """
    adaptive = current.get("adaptive")
    if not isinstance(adaptive, dict):
        return []
    failures: list[str] = []

    writes = adaptive.get("write_tool_routes", 0)
    if writes:
        failures.append(
            f"adaptive routed to a write tool {writes} time(s) "
            f"(required maximum {PROMOTION_GATES['write_tool_routes_max']})")

    violations = adaptive.get("budget_cap_violations") or {}
    for counter, count in violations.items():
        if count:
            failures.append(
                f"adaptive budget cap exceeded for {counter} in {count} request(s) "
                "(hard cap must never be exceeded)")
    return failures


def phase22_checks(current: dict) -> list[str]:
    """Phase 2.2 acceptance failures (2.2.1.3 Step 5). Empty list = all clear.

    For each Phase 2.2 metric present in the current summary:
      - zero eligible fixtures for a required category → fail (a missing
        category must never look like a pass);
      - eligible but unscored (score None) → skip (don't manufacture a failure
        from missing data — matches the null-metric placeholder baseline);
      - scored → enforce the absolute floor/ceiling.

    A metric block absent from the summary (a run with no Phase 2.2 fixtures at
    all) is not activated, so pre-2.2 single-turn runs are unaffected.
    """
    failures: list[str] = []
    for metric, (direction, threshold) in PHASE22_THRESHOLDS.items():
        block = current.get(metric)
        if not isinstance(block, dict):
            continue  # metric not present in this run → threshold not activated
        n_elig = block.get("n_eligible")
        if n_elig == 0:
            failures.append(
                f"{metric}: zero eligible fixtures (required category missing)")
            continue
        score = block.get("score")
        if score is None:
            continue  # eligible but unscored — don't fail on missing data
        if direction == "min" and score < threshold:
            failures.append(f"{metric} {score:.2f} < required minimum {threshold:.2f}")
        elif direction == "max" and score > threshold:
            failures.append(f"{metric} {score:.2f} > allowed maximum {threshold:.2f}")
    return failures


def _metric_value(summary: dict, name: str):
    """Extract a scalar metric value (judge metrics pull .score)."""
    if name in M.JUDGE_METRICS:
        block = summary.get(name)
        if isinstance(block, dict):
            return block.get("score")
        return block
    return summary.get(name)


def _eligible(summary: dict, name: str) -> Optional[int]:
    block = summary.get(name)
    return block.get("n_eligible") if isinstance(block, dict) else None


def pretrace_checks(current: dict, baseline: dict, *, policy: Optional[str] = None) -> list[str]:
    """Hard pre-metric failures (2.2.1.2 Step 5). Empty list = all clear."""
    failures: list[str] = []

    base_policy = baseline.get("answer_policy")
    cur_policy = current.get("answer_policy")
    if policy and base_policy and policy != base_policy:
        failures.append(
            f"policy mismatch: --policy {policy} requested but baseline is for {base_policy}")
    if cur_policy and base_policy and cur_policy != base_policy:
        failures.append(
            f"policy mismatch: current run answer_policy={cur_policy} vs "
            f"baseline answer_policy={base_policy}")

    cur_trace_v = current.get("trace_schema_version")
    base_trace_v = baseline.get("trace_schema_version")
    if cur_trace_v is not None and base_trace_v is not None and cur_trace_v != base_trace_v:
        failures.append(
            f"evidence-trace schema mismatch: current {cur_trace_v} vs baseline {base_trace_v}")

    cur_score_v = current.get("score_schema_version")
    base_score_v = baseline.get("score_schema_version")
    if cur_score_v is not None and base_score_v is not None and cur_score_v != base_score_v:
        failures.append(
            f"score schema mismatch: current {cur_score_v} vs baseline {base_score_v}")

    cur_digest = current.get("dataset_digest")
    base_digest = baseline.get("dataset_digest")
    if cur_digest and base_digest and cur_digest != base_digest:
        failures.append(
            f"dataset digest mismatch: current {cur_digest} vs baseline {base_digest} "
            "(golden inputs changed since the baseline was set)")

    n_trace_errors = current.get("n_trace_errors", 0)
    if n_trace_errors:
        failures.append(
            f"{n_trace_errors} answerable model row(s) lack a complete evidence trace "
            f"(see summary.trace_errors)")

    for metric in M.JUDGE_METRICS:
        if metric == "answer_relevance":
            continue  # not evidence-gated; eligibility tracks answer presence only
        cur_elig = _eligible(current, metric)
        base_elig = baseline.get(metric, {}).get("n_eligible") \
            if isinstance(baseline.get(metric), dict) else None
        if isinstance(cur_elig, int) and isinstance(base_elig, int) and cur_elig < base_elig:
            failures.append(
                f"{metric} eligible denominator dropped: current {cur_elig} < "
                f"baseline {base_elig}")

    return failures


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


def render_report(result: dict, pretrace_failures: Optional[list[str]] = None) -> str:
    """Human-readable pass/fail report."""
    lines = ["=== Regression gate ==="]

    if pretrace_failures:
        lines.append(f"RESULT: FAIL - {len(pretrace_failures)} pre-metric check(s) failed "
                     "(metrics were not compared).")
        lines.append("")
        lines.append("Pre-metric failures:")
        for f in pretrace_failures:
            lines.append(f"  - {f}")
        return "\n".join(lines)

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


def baseline_path(policy: str, baselines_dir: Optional[Path] = None) -> Path:
    """Reads the module-level ``BASELINES_DIR`` at call time (not as a bound
    default) so tests can monkeypatch it."""
    return (baselines_dir or BASELINES_DIR) / f"{policy}.json"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Regression gate vs a policy-specific baseline.")
    p.add_argument("--summary", type=Path, default=None,
                   help="Summary to gate (default: latest in eval/runs).")
    p.add_argument("--policy", choices=["strict", "graded"], default="graded",
                   help="Answer policy this run must match (default: graded).")
    p.add_argument("--baseline", type=Path, default=None,
                   help="Baseline file (default: eval/baselines/<policy>.json).")
    args = p.parse_args(argv)

    summary_path = args.summary or latest_summary()
    if summary_path is None or not summary_path.exists():
        print("No summary found. Run `python eval/score.py` first.", file=sys.stderr)
        return 2

    baseline_file = args.baseline or baseline_path(args.policy)
    if not baseline_file.exists():
        print(f"Baseline not found: {baseline_file}. "
              f"Run `python eval/score.py --set-baseline --policy {args.policy}` first.",
              file=sys.stderr)
        return 2

    current = _load_json(summary_path)
    baseline = _load_json(baseline_file)

    # current summaries produced before this run's score.py call may not carry
    # trace_schema_version — backfill from the live module so an old-but-
    # otherwise-valid summary isn't falsely flagged as a schema mismatch.
    current.setdefault("trace_schema_version", evidence_trace.SCHEMA_VERSION)
    current.setdefault("score_schema_version", M.SCORE_SCHEMA_VERSION)

    # Hard pre-metric failures: measurement-fidelity checks (2.2.1.2) plus the
    # Phase 2.2 acceptance thresholds (2.2.1.3). Either kind fails the gate
    # before baseline regressions are even compared.
    hard_failures = (pretrace_checks(current, baseline, policy=args.policy)
                     + phase22_checks(current)
                     + adaptive_safety_checks(current)
                     + evidence_sufficiency_checks(current, baseline)
                     + citation_validation_checks(current, baseline))
    if hard_failures:
        print(render_report({}, hard_failures))
        print(f"\nsummary:  {summary_path}")
        print(f"baseline: {baseline_file}")
        return 1

    result = compare_to_baseline(current, baseline)

    print(render_report(result))
    print(f"\nsummary:  {summary_path}")
    print(f"baseline: {baseline_file}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
tests/test_eval_harness.py
Phase 2.1.1 / 2.2.1.2 — Tests for the evaluation harness (golden dataset,
runner, metrics, scoring, and the regression gate).

Unit tests cover the dataset, metrics, and gate (pass + fail paths) with the
LLM-judge mocked — no live dependency. The live-runner test is marked
``integration``/``network`` and skips cleanly when the middleware on :8000 is
down.

2.2.1.2 replaced the single ``faithfulness`` metric (scored against a
truncated, re-retrieved ``context`` string) with ``grounded_faithfulness``
and ``policy_compliance`` (both scored against a row's exact
``evidence_trace``). Rows now carry ``evidence_trace``, ``trace_complete``,
``answer_policy``, ``grounding``, and ``dataset_digest``.
"""

import json
from pathlib import Path

import pytest

from eval import gate, metrics as M
from eval import run_eval as R
from eval import score

# ── Fixtures ───────────────────────────────────────────────────────────

GOLDEN = R.GOLDEN


def _trace(*, facts=None, documents=None, tool_results=None,
           system_prompt="SYSTEM", user_prompt="USER",
           answer_policy="graded", grounding_level="grounded",
           raw_question="q", retrieval_query="q") -> dict:
    """A minimal, structurally-complete evidence trace dict (2.2.1.2)."""
    return {
        "schema_version": 1,
        "answer_policy": answer_policy,
        "grounding_level": grounding_level,
        "raw_question": raw_question,
        "retrieval_query": retrieval_query,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "facts": facts if facts is not None else [
            {"metric": "revenue", "value": 26.0, "unit": "B", "period": "2026-Q1",
             "ticker": "NVDA", "source_type": "yfinance"},
        ],
        "documents": documents if documents is not None else [],
        "tool_results": tool_results if tool_results is not None else [],
    }


def _row(cid="c1", *, answer="Revenue was 26B. [Source: yfinance/NVDA]",
         detected_ticker="NVDA", detected_intent="fact_lookup",
         facts_used=3, documents_used=1, retrieved_sources=("yfinance",),
         context="Facts:\n- revenue: 26", model_available=True, error=None,
         evidence_trace=None, trace_complete=True, answer_policy="graded",
         grounding="grounded", dataset_digest="digest-1",
         expected_ticker="NVDA", expected_intent="fact_lookup",
         must_mention=("revenue",), expected_sources=("yfinance",),
         category="fact_lookup", question="q"):
    """Build a row with its case attached (matches run_eval output shape)."""
    if evidence_trace is None and model_available:
        evidence_trace = _trace(grounding_level=grounding or "grounded")
    return {
        "id": cid, "question": question, "answer": answer,
        "detected_ticker": detected_ticker, "detected_intent": detected_intent,
        "facts_used": facts_used, "documents_used": documents_used,
        "retrieved_sources": list(retrieved_sources), "context": context,
        "evidence_trace": evidence_trace, "trace_complete": trace_complete,
        "answer_policy": answer_policy, "grounding": grounding,
        "dataset_digest": dataset_digest,
        "latency_ms": 12.3, "model_available": model_available, "error": error,
        "case": {"id": cid, "question": question, "expected_ticker": expected_ticker,
                 "expected_intent": expected_intent, "must_mention": list(must_mention),
                 "expected_sources": list(expected_sources), "category": category},
    }


def _write_run(tmp_path: Path, rows: list[dict], name: str = "run.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


# ── Golden dataset ─────────────────────────────────────────────────────

class TestGoldenDataset:
    def test_golden_dataset_valid(self):
        lines = [
            line
            for line in GOLDEN.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) >= 30, f"expected >=30 cases, got {len(lines)}"
        ids = []
        for i, line in enumerate(lines, 1):
            obj = json.loads(line)  # raises on invalid JSON
            for key in ("id", "question", "category"):
                assert key in obj, f"line {i} missing {key}"
            ids.append(obj["id"])
        assert len(ids) == len(set(ids)), "duplicate ids in golden dataset"

    def test_golden_intents_covered(self):
        cases = R.load_cases()
        intents = {c.get("expected_intent") for c in cases}
        required = {"fact_lookup", "comparison", "trend", "explanation",
                    "sentiment", "news", "risk", "general"}
        missing = required - intents
        assert not missing, f"golden set missing intents: {missing}"

    def test_golden_expected_sources_canonical(self):
        """Every expected_source must be a known canonical name."""
        cases = R.load_cases()
        for c in cases:
            for s in c.get("expected_sources") or []:
                assert s in R.CANONICAL_SOURCES, (
                    f"{c['id']}: unknown expected_source {s!r}")

    def test_load_cases_returns_dicts(self):
        cases = R.load_cases()
        assert cases and isinstance(cases[0], dict)
        assert "id" in cases[0] and "question" in cases[0]

    def test_dataset_digest_deterministic(self):
        """The digest is stable across calls and sensitive to content changes."""
        cases = R.load_cases()
        d1 = R.dataset_digest(cases)
        d2 = R.dataset_digest(list(cases))  # new list, same content
        assert d1 == d2
        mutated = [dict(cases[0], question="different question")] + cases[1:]
        assert R.dataset_digest(mutated) != d1


# ── Runner ─────────────────────────────────────────────────────────────

def _middleware_up(url="http://127.0.0.1:8000/health", timeout=3):
    import httpx
    try:
        return httpx.get(url, timeout=timeout).status_code == 200
    except Exception:
        return False


class TestRunner:
    def test_result_keys_constant(self):
        # Every row produced by run_case must carry exactly these keys.
        assert "evidence_trace" in R.RESULT_KEYS
        assert "trace_complete" in R.RESULT_KEYS
        assert "answer_policy" in R.RESULT_KEYS
        assert "grounding" in R.RESULT_KEYS
        assert "dataset_digest" in R.RESULT_KEYS
        assert "retrieved_sources" in R.RESULT_KEYS

    def test_runner_injected_shapes(self):
        """An injected query_fn (no network) is shaped into a full result row."""
        trace = _trace()

        def fake_endpoint(case):
            return {"answer": "A", "detected_ticker": "NVDA",
                    "detected_intent": "fact_lookup", "facts_used": 3,
                    "documents_used": 1,
                    "citations": [{"source_type": "yfinance"}],
                    "model_available": True, "grounding": "grounded",
                    "evidence_trace": trace}
        row = R.run_case({"id": "x", "question": "q"}, query_fn=fake_endpoint,
                         allow_direct=False)
        assert all(k in row for k in R.RESULT_KEYS)
        assert row["answer"] == "A"
        assert row["retrieved_sources"] == ["yfinance"]
        assert row["facts_used"] == 3 and row["documents_used"] == 1
        assert row["model_available"] is True
        assert row["error"] is None
        assert row["evidence_trace"] == trace
        assert row["trace_complete"] is True
        assert row["answer_policy"] == "graded"
        assert row["grounding"] == "grounded"

    def test_runner_offline_fallback(self, monkeypatch):
        """When the endpoint is down and no fallback is allowed, the runner
        still emits a well-formed row (never raises)."""
        def boom(*a, **k):
            raise RuntimeError("endpoint down")
        monkeypatch.setattr(R, "call_live_endpoint", boom)
        row = R.run_case({"id": "x", "question": "q"},
                        use_live=True, allow_direct=False)
        assert all(k in row for k in R.RESULT_KEYS), row.keys()
        assert row["model_available"] is False
        assert row["error"] and "endpoint down" in row["error"]
        assert row["answer"] == ""
        assert row["retrieved_sources"] == []
        assert row["evidence_trace"] is None
        assert row["trace_complete"] is True  # no model answer -> nothing omitted

    def test_format_trace_context_includes_document_body(self):
        """A trace's document body (Chroma ``document`` naming) must render
        in the display context, not an empty excerpt (regression: 2.1.7.1
        found doc bodies silently dropped)."""
        trace = _trace(facts=[], documents=[{
            "id": "news/AMD/1",
            "document": "AMD beat revenue estimates for Q3 2026.",
            "metadata": {"source": "yfinance_news", "ticker": "AMD"},
        }])
        ctx = R._format_trace_context(trace)
        assert "AMD beat revenue estimates" in ctx
        # Older/injected shapes still work (legacy text alias).
        trace["documents"][0] = {"id": "d", "text": "TEXT BODY",
                                 "metadata": {"source": "sec", "ticker": "N"}}
        assert "TEXT BODY" in R._format_trace_context(trace)

    def test_runner_citation_sources_normalized(self):
        """Citation source types are normalized to canonical names."""
        def fake_endpoint(case):
            return {"answer": "x [Source: sec_10q/NVDA]",
                    "citations": [{"source_type": "sec_10q"}],
                    "model_available": True}
        row = R.run_case({"id": "x", "question": "q"}, query_fn=fake_endpoint,
                         allow_direct=False)
        assert row["retrieved_sources"] == ["sec"]

    def test_live_runner_uses_endpoint_trace_without_second_retrieval(self, monkeypatch):
        """2.2.1.2: the live path must never re-run retrieval to build judge
        context — everything comes from the endpoint's evidence_trace."""
        def boom(*a, **k):
            raise AssertionError("retrieval must not run on the live path")
        monkeypatch.setattr(R, "retrieve_for_question", boom)

        trace = _trace(
            facts=[{"metric": "total_revenue", "value": 26.0, "unit": "B",
                    "period": "2026-Q1", "ticker": "NVDA", "source_type": "yfinance"}],
            documents=[],
        )

        def fake_endpoint(case):
            return {"answer": "Revenue was 26B. [Source: yfinance/NVDA]",
                    "detected_ticker": "NVDA", "detected_intent": "fact_lookup",
                    "facts_used": 1, "documents_used": 0,
                    "citations": [{"source_type": "yfinance"}],
                    "model_available": True, "grounding": "grounded",
                    "evidence_trace": trace}

        row = R.run_case({"id": "x", "question": "q"}, query_fn=fake_endpoint,
                         allow_direct=False)
        assert row["evidence_trace"] == trace
        assert row["trace_complete"] is True
        assert row["error"] is None
        assert row["retrieved_sources"] == ["yfinance"]
        # context is derived purely from the trace, not a fresh retrieval.
        assert "total_revenue" in row["context"]

    def test_missing_trace_is_an_eval_error(self):
        """A model-produced answer without a complete evidence trace is an
        evaluation error, not silently scorable context (2.2.1.2)."""
        def fake_endpoint(case):
            return {"answer": "Revenue was 26B.", "model_available": True,
                    "detected_ticker": "NVDA", "detected_intent": "fact_lookup",
                    "facts_used": 3, "documents_used": 1, "grounding": "grounded"}
            # no evidence_trace key at all

        row = R.run_case({"id": "x", "question": "q"}, query_fn=fake_endpoint,
                         allow_direct=False)
        assert row["model_available"] is True
        assert row["evidence_trace"] is None
        assert row["trace_complete"] is False
        assert row["error"] is not None
        assert "trace" in row["error"].lower()

    @pytest.mark.integration
    @pytest.mark.network
    def test_runner_shapes_result(self):
        """Run one case against the live middleware; skip if :8000 is down."""
        if not _middleware_up():
            pytest.skip("middleware not running on :8000")
        cases = R.load_cases()
        row = R.run_case(cases[0], use_live=True, allow_direct=False)
        assert all(k in row for k in R.RESULT_KEYS)
        assert isinstance(row["facts_used"], int)
        assert isinstance(row["documents_used"], int)
        assert isinstance(row["latency_ms"], (int, float))
        assert isinstance(row["model_available"], bool)
        assert isinstance(row["retrieved_sources"], list)


# ── Metrics ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_refusal_rate(self):
        rows = [
            _row(cid="a", answer="Revenue was 26B.", model_available=True),
            _row(cid="b", answer="I do not have enough data to answer.", model_available=True),
            _row(cid="c", answer="Cannot answer this question.", model_available=True),
            _row(cid="d", answer="NVIDIA's EPS is 2.84.", model_available=True),
        ]
        assert M.refusal_rate(rows) == 0.5  # 2 of 4

    def test_refusal_rate_ignores_degraded(self):
        """A model-unavailable (degraded) answer is NOT counted as a refusal."""
        rows = [
            _row(cid="a", answer="I do not have enough data.", model_available=True),
            _row(cid="b", answer="Model unavailable — raw data", model_available=False,
                 evidence_trace=None),
        ]
        assert M.refusal_rate(rows) == 1.0  # 1 of 1 model-generated answers

    def test_retrieval_hit_rate(self):
        # matching source -> hit; non-matching -> miss; empty expected + counts -> hit
        rows = [
            _row(cid="a", expected_sources=("yfinance",), retrieved_sources=("yfinance",)),  # hit
            _row(cid="b", expected_sources=("fred",), retrieved_sources=("yfinance",), facts_used=3),  # miss
            _row(cid="c", expected_sources=(), retrieved_sources=(), facts_used=0, documents_used=0),  # miss (nothing)
            _row(cid="d", expected_sources=(), retrieved_sources=(), facts_used=2),  # hit (count fallback)
        ]
        assert M.retrieval_hit_rate(rows) == 0.5

    def test_keyword_coverage_partial(self):
        rows = [
            _row(cid="a", must_mention=("revenue",), answer="Revenue was 26B."),       # 1.0
            _row(cid="b", must_mention=("revenue", "growth"), answer="Revenue was 26B."),  # 0.5
            _row(cid="c", must_mention=("eps",), answer="No such token here."),        # 0.0
            _row(cid="d", must_mention=(), answer="anything"),  # excluded (empty must_mention)
        ]
        assert M.keyword_coverage(rows) == 0.5  # mean(1.0, 0.5, 0.0)

    def test_intent_accuracy(self):
        rows = [
            _row(cid="a", expected_intent="fact_lookup", detected_intent="fact_lookup"),  # hit
            _row(cid="b", expected_intent="fact_lookup", detected_intent="general"),    # miss
            _row(cid="c", expected_intent="comparison", detected_intent="comparison"),  # hit
        ]
        assert M.intent_accuracy(rows) == round(2 / 3, 4)

    def test_ticker_accuracy_none_equal(self):
        """None==None is a hit; a false-positive ticker is a miss."""
        rows = [
            _row(cid="a", expected_ticker="NVDA", detected_ticker="NVDA"),  # hit
            _row(cid="b", expected_ticker=None, detected_ticker=None),      # hit
            _row(cid="c", expected_ticker=None, detected_ticker="US"),      # miss (false positive)
        ]
        assert M.ticker_accuracy(rows) == round(2 / 3, 4)

    def test_parse_score_formats(self):
        assert M._parse_score("blah\nScore: 0.83") == 0.83
        assert M._parse_score("Score: 8") == 0.8          # 0-10 scale normalized
        assert M._parse_score("Score: 0.95 extra") == 0.95
        assert M._parse_score("no number") is None
        assert M._parse_score("Score: 1.5") is None      # >1 and >10 → invalid
        assert M._parse_score(None) is None

    def test_trace_errors_reported(self):
        rows = [
            _row(cid="a", model_available=True, trace_complete=True),
            _row(cid="b", model_available=True, trace_complete=False,
                 evidence_trace=None, error="incomplete evidence trace"),
            _row(cid="c", model_available=False, trace_complete=True,
                 evidence_trace=None),
        ]
        errs = M.trace_errors(rows)
        assert [e["id"] for e in errs] == ["b"]


# ── LLM-judge (mocked) ──────────────────────────────────────────────────

class TestJudge:
    def test_grounded_faithfulness_mocked(self):
        calls = []

        def fake_judge(prompt):
            calls.append(prompt)
            return "Looks grounded. Score: 0.9"

        rows = [_row(cid="a", grounding="grounded"),
                _row(cid="b", grounding="partial")]
        res = M.grounded_faithfulness(rows, judge=fake_judge)
        assert res["score"] == 0.9
        assert res["n_scored"] == 2
        assert res["n_eligible"] == 2
        assert res["model_available"] is True
        assert len(calls) == 2

    def test_grounded_faithfulness_skips_empty_answer(self):
        rows = [_row(cid="a", answer="", grounding="grounded")]
        res = M.grounded_faithfulness(rows, judge=lambda p: "Score: 0.5")
        assert res["score"] is None
        assert res["n_eligible"] == 0
        assert res["n_skipped"] == 1

    def test_grounded_faithfulness_model_down(self):
        """When the judge client returns None, the score degrades to None."""
        rows = [_row(cid="a", grounding="grounded")]
        res = M.grounded_faithfulness(rows, judge=lambda p: None)
        assert res["score"] is None
        assert res["model_available"] is False
        assert res["n_eligible"] == 1
        assert res["n_skipped"] == 1

    def test_refusal_does_not_raise_grounded_faithfulness(self):
        """2.2.1.2: refused answers never enter the grounded_faithfulness
        denominator, so a refusal can't inflate the score."""
        rows = [
            _row(cid="a", answer="I don't have enough data to answer this.",
                 grounding="refused"),
            _row(cid="b", answer="Revenue was 26B [Source: yfinance/NVDA]",
                 grounding="grounded"),
        ]
        res = M.grounded_faithfulness(rows, judge=lambda p: "Score: 1.0")
        assert res["n_eligible"] == 1
        assert res["n_scored"] == 1
        refused_case = [pc for pc in res["per_case"] if pc["id"] == "a"][0]
        assert refused_case["score"] is None
        assert "refused" in refused_case["reason"]

    def test_general_fallback_scores_policy_compliance_not_grounded_faithfulness(self):
        """2.2.1.2: a labeled general-knowledge answer is out of scope for
        grounded_faithfulness (it claims no grounding) but is still judged
        by policy_compliance (labeling/caveat/no-fabrication rules)."""
        rows = [
            _row(cid="a",
                 answer="Not from your data - general knowledge: buybacks reduce "
                        "share count.\n\nPlease verify against a primary source "
                        "before relying on it.",
                 grounding="general", evidence_trace=_trace(facts=[], documents=[])),
        ]
        gf = M.grounded_faithfulness(rows, judge=lambda p: "Score: 1.0")
        assert gf["n_eligible"] == 0
        reason = gf["per_case"][0]["reason"]
        assert "general" in reason

        pc = M.policy_compliance(rows, judge=lambda p: "Score: 0.9")
        assert pc["n_eligible"] == 1
        assert pc["n_scored"] == 1
        assert pc["score"] == 0.9

    def test_tool_grounded_answer_is_judged_with_tool_evidence(self):
        """2.2.1.2: grounded_faithfulness must judge against tool results,
        not just facts/documents — the prompt sent to the judge must contain
        the exact tool evidence."""
        trace = _trace(facts=[], documents=[], tool_results=[
            {"name": "query_facts",
             "arguments": {"metric": "forward_pe", "order": "asc", "limit": 1},
             "result": {"metric": "forward_pe",
                        "results": [{"ticker": "META", "value": 15.9}]}},
        ])
        rows = [_row(cid="a", answer="META has the lowest forward P/E. [Source: sqlite/META]",
                     grounding="grounded", evidence_trace=trace)]
        captured = []

        def fake_judge(prompt):
            captured.append(prompt)
            return "Score: 0.95"

        res = M.grounded_faithfulness(rows, judge=fake_judge)
        assert res["n_eligible"] == 1
        assert res["score"] == 0.95
        assert "query_facts" in captured[0]
        assert "META" in captured[0]

    def test_grounded_faithfulness_missing_trace_never_faithful(self):
        rows = [_row(cid="a", model_available=True, trace_complete=False,
                     evidence_trace=None, grounding="grounded")]
        res = M.grounded_faithfulness(rows, judge=lambda p: "Score: 1.0")
        assert res["score"] is None
        assert res["n_eligible"] == 0
        reason = res["per_case"][0]["reason"]
        assert "trace" in reason

    def test_policy_compliance_excludes_degraded_answers(self):
        rows = [_row(cid="a", model_available=False, evidence_trace=None,
                     answer="⚠️ Model unavailable — showing raw retrieved data")]
        res = M.policy_compliance(rows, judge=lambda p: "Score: 1.0")
        assert res["n_eligible"] == 0
        assert res["score"] is None

    def test_relevance_mocked(self):
        rows = [_row(cid="a"), _row(cid="b")]
        res = M.answer_relevance(rows, judge=lambda p: "Relevant. Score: 0.8")
        assert res["score"] == 0.8
        assert res["n_scored"] == 2
        assert res["n_eligible"] == 2

    def test_score_all_shape(self):
        rows = [_row(cid="a"), _row(cid="b", answer="I do not have enough data.",
                                    grounding="refused")]
        s = M.score_all(rows, judge=lambda p: "Score: 0.7", run_judge=True)
        for k in ("intent_accuracy", "ticker_accuracy", "retrieval_hit_rate",
                 "keyword_coverage", "refusal_rate", "grounded_faithfulness",
                 "policy_compliance", "answer_relevance", "per_category", "n_cases",
                 "n_trace_errors", "answer_policy", "dataset_digest", "denominators"):
            assert k in s
        assert s["policy_compliance"]["score"] == 0.7
        assert s["answer_relevance"]["score"] == 0.7
        assert s["answer_policy"] == "graded"
        assert s["dataset_digest"] == "digest-1"

    def test_score_all_no_judge(self):
        rows = [_row(cid="a")]
        s = M.score_all(rows, run_judge=False)
        assert s["grounded_faithfulness"]["score"] is None
        assert s["policy_compliance"]["score"] is None
        assert s["answer_relevance"]["score"] is None


# ── Gate ────────────────────────────────────────────────────────────────

def _baseline_from(summary):
    """Promote a summary to a baseline (with tolerances)."""
    return {**summary, "tolerances": dict(score.DEFAULT_TOLERANCES)}


class TestGate:
    def _summary(self, **overrides):
        s = {
            "answer_policy": "graded",
            "dataset_digest": "digest-1",
            "trace_schema_version": 1,
            "score_schema_version": 1,
            "n_trace_errors": 0,
            "intent_accuracy": 0.90, "ticker_accuracy": 0.85,
            "retrieval_hit_rate": 0.80, "keyword_coverage": 0.70,
            "refusal_rate": 0.20,
            "grounded_faithfulness": {"score": 0.83, "n_scored": 10, "n_eligible": 10,
                                      "n_skipped": 0, "model_available": True, "per_case": []},
            "policy_compliance": {"score": 0.88, "n_scored": 12, "n_eligible": 12,
                                  "n_skipped": 0, "model_available": True, "per_case": []},
            "answer_relevance": {"score": 0.79, "n_scored": 10, "n_eligible": 10,
                                 "n_skipped": 0, "model_available": True, "per_case": []},
        }
        s.update(overrides)
        return s

    def test_gate_passes_on_baseline(self):
        baseline = self._summary()
        current = self._summary()  # identical
        res = gate.compare_to_baseline(current, baseline)
        assert res["passed"] is True
        assert res["regressions"] == []

    def test_gate_fails_on_regression(self):
        baseline = self._summary()
        current = self._summary(
            grounded_faithfulness={"score": 0.50, "n_scored": 10, "n_eligible": 10,
                                   "n_skipped": 0, "model_available": True, "per_case": []},
            intent_accuracy=0.40)
        res = gate.compare_to_baseline(current, baseline)
        assert res["passed"] is False
        regressed = {r["metric"] for r in res["regressions"]}
        assert "grounded_faithfulness" in regressed   # -0.33 > tol 0.05
        assert "intent_accuracy" in regressed  # -0.50 > tol 0.05

    def test_gate_lower_is_better(self):
        """refusal_rate rising beyond tolerance is a regression."""
        baseline = self._summary(refusal_rate=0.20)
        current = self._summary(refusal_rate=0.30)  # +0.10 > 0.05
        res = gate.compare_to_baseline(current, baseline)
        assert res["passed"] is False
        assert any(r["metric"] == "refusal_rate" for r in res["regressions"])

    def test_gate_within_tolerance_passes(self):
        """A small drop within tolerance is not a regression."""
        baseline = self._summary(intent_accuracy=0.90)
        current = self._summary(intent_accuracy=0.87)  # -0.03 < 0.05
        res = gate.compare_to_baseline(current, baseline)
        assert res["passed"] is True

    def test_gate_skips_null(self):
        """Null judge scores on either side are skipped, not failed."""
        baseline = self._summary(
            grounded_faithfulness={"score": None, "n_scored": 0, "n_eligible": 0,
                                   "n_skipped": 10, "model_available": False, "per_case": []})
        current = self._summary()  # grounded_faithfulness 0.83
        res = gate.compare_to_baseline(current, baseline)
        assert res["passed"] is True  # grounded_faithfulness skipped (baseline null)
        assert any(s["metric"] == "grounded_faithfulness" for s in res["skipped"])

    def test_gate_fails_when_metric_unavailable(self):
        """If the baseline scored a metric but the current run could not (e.g.
        the model went down), the gate fails — a model-down run can't pass."""
        baseline = self._summary()  # grounded_faithfulness 0.83
        current = self._summary(
            grounded_faithfulness={"score": None, "n_scored": 0, "n_eligible": 0,
                                   "n_skipped": 42, "model_available": False, "per_case": []})
        res = gate.compare_to_baseline(current, baseline)
        assert res["passed"] is False
        reg = {r["metric"]: r for r in res["regressions"]}
        assert "grounded_faithfulness" in reg
        assert reg["grounded_faithfulness"]["current"] is None  # unavailable vs baseline

    def test_gate_rejects_policy_dataset_or_schema_mismatch(self):
        baseline = self._summary()

        policy_mismatch = self._summary(answer_policy="strict")
        failures = gate.pretrace_checks(policy_mismatch, baseline, policy="graded")
        assert any("policy mismatch" in f for f in failures)

        digest_mismatch = self._summary(dataset_digest="different-digest")
        failures = gate.pretrace_checks(digest_mismatch, baseline, policy="graded")
        assert any("dataset digest mismatch" in f for f in failures)

        schema_mismatch = self._summary(trace_schema_version=99)
        failures = gate.pretrace_checks(schema_mismatch, baseline, policy="graded")
        assert any("schema mismatch" in f for f in failures)

        # A fully matching current summary has no pre-metric failures.
        assert gate.pretrace_checks(self._summary(), baseline, policy="graded") == []

    def test_gate_rejects_lower_eligible_denominator(self):
        baseline = self._summary(
            grounded_faithfulness={"score": 0.8, "n_scored": 10, "n_eligible": 10,
                                   "n_skipped": 0, "model_available": True, "per_case": []})
        current = self._summary(
            grounded_faithfulness={"score": 0.8, "n_scored": 5, "n_eligible": 5,
                                   "n_skipped": 0, "model_available": True, "per_case": []})
        failures = gate.pretrace_checks(current, baseline, policy="graded")
        assert any("eligible denominator dropped" in f for f in failures)

    def test_gate_rejects_incomplete_traces(self):
        baseline = self._summary()
        current = self._summary(n_trace_errors=2)
        failures = gate.pretrace_checks(current, baseline, policy="graded")
        assert any("complete evidence trace" in f for f in failures)

    def test_gate_cli_exit_code(self, tmp_path, monkeypatch):
        """The CLI returns 0 on pass, 1 on regression."""
        baseline = tmp_path / "baseline.json"
        base_summary = self._summary()
        # set_baseline writes the file (and returns the path, not the JSON).
        score.set_baseline(base_summary, path=baseline)
        assert baseline.exists()

        # Identical summary -> exit 0
        ok = tmp_path / "ok.summary.json"
        ok.write_text(json.dumps(self._summary()), encoding="utf-8")
        assert gate.main(["--summary", str(ok), "--baseline", str(baseline)]) == 0

        # Degraded summary -> exit 1
        bad = tmp_path / "bad.summary.json"
        bad.write_text(json.dumps(self._summary(
            grounded_faithfulness={"score": 0.50, "n_scored": 10, "n_eligible": 10,
                                   "n_skipped": 0, "model_available": True, "per_case": []})),
            encoding="utf-8")
        assert gate.main(["--summary", str(bad), "--baseline", str(baseline)]) == 1


# ── Score CLI ──────────────────────────────────────────────────────────

class TestScoreCLI:
    def test_score_main_no_judge(self, tmp_path, monkeypatch):
        rows = [_row(cid="a"), _row(cid="b", answer="I do not have enough data.",
                                    grounding="refused")]
        run_path = _write_run(tmp_path, rows)
        rc = score.main(["--run", str(run_path), "--no-judge"])
        assert rc == 0
        summary_path = run_path.with_suffix(".summary.json")
        assert summary_path.exists()
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "intent_accuracy" in data and "per_category" in data
        assert data["grounded_faithfulness"]["score"] is None  # judge skipped
        assert data["policy_compliance"]["score"] is None

    def test_score_main_report(self, tmp_path, monkeypatch):
        rows = [_row(cid="a")]
        run_path = _write_run(tmp_path, rows)
        # Keep latest_summary() from picking up real runs by pointing RUNS_DIR at tmp.
        monkeypatch.setattr(score, "RUNS_DIR", tmp_path)
        rc = score.main(["--run", str(run_path), "--no-judge", "--report"])
        assert rc == 0
        assert run_path.with_suffix(".report.md").exists()

    def test_set_baseline_writes_tolerances(self, tmp_path):
        """set_baseline writes a baseline file with tolerances + metrics (tmp path)."""
        rows = [_row(cid="a")]
        _write_run(tmp_path, rows)
        summary = M.score_all(rows, run_judge=False)
        base_path = tmp_path / "baseline.json"
        out = score.set_baseline(summary, path=base_path)
        assert out == base_path and base_path.exists()
        data = json.loads(base_path.read_text(encoding="utf-8"))
        assert data["is_baseline"] is True
        assert "tolerances" in data and data["tolerances"]["grounded_faithfulness"] == 0.05
        assert "intent_accuracy" in data
        assert data["answer_policy"] == "graded"
        assert data["trace_schema_version"] == 1
        assert "eligible_counts" in data

    def test_set_baseline_policy_selects_path(self, tmp_path, monkeypatch):
        """--policy (or an explicit ``policy=``) selects eval/baselines/<policy>.json
        when no explicit path is given."""
        monkeypatch.setattr(score, "BASELINES_DIR", tmp_path)
        rows = [_row(cid="a")]
        summary = M.score_all(rows, run_judge=False)
        out = score.set_baseline(summary, policy="strict")
        assert out == tmp_path / "strict.json"
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["answer_policy"] == "strict"


# ══════════════════════════════════════════════════════════════════════
# 2.2.1.3 — Conversational & compound-query golden set
# ══════════════════════════════════════════════════════════════════════


def _crow(cid="t0", *, conversation_id="convA", turn_index=0, answer="ok",
          detected_intent="fact_lookup", resolved_tickers=(), resolved_metrics=(),
          resolved_timeframe=None, expected_tickers=(), expected_metrics=(),
          expected_timeframe=None, expected_carryover=None, subquestions=None,
          paraphrase_group=None, requires_stale_disclosure=False,
          answerability=None, category="conversation"):
    """Build a conversation-turn result row for the deterministic metric tests.

    Carries both the row-level resolved/expected fields and an attached ``case``
    so ``metrics._case(row)`` sees the fixture expectations."""
    case = {
        "id": cid, "question": "q", "category": category,
        "expected_tickers": list(expected_tickers),
        "expected_metrics": list(expected_metrics),
        "expected_timeframe": expected_timeframe,
        "expected_carryover": expected_carryover,
        "subquestions": subquestions or [],
        "paraphrase_group": paraphrase_group,
        "requires_stale_disclosure": requires_stale_disclosure,
        "answerability": answerability,
    }
    return {
        "id": cid, "answer": answer, "detected_intent": detected_intent,
        "conversation_id": conversation_id, "turn_index": turn_index,
        "resolved_tickers": list(resolved_tickers),
        "resolved_metrics": list(resolved_metrics),
        "resolved_timeframe": resolved_timeframe,
        "expected_tickers": list(expected_tickers),
        "case": case,
    }


# ── Fixtures: conversation file + challenge-set minimums ────────────────

class TestConversationFixtures:
    def test_conversation_file_valid(self):
        convs = R.load_conversations()
        assert convs, "no conversations loaded"
        for cv in convs:
            assert cv.get("id") and cv.get("turns"), cv
            for t in cv["turns"]:
                assert t.get("id") and t.get("question"), t

    def test_all_ids_unique_across_files(self):
        cases = R.load_cases()
        convs = R.load_conversations()
        assert R.fixture_id_collisions(cases, convs) == []

    def test_challenge_set_minimums(self):
        """The validation report meets every per-dimension minimum (2.2.1.3)."""
        cases = R.load_cases()
        convs = R.load_conversations()
        counts, problems = R.validate_fixtures(cases, convs)
        assert problems == [], problems
        assert counts["conversations"] >= 12
        assert counts["conversation_turns"] >= 30
        assert counts["paraphrase_pairs"] >= 6
        assert counts["compound"] >= 6
        assert counts["multi_ticker"] >= 6
        assert counts["timeframe"] >= 6
        assert counts["stale"] >= 4
        assert counts["unanswerable"] >= 4

    def test_validate_fixtures_flags_missing_dimension(self):
        """A thinned-out set is reported, not silently accepted."""
        counts, problems = R.validate_fixtures([], [])
        assert problems  # every minimum unmet
        assert any("conversations" in p for p in problems)

    def test_validate_fixtures_flags_id_collision(self):
        cases = [{"id": "dup", "question": "q", "category": "x"}]
        convs = [{"id": "dup", "turns": [{"id": "t", "question": "q"}]}]
        assert "dup" in R.fixture_id_collisions(cases, convs)


# ── Runner: sequential + interleaved history ────────────────────────────

class TestConversationRunner:
    def test_sequential_history_is_runner_owned(self):
        """History starts empty and grows one completed turn at a time."""
        seen = []

        def qfn(case, history):
            seen.append([h["question"] for h in history])
            return {"answer": f"a-{case['id']}", "model_available": True}

        conv = {"id": "c", "category": "follow_up", "turns": [
            {"id": "c0", "question": "q0"},
            {"id": "c1", "question": "q1"},
            {"id": "c2", "question": "q2"}]}
        rows = R.run_conversation(conv, query_fn=qfn, allow_direct=False)
        assert seen == [[], ["q0"], ["q0", "q1"]]
        assert [r["history_sent"] for r in rows] == [0, 1, 2]
        assert [r["turn_index"] for r in rows] == [0, 1, 2]
        assert all(r["conversation_id"] == "c" for r in rows)
        assert all(k in rows[0] for k in R.RESULT_KEYS)

    def test_interleaved_conversations_never_cross(self):
        """Two conversations driven in interleaved order never see each other's
        turns — histories are runner-owned per conversation (2.2.1.3 Step 3)."""
        seen = {}

        def qfn(case, history):
            seen[case["id"]] = [h["question"] for h in history]
            return {"answer": f"a-{case['id']}", "model_available": True}

        convA = {"id": "A", "turns": [{"id": "A0", "question": "A q0"},
                                      {"id": "A1", "question": "A q1"},
                                      {"id": "A2", "question": "A q2"}]}
        convB = {"id": "B", "turns": [{"id": "B0", "question": "B q0"},
                                      {"id": "B1", "question": "B q1"}]}
        histA, histB = [], []
        order = [(convA, 0, histA), (convB, 0, histB), (convA, 1, histA),
                 (convB, 1, histB), (convA, 2, histA)]
        for conv, i, hist in order:
            turn = conv["turns"][i]
            tc = R._turn_case(conv, turn, i, "cat")
            row = R.run_case(tc, query_fn=qfn, allow_direct=False,
                             history=list(hist), conversation_id=conv["id"],
                             turn_index=i)
            hist.append({"question": turn["question"], "answer": row["answer"]})

        assert seen["A0"] == []
        assert seen["A1"] == ["A q0"]
        assert seen["A2"] == ["A q0", "A q1"]
        assert seen["B0"] == []
        assert seen["B1"] == ["B q0"]
        # No conversation ever saw the other's questions.
        for k in ("A0", "A1", "A2"):
            assert all("B" not in q for q in seen[k])
        for k in ("B0", "B1"):
            assert all("A" not in q for q in seen[k])

    def test_mixed_single_and_conversation_artifacts(self):
        """A run mixing single-turn cases and conversation turns yields rows of
        one shape; single-turn rows carry no conversation position."""
        def qfn_single(case):
            return {"answer": "a", "model_available": True}

        def qfn_turn(case, history):
            return {"answer": "a", "model_available": True}

        single = R.run_case({"id": "s1", "question": "q"}, query_fn=qfn_single,
                            allow_direct=False)
        conv = {"id": "cv", "category": "follow_up",
                "turns": [{"id": "cv0", "question": "q0"},
                          {"id": "cv1", "question": "q1"}]}
        turns = R.run_conversation(conv, query_fn=qfn_turn, allow_direct=False)
        rows = [single] + turns
        assert all(all(k in r for k in R.RESULT_KEYS) for r in rows)
        assert single["conversation_id"] is None and single["turn_index"] is None
        assert turns[0]["conversation_id"] == "cv" and turns[1]["turn_index"] == 1

    def test_resolved_fields_derived_from_trace(self):
        """With no explicit resolved_* fields, the runner derives them from the
        evidence trace + detected ticker (forward-compatible with 2.2.2)."""
        trace = _trace(
            facts=[{"metric": "gross_margin", "value": 0.7, "unit": "ratio",
                    "period": "2026-Q2", "ticker": "NVDA", "source_type": "yfinance"}],
            documents=[])

        def qfn(case):
            return {"answer": "a", "model_available": True,
                    "detected_ticker": "NVDA", "detected_intent": "fact_lookup",
                    "evidence_trace": trace}

        row = R.run_case({"id": "x", "question": "q"}, query_fn=qfn, allow_direct=False)
        assert row["resolved_tickers"] == ["NVDA"]
        assert row["resolved_metrics"] == ["gross_margin"]
        assert row["resolved_timeframe"] == "2026-Q2"

    def test_explicit_resolved_fields_preferred(self):
        """An explicit resolved_* field wins over trace derivation."""
        def qfn(case):
            return {"answer": "a", "model_available": True,
                    "resolved_tickers": ["AMD"], "resolved_metrics": ["total_revenue"],
                    "resolved_timeframe": "FY2025"}

        row = R.run_case({"id": "x", "question": "q"}, query_fn=qfn, allow_direct=False)
        assert row["resolved_tickers"] == ["AMD"]
        assert row["resolved_metrics"] == ["total_revenue"]
        assert row["resolved_timeframe"] == "FY2025"


# ── Deterministic conversational metrics ────────────────────────────────

class TestConversationalMetrics:
    def test_entity_carryover_accuracy(self):
        rows = [
            _crow("a", expected_carryover={"ticker": "NVDA"}, resolved_tickers=["NVDA"]),
            _crow("b", expected_carryover={"ticker": "META"}, resolved_tickers=["AAPL"]),
            _crow("c", expected_carryover={"metrics": ["eps"]}, resolved_tickers=["X"]),
        ]
        res = M.entity_carryover_accuracy(rows)
        assert res["n_eligible"] == 2  # only ticker-carry turns
        assert res["score"] == 0.5

    def test_metric_carryover_accuracy_subset(self):
        rows = [
            _crow("a", expected_carryover={"metrics": ["total_revenue"]},
                  resolved_metrics=["total_revenue", "gross_margin"]),  # superset -> pass
            _crow("b", expected_carryover={"metrics": ["net_income"]},
                  resolved_metrics=["total_revenue"]),  # missing -> fail
        ]
        res = M.metric_carryover_accuracy(rows)
        assert res["n_eligible"] == 2
        assert res["score"] == 0.5

    def test_timeframe_carryover_accuracy(self):
        rows = [
            _crow("a", expected_carryover={"timeframe": "2026-Q2"},
                  resolved_timeframe="2026-Q2"),  # pass (normalized equal)
            _crow("b", expected_carryover={"timeframe": "FY2025"},
                  resolved_timeframe="2026-Q1"),  # fail
        ]
        res = M.timeframe_carryover_accuracy(rows)
        assert res["score"] == 0.5

    def test_carryover_no_eligible_returns_none(self):
        rows = [_crow("a")]  # declares no carryover
        res = M.entity_carryover_accuracy(rows)
        assert res["n_eligible"] == 0 and res["score"] is None

    def test_verbose_paraphrase_parity(self):
        same = [
            _crow("v", paraphrase_group="g1", resolved_tickers=["NVDA"],
                  resolved_metrics=["total_revenue"], resolved_timeframe="2026-Q2"),
            _crow("c", paraphrase_group="g1", resolved_tickers=["NVDA"],
                  resolved_metrics=["total_revenue"], resolved_timeframe="2026-Q2"),
        ]
        assert M.verbose_paraphrase_parity(same)["score"] == 1.0
        diff = [
            _crow("v", paraphrase_group="g2", resolved_metrics=["total_revenue"]),
            _crow("c", paraphrase_group="g2", resolved_metrics=["gross_margin"]),
        ]
        res = M.verbose_paraphrase_parity(diff)
        assert res["n_eligible"] == 1 and res["score"] == 0.0

    def test_compound_subquestion_coverage(self):
        subs = [{"id": "s1", "must_mention": ["revenue"]},
                {"id": "s2", "must_mention": ["gross margin"]}]
        full = _crow("a", subquestions=subs, answer="revenue up, gross margin steady")
        partial = _crow("b", subquestions=subs, answer="revenue up only")
        res = M.compound_subquestion_coverage([full, partial])
        assert res["n_eligible"] == 2
        assert res["score"] == 0.75  # mean(1.0, 0.5)
        assert res["n_pass"] == 1    # only the fully-covered row

    def test_stale_disclosure_rate(self):
        rows = [
            _crow("a", requires_stale_disclosure=True,
                  answer="Revenue was strong (data may be out of date)."),
            _crow("b", requires_stale_disclosure=True, answer="Revenue was strong."),
            _crow("c", requires_stale_disclosure=False, answer="ignored"),
        ]
        res = M.stale_disclosure_rate(rows)
        assert res["n_eligible"] == 2
        assert res["score"] == 0.5

    def test_unanswerable_numeric_hallucination_rate(self):
        rows = [
            _crow("a", answerability="unanswerable",
                  answer="I don't have that data; cannot answer."),  # clean
            _crow("b", answerability="unanswerable",
                  answer="Their revenue was about $5.2 billion."),  # invented figure
            _crow("c", answerability="unanswerable",
                  answer="No data for the fourth quarter of 2026."),  # year/quarter ok
        ]
        res = M.unanswerable_numeric_hallucination_rate(rows)
        assert res["n_eligible"] == 3
        assert res["score"] == round(1 / 3, 4)
        assert res["n_pass"] == 2

    def test_cross_session_leakage_rate(self):
        rows = [
            # convA legit = {NVDA}; convB legit = {AMD}
            _crow("a0", conversation_id="A", expected_tickers=["NVDA"],
                  resolved_tickers=["NVDA"]),  # clean
            _crow("b0", conversation_id="B", expected_tickers=["AMD"],
                  resolved_tickers=["NVDA"]),  # leaked A's ticker into B
        ]
        res = M.cross_session_leakage_rate(rows)
        assert res["n_eligible"] == 2
        assert res["score"] == 0.5

    def test_cross_session_leakage_zero_when_clean(self):
        rows = [
            _crow("a0", conversation_id="A", expected_tickers=["NVDA"],
                  resolved_tickers=["NVDA"]),
            _crow("b0", conversation_id="B", expected_tickers=["AMD"],
                  resolved_tickers=["AMD"]),
        ]
        assert M.cross_session_leakage_rate(rows)["score"] == 0.0

    def test_financial_figure_and_staleness_helpers(self):
        assert M.contains_financial_figure("about $5.2 billion")
        assert M.contains_financial_figure("margin of 42.5%")
        assert M.contains_financial_figure("trades at 15.9x earnings")
        assert not M.contains_financial_figure("in the fourth quarter of 2026")
        assert not M.contains_financial_figure("no data available")
        assert M.discloses_staleness("This may be out of date.")
        assert not M.discloses_staleness("Revenue grew strongly.")

    def test_score_all_includes_conversational_when_present(self):
        rows = [_crow("a", expected_carryover={"ticker": "NVDA"},
                      resolved_tickers=["NVDA"])]
        s = M.score_all(rows, run_judge=False)
        for m in M.PHASE22_METRICS:
            assert m in s
        assert "conversational_denominators" in s
        assert s["entity_carryover_accuracy"]["score"] == 1.0

    def test_score_all_omits_conversational_when_absent(self):
        rows = [_row(cid="a")]  # plain single-turn, no Phase 2.2 fields
        s = M.score_all(rows, run_judge=False)
        assert "entity_carryover_accuracy" not in s
        assert "conversational_denominators" not in s


# ── Phase 2.2 acceptance gate ───────────────────────────────────────────

class TestPhase22Gate:
    def _block(self, score, n_eligible=5):
        return {"score": score, "n_eligible": n_eligible, "n_pass": n_eligible}

    def _passing(self):
        return {
            "entity_carryover_accuracy": self._block(0.90),
            "metric_carryover_accuracy": self._block(0.95),
            "timeframe_carryover_accuracy": self._block(0.92),
            "verbose_paraphrase_parity": self._block(1.0),
            "compound_subquestion_coverage": self._block(0.85),
            "stale_disclosure_rate": self._block(1.0),
            "unanswerable_numeric_hallucination_rate": self._block(0.0),
            "cross_session_leakage_rate": self._block(0.0),
        }

    def test_phase22_pass_at_thresholds(self):
        assert gate.phase22_checks(self._passing()) == []

    def test_phase22_fail_below_min(self):
        s = self._passing()
        s["entity_carryover_accuracy"] = self._block(0.89)  # < 0.90
        failures = gate.phase22_checks(s)
        assert any("entity_carryover_accuracy" in f for f in failures)

    def test_phase22_fail_above_max(self):
        s = self._passing()
        s["cross_session_leakage_rate"] = self._block(0.05)  # > 0.00
        failures = gate.phase22_checks(s)
        assert any("cross_session_leakage_rate" in f for f in failures)

    def test_phase22_fail_zero_eligible(self):
        s = self._passing()
        s["unanswerable_numeric_hallucination_rate"] = {
            "score": None, "n_eligible": 0, "n_pass": 0}
        failures = gate.phase22_checks(s)
        assert any("zero eligible fixtures" in f for f in failures)

    def test_phase22_eligible_but_unscored_is_not_a_failure(self):
        s = self._passing()
        s["stale_disclosure_rate"] = {"score": None, "n_eligible": 4, "n_pass": 0}
        assert gate.phase22_checks(s) == []

    def test_phase22_absent_metrics_not_activated(self):
        """A run with no Phase 2.2 fixtures (no metric blocks) is unaffected."""
        assert gate.phase22_checks({"intent_accuracy": 0.9}) == []

    def test_gate_main_fails_on_phase22(self, tmp_path):
        """The gate CLI returns non-zero when a Phase 2.2 threshold is missed,
        even against an all-null placeholder baseline."""
        baseline = tmp_path / "baseline.json"
        base = {"answer_policy": "graded", "dataset_digest": "d",
                "trace_schema_version": 1, "score_schema_version": 1,
                "n_trace_errors": 0}
        score.set_baseline(base, path=baseline)
        summary = tmp_path / "s.summary.json"
        s = dict(base)
        s.update(self._passing())
        s["cross_session_leakage_rate"] = self._block(0.5)  # leak -> fail
        summary.write_text(json.dumps(s), encoding="utf-8")
        assert gate.main(["--summary", str(summary), "--baseline", str(baseline)]) == 1

    def test_gate_main_passes_with_good_phase22(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        base = {"answer_policy": "graded", "dataset_digest": "d",
                "trace_schema_version": 1, "score_schema_version": 1,
                "n_trace_errors": 0}
        score.set_baseline(base, path=baseline)
        summary = tmp_path / "s.summary.json"
        s = dict(base)
        s.update(self._passing())
        summary.write_text(json.dumps(s), encoding="utf-8")
        assert gate.main(["--summary", str(summary), "--baseline", str(baseline)]) == 0


# ── 2.2.2.2: compiled query + carried slots on run rows ────────────────────

class TestCarriedSlotCapture:
    """A conversation turn's compiled query and resolved slots are captured on
    the run row from the endpoint response, and drive the carryover metric."""

    def _endpoint_data(self):
        return {
            "answer": "NVDA revenue grew. [Source: yfinance/NVDA]",
            "detected_ticker": "NVDA",
            "detected_intent": "explanation",
            "facts_used": 1,
            "documents_used": 0,
            "model_available": True,
            "grounding": "grounded",
            "evidence_trace": _trace(
                raw_question="Why did it grow?",
                retrieval_query="NVDA total revenue FY2025",
            ),
            # 2.2.2.2 response metadata:
            "retrieval_query": "NVDA total revenue FY2025",
            "carried_context": {
                "entities": ["NVDA"], "metrics": ["total_revenue"],
                "timeframe": "fy2025", "topic_reset": False,
                "ambiguous_slots": [], "resolution_sources": ["history"],
            },
            "resolved_tickers": ["NVDA"],
            "resolved_metrics": ["total_revenue"],
            "resolved_timeframe": "fy2025",
        }

    def test_row_captures_compiled_query_and_resolved_slots(self):
        case = {"id": "conv#1", "question": "Why did it grow?",
                "expected_carryover": {"ticker": "NVDA", "metrics": ["total_revenue"]}}
        row = R._row_from_endpoint(
            case, self._endpoint_data(), 0.1, dataset_digest="d",
            ctx={"conversation_id": "conv", "turn_index": 1, "history_sent": 2})
        # Raw question and compiled retrieval query are distinct fields.
        assert row["raw_question"] == "Why did it grow?"
        assert row["retrieval_query"] == "NVDA total revenue FY2025"
        assert row["resolved_tickers"] == ["NVDA"]
        assert row["resolved_metrics"] == ["total_revenue"]

    def test_captured_slots_score_carryover(self):
        case = {"id": "conv#1", "question": "Why did it grow?",
                "expected_carryover": {"ticker": "NVDA", "metrics": ["total_revenue"]}}
        row = R._row_from_endpoint(
            case, self._endpoint_data(), 0.1, dataset_digest="d",
            ctx={"conversation_id": "conv", "turn_index": 1, "history_sent": 2})
        ent = M.entity_carryover_accuracy([row])
        met = M.metric_carryover_accuracy([row])
        assert ent["n_eligible"] == 1 and ent["score"] == 1.0
        assert met["n_eligible"] == 1 and met["score"] == 1.0


# ── Phase 2.2.3.4 — adaptive orchestration eval scaffolding ─────────────

def _orch_row(cid, *, lane="standard", fallback_reason=None, reason_codes=(),
              tools=(), subqueries=1, rounds=1, planning=0, rerank=0,
              config_label="adaptive_tools_rerank", **row_kwargs):
    """A run row carrying an orchestration block (matches run_eval._row shape)."""
    row = _row(cid, **row_kwargs)
    orch = {
        "lane": lane, "reason_codes": list(reason_codes),
        "subqueries_executed": subqueries, "retrieval_rounds": rounds,
        "planning_calls": planning, "reranker_calls": rerank,
        "deterministic_tools": list(tools), "context_chars": 500,
        "evidence_dropped": 0, "fallback_reason": fallback_reason,
    }
    row["orchestration"] = orch
    row["lane"] = lane
    row["fallback_reason"] = fallback_reason
    row["config_label"] = config_label
    return row


class TestAdaptiveMetrics:
    def test_score_all_omits_adaptive_when_absent(self):
        s = M.score_all([_row("c1"), _row("c2")], run_judge=False)
        assert "adaptive" not in s
        assert "config_label" not in s

    def test_adaptive_metrics_present_and_shaped(self):
        rows = [
            _orch_row("c1", lane="fast", tools=("get_fundamentals",),
                      subqueries=1, rounds=0),
            _orch_row("c2", lane="standard", subqueries=1, rounds=1),
            _orch_row("c3", lane="complex", subqueries=3, rounds=2, rerank=1,
                      reason_codes=("subquery_budget_exhausted",)),
            _orch_row("c4", lane="standard", fallback_reason="adaptive_error"),
        ]
        s = M.score_all(rows, run_judge=False)

        assert s["config_label"] == "adaptive_tools_rerank"
        a = s["adaptive"]
        assert a["n_adaptive"] == 4
        assert a["lane_distribution"] == {"fast": 1, "standard": 2, "complex": 1}
        assert a["fallback_rate"]["rate"] == 0.25
        assert a["budget_exhaustion_rate"]["n_exhausted"] == 1
        assert a["write_tool_routes"] == 0
        assert a["budget_cap_violations"] == {
            "subqueries_executed": 0, "retrieval_rounds": 0,
            "planning_calls": 0, "reranker_calls": 0}
        assert set(a["per_lane"]) == {"fast", "standard", "complex"}
        assert a["max_counters"]["subqueries_executed"] == 3

    def test_config_label_only_run_emits_adaptive_block(self):
        # A labeled legacy run (no orchestration) still gets the config_label so
        # the three-config comparison can line up the legacy baseline.
        rows = [_row("c1"), _row("c2")]
        for r in rows:
            r["config_label"] = "legacy"
        s = M.score_all(rows, run_judge=False)
        assert s["config_label"] == "legacy"
        assert s["adaptive"]["n_adaptive"] == 0

    def test_write_tool_route_and_cap_violation_detected(self):
        rows = [
            _orch_row("c1", tools=("refresh_data",)),          # write route
            _orch_row("c2", subqueries=4),                      # exceeds cap 3
        ]
        assert M.write_tool_route_count(rows) == 1
        assert M.budget_cap_violations(rows)["subqueries_executed"] == 1


class TestAdaptiveGate:
    def test_safety_checks_pass_when_clean(self):
        rows = [_orch_row("c1", lane="fast"), _orch_row("c2", lane="standard")]
        summary = M.score_all(rows, run_judge=False)
        assert gate.adaptive_safety_checks(summary) == []

    def test_safety_checks_flag_write_and_budget(self):
        rows = [
            _orch_row("c1", tools=("refresh_data",)),
            _orch_row("c2", rounds=3),
        ]
        summary = M.score_all(rows, run_judge=False)
        failures = gate.adaptive_safety_checks(summary)
        assert any("write tool" in f for f in failures)
        assert any("budget cap" in f for f in failures)

    def test_safety_checks_inert_without_adaptive_block(self):
        assert gate.adaptive_safety_checks({"n_cases": 2}) == []


class TestEvidenceSufficiencyMetrics:
    @staticmethod
    def _suff_row(
        row_id, *, expected_covered=(), covered=(), should_abstain=False,
        status="grounded", retry=False, simple=True, unsupported=0,
        numerical=0, rounds=1, latency=10.0,
    ):
        row = _row(row_id)
        row["case"].update({
            "expected_covered_subqueries": list(expected_covered),
            "should_abstain": should_abstain,
            "complexity": "simple" if simple else "complex",
        })
        row.update({
            "grounding": status,
            "latency_ms": latency,
            "unsupported_number_count": unsupported,
            "numerical_claim_count": numerical,
            "evidence_sufficiency": {
                "status": status,
                "covered_subqueries": list(covered),
                "missing_subqueries": [],
                "reason_codes": [],
                "corrective_action": "none",
                "retry_performed": retry,
            },
            "orchestration": {"retrieval_rounds": rounds},
        })
        return row

    def test_metrics_measure_coverage_retry_abstention_numbers_latency_and_rounds(self):
        rows = [
            self._suff_row("a", expected_covered=("sq0",), covered=("sq0",),
                           numerical=2, latency=10, rounds=1),
            self._suff_row("b", expected_covered=("sq0", "sq1"), covered=("sq0",),
                           retry=True, simple=True, unsupported=1, numerical=2,
                           latency=30, rounds=2),
            self._suff_row("c", should_abstain=True, status="refused",
                           simple=False, latency=20, rounds=1),
        ]

        block = M.evidence_sufficiency_metrics(rows)

        assert block["coverage_precision"] == 1.0
        assert block["coverage_recall"] == 0.6667
        assert block["unnecessary_retry_rate"] == 0.5
        assert block["abstention_f1"] == 1.0
        assert block["unsupported_number_rate"] == 0.25
        assert block["latency_ms"] == {"mean": 20.0, "p95": 30.0}
        assert block["retrieval_rounds"] == {"mean": 1.333, "max": 2,
                                               "above_two": 0}

    def test_score_all_emits_block_only_when_feature_metadata_is_present(self):
        assert "evidence_sufficiency" not in M.score_all([_row("legacy")], run_judge=False)
        row = self._suff_row("enabled", expected_covered=("sq0",), covered=("sq0",))
        assert "evidence_sufficiency" in M.score_all([row], run_judge=False)


class TestEvidenceSufficiencyGate:
    def test_absolute_and_relative_promotion_gates(self):
        baseline = {"evidence_sufficiency": {"unsupported_number_rate": 0.20},
                    "keyword_coverage": 0.90}
        passing = {"evidence_sufficiency": {
            "abstention_f1": 0.85,
            "unsupported_number_rate": 0.14,
            "unnecessary_retry_rate": 0.15,
            "retrieval_rounds": {"above_two": 0},
        }, "keyword_coverage": 0.88}
        assert gate.evidence_sufficiency_checks(passing, baseline) == []

        failing = {"evidence_sufficiency": {
            "abstention_f1": 0.80,
            "unsupported_number_rate": 0.19,
            "unnecessary_retry_rate": 0.20,
            "retrieval_rounds": {"above_two": 1},
        }, "keyword_coverage": 0.85}
        failures = gate.evidence_sufficiency_checks(failing, baseline)
        assert len(failures) == 5


class TestDecompositionMetrics:
    @staticmethod
    def _decomp_row(row_id, *, proposed=0, derived=0, drift=(), complexity="complex",
                    subqueries_executed=1, rounds=1):
        row = _row(row_id)
        row["case"]["complexity"] = complexity
        row["decomposition"] = {
            "proposed_subqueries": proposed,
            "derived_subqueries": derived,
            "drift_reason_codes": list(drift),
        }
        row["orchestration"] = {
            "subqueries_executed": subqueries_executed,
            "retrieval_rounds": rounds,
        }
        return row

    def test_measures_drift_rate_and_derived_counts(self):
        rows = [
            self._decomp_row("a", proposed=2, derived=2, rounds=2,
                             subqueries_executed=3),
            self._decomp_row("b", proposed=2, derived=1,
                             drift=["drift_invented_entity"], rounds=2),
        ]
        block = M.decomposition_metrics(rows)
        assert block["n_eligible"] == 2
        assert block["proposed_subqueries"] == 4
        assert block["accepted_subqueries"] == 3
        assert block["drift_rate"] == 0.25          # 1 of 4 proposed rejected
        assert block["max_derived_subqueries"] == 2
        assert block["drift_reason_codes"] == {"drift_invented_entity": 1}
        assert block["subquery_cap_violations"] == 0
        assert block["retrieval_round_cap_violations"] == 0

    def test_flags_simple_query_with_derived_and_cap_violations(self):
        rows = [
            self._decomp_row("s", proposed=1, derived=1, complexity="simple"),
            self._decomp_row("v", proposed=1, derived=1, subqueries_executed=4,
                             rounds=3),
        ]
        block = M.decomposition_metrics(rows)
        assert block["n_simple"] == 1
        assert block["simple_with_derived"] == 1           # invariant violated
        assert block["subquery_cap_violations"] == 1       # 4 > 3
        assert block["retrieval_round_cap_violations"] == 1  # 3 > 2

    def test_score_all_emits_block_only_when_present(self):
        assert "decomposition" not in M.score_all([_row("legacy")], run_judge=False)
        row = self._decomp_row("d", proposed=2, derived=2)
        assert "decomposition" in M.score_all([row], run_judge=False)

    def test_row_from_endpoint_captures_decomposition(self):
        case = {"id": "c1", "question": "NVDA revenue and risk"}
        data = {
            "answer": "ok", "model_available": True, "evidence_trace": _trace(),
            "decomposition": {
                "proposed_subqueries": 2, "derived_subqueries": 1,
                "drift_reason_codes": ["drift_new_number"]},
        }
        row = R._row_from_endpoint(case, data, latency_s=0.02)
        assert row["decomposition"]["derived_subqueries"] == 1
        assert row["decomposition"]["drift_reason_codes"] == ["drift_new_number"]


class TestRunEvalOrchestrationCapture:
    def test_row_from_endpoint_captures_orchestration(self):
        case = {"id": "c1", "question": "What is NVDA revenue?",
                "expected_ticker": "NVDA", "expected_intent": "fact_lookup"}
        data = {
            "answer": "Revenue was 26B.", "detected_ticker": "NVDA",
            "detected_intent": "fact_lookup", "facts_used": 3,
            "documents_used": 1, "citations": [], "model_available": True,
            "grounding": "grounded", "evidence_trace": _trace(),
            "orchestration": {
                "lane": "fast", "reason_codes": ["lane_fast"],
                "subqueries_executed": 1, "retrieval_rounds": 0,
                "planning_calls": 0, "reranker_calls": 0,
                "deterministic_tools": ["get_fundamentals"],
                "context_chars": 400, "evidence_dropped": 0,
                "fallback_reason": None},
            "evidence_sufficiency": {
                "status": "grounded", "reason_codes": [],
                "covered_subqueries": ["sq0"], "missing_subqueries": [],
                "corrective_action": "none", "retry_performed": False,
            },
        }
        row = R._row_from_endpoint(case, data, latency_s=0.05)

        assert row["lane"] == "fast"
        assert row["fallback_reason"] is None
        assert row["orchestration"]["deterministic_tools"] == ["get_fundamentals"]
        assert row["evidence_sufficiency"]["covered_subqueries"] == ["sq0"]
        # config_label defaults None until main() denormalizes the run label.
        assert row["config_label"] is None


# ── Phase 2.2.4.3 — citation provenance / numeric validation eval ───────────

class TestCitationValidationMetrics:
    @staticmethod
    def _val_row(row_id, **av):
        row = _row(row_id)
        row["answer_validation"] = {
            "validation_status": av.get("status", "supported"),
            "citation_support_rate": av.get("support_rate", 1.0),
            "numeric_claims_supported": av.get("supported", 0),
            "numeric_claims_unsupported": av.get("unsupported", 0),
            "numeric_claims_ambiguous": av.get("ambiguous", 0),
            "citations_total": av.get("cit_total", 0),
            "citations_resolved": av.get("cit_resolved", 0),
            "citations_missing": av.get("cit_missing", 0),
            "citations_malformed": av.get("cit_malformed", 0),
            "mismatch_counts": av.get(
                "mismatch", {"unit": 0, "period": 0, "entity": 0, "value": 0}),
            "enforcement": av.get("enforcement", "none"),
        }
        return row

    def test_metrics_measure_citation_and_numeric_support(self):
        rows = [
            self._val_row("a", supported=3, unsupported=0, cit_total=3,
                          cit_resolved=3, support_rate=1.0),
            self._val_row("b", supported=1, unsupported=1, cit_total=2,
                          cit_resolved=1, cit_missing=1, support_rate=0.5,
                          enforcement="downgrade",
                          mismatch={"unit": 1, "period": 0, "entity": 0, "value": 1}),
        ]
        block = M.citation_validation_metrics(rows)
        assert block["n_eligible"] == 2
        assert block["numeric_claims_supported"] == 4
        assert block["numeric_claims_unsupported"] == 1
        assert block["numeric_support_rate"] == round(4 / 5, 4)
        assert block["numeric_unsupported_rate"] == round(1 / 5, 4)
        assert block["citation_support_rate"] == 0.75          # mean(1.0, 0.5)
        assert block["citation_existence_rate"] == 0.8         # 4 resolved / 5 resolvable
        assert block["accepted_absent_citations"] == 0
        assert block["downgrade_rate"] == 0.5
        assert block["mismatch_counts"]["unit"] == 1

    def test_score_all_emits_block_only_when_present(self):
        assert "citation_validation" not in M.score_all([_row("legacy")], run_judge=False)
        row = self._val_row("x", supported=1, cit_total=1, cit_resolved=1)
        assert "citation_validation" in M.score_all([row], run_judge=False)

    def test_row_from_endpoint_captures_answer_validation(self):
        case = {"id": "c1", "question": "NVDA revenue"}
        data = {
            "answer": "Revenue was $26B [E1].", "model_available": True,
            "evidence_trace": _trace(),
            "answer_validation": {
                "validation_status": "supported", "numeric_claims_total": 1,
                "numeric_claims_unsupported": 0},
            "evidence_citations": [{"evidence_id": "E1", "support_status": "supported"}],
        }
        row = R._row_from_endpoint(case, data, latency_s=0.02)
        assert row["answer_validation"]["validation_status"] == "supported"
        # Bridges into the 2.2.4.1 sufficiency unsupported-number rate.
        assert row["numerical_claim_count"] == 1
        assert row["unsupported_number_count"] == 0


class TestCitationValidationGate:
    def test_absolute_and_relative_gates(self):
        baseline = {"citation_validation": {"numeric_unsupported_rate": 0.20}}
        passing = {"citation_validation": {
            "citation_support_rate": 0.96,
            "accepted_absent_citations": 0,
            "numeric_unsupported_rate": 0.10,
        }}
        assert gate.citation_validation_checks(passing, baseline) == []

        failing = {"citation_validation": {
            "citation_support_rate": 0.90,          # < 0.95
            "accepted_absent_citations": 2,          # must be 0
            "numeric_unsupported_rate": 0.18,        # only 10% relative reduction
        }}
        failures = gate.citation_validation_checks(failing, baseline)
        assert len(failures) == 3

    def test_inert_without_block(self):
        assert gate.citation_validation_checks({"n_cases": 1}, {}) == []


# ── Long-document / hierarchical retrieval eval (2.2.5.3) ───────────────

class TestLongDocumentEval:
    """Deterministic flat-vs-hierarchical comparison over the fixture corpus."""

    def test_corpus_covers_seven_case_types(self):
        corpus = R.load_hierarchical_corpus()
        types = {c["case_type"] for c in corpus["cases"]}
        required = {
            "exact_child", "heading_plus_child", "adjacent_table",
            "two_distant_sections", "fact_plus_explanation",
            "annual_and_quarterly", "missing_evidence", "conflicting_evidence",
        }
        assert required <= types

    def test_recall_primitive(self):
        assert M.recall_at_k(["a", "b", "c"], ["b"], 5) == 1.0
        assert M.recall_at_k(["a", "b"], ["c"], 5) == 0.0
        assert M.recall_at_k(["a"], ["a", "b"], 5) == 0.5
        # No relevant evidence → excluded from the recall denominator.
        assert M.recall_at_k(["a"], [], 5) is None

    def test_context_precision_primitive(self):
        assert M.context_precision(["a", "b"], ["a"]) == 0.5
        assert M.context_precision([], ["a"]) is None

    def test_hierarchical_beats_flat_recall(self):
        res = R.evaluate_long_document_configs()
        s = res["summary"]
        assert s["hierarchical"]["recall_at_10"] >= s["flat"]["recall_at_10"]
        # Bounded expansion actually fired on the context-requiring cases.
        assert s["hierarchical"]["expansion_count"] > 0

    def test_hierarchical_cheaper_than_large_k(self):
        s = R.evaluate_long_document_configs()["summary"]
        # Reaches larger-top-k recall without the larger-top-k prompt cost.
        assert s["hierarchical"]["prompt_chars"] <= s["flat_large_k"]["prompt_chars"]

    def test_hierarchical_precision_not_worse(self):
        s = R.evaluate_long_document_configs()["summary"]
        assert (s["hierarchical"]["context_precision"]
                >= s["flat_large_k"]["context_precision"])

    def test_promotion_gate_passes_on_fixture(self):
        res = R.evaluate_long_document_configs()
        gate_result = res["gate"]
        assert gate_result["passed"], gate_result["checks"]

    def test_reports_all_scalar_metrics(self):
        res = R.evaluate_long_document_configs()
        for config in ("flat", "flat_large_k", "hierarchical"):
            block = res["summary"][config]
            for key in M.LONG_DOC_SCALARS:
                assert key in block

    def test_unanswerable_case_flags_no_fabricated_numbers(self):
        # The missing-evidence case retrieves nothing relevant, so its clean
        # (no-fabrication) count stays low; hierarchical never invents figures.
        res = R.evaluate_long_document_configs()
        rows = [r for r in res["per_case"]
                if r["id"] == "hd-missing-evidence" and r["config"] == "hierarchical"]
        assert rows and rows[0]["unsupported_numeric_claims"] == 0

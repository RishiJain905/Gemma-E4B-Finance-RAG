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

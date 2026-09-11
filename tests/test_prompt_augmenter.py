# tests/test_prompt_augmenter.py
# Unit tests for PromptAugmenter — Phase 1.5.4


class TestPromptAugmenter:
    """Tests for the PromptAugmenter module."""

    def test_import(self):
        """PromptAugmenter imports successfully."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        assert PromptAugmenter is not None

    def test_build_prompt_with_facts(self):
        """Prompt includes structured facts with citations."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="What is NVDA revenue?",
            intent={"ticker": "NVDA", "metrics": ["total_revenue"],
                    "question_type": "fact_lookup"},
            retrieval={
                "facts": [
                    {"metric": "total_revenue", "value": 26.0, "unit": "billion_usd",
                     "period": "2026-Q1", "source_type": "sec_10q"},
                ],
                "documents": [],
                "ticker": "NVDA",
            },
        )

        assert "Retrieved Financial Facts" in prompt
        assert "total_revenue" in prompt
        assert "26.0" in prompt
        assert "billion_usd" in prompt
        assert "Source: sec_10q/NVDA" in prompt
        assert "User Question" in prompt
        assert "What is NVDA revenue?" in prompt
        assert "Output Format" in prompt

    def test_build_prompt_with_documents(self):
        """Prompt includes document context with metadata."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="What is the outlook for NVDA?",
            intent={"ticker": "NVDA", "question_type": "sentiment"},
            retrieval={
                "facts": [],
                "documents": [
                    {
                        "id": "sec_10k/NVDA/10-K-2025",
                        "text": "NVIDIA reported strong growth in datacenter revenue.",
                        "ticker": "NVDA",
                        "source": "10-K",
                        "date": "2025-03-15",
                        "metadata": {"ticker": "NVDA", "source": "10-K"},
                    },
                ],
                "ticker": "NVDA",
            },
        )

        assert "Retrieved Documents" in prompt
        assert "NVIDIA reported" in prompt
        assert "Document 1" in prompt
        assert "User Question" in prompt

    def test_document_field_body_is_rendered(self):
        """A retriever-shaped row containing only "document" (Chroma naming)
        renders verbatim — the confirmed P0 evidence-loss bug (2.2.1.1)."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="What is the outlook for NVDA?",
            intent={"ticker": "NVDA", "question_type": "sentiment"},
            retrieval={
                "facts": [],
                "documents": [
                    {
                        "id": "sec_10k/NVDA/10-K-2025",
                        "document": "NVIDIA datacenter revenue grew sharply.",
                        "metadata": {"ticker": "NVDA", "source": "10-K"},
                    },
                ],
                "ticker": "NVDA",
            },
        )

        assert "Retrieved Documents" in prompt
        assert "NVIDIA datacenter revenue grew sharply." in prompt

    def test_legacy_text_alias_remains_supported(self):
        """An injected "text" row still renders during the compatibility
        window — document_body() accepts it as a legacy alias."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="What is the outlook for NVDA?",
            intent={"ticker": "NVDA", "question_type": "sentiment"},
            retrieval={
                "facts": [],
                "documents": [
                    {
                        "id": "doc-1",
                        "text": "Legacy text-field body for compatibility.",
                        "metadata": {"ticker": "NVDA", "source": "10-K"},
                    },
                ],
                "ticker": "NVDA",
            },
        )

        assert "Legacy text-field body for compatibility." in prompt

    def test_blank_document_never_renders_empty_section(self):
        """Metadata-only rows with a blank/whitespace body must not produce
        an empty "## Retrieved Documents" section."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="What is the outlook for AAPL?",
            intent={"ticker": "AAPL", "question_type": "sentiment"},
            retrieval={
                "facts": [],
                "documents": [
                    {"id": "doc-1", "document": "   ", "metadata": {"ticker": "AAPL"}},
                ],
                "ticker": "AAPL",
            },
        )

        assert "Retrieved Documents" not in prompt
        assert "No data was found" in prompt

    def test_multi_ticker_fact_provenance_preserved(self):
        """Each fact keeps its own ticker/period/source/unit citation; the
        request-level ticker is only a fallback and must not relabel a fact
        from a different ticker (comparison retrieval mixes tickers)."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="Compare NVDA and AMD revenue.",
            intent={"ticker": "NVDA", "question_type": "comparison"},
            retrieval={
                "facts": [
                    {"metric": "total_revenue", "value": 26.0, "unit": "billion_usd",
                     "period": "2026-Q1", "source_type": "sec_10q", "ticker": "NVDA"},
                    {"metric": "total_revenue", "value": 5.8, "unit": "billion_usd",
                     "period": "2026-Q1", "source_type": "sec_10q", "ticker": "AMD"},
                ],
                "documents": [],
                "ticker": "NVDA",
            },
        )

        assert "[Source: sec_10q/NVDA]" in prompt
        assert "[Source: sec_10q/AMD]" in prompt

    def test_build_prompt_empty_retrieval(self):
        """Empty retrieval generates a 'no data found' note."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="What about AAPL?",
            intent={"ticker": "AAPL", "question_type": "general"},
            retrieval={"facts": [], "documents": [], "ticker": "AAPL"},
        )

        assert "No data was found" in prompt
        assert "AAPL" in prompt

    def test_prompt_excludes_duplicated_policy_rules(self):
        """The augmented user prompt carries retrieved evidence, intent-
        specific task guidance, the question, and output format — the
        authoritative answer-policy rules (grounding modes, "ONLY the
        provided context", etc.) live solely in the system message built by
        prompt_policy.py, not duplicated here (2.2.1.1 step 4)."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="Test?",
            intent={"ticker": None, "question_type": "general"},
            retrieval={"facts": [], "documents": [], "ticker": None},
        )

        assert "## Instructions" in prompt
        assert "User Question" in prompt
        assert "Output Format" in prompt
        assert "financial research assistant" not in prompt
        assert "Answer using ONLY the provided context" not in prompt

    def test_question_type_instructions(self):
        """Question-type-specific instructions are included."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        for qtype in ("fact_lookup", "comparison", "trend", "explanation",
                      "sentiment", "news", "risk"):
            prompt = augmenter.build_prompt(
                question="Test?",
                intent={"ticker": "NVDA", "question_type": qtype},
                retrieval={"facts": [], "documents": [], "ticker": "NVDA"},
            )
            assert "## Instructions" in prompt

    def test_truncation(self):
        """Very long prompts are truncated gracefully."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        long_docs = [
            {
                "id": f"doc_{i}",
                "text": "X" * 5000,
                "ticker": "NVDA",
                "source": "10-K",
                "metadata": {"ticker": "NVDA", "source": "10-K"},
            }
            for i in range(10)
        ]

        prompt = augmenter.build_prompt(
            question="Test?",
            intent={"ticker": "NVDA", "question_type": "general"},
            retrieval={"facts": [], "documents": long_docs, "ticker": "NVDA"},
        )

        truncated = augmenter.truncate_if_needed(prompt, max_tokens=2000)
        assert len(truncated) < len(prompt)
        assert "truncated for length" in truncated or len(truncated) < len(prompt)

    def test_estimate_tokens(self):
        """Token estimation is reasonable."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()
        text = "Hello world, this is a test " * 100
        estimate = augmenter.estimate_tokens(text)
        assert estimate > 0
        assert estimate < len(text)  # Should be less than char count

    def test_long_or_short_requires_classify_trade_bias(self):
        """Long vs short questions tell the model it must call classify_trade_bias."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()
        prompt = augmenter.build_prompt(
            question="is NVDA a long or short trade",
            intent={"ticker": "NVDA", "question_type": "general"},
            retrieval={
                "facts": [{"metric": "recommendation_mean", "value": 1.8,
                           "ticker": "NVDA", "source_type": "estimates"}],
                "documents": [],
                "ticker": "NVDA",
            },
        )
        assert "classify_trade_bias" in prompt
        assert "long, short, or neutral" in prompt


# ── Evidence ledger header (2.2.4.3) ───────────────────────────────────────

class TestEvidenceLedgerHeader:
    """The [E#] evidence header renders only when a ledger is supplied."""

    def _augmenter(self):
        from types import SimpleNamespace
        from src.middleware.prompt_augmenter import PromptAugmenter
        return PromptAugmenter(config=SimpleNamespace(answer_policy="graded"))

    def _retrieval(self):
        return {
            "facts": [{"metric": "total_revenue", "value": 26.0,
                       "unit": "billion_usd", "period": "2026-Q1",
                       "source_type": "sec_10q", "ticker": "NVDA"}],
            "documents": [{"id": "sec_10k/NVDA/2025",
                           "document": "NVIDIA datacenter revenue grew.",
                           "metadata": {"ticker": "NVDA", "source": "sec_10k"}}],
            "ticker": "NVDA",
        }

    def test_no_ledger_means_no_header(self):
        """Without a ledger the prompt is the legacy prompt (no ## Evidence)."""
        prompt = self._augmenter().build_prompt(
            question="What is NVDA revenue?",
            intent={"ticker": "NVDA", "question_type": "fact_lookup"},
            retrieval=self._retrieval(),
        )
        assert "## Evidence\n" not in prompt
        assert "[E1]" not in prompt

    def test_ledger_renders_header_and_ids(self):
        from src.middleware.evidence import assign_evidence_ids, build_evidence_items
        retrieval = self._retrieval()
        ledger = assign_evidence_ids(
            build_evidence_items(retrieval["facts"], retrieval["documents"]))
        prompt = self._augmenter().build_prompt(
            question="What is NVDA revenue?",
            intent={"ticker": "NVDA", "question_type": "fact_lookup"},
            retrieval=retrieval,
            evidence_ledger=ledger,
        )
        assert "## Evidence" in prompt
        assert "[E1]" in prompt and "[E2]" in prompt
        assert "total_revenue=26.00 billion_usd" in prompt
        # The output-format section now hints at [E#] citations.
        assert "cite each figure with its bracketed id" in prompt
        # Legacy sections still render underneath.
        assert "## Retrieved Financial Facts" in prompt
        assert "## Retrieved Documents" in prompt

    def test_ledger_and_document_header_are_self_describing(self):
        from src.middleware.evidence import assign_evidence_ids, build_evidence_items
        from src.middleware.evidence_taxonomy import normalize_evidence

        retrieval = {
            "facts": [],
            "documents": [normalize_evidence({
                "id": "sec-1",
                "document": "Filed financing evidence.",
                "metadata": {
                    "ticker": "ORCL", "security_id": "sec-orcl", "source": "sec",
                    "source_category": "regulatory_filing", "item_type": "sec_filing",
                    "event_type": "debt_raise", "authority_tier": "direct_sec",
                    "published_at": "2026-07-10T13:00:00Z", "coverage_tier": "broad",
                },
            })],
            "ticker": "ORCL",
        }
        ledger = assign_evidence_ids(build_evidence_items([], retrieval["documents"]))
        prompt = self._augmenter().build_prompt(
            question="How did Oracle finance itself?",
            intent={"ticker": "ORCL", "question_type": "news"},
            retrieval=retrieval,
            evidence_ledger=ledger,
        )
        for expected in (
            "item:filing", "event:debt_raise", "authority:direct_sec",
            "source:sec", "canonical:ORCL", "coverage:broad",
            "published_at=2026-07-10T13:00:00Z",
        ):
            assert expected in prompt

    def test_empty_ledger_renders_no_header(self):
        prompt = self._augmenter().build_prompt(
            question="Test?",
            intent={"ticker": "NVDA", "question_type": "fact_lookup"},
            retrieval={"facts": [], "documents": [], "ticker": "NVDA"},
            evidence_ledger=[],
        )
        assert "## Evidence" not in prompt

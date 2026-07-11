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

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

    def test_prompt_includes_system_instruction(self):
        """Prompt always includes system instruction with rules."""
        from src.middleware.prompt_augmenter import PromptAugmenter
        augmenter = PromptAugmenter()

        prompt = augmenter.build_prompt(
            question="Test?",
            intent={"ticker": None, "question_type": "general"},
            retrieval={"facts": [], "documents": [], "ticker": None},
        )

        assert "financial research assistant" in prompt
        assert "Answer using ONLY the provided context" in prompt
        assert "Cite sources inline" in prompt

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

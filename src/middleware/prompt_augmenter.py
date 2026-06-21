"""
src/middleware/prompt_augmenter.py
Prompt augmentation — assembles retrieved facts + documents into
a structured, grounded prompt for the TraceAlchemy model.

Usage:
    augmenter = PromptAugmenter(config=config)
    prompt = augmenter.build_prompt(
        question="What is NVDA's revenue?",
        intent={"ticker": "NVDA", "metrics": ["total_revenue"], ...},
        retrieval={"facts": [...], "documents": [...], ...},
    )
"""

import json
import logging
from typing import Optional

from .config import MiddlewareConfig

logger = logging.getLogger(__name__)


class PromptAugmenter:
    """
    Builds structured, grounded prompts for the TraceAlchemy model.

    Prompt structure:
      1. System instruction (role + rules)
      2. Retrieved facts (structured data from SQLite)
      3. Retrieved documents (semantic context from ChromaDB)
      4. User question
      5. Output format instruction

    Adapts the prompt template based on question type.
    """

    MACRO_LABELS = {
        "GDP": "GDP", "GDPC1": "Real GDP", "CPIAUCSL": "CPI (Inflation)",
        "PCEPILFE": "Core PCE", "FEDFUNDS": "Fed Funds Rate", "DFF": "Fed Funds Rate (Daily)",
        "DGS10": "10-Year Treasury", "DGS2": "2-Year Treasury", "T10Y2Y": "10Y-2Y Spread",
        "UNRATE": "Unemployment Rate", "PAYEMS": "Nonfarm Payrolls",
        "UMCSENT": "Consumer Sentiment", "HOUST": "Housing Starts", "INDPRO": "Industrial Production",
    }

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        self.config = config or MiddlewareConfig()

    # ── Public API ─────────────────────────────────────

    def build_prompt(self, question: str, intent: dict,
                     retrieval: dict) -> str:
        """
        Build the full augmented prompt.

        Args:
            question: Original user question
            intent: Parsed intent from IntentParser
            retrieval: Retrieved data from Retriever

        Returns:
            The complete prompt string ready to send to the model
        """
        question_type = intent.get("question_type", "general")
        ticker = intent.get("ticker")
        facts = retrieval.get("facts", [])
        documents = retrieval.get("documents", [])

        sections = []

        # 1. System instruction
        sections.append(self._build_system_instruction(question_type))

        # 2. Retrieved facts (if any)
        if facts:
            sections.append(self._format_facts_section(facts, ticker))

        # 2.5 Macro context (if any)
        macro_section = self._format_macro_context(facts)
        if macro_section:
            sections.append(macro_section)

        # 3. Retrieved documents (if any)
        if documents:
            sections.append(self._format_documents_section(documents))

        # 4. Handle empty retrieval
        if not facts and not documents:
            sections.append(self._build_no_data_section(ticker))

        # 5. Question-specific instructions
        sections.append(self._build_question_instruction(question_type, ticker))

        # 6. The actual question
        sections.append(f"## User Question\n\n{question}")

        # 7. Output format
        sections.append(self._build_output_format(question_type))

        return "\n\n".join(sections)

    # ── System Instruction ─────────────────────────────

    def _build_system_instruction(self, question_type: str) -> str:
        """Build the system-level instruction for the model."""
        base = (
            "You are a financial research assistant powered by the TraceAlchemy model. "
            "You answer questions about stocks, markets, and financial data using "
            "the provided context below."
        )

        rules = [
            "Answer using ONLY the provided context. Do not use your training data.",
            "If the context doesn't contain enough information, say so clearly.",
            "Cite sources inline using [Source: type/ticker] notation.",
            "Use precise numbers from the context — do not approximate or round.",
            "If a metric is not found in the context, state that it's unavailable.",
            "Be concise but thorough. Prioritize accuracy over verbosity.",
        ]

        type_specific = {
            "fact_lookup": (
                "The user wants a specific financial metric. Provide the exact "
                "value and the period it covers."
            ),
            "comparison": (
                "The user wants a comparison. Present data for each ticker side-by-side "
                "and highlight key differences."
            ),
            "trend": (
                "The user wants to understand a trend over time. Present historical "
                "data points and describe the trajectory."
            ),
            "explanation": (
                "The user wants an explanation. Use the context to explain the "
                "underlying factors or causes."
            ),
            "sentiment": (
                "The user wants market sentiment or analyst views. Summarize the "
                "tone and key opinions from the provided documents."
            ),
            "news": (
                "The user wants recent news or developments. Summarize the key "
                "events and their potential impact."
            ),
            "risk": (
                "The user wants risk factors or concerns. Extract relevant risk "
                "information from the provided documents."
            ),
        }

        instruction = base + "\n\n## Rules\n" + "\n".join(f"- {r}" for r in rules)

        if question_type in type_specific:
            instruction += f"\n\n## Question Type Guidance\n{type_specific[question_type]}"

        return instruction

    # ── Facts Section ─────────────────────────────────

    def _format_facts_section(self, facts: list[dict],
                              ticker: Optional[str]) -> str:
        """Format structured facts into a readable table-like section."""
        lines = ["## Retrieved Financial Facts\n"]

        if ticker:
            lines.append(f"Ticker: **{ticker}**\n")

        for fact in facts:
            metric = fact.get("metric", "unknown")
            value = fact.get("value")
            unit = fact.get("unit", "")
            period = fact.get("period", "")
            source = fact.get("source_type", "unknown")

            # Format value nicely
            if value is not None:
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        value_str = f"{value:.6f}"
                    elif abs(value) < 1:
                        value_str = f"{value:.4f}"
                    elif abs(value) < 100:
                        value_str = f"{value:.2f}"
                    else:
                        value_str = f"{value:,.2f}"
                else:
                    value_str = str(value)
            else:
                value_str = "N/A"

            unit_str = f" {unit}" if unit else ""
            period_str = f" ({period})" if period else ""
            lines.append(
                f"- **{metric}**: {value_str}{unit_str}{period_str} "
                f"[Source: {source}/{ticker or 'unknown'}]"
            )

        return "\n".join(lines)

    # ── Macro Context Section ──────────────────────────

    def _format_macro_context(self, facts: list[dict]) -> str:
        """Format macro-economic context into a readable section."""
        macro_facts = [f for f in facts if f.get("ticker") == "MACRO"]
        if not macro_facts:
            return ""

        lines = ["## Macro-Economic Context\n"]
        for fact in macro_facts:
            metric = fact.get("metric", "unknown")
            value = fact.get("value")
            unit = fact.get("unit", "")
            period = fact.get("period", "")

            label = self.MACRO_LABELS.get(metric, metric)
            value_str = f"{value:,.2f}" if isinstance(value, float) else str(value)
            period_str = f" ({period})" if period else ""
            lines.append(f"- **{label}**: {value_str}{unit}{period_str}")

        return "\n".join(lines)

    # ── Documents Section ──────────────────────────────

    def _format_documents_section(self, documents: list[dict]) -> str:
        """Format retrieved documents into a readable context section."""
        lines = ["## Retrieved Documents\n"]

        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            doc_id = doc.get("id", f"doc_{i}")
            ticker = metadata.get("ticker", doc.get("ticker", "unknown"))
            source = metadata.get("source", doc.get("source", "unknown"))
            date = metadata.get("date", doc.get("date", ""))

            # Truncate very long documents
            if len(text) > 2000:
                text = text[:2000] + "..."

            header_parts = [f"### Document {i}: {doc_id}"]
            if ticker:
                header_parts.append(f"Ticker: {ticker}")
            if source:
                header_parts.append(f"Source: {source}")
            if date:
                header_parts.append(f"Date: {date}")

            lines.append(" | ".join(header_parts))
            lines.append("")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    # ── No Data Section ────────────────────────────────

    def _build_no_data_section(self, ticker: Optional[str]) -> str:
        """Build a section for when no data was retrieved."""
        ticker_str = f" for {ticker}" if ticker else ""
        return (
            "## Note\n\n"
            f"No data was found in the knowledge base{ticker_str}. "
            "If you have general knowledge about this topic, you may use it, "
            "but clearly state that no specific data was found in the system."
        )

    # ── Question-Specific Instructions ─────────────────

    def _build_question_instruction(self, question_type: str,
                                    ticker: Optional[str]) -> str:
        """Build question-type-specific instructions."""
        instructions = {
            "fact_lookup": (
                "Provide the exact value requested. Include the period and source. "
                "If the exact metric isn't available, say so."
            ),
            "comparison": (
                "Present a side-by-side comparison. Use a structured format with "
                "each ticker's data clearly labeled. Highlight the most significant "
                "differences."
            ),
            "trend": (
                "Describe the trend over the available time periods. Note whether "
                "the metric is increasing, decreasing, or stable. Include specific "
                "data points."
            ),
            "explanation": (
                "Explain the factors or causes behind the situation. Use the "
                "retrieved context to support your explanation."
            ),
            "sentiment": (
                "Summarize the overall sentiment from the documents. Note whether "
                "it's positive, negative, or mixed, and what specific points "
                "drive that sentiment."
            ),
            "news": (
                "Summarize the key recent developments. Focus on what happened, "
                "when, and the potential market impact."
            ),
            "risk": (
                "List the key risk factors found in the documents. For each risk, "
                "note its potential impact if mentioned."
            ),
        }

        instruction = instructions.get(
            question_type,
            "Answer the question using the provided context. Be accurate and concise."
        )

        return f"## Instructions\n\n{instruction}"

    # ── Output Format ─────────────────────────────────

    def _build_output_format(self, question_type: str) -> str:
        """Build the output format instruction."""
        return (
            "## Output Format\n\n"
            "Provide your answer in plain text. Use inline citations like "
            "[Source: sec_10k/NVDA] or [Source: yfinance/NVDA] for each "
            "fact you reference. If you use multiple sources, cite each one.\n\n"
            "If the data is insufficient, say: "
            '"I don\'t have enough data in my knowledge base to answer this fully."'
        )

    # ── Prompt Length Management ──────────────────────

    def estimate_tokens(self, prompt: str) -> int:
        """Rough estimate of token count (4 chars ≈ 1 token)."""
        return len(prompt) // 4

    def truncate_if_needed(self, prompt: str, max_tokens: int = 12000) -> str:
        """
        Truncate the prompt if it exceeds the token budget.
        Truncates documents first (oldest/longest), then facts.
        """
        if self.estimate_tokens(prompt) <= max_tokens:
            return prompt

        logger.warning(
            "Prompt exceeds %d tokens (~%d chars), truncating...",
            max_tokens, len(prompt),
        )

        # Split into sections
        sections = prompt.split("\n\n## ")

        # Keep system instruction + question + output format
        keep = []
        docs_section = None
        facts_section = None

        for section in sections:
            if section.startswith("Retrieved Documents"):
                docs_section = section
            elif section.startswith("Retrieved Financial Facts"):
                facts_section = section
            else:
                keep.append(section)

        # Truncate documents first
        if docs_section:
            lines = docs_section.split("\n")
            # Keep header + first 3 documents
            truncated = lines[:1]  # Header
            doc_count = 0
            for line in lines[1:]:
                truncated.append(line)
                if line.startswith("### Document"):
                    doc_count += 1
                    if doc_count >= 3:
                        truncated.append("*(Additional documents truncated for length)*")
                        break
            keep.append("## " + "\n".join(truncated))

        # Truncate facts if still too long
        if facts_section and self.estimate_tokens("\n\n".join(keep)) > max_tokens * 0.8:
            lines = facts_section.split("\n")
            truncated = lines[:min(len(lines), 15)]  # Keep first 15 lines
            keep.append("## " + "\n".join(truncated))

        result = "\n\n".join(keep)
        logger.info("Truncated prompt to ~%d tokens", self.estimate_tokens(result))
        return result

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

import logging
from typing import Optional

from .config import MiddlewareConfig
from .evidence import document_body, evidence_counts, usable_documents

logger = logging.getLogger(__name__)


class PromptAugmenter:
    """
    Builds the augmented USER message for the TraceAlchemy model — retrieved
    evidence, intent-specific task guidance, the raw question, and output
    format. Authoritative answer-policy rules (grounding modes, citation
    requirements, the "no fabricated numbers" rule) are not duplicated here;
    they live solely in the system message built by
    ``src/middleware/prompt_policy.py``.

    Prompt structure:
      1. Retrieved facts (structured data from SQLite)
      2. Retrieved documents (semantic context from ChromaDB)
      3. Question-type-specific task guidance
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
    ESTIMATE_LABELS = {
        "estimate_revenue_current_q": "Revenue estimate (current quarter)",
        "estimate_revenue_next_q": "Revenue estimate (next quarter)",
        "estimate_revenue_current_y": "Revenue estimate (current FY)",
        "estimate_revenue_next_y": "Revenue estimate (next FY)",
        "estimate_eps_current_q": "EPS estimate (current quarter)",
        "estimate_eps_next_q": "EPS estimate (next quarter)",
        "estimate_eps_current_y": "EPS estimate (current FY)",
        "estimate_eps_next_y": "EPS estimate (next FY)",
        "price_target_mean": "Price target (mean)",
        "price_target_high": "Price target (high)",
        "price_target_low": "Price target (low)",
        "num_analysts": "Analyst count",
        "recommendation_mean": "Analyst rating (1=Strong Buy, 5=Sell)",
    }

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        self.config = config or MiddlewareConfig()

    # ── Public API ─────────────────────────────────────

    def build_prompt(
        self,
        question: str,
        intent: dict,
        retrieval: dict,
        grounding_level: Optional[str] = None,
        preselected: Optional[dict] = None,
    ) -> str:
        """
        Build the full augmented prompt.

        Args:
            question: Original user question
            intent: Parsed intent from IntentParser
            retrieval: Retrieved data from Retriever
            grounding_level: Optional grounding label; derived from ``retrieval``
                when omitted.
            preselected: Optional pre-budgeted evidence from the adaptive
                orchestrator's :class:`ContextBudget` (2.2.3.3), a dict with
                ``facts``/``documents`` already selected, de-duplicated, and
                sized. When provided, it is the single budget owner: these rows
                are rendered as-is (no re-filtering, no independent per-document
                truncation). When ``None``, behavior is byte-for-byte identical
                to the pre-2.2.3.3 path.

        Returns:
            The complete prompt string ready to send to the model
        """
        question_type = intent.get("question_type", "general")
        ticker = intent.get("ticker")
        adaptive = preselected is not None
        if adaptive:
            facts = list(preselected.get("facts", []))
            # Pre-budgeted rows: already usable (blank bodies dropped upstream)
            # and pre-sized, so no re-filtering here.
            documents = list(preselected.get("documents", []))
        else:
            facts = retrieval.get("facts", [])
            # Filter unusable rows before prompt assembly so an empty
            # "## Retrieved Documents" section is never emitted (2.2.1.1).
            documents = usable_documents(retrieval)
        if grounding_level is None:
            if adaptive:
                n = len(facts) + len(documents)
            else:
                n_facts, n_docs = evidence_counts(retrieval)
                n = n_facts + n_docs
            grounding_level = "grounded" if n >= 3 else "partial" if n >= 1 else "none"
        estimate_facts = facts if question_type == "projection" else []
        realized_facts = facts
        if question_type == "projection":
            realized_facts = [f for f in facts if not self._is_estimate_fact(f)]

        sections = []

        # 1. Projection context (if any)
        projection_section = self._format_projection_section(estimate_facts)
        if projection_section:
            sections.append(projection_section)

        # 2. Retrieved realized facts (if any)
        if realized_facts:
            sections.append(self._format_facts_section(realized_facts, ticker))

        # 2.5 Macro context (if any)
        macro_section = self._format_macro_context(realized_facts)
        if macro_section:
            sections.append(macro_section)

        # 3. Retrieved documents (if any). On the adaptive path the ContextBudget
        # already sized every chunk, so render bodies whole (no 2000-char cut).
        if documents:
            sections.append(self._format_documents_section(
                documents, max_body=None if adaptive else 2000))

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
            # The request-level ticker is only a fallback for facts that
            # lack their own — never overwrite a fact from a different
            # ticker (comparison/macro retrieval mixes tickers per row).
            fact_ticker = fact.get("ticker") or ticker or "unknown"

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
                f"[Source: {source}/{fact_ticker}]"
            )

        return "\n".join(lines)

    # ── Macro Context Section ──────────────────────────

    def _format_projection_section(self, facts: list[dict]) -> str:
        """Format forward-looking analyst estimate facts."""
        estimate_facts = [f for f in facts if self._is_estimate_fact(f)]
        if not estimate_facts:
            return ""

        lines = ["## Analyst Consensus (Forward-Looking Estimates)\n"]
        analyst_fact = next(
            (f for f in estimate_facts if f.get("metric") == "num_analysts"),
            None,
        )
        analyst_count = self._format_plain_number(analyst_fact.get("value")) if analyst_fact else None
        if analyst_count:
            lines.append(f"Based on {analyst_count} analysts.\n")

        for fact in estimate_facts:
            metric = fact.get("metric", "unknown")
            value = fact.get("value")
            unit = fact.get("unit", "")
            period = fact.get("period", "")
            source = fact.get("source_type", "estimates")
            ticker = fact.get("ticker", "unknown")
            label = self.ESTIMATE_LABELS.get(metric, metric)
            value_str = self._format_value(value)
            unit_str = f" {unit}" if unit else ""
            period_str = f" ({period})" if period else ""
            lines.append(
                f"- **{label}**: {value_str}{unit_str}{period_str} "
                f"[Source: {source}/{ticker}]"
            )

        return "\n".join(lines)

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

    def _format_documents_section(self, documents: list[dict],
                                  max_body: Optional[int] = 2000) -> str:
        """Format retrieved documents into a readable context section.

        ``documents`` must already be filtered to usable rows (see
        ``evidence.usable_documents``) — this only renders bodies, it does
        not re-check for blanks. ``max_body`` caps each rendered body at that
        many characters (legacy default 2000); pass ``None`` on the adaptive
        path where the ContextBudget already owns sizing.
        """
        lines = ["## Retrieved Documents\n"]

        for i, doc in enumerate(documents, 1):
            text = document_body(doc)
            metadata = doc.get("metadata", {})
            doc_id = doc.get("id", f"doc_{i}")
            ticker = metadata.get("ticker", doc.get("ticker", "unknown"))
            source = metadata.get("source", doc.get("source", "unknown"))
            date = metadata.get("date", doc.get("date", ""))

            # Truncate very long documents (legacy path only).
            if max_body is not None and len(text) > max_body:
                text = text[:max_body] + "..."

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
            "projection": (
                "Lead with the consensus figures and analyst count from the Analyst "
                "Consensus section. Label every number with its period (for example, "
                "FY2027E). Clearly separate sourced figures from interpretation, and "
                "end with a one-sentence caveat that these are analyst estimates, "
                "not guarantees, and not financial advice. Never state a "
                "self-invented price target or forecast; if no consensus data is "
                "provided, say so plainly."
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
        if getattr(self.config, "answer_policy", "graded") != "strict":
            return (
                "## Output Format\n\n"
                "Provide your answer in plain text. Use inline citations like "
                "[Source: sec_10k/NVDA] or [Source: yfinance/NVDA] for each "
                "retrieved fact you reference. If you use multiple sources, cite "
                "each one.\n\n"
                "For general fallback answers, start with: "
                '"Not from your data - general knowledge:" and include a '
                "primary-source verification caveat. If you refuse, briefly say "
                "why the request cannot be answered."
            )
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

    def _is_estimate_fact(self, fact: dict) -> bool:
        metric = str(fact.get("metric", ""))
        return (
            fact.get("source_type") == "estimates"
            or metric.startswith(("estimate_", "price_target_"))
            or metric in {"num_analysts", "recommendation_mean"}
        )

    def _format_value(self, value) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            if abs(value) < 0.01:
                return f"{value:.6f}"
            if abs(value) < 1:
                return f"{value:.4f}"
            if abs(value) < 100:
                return f"{value:.2f}"
            return f"{value:,.2f}"
        return str(value)

    def _format_plain_number(self, value) -> str:
        if value is None:
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return str(int(number)) if number.is_integer() else str(number)

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

        # Split into sections. Prepend a boundary marker first: the prompt
        # may now start directly with a "## "-headed section (there is no
        # mandatory non-"##" preamble since 2.2.1.1 removed the duplicated
        # system instruction), and without this the leading section would
        # keep its "## " prefix and fail the startswith() checks below.
        sections = ("\n\n" + prompt).split("\n\n## ")

        # Keep task guidance + question + output format
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

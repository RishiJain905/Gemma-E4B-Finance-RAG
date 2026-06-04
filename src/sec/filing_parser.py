"""
src/sec/filing_parser.py
TraceAlchemy filing parser — sends SEC filing text to the local model
and extracts structured financial facts.

The model acts as a parser, not a generator:
  - Input: SEC filing text (10-K or 10-Q excerpt)
  - Output: Structured JSON with extracted financial metrics
  - Model config: temperature=0.1, max_tokens=1024 (deterministic extraction)

Usage:
    parser = TraceAlchemyFilingParser()
    text = download_filing_text_from_somewhere()
    facts = parser.extract_facts_from_filing("NVDA", "10-Q", text)
    # Returns: [{"metric": "total_revenue", "value": 26.0, "unit": "billion_usd", "period": "2026-Q1"}, ...]
"""

import json
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class TraceAlchemyFilingParser:
    """Parses SEC filing text into structured financial facts using TraceAlchemy.

    The TraceAlchemy model is served via llama-server at the configured endpoint.
    It's fine-tuned on financial data — this parser crafts a deterministic
    extraction prompt and validates the structured output.

    Attributes:
        endpoint: llama-server chat completions endpoint
        model: Model identifier sent to the API
        extraction_fields: The list of financial metrics the parser extracts
    """

    # ── Endpoint Configuration ─────────────────────────

    DEFAULT_ENDPOINT = "http://127.0.0.1:8087/v1/chat/completions"
    DEFAULT_MODEL = "tracealchemy"

    # ── Extraction Schema ──────────────────────────────

    # Metrics extracted from 10-K / 10-Q filings
    # Each entry: (field_name, label, unit, description)
    EXTRACTION_FIELDS = [
        # Income Statement
        ("total_revenue", "Total Revenue / Revenue", "billion_usd",
         "Total revenue, sometimes labeled 'Total revenue' or 'Revenue'"),
        ("cost_of_revenue", "Cost of Revenue", "billion_usd",
         "Cost of revenue / cost of goods sold"),
        ("gross_profit", "Gross Profit", "billion_usd",
         "Gross profit (revenue minus cost of revenue)"),
        ("research_development", "Research and Development", "billion_usd",
         "R&D expenses"),
        ("sales_marketing", "Sales and Marketing", "billion_usd",
         "Sales and marketing expenses"),
        ("general_administrative", "General and Administrative", "billion_usd",
         "G&A expenses"),
        ("operating_income", "Operating Income / Income from Operations", "billion_usd",
         "Operating income (EBIT)"),
        ("interest_income_expense", "Interest Income (Expense)", "billion_usd",
         "Net interest income or expense"),
        ("other_income_expense", "Other Income (Expense)", "billion_usd",
         "Other income or expense items"),
        ("pretax_income", "Income Before Income Taxes", "billion_usd",
         "Income before taxes"),
        ("income_tax_provision", "Provision for Income Taxes", "billion_usd",
         "Income tax expense"),
        ("net_income", "Net Income", "billion_usd",
         "Net income (earnings)"),

        # Per Share
        ("eps_basic", "Earnings Per Share (Basic)", "usd",
         "Basic earnings per share"),
        ("eps_diluted", "Earnings Per Share (Diluted)", "usd",
         "Diluted earnings per share"),
        ("weighted_avg_shares_basic", "Weighted Average Shares (Basic)", "millions",
         "Basic weighted average shares outstanding"),
        ("weighted_avg_shares_diluted", "Weighted Average Shares (Diluted)", "millions",
         "Diluted weighted average shares outstanding"),

        # Balance Sheet (if available)
        ("total_assets", "Total Assets", "billion_usd",
         "Total assets"),
        ("total_liabilities", "Total Liabilities", "billion_usd",
         "Total liabilities"),
        ("total_equity", "Total Stockholders' Equity", "billion_usd",
         "Total shareholders' equity"),

        # Cash Flow
        ("operating_cash_flow", "Net Cash Provided by Operating Activities", "billion_usd",
         "Cash from operating activities"),
        ("investing_cash_flow", "Net Cash Used in Investing Activities", "billion_usd",
         "Cash from investing activities"),
        ("financing_cash_flow", "Net Cash Used in Financing Activities", "billion_usd",
         "Cash from financing activities"),
        ("free_cash_flow", "Free Cash Flow", "billion_usd",
         "Free cash flow (operating CF minus capex)"),
    ]

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
    ):
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __del__(self):
        if hasattr(self, "_client"):
            self._client.close()

    # ── Public API ─────────────────────────────────────

    def extract_facts_from_filing(
        self,
        ticker: str,
        filing_type: str,
        filing_text: str,
        period: Optional[str] = None,
    ) -> list[dict]:
        """Extract structured financial facts from an SEC filing text.

        Args:
            ticker: Stock ticker symbol
            filing_type: "10-K" or "10-Q"
            filing_text: Full or excerpted filing text
            period: Fiscal period label (e.g., "2026", "2026-Q1")

        Returns:
            List of fact dicts:
                {metric, value, unit, period, period_type, source_type}
        """
        # For long filings, use the income statement + balance sheet sections
        # to keep within context window limits
        extracted_text = self._select_extraction_sections(filing_text, filing_type)

        prompt = self._build_extraction_prompt(
            ticker, filing_type, extracted_text,
        )

        raw_response = self._call_model(prompt)
        facts = self._parse_model_response(raw_response, ticker, period or "")

        logger.info(
            "Parsed %s %s filing for %s: %d facts extracted",
            filing_type, period, ticker, len(facts),
        )
        return facts

    def extract_facts_from_filing_chunked(
        self,
        ticker: str,
        filing_type: str,
        filing_text: str,
        period: Optional[str] = None,
        chunk_size: int = 8000,
    ) -> list[dict]:
        """Extract facts from a very long filing by chunking the text.

        Falls back to the main method for most filings; this is for
        complete 10-K filings that can be 50K+ characters.

        Strategy: extract from the income statement section first,
        then supplement with balance sheet if available.
        """
        # Try the simple approach first
        facts = self.extract_facts_from_filing(ticker, filing_type, filing_text, period)

        # If we got the core income statement metrics, that's sufficient
        core_metrics = {"total_revenue", "net_income", "eps_diluted"}
        extracted_metrics = {f["metric"] for f in facts}
        if core_metrics.issubset(extracted_metrics):
            return facts

        # If not, try chunking — look for specific sections
        sections = self._split_filing_sections(filing_text)
        all_facts = []

        for section_name, section_text in sections.items():
            if len(section_text) < 100:
                continue
            section_facts = self.extract_facts_from_filing(
                ticker, filing_type, section_text[:chunk_size], period,
            )
            all_facts.extend(section_facts)

        # Deduplicate by (metric, period)
        seen: set[tuple] = set()
        deduped = []
        for f in all_facts:
            key = (f["metric"], f["period"])
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        return deduped

    # ── Prompt Engineering ─────────────────────────────

    def _build_extraction_prompt(
        self, ticker: str, filing_type: str, text: str,
    ) -> str:
        """Build the extraction prompt sent to the model.

        The prompt is designed for determinism:
          - Explicit field list from EXTRACTION_FIELDS
          - JSON-only output (no explanations)
          - Specific example format
          - Temperature should be 0.1 in the API call
        """
        field_descriptions = "\n".join(
            f"  - \"{field_name}\": {label} ({unit}) — {desc}"
            for field_name, label, unit, desc in self.EXTRACTION_FIELDS
        )

        return f"""You are a financial document parser. Extract structured financial metrics from the following {ticker} {filing_type} filing text.

Extract ONLY the metrics listed below. For each metric you find in the text, include:
  - "metric": the field name from the list
  - "value": the numeric value (as a float, in the specified unit)
  - "unit": the unit from the list
  - "confidence": 1.0 if explicitly stated, 0.5 if derived/calculated

Fields to extract:
{field_descriptions}

RULES:
1. Return ONLY valid JSON — no markdown, no explanations, no code fences.
2. If a metric is NOT found in the text, DO NOT include it in the output.
3. Values must be in the specified unit (billion_usd means billions of dollars).
4. Convert values to the correct unit (e.g., $26,000,000,000 → 26.0 for billion_usd).
5. Report the filing period as you find it.

FILING TEXT:
--- START OF {filing_type} FOR {ticker} ---
{text[:12000]}
--- END OF {filing_type} FOR {ticker} ---

Return the JSON array now."""

    def _select_extraction_sections(self, filing_text: str, filing_type: str) -> str:
        """Select the most relevant sections of a filing for extraction.

        For 10-K/Q, the key sections are:
          - Consolidated Income Statements
          - Consolidated Balance Sheets
          - Consolidated Statements of Cash Flows

        If we can find these sections, extract just them to save context.
        Otherwise, fall back to the first N characters of the filing.
        """
        text = filing_text[:15000]  # Cap to avoid context overflow

        # Try to find income statement section using common section markers
        section_markers = [
            "CONSOLIDATED INCOME STATEMENT",
            "CONSOLIDATED STATEMENTS OF INCOME",
            "CONSOLIDATED STATEMENT OF INCOME",
            "INCOME STATEMENT",
            "STATEMENT OF INCOME",
            "CONSOLIDATED STATEMENTS OF OPERATIONS",
            "STATEMENT OF OPERATIONS",
        ]

        for marker in section_markers:
            idx = text.upper().find(marker)
            if idx >= 0:
                # Found the income statement — grab from here
                start = max(0, idx - 200)
                # Also get balance sheet if available
                bs_markers = [
                    "CONSOLIDATED BALANCE SHEET",
                    "CONSOLIDATED BALANCE SHEETS",
                    "BALANCE SHEET",
                    "BALANCE SHEETS",
                ]
                end_idx = len(text)
                for bs_marker in bs_markers:
                    bs_pos = text.upper().find(bs_marker, idx + 100)
                    if bs_pos > 0:
                        end_idx = bs_pos + 3000
                        break

                return text[start:end_idx]

        # Fallback: first 12K chars
        return text[:12000]

    def _split_filing_sections(self, filing_text: str) -> dict[str, str]:
        """Split a filing into logical sections by common headers."""
        sections = {}
        current_section = "preamble"
        current_text = []

        section_headers = [
            "CONSOLIDATED INCOME STATEMENT",
            "CONSOLIDATED STATEMENTS OF INCOME",
            "CONSOLIDATED BALANCE SHEET",
            "CONSOLIDATED STATEMENTS OF CASH FLOWS",
            "CONSOLIDATED STATEMENTS OF STOCKHOLDERS",
            "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
            "MANAGEMENT'S DISCUSSION AND ANALYSIS",
            "RISK FACTORS",
            "BUSINESS",
        ]

        for line in filing_text.split("\n"):
            upper_line = line.strip().upper()
            matched = False
            for header in section_headers:
                if header in upper_line:
                    if current_text:
                        sections[current_section] = "\n".join(current_text)
                    current_section = header.lower().replace(" ", "_").replace("'", "")
                    current_text = [line]
                    matched = True
                    break

            if not matched:
                current_text.append(line)

        if current_text:
            sections[current_section] = "\n".join(current_text)

        return sections

    # ── Model API Call ────────────────────────────────

    def _call_model(self, prompt: str) -> Optional[str]:
        """Send the extraction prompt to the TraceAlchemy model.

        Uses the fact_extraction task parameters (temperature=0.1).
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise financial document parser. "
                        "Extract only the requested fields. Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 0.1,  # Narrow sampling for deterministic extraction
        }

        try:
            response = self._client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                logger.warning("Model returned empty response")
                return None

            return content

        except httpx.HTTPStatusError as e:
            logger.error("Model API error (HTTP %d): %s", e.response.status_code, e)
            return None
        except httpx.RequestError as e:
            logger.error("Model API request failed: %s", e)
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("Failed to parse model API response: %s", e)
            return None

    # ── Response Parsing ──────────────────────────────

    def _parse_model_response(
        self, raw_response: Optional[str], ticker: str, period: str,
    ) -> list[dict]:
        """Parse the model's JSON response into our standard fact format.

        Handles:
          - Pure JSON array with no wrapper
          - JSON array in a markdown code block
          - JSON with extra text before/after
          - Failed parse = empty list (logged)
        """
        if not raw_response:
            return []

        # Strip markdown code fences if present
        text = raw_response.strip()
        if text.startswith("```"):
            # Remove opening fence (possibly with language tag)
            text = re.sub(r"^```\w*\n?", "", text)
            # Remove closing fence
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        # Try to find a JSON array in the response
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON array with regex
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.error("Failed to parse model response as JSON")
                    return []
            else:
                logger.error("No JSON array found in model response")
                return []

        if not isinstance(data, list):
            logger.warning("Model response is not a JSON array")
            return []

        # Convert to our standard fact format
        period_type = "annual" if "Q" not in (period or "") else "quarterly"
        facts = []

        for item in data:
            if not isinstance(item, dict):
                continue

            metric = item.get("metric", "")
            value = item.get("value")
            unit = item.get("unit", "billion_usd")

            # Validate we have the minimum fields
            if not metric or value is None:
                continue

            # Normalize the value to float
            try:
                value = float(value)
            except (TypeError, ValueError):
                logger.debug("Skipping metric %s: non-numeric value %s", metric, value)
                continue

            facts.append({
                "metric": metric,
                "value": value,
                "unit": unit,
                "period": period,
                "period_type": period_type,
                "source_type": f"sec_{filing_type_from_period(period)}",
            })

        return facts

    # ── Cleanup ────────────────────────────────────────

    def close(self):
        """Close the HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()


def filing_type_from_period(period: str) -> str:
    """Derive the filing type suffix from a period string.

    Examples:
      "2025"    → "10-K"
      "2026-Q1" → "10-Q"
      "2026-Q2" → "10-Q"
    """
    if not period:
        return "10-K"
    if "-Q" in period:
        return "10-Q"
    return "10-K"

"""
src/middleware/intent_parser.py
Intent parsing — extracts ticker, metrics, question type, and timeframe
from natural language financial questions.

Usage:
    parser = IntentParser()
    intent = parser.parse("What is NVDA's revenue for Q1 2026?")
    # Returns:
    # {
    #     "ticker": "NVDA",
    #     "metrics": ["total_revenue"],
    #     "question_type": "fact_lookup",
    #     "timeframe": "2026-Q1",
    #     "original_question": "What is NVDA's revenue for Q1 2026?"
    # }
"""

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .query_plan import QueryEntity, QueryPlan
    from .symbol_resolver import SymbolResolver

logger = logging.getLogger(__name__)


class IntentParser:
    """
    Parses natural language financial questions into structured intent.

    Handles:
      - Ticker detection (symbols + company names)
      - Metric extraction (revenue, EPS, PE ratio, margins, etc.)
      - Question type classification
      - Timeframe detection
    """

    # ── Known Ticker Mapping ───────────────────────────

    COMPANY_TO_TICKER = {
        "nvidia": "NVDA",
        "advanced micro devices": "AMD",
        "amd": "AMD",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "alphabet": "GOOGL",
        "google": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "intel": "INTC",
        "salesforce": "CRM",
        "broadcom": "AVGO",
        "oracle": "ORCL",
        "cisco": "CSCO",
        "ibm": "IBM",
        "qualcomm": "QCOM",
        "texas instruments": "TXN",
        "micron": "MU",
        "marvell": "MRVL",
        "palantir": "PLTR",
        "snowflake": "SNOW",
        "crowdstrike": "CRWD",
        "palo alto networks": "PANW",
        "uber": "UBER",
        "block": "SQ",
        "robinhood": "HOOD",
        "coinbase": "COIN",
        "microstrategy": "MSTR",
        "strategy": "MSTR",
        "netflix": "NFLX",
        "spotify": "SPOT",
        "servicenow": "NOW",
        "workday": "WDAY",
        "adobe": "ADBE",
        "datadog": "DDOG",
        "mongodb": "MDB",
        "cloudflare": "NET",
        "zoom": "ZM",
        "twilio": "TWLO",
        "shopify": "SHOP",
        "stripe": "STRIPE",  # Private, but commonly asked about
        "openai": "OPENAI",  # Private
        "databricks": "DATABRICKS",  # Private
    }

    KNOWN_TICKERS = set(COMPANY_TO_TICKER.values())

    # Uppercase words that look like tickers but are almost always English.
    # Shared by the step-3 fallback below and SymbolResolver's catalog ticker
    # match (several of these ARE real symbols — e.g. ARE, ALL, IT, CAN — and
    # must not resolve from prose).
    COMMON_QUERY_WORDS = {"I", "A", "AN", "THE", "IT", "IS", "BE", "TO",
                          "OF", "IN", "ON", "AT", "BY", "AS", "OR", "IF",
                          "NO", "GO", "DO", "WE", "HE", "SHE", "ALL", "FOR",
                          "AND", "NOT", "ARE", "WAS", "HAS", "HAD", "CAN",
                          "WILL", "MAY", "YOUR", "YOU", "THAT", "THIS", "WITH"}

    # ── Metric Keywords ────────────────────────────────

    METRIC_PATTERNS = [
        # Revenue / Sales
        (r"\brevenue\b", "total_revenue"),
        (r"\bsales\b", "total_revenue"),
        (r"\btop[- ]line\b", "total_revenue"),
        # "gross profit" -> gross_profit; "gross margin" -> gross_margin_pct
        # only (2.2.3.4 review finding 6). The old `gross (profit|margin)`
        # alternation double-matched "gross margin" as both metrics, creating a
        # false multi-metric ambiguity that made the router abstain.
        (r"\bgross profit\b", "gross_profit"),
        (r"\bgross margin\b", "gross_margin_pct"),

        # Earnings / Profit
        (r"\bnet income\b", "net_income"),
        (r"\bnet profit\b", "net_income"),
        (r"\bprofit\b", "net_income"),
        (r"\beps\b", "eps_diluted"),
        (r"\bearnings per share\b", "eps_diluted"),
        (r"\boperating income\b", "operating_income"),
        (r"\bebit\b", "operating_income"),
        (r"\bebitda\b", "ebitda"),

        # Costs / Expenses
        (r"\bcogs\b", "cost_of_revenue"),
        (r"\bcost of (revenue|goods|sales)\b", "cost_of_revenue"),
        (r"\br&d\b", "research_development"),
        (r"\bresearch and development\b", "research_development"),
        (r"\bsga\b", "sales_marketing"),

        # Balance Sheet
        (r"\btotal assets\b", "total_assets"),
        (r"\btotal liabilities\b", "total_liabilities"),
        (r"\b(equity|shareholders equity|book value)\b", "total_equity"),
        (r"\bcash and (equivalents|short.?term)\b", "cash_and_equivalents"),
        (r"\bdebt\b", "total_debt"),

        # Cash Flow
        (r"\boperating cash flow\b", "operating_cash_flow"),
        (r"\bfree cash flow\b", "free_cash_flow"),
        (r"\bfcf\b", "free_cash_flow"),
        (r"\bcash flow\b", "operating_cash_flow"),

        # Valuation
        (r"\bprice target\b", "price_target_mean"),
        (r"\bpe ratio\b", "pe_ratio"),
        (r"\bprice to earnings\b", "pe_ratio"),
        (r"\bp/e\b", "pe_ratio"),
        (r"\bforward pe\b", "forward_pe"),
        (r"\bprice to book\b", "pb_ratio"),
        (r"\bp/b\b", "pb_ratio"),
        (r"\bprice to sales\b", "ps_ratio"),
        (r"\bp/s\b", "ps_ratio"),
        (r"\bev/ebitda\b", "ev_ebitda"),
        (r"\benterprise value\b", "enterprise_value"),
        (r"\bmarket cap\b", "market_cap"),
        (r"\bmarket capitalization\b", "market_cap"),

        # Growth
        (r"\brevenue growth\b", "revenue_growth"),
        (r"\b(yoy|year over year|year-on-year) growth\b", "yoy_growth"),
        (r"\bgrowth rate\b", "growth_rate"),

        # Per Share
        (r"\bdividend\b", "dividend_yield"),
        (r"\byield\b", "dividend_yield"),

        # Margins
        (r"\bgross margin\b", "gross_margin_pct"),
        (r"\boperating margin\b", "operating_margin_pct"),
        (r"\bnet margin\b", "net_margin_pct"),
        (r"\bprofit margin\b", "net_margin_pct"),

        # Returns
        (r"\broe\b", "roe"),
        (r"\breturn on equity\b", "roe"),
        (r"\broa\b", "roa"),
        (r"\breturn on assets\b", "roa"),
        (r"\broic\b", "roic"),
    ]

    # ── Question Type Patterns ──────────────────────────

    QUESTION_TYPE_PATTERNS = {
        "fact_lookup": [
            r"\bwhat (is|was|are|were)\b",
            r"\bhow much\b",
            r"\bhow many\b",
            r"\bgive me\b",
            r"\bshow me\b",
            r"\btell me\b",
        ],
        "comparison": [
            r"\bcompare\b",
            r"\bversus\b",
            r"\bvs\b",
            r"\bdifference between\b",
            r"\bwhich (is|has|performs)\b",
            r"\bwho (has|performs)\b",
        ],
        "trend": [
            r"\btrend\b",
            r"\bover time\b",
            r"\bhistory\b",
            r"\bhistorical\b",
            r"\bhow has\b",
            r"\bchange (in|over)\b",
            r"\btrajectory\b",
        ],
        "explanation": [
            r"\bwhy\b",
            r"\bexplain\b",
            r"\bhow does\b",
            r"\bwhat (causes|drives|impacts|affects)\b",
            r"\breason\b",
        ],
        "projection": [
            r"\bprice target\b",
            r"\bprojection[s]?\b",
            r"\bconsensus\b",
            r"\bnext (quarter|year)\b",
            r"\bforecast\b",
            r"\bexpect(s|ed|ation|ations)?\b",
            r"\bguidance\b",
            r"\bestimate[ds]?\b",
            r"\bwill .* (grow|reach|hit)\b",
        ],
        "sentiment": [
            r"\bsentiment\b",
            r"\bmarket (mood|feeling|attitude)\b",
            r"\bwhat (do|does) analysts\b",
            r"\boutlook\b",
            r"\bforecast\b",
            r"\bexpectation\b",
        ],
        "news": [
            r"\bnews\b",
            r"\brecent (developments|events|updates)\b",
            r"\bwhat happened\b",
            r"\bwhat's new\b",
            r"\bwhat is happening\b",
        ],
        "risk": [
            r"\brisk\b",
            r"\brisks\b",
            r"\bconcern\b",
            r"\bthreat\b",
            r"\bchallenge\b",
            r"\bheadwind\b",
            r"\bdownside\b",
        ],
    }

    _TYPE_PRIORITY = (
        "comparison",
        "trend",
        "explanation",
        "projection",
        "sentiment",
        "news",
        "risk",
        "fact_lookup",
    )

    _WEAK_FACT_LOOKUP_PATTERNS = (
        r"\bgive me\b",
        r"\bshow me\b",
        r"\btell me\b",
    )

    # ── Timeframe Patterns ─────────────────────────────

    TIMEFRAME_PATTERNS = [
        (r"\bq[1-4]\s*20\d{2}\b", "quarter"),       # Q1 2026
        (r"\b20\d{2}\s*q[1-4]\b", "quarter"),       # 2026 Q1
        (r"\b(fy|fiscal year)\s*20\d{2}\b", "annual"),  # FY 2026
        (r"\b20\d{2}\b", "annual"),                  # 2026
        (r"\blatest (quarter|q)\b", "latest_quarter"),
        (r"\b(last|most recent|current) quarter\b", "latest_quarter"),
        (r"\b(this|current) (fiscal )?year\b", "current_year"),
        (r"\blast (fiscal )?year\b", "last_year"),
        (r"\b(trailing|ttm|last twelve months)\b", "ttm"),
        (r"\bytd\b", "ytd"),
    ]

    # Macro series that answer without a company entity — used only as an
    # advisory retrieval-mode hint on the query plan (2.2.3.1).
    _MACRO_TERM_RE = re.compile(
        r"\b(gdp|cpi|inflation|unemployment|treasury|interest rate|fed funds"
        r"|federal funds|yield curve|ppi|nonfarm|payroll|jobs report"
        r"|consumer confidence|retail sales)\b",
        re.IGNORECASE,
    )

    _EVIDENCE_TOPIC_PATTERNS = (
        ("financing", re.compile(
            r"\b(financ(?:e|ed|ing)|debt raise|raise(?:d|s|ing)? (?:debt|capital)|"
            r"equity offering|convertible offering|shelf registration|prospectus)\b", re.I)),
        ("corporate_action", re.compile(
            r"\b(corporate actions?|buybacks?|repurchases?|dividends?|stock splits?|"
            r"acqui(?:re[ds]?|sitions?)|mergers?|divest(?:ed|itures?)|dispositions?)\b", re.I)),
        ("ownership", re.compile(
            r"\b(beneficial ownership|ownership changes?|insider transactions?|"
            r"schedule 13[DG]|form [345])\b", re.I)),
        ("regulatory", re.compile(
            r"\b(regulatory|regulator|enforcement|investigation|recall|safety alert)\b", re.I)),
        ("macro_release", re.compile(
            r"\b(inflation|cpi|gdp|jobs report|payroll|unemployment|economic release|"
            r"policy decision|rate decision|treasury auction)\b", re.I)),
        ("company_news", re.compile(
            r"\b(company news|press release|latest news|recent news|what happened)\b", re.I)),
    )

    _EVIDENCE_TOPIC_FILTERS = {
        "financing": {
            "item_type": "filing",
            "event_types": [
                "debt_raise", "equity_raise", "convertible_offering",
                "shelf_registration", "prospectus_update",
            ],
        },
        "corporate_action": {
            "item_type": "corporate_action",
            "event_types": [
                "acquisition", "divestiture", "buyback", "dividend_change",
                "split", "dividend", "corporate_action",
            ],
        },
        "ownership": {
            "item_type": "filing",
            "event_types": ["beneficial_ownership_change", "insider_transaction"],
        },
        "regulatory": {
            "item_type": "regulatory_event",
            "event_types": ["enforcement_action", "investigation", "recall", "safety_alert"],
            "source_categories": ["sec", "regulator", "sector_agency"],
        },
        "macro_release": {
            "item_type": "economic_release",
            "event_types": ["economic_release", "monetary_policy_decision", "treasury_auction"],
            "source_categories": ["central_bank", "treasury", "economic_agency"],
        },
        "company_news": {
            "item_type": "news",
            "source_categories": ["issuer", "company_news"],
        },
    }

    def __init__(self, resolver: Optional["SymbolResolver"] = None):
        from .symbol_resolver import NO_MATCH, get_default_resolver

        # Pre-compile regex patterns for performance
        self._metric_regexes = [
            (re.compile(pattern, re.IGNORECASE), metric)
            for pattern, metric in self.METRIC_PATTERNS
        ]
        self._type_regexes = {
            qtype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for qtype, patterns in self.QUESTION_TYPE_PATTERNS.items()
        }
        self._timeframe_regexes = [
            (re.compile(pattern, re.IGNORECASE), tf_type)
            for pattern, tf_type in self.TIMEFRAME_PATTERNS
        ]
        self._resolver = resolver or get_default_resolver()
        self._last_resolution = NO_MATCH

    # ── Public API ─────────────────────────────────────

    def parse(self, question: str,
              override_ticker: Optional[str] = None) -> dict:
        """
        Parse a natural language financial question into structured intent.

        Args:
            question: The user's question
            override_ticker: Optional ticker override (from request)

        Returns:
            {
                "ticker": str or None,
                "metrics": list[str],
                "question_type": str,
                "timeframe": str or None,
                "timeframe_type": str or None,
                "original_question": str,
            }
        """
        if override_ticker:
            from .symbol_resolver import Resolution

            ticker = override_ticker
            ticker_resolution = Resolution(override_ticker, 1.0, None, "override")
            self._last_resolution = ticker_resolution
        else:
            ticker = self._detect_ticker(question)
            ticker_resolution = self._last_resolution

        evidence_topic, evidence_filters = self._extract_evidence_intent(question)
        intent = {
            "ticker": ticker,
            "metrics": self._extract_metrics(question),
            "question_type": self._classify_question_type(question),
            "timeframe": self._extract_timeframe(question),
            "timeframe_type": self._extract_timeframe_type(question),
            "original_question": question,
            "ticker_confidence": ticker_resolution.confidence,
            "resolved_name": ticker_resolution.resolved_name,
            "ticker_source": ticker_resolution.source,
            "evidence_topic": evidence_topic,
            "evidence_filters": evidence_filters,
        }
        logger.debug(
            "Parsed intent: ticker=%s type=%s metrics=%s timeframe=%s",
            intent["ticker"], intent["question_type"],
            intent["metrics"], intent["timeframe"],
        )
        return intent

    def parse_plan(
        self,
        question: str,
        retrieval_query: Optional[str] = None,
        override_ticker: Optional[str] = None,
    ) -> "QueryPlan":
        """Parse a question into an explicit, validated multi-entity query plan.

        ``question`` is preserved verbatim as ``original_question``; all
        entity/intent/metric/period matching runs on ``retrieval_query or
        question`` (the 2.2.2-compiled standalone query when one exists, else the
        raw question). Every matched intent (priority order), every unique
        metric, and every explicit period are retained — not just the first of
        each. ``primary_intent``/``primary_period`` mirror the legacy single
        result for backward compatibility.

        Only baseline ``sq0`` is populated; selective decomposition into further
        subqueries belongs to 2.2.4.2. The returned plan is validated before
        return, so a caller that catches :class:`QueryPlanError` can fall back to
        :meth:`parse` without ever failing the user request.
        """
        from .query_plan import QueryPlan, QuerySubquery, normalize_question

        original_question = question
        match_text = retrieval_query if retrieval_query else question

        entities = self._plan_entities(match_text, override_ticker)
        intents = self._classify_all_types(match_text) or ["general"]
        primary_intent = intents[0]
        metrics = self._extract_metrics(match_text)
        periods = self._extract_all_timeframes(match_text)
        evidence_topic, evidence_filters = self._extract_evidence_intent(match_text)

        reason_codes = self._plan_reason_codes(
            entities, intents, metrics, periods, override_ticker
        )
        modes = self._subquery_modes(primary_intent, metrics, entities, match_text)

        sq0 = QuerySubquery(
            id="sq0",
            text=match_text,
            entity_tickers=tuple(entity.ticker for entity in entities),
            intents=tuple(intents),
            metrics=tuple(metrics),
            periods=tuple(periods),
            retrieval_modes=modes,
            derived=False,
            parent_id=None,
        )

        plan = QueryPlan(
            original_question=original_question,
            retrieval_query=match_text,
            normalized_question=normalize_question(match_text),
            entities=entities,
            intents=intents,
            metrics=metrics,
            periods=periods,
            subqueries=[sq0],
            primary_intent=primary_intent,
            primary_period=self._extract_timeframe(match_text),
            primary_period_type=self._extract_timeframe_type(match_text),
            evidence_topic=evidence_topic,
            evidence_filters=evidence_filters,
            reason_codes=reason_codes,
        )
        plan.validate()
        return plan

    def _plan_entities(
        self, text: str, override_ticker: Optional[str]
    ) -> list["QueryEntity"]:
        """Resolve all explicit entities; place any override first.

        The override is added with ``source="override"`` ahead of the resolved
        entities and other explicitly mentioned tickers are kept for
        comparisons. When the text names no other entity, the override is the
        only entity.
        """
        from .query_plan import QueryEntity

        entities: list[QueryEntity] = []
        seen: set[str] = set()

        if override_ticker:
            override = str(override_ticker).strip().upper()
            if override:
                entities.append(
                    QueryEntity(
                        ticker=override,
                        resolved_name=None,
                        confidence=1.0,
                        source="override",
                        mention=override,
                        start=-1,
                    )
                )
                seen.add(override)

        for resolution in self._resolver.resolve_all(text):
            if not resolution.ticker:
                continue
            ticker = resolution.ticker.upper()
            if ticker in seen:
                continue
            seen.add(ticker)
            entities.append(
                QueryEntity(
                    ticker=ticker,
                    resolved_name=resolution.resolved_name,
                    confidence=resolution.confidence,
                    source=resolution.source,
                    mention=resolution.mention or ticker,
                    start=resolution.start,
                )
            )
        return entities

    def _classify_all_types(self, text: str) -> list[str]:
        """All matched question types in the existing priority order.

        Mirrors :meth:`_classify_question_type` (same fact_lookup weak-cue guard)
        but collects every match instead of returning the first, so a compound
        request ("revenue trend, then risks") retains ``trend`` and ``risk``.
        """
        normalized = text.lower()
        matched: list[str] = []
        for qtype in self._TYPE_PRIORITY:
            if qtype == "fact_lookup":
                if self._matches_fact_lookup(normalized, text):
                    matched.append(qtype)
                continue
            for pattern in self._type_regexes[qtype]:
                if pattern.search(normalized):
                    matched.append(qtype)
                    break
        return matched

    def _extract_all_timeframes(self, text: str) -> list[str]:
        """Every explicit timeframe in mention order (deduped).

        Unlike :meth:`_extract_timeframe` (first match only), this retains all
        distinct periods — "2023 through 2025" yields both years — while
        dropping matches fully contained in a more specific one (the bare year
        inside "Q1 2026").
        """
        normalized = text.lower()
        spans: list[tuple[int, int, str]] = []
        for regex, _ in self._timeframe_regexes:
            for match in regex.finditer(normalized):
                spans.append((match.start(), match.end(), match.group(0).strip()))

        # Longest-first per start so a specific period supersedes a contained one.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        kept: list[tuple[int, int, str]] = []
        for start, end, value in spans:
            if any(k_start <= start and end <= k_end for k_start, k_end, _ in kept):
                continue
            kept.append((start, end, value))

        kept.sort(key=lambda s: s[0])
        periods: list[str] = []
        seen: set[str] = set()
        for _, _, value in kept:
            if value and value not in seen:
                seen.add(value)
                periods.append(value)
        return periods

    def _subquery_modes(
        self,
        primary_intent: str,
        metrics: list[str],
        entities: list["QueryEntity"],
        text: str,
    ) -> tuple[str, ...]:
        """Advisory retrieval-mode hints for ``sq0`` (the 2.2.3.2 router decides).

        Deterministic and conservative: structured ``facts`` for metric/exact
        intents, ``documents`` for qualitative intents, ``macro`` when the
        request has no company entity but names a macro series. The router in
        2.2.3.2 makes the binding lane decision; these are only hints.
        """
        modes: list[str] = []
        if metrics or primary_intent in {"fact_lookup", "comparison", "projection"}:
            modes.append("facts")
        if primary_intent in {
            "explanation", "news", "risk", "sentiment", "trend",
        } or not metrics:
            modes.append("documents")
        if not entities and self._MACRO_TERM_RE.search(text.lower()):
            modes.append("macro")
        if not modes:
            modes.append("documents")
        return tuple(dict.fromkeys(modes))

    def _extract_evidence_intent(self, text: str) -> tuple[Optional[str], dict]:
        """Return provider-independent finance evidence routing facets."""
        topic = next((name for name, pattern in self._EVIDENCE_TOPIC_PATTERNS if pattern.search(text)), None)
        filters = dict(self._EVIDENCE_TOPIC_FILTERS.get(topic, {}))
        years = re.findall(r"\b(?:19|20)\d{2}\b", text)
        if years:
            filters["as_of"] = f"{years[0]}-12-31"
        if re.search(r"\b(latest|newest|current|recent|today)\b", text, re.I):
            filters["recency"] = "strong"
        elif re.search(r"\b(historical|history)\b", text, re.I) or years:
            filters["recency"] = "weak"
        return topic, filters

    @staticmethod
    def _plan_reason_codes(
        entities: list["QueryEntity"],
        intents: list[str],
        metrics: list[str],
        periods: list[str],
        override_ticker: Optional[str],
    ) -> list[str]:
        """Observable rule matches for the plan (telemetry / debugging)."""
        codes: list[str] = []
        if override_ticker:
            codes.append("override_entity")
        codes.append(
            "multi_entity" if len(entities) > 1
            else "single_entity" if entities else "no_entity"
        )
        if len(intents) > 1:
            codes.append("multi_intent")
        if len(metrics) > 1:
            codes.append("multi_metric")
        if len(periods) > 1:
            codes.append("multi_period")
        return codes

    # ── Ticker Detection ────────────────────────────────

    def _detect_ticker(self, text: str) -> Optional[str]:
        """Detect ticker from company name or symbol in the question."""
        from .symbol_resolver import NO_MATCH, Resolution

        self._last_resolution = NO_MATCH
        normalized = text.lower()

        # 1. Check company names first (more specific)
        for company_name, ticker in self.COMPANY_TO_TICKER.items():
            if company_name in normalized:
                self._last_resolution = Resolution(ticker, 1.0, company_name, "local_map")
                return ticker

        # 2. Check for uppercase ticker symbols (1-5 letters)
        candidates = set(re.findall(r'\b[A-Z]{1,5}\b', text))
        for c in candidates:
            if c in self.KNOWN_TICKERS:
                self._last_resolution = Resolution(c, 1.0, c, "known_ticker")
                return c

        resolved = self._resolver.resolve(text)
        if resolved.ticker:
            self._last_resolution = resolved
            return resolved.ticker

        # 3. Fallback: return first uppercase word that looks like a ticker
        for c in candidates:
            if c not in self.COMMON_QUERY_WORDS and len(c) >= 1:
                self._last_resolution = Resolution(c, 0.3, None, "fallback")
                return c

        return None

    # ── Metric Extraction ──────────────────────────────

    def _extract_metrics(self, text: str) -> list[str]:
        """Extract financial metric names from the question."""
        normalized = text.lower()
        metrics = []
        seen = set()

        for pattern, metric_name in self._metric_regexes:
            if pattern.search(normalized) and metric_name not in seen:
                metrics.append(metric_name)
                seen.add(metric_name)

        return metrics

    # ── Question Type Classification ───────────────────

    def _classify_question_type(self, text: str) -> str:
        """Classify the question into a type category."""
        normalized = text.lower()

        for qtype in self._TYPE_PRIORITY:
            if qtype == "fact_lookup":
                if self._matches_fact_lookup(normalized, text):
                    return qtype
                continue
            for pattern in self._type_regexes[qtype]:
                if pattern.search(normalized):
                    return qtype

        return "general"

    def _matches_fact_lookup(self, normalized: str, text: str) -> bool:
        """Match fact_lookup; weak cues require a known ticker or metrics."""
        matched_weak = False
        for pattern in self._type_regexes["fact_lookup"]:
            if not pattern.search(normalized):
                continue
            if pattern.pattern in self._WEAK_FACT_LOOKUP_PATTERNS:
                matched_weak = True
            else:
                return True

        if not matched_weak:
            return False

        ticker = self._detect_ticker(text)
        if ticker and ticker in self.KNOWN_TICKERS:
            return True
        return bool(self._extract_metrics(text))

    # ── Timeframe Extraction ───────────────────────────

    def _extract_timeframe(self, text: str) -> Optional[str]:
        """Extract a timeframe string from the question."""
        normalized = text.lower()

        for pattern, _ in self._timeframe_regexes:
            match = pattern.search(normalized)
            if match:
                return match.group(0).strip()

        return None

    def _extract_timeframe_type(self, text: str) -> Optional[str]:
        """Extract the type of timeframe (quarter, annual, ttm, etc.)."""
        normalized = text.lower()

        for pattern, tf_type in self._timeframe_regexes:
            if pattern.search(normalized):
                return tf_type

        return None

"""
src/middleware/answer_validator.py
Deterministic numeric-claim and citation validation for a produced answer
(2.2.4.3) — standard library + ``decimal.Decimal`` only, no model call.

Given an answer string and the request-local evidence ledger (see
``src/middleware/evidence.py``: ``[E#]`` items in final packed order), this
module:

  * extracts currency / percentage / signed / ratio / K-M-B-T-scaled numbers
    from answer sentences (dates, evidence labels, and clearly-marked general
    examples are excluded so they never create false positives);
  * associates the ``[E#]`` (and legacy ``[Source: type/ticker]``) citations
    that appear in the same sentence with each number;
  * checks — by exact/rounded-display magnitude with a compatible unit,
    entity, and period — whether a cited evidence item (or a recorded
    deterministic calculation) contains that number;
  * returns supported / unsupported / ambiguous claim records plus structured
    citation records.

It performs no natural-language entailment and never rewrites the prose. All
matching is deterministic: a displayed value ``D`` with precision unit ``P``
matches a true value ``V`` iff ``|V - D| <= P/2`` (i.e. ``V`` rounds to ``D``),
so exact and rounded display formats are handled uniformly.
"""

from __future__ import annotations

import logging
import re
from time import perf_counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from .evidence import EvidenceItem

logger = logging.getLogger(__name__)

# Scale multipliers for magnitude words and single-letter shorthand.
_SCALE = {
    "k": Decimal(10) ** 3, "thousand": Decimal(10) ** 3,
    "m": Decimal(10) ** 6, "mn": Decimal(10) ** 6, "million": Decimal(10) ** 6,
    "b": Decimal(10) ** 9, "bn": Decimal(10) ** 9, "billion": Decimal(10) ** 9,
    "t": Decimal(10) ** 12, "tn": Decimal(10) ** 12, "trillion": Decimal(10) ** 12,
}
_CURRENCY = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

# One left-to-right, non-overlapping pass over a citation-stripped sentence.
# Order matters: currency (may carry a scale word) before bare scaled/suffix.
_NUMBER_RE = re.compile(
    r"""
      (?P<curpre>[-+]?)(?P<cur>[$€£¥])\s?(?P<cursign>[-+]?)(?P<curnum>\d[\d,]*(?:\.\d+)?)
        (?:\s*(?P<curscale>trillion|billion|million|thousand|bn|mn|tn|[kmbt])\b)?
    | (?P<pctsign>[-+]?)(?P<pctnum>\d[\d,]*(?:\.\d+)?)\s*(?P<pct>%)
    | (?P<ratsign>[-+]?)(?P<ratnum>\d[\d,]*(?:\.\d+)?)\s*(?P<rat>x)(?![\w])
    | (?P<scsign>[-+]?)(?P<scnum>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>trillion|billion|million|thousand|bn|mn|tn)\b
    | (?P<sufsign>[-+]?)(?P<sufnum>\d[\d,]*(?:\.\d+)?)(?P<suf>[kmbt])\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_BRACKET_RE = re.compile(r"\[([^\]]*)\]")
_LEGACY_RE = re.compile(r"^\s*Source:\s*(.+?)\s*$", re.IGNORECASE)
_YEAR_RE = re.compile(r"20\d\d")
_UPPER_TOKEN_RE = re.compile(r"\b[A-Z]{1,5}\b")
# Markers that make a sentence a clearly-labelled hypothetical / illustration.
_EXAMPLE_MARKERS = (
    "for example", "e.g.", "e.g ", "for instance", "such as", "hypothetical",
    "illustrat", "say a company", "imagine", "suppose",
)


# ── Public result records ─────────────────────────────────────────────────

@dataclass
class NumericClaim:
    """One specific financial number extracted from the answer."""

    text: str
    sentence_index: int
    kind: str
    citations: tuple[str, ...]
    status: str  # supported | unsupported | ambiguous
    matched_evidence_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class CitationRecord:
    """One structured citation link parsed from the answer."""

    evidence_id: Optional[str]
    source_type: Optional[str]
    ticker: Optional[str]
    metric: Optional[str]
    period: Optional[str]
    source_url: Optional[str]
    support_status: str  # supported | missing | malformed
    item_type: Optional[str] = None
    event_type: Optional[str] = None
    authority_tier: Optional[str] = None
    source: Optional[str] = None
    date_semantics: Optional[dict] = None
    canonical_security: Optional[str] = None
    coverage_tier: Optional[str] = None
    source_category: Optional[str] = None
    provider: Optional[str] = None
    publisher: Optional[str] = None

    def to_dict(self) -> dict:
        data = {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "ticker": self.ticker,
            "metric": self.metric,
            "period": self.period,
            "source_url": self.source_url,
            "support_status": self.support_status,
        }
        data.update(self._taxonomy_reference())
        return data

    def _taxonomy_reference(self) -> dict:
        """Return populated stable taxonomy fields without changing legacy rows."""
        values = {
            "item_type": self.item_type,
            "event_type": self.event_type,
            "authority_tier": self.authority_tier,
            "source": self.source,
            "date_semantics": self.date_semantics,
            "canonical_security": self.canonical_security,
            "coverage_tier": self.coverage_tier,
            "source_category": self.source_category,
            "provider": self.provider,
            "publisher": self.publisher,
        }
        return {key: value for key, value in values.items() if value not in (None, {})}

    def graph_reference(self) -> dict:
        """Return the allowlisted provenance fields used by graph citations."""
        data = {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "ticker": self.ticker,
            "metric": self.metric,
            "period": self.period,
            "support_status": self.support_status,
        }
        data.update(self._taxonomy_reference())
        return data


@dataclass
class AnswerValidation:
    """Deterministic validation outcome for one answer."""

    status: str  # supported | unsupported | no_claims | report_unavailable
    citation_support_rate: Optional[float]
    numeric_claims_supported: int
    numeric_claims_unsupported: int
    numeric_claims_ambiguous: int
    citations: tuple[CitationRecord, ...]
    claims: tuple[NumericClaim, ...]
    mismatch_counts: dict = field(default_factory=dict)

    # ── Derived counts ────────────────────────────────────
    @property
    def numeric_claims_total(self) -> int:
        return (self.numeric_claims_supported + self.numeric_claims_unsupported
                + self.numeric_claims_ambiguous)

    @property
    def citations_total(self) -> int:
        return len(self.citations)

    @property
    def citations_resolved(self) -> int:
        return sum(1 for c in self.citations
                   if c.support_status == "supported" and c.evidence_id)

    @property
    def citations_missing(self) -> int:
        return sum(1 for c in self.citations if c.support_status == "missing")

    @property
    def citations_malformed(self) -> int:
        return sum(1 for c in self.citations if c.support_status == "malformed")

    def wholly_unsupported(self) -> bool:
        """True when the answer states numbers and none are supported/ambiguous."""
        return (self.numeric_claims_total > 0
                and self.numeric_claims_supported == 0
                and self.numeric_claims_ambiguous == 0)

    def has_violations(self, *, require_evidence_ids: bool = False) -> bool:
        """Whether enforcement should act (downgrade/warn) on this answer."""
        if self.numeric_claims_unsupported > 0:
            return True
        if self.citations_missing > 0 or self.citations_malformed > 0:
            return True
        if require_evidence_ids and self.numeric_claims_total > 0 and self.citations_resolved == 0:
            return True
        return False

    def support_warning(self) -> Optional[str]:
        """A short, deterministic warning naming the affected sentences/claims."""
        problems = [c for c in self.claims if c.status == "unsupported"]
        if not problems and not (self.citations_missing or self.citations_malformed):
            return None
        parts: list[str] = []
        if problems:
            listed = "; ".join(
                f"'{c.text}' (sentence {c.sentence_index + 1}: {c.reason})"
                for c in problems[:5]
            )
            parts.append(f"unverified figures: {listed}")
        if self.citations_missing or self.citations_malformed:
            parts.append(
                f"unresolved citations: {self.citations_missing + self.citations_malformed}")
        return "Support warning — " + "; ".join(parts) + "."

    def to_metadata(self) -> dict:
        """The response/eval-safe validation metadata block."""
        return {
            "validation_status": self.status,
            "citation_support_rate": self.citation_support_rate,
            "numeric_claims_supported": self.numeric_claims_supported,
            "numeric_claims_unsupported": self.numeric_claims_unsupported,
            "numeric_claims_ambiguous": self.numeric_claims_ambiguous,
            "numeric_claims_total": self.numeric_claims_total,
            "citations_total": self.citations_total,
            "citations_resolved": self.citations_resolved,
            "citations_missing": self.citations_missing,
            "citations_malformed": self.citations_malformed,
            "mismatch_counts": dict(self.mismatch_counts),
        }

    @classmethod
    def unavailable(cls) -> "AnswerValidation":
        """Fail-soft sentinel used when the validator itself errors."""
        return cls(
            status="report_unavailable",
            citation_support_rate=None,
            numeric_claims_supported=0,
            numeric_claims_unsupported=0,
            numeric_claims_ambiguous=0,
            citations=(),
            claims=(),
            mismatch_counts={},
        )


# ── Internal numeric view ─────────────────────────────────────────────────

@dataclass(frozen=True)
class _Claim:
    text: str
    magnitude: Decimal
    precision: Decimal
    kind: str  # currency | percent | ratio | scaled
    currency: Optional[str]


@dataclass(frozen=True)
class _Evidence:
    magnitude: Decimal
    kind: str  # currency | percent | ratio | scaled | number
    currency: Optional[str]


def _num(raw: str) -> tuple[Decimal, int]:
    clean = raw.replace(",", "")
    value = Decimal(clean)
    decimals = len(clean.split(".")[1]) if "." in clean else 0
    return value, decimals


def _scale_mult(word: Optional[str]) -> Decimal:
    if not word:
        return Decimal(1)
    return _SCALE.get(word.strip().lower(), Decimal(1))


def _signed(value: Decimal, *sign_tokens: str) -> Decimal:
    negative = any("-" in (tok or "") for tok in sign_tokens)
    return -value if negative else value


def _extract_numbers(text: str) -> list[_Claim]:
    """Extract unit-bearing financial numbers from citation-stripped text."""
    claims: list[_Claim] = []
    for match in _NUMBER_RE.finditer(text):
        gd = match.groupdict()
        try:
            if gd.get("cur"):
                value, decimals = _num(gd["curnum"])
                scale = _scale_mult(gd.get("curscale"))
                mag = _signed(value, gd.get("curpre"), gd.get("cursign")) * scale
                prec = scale * (Decimal(10) ** (-decimals))
                claims.append(_Claim(match.group(0).strip(), mag, prec, "currency",
                                     _CURRENCY.get(gd["cur"])))
            elif gd.get("pct"):
                value, decimals = _num(gd["pctnum"])
                mag = _signed(value, gd.get("pctsign"))
                claims.append(_Claim(match.group(0).strip(), mag,
                                     Decimal(10) ** (-decimals), "percent", None))
            elif gd.get("rat"):
                value, decimals = _num(gd["ratnum"])
                mag = _signed(value, gd.get("ratsign"))
                claims.append(_Claim(match.group(0).strip(), mag,
                                     Decimal(10) ** (-decimals), "ratio", None))
            elif gd.get("scale"):
                value, decimals = _num(gd["scnum"])
                scale = _scale_mult(gd["scale"])
                mag = _signed(value, gd.get("scsign")) * scale
                prec = scale * (Decimal(10) ** (-decimals))
                claims.append(_Claim(match.group(0).strip(), mag, prec, "scaled", None))
            elif gd.get("suf"):
                value, decimals = _num(gd["sufnum"])
                scale = _scale_mult(gd["suf"])
                mag = _signed(value, gd.get("sufsign")) * scale
                prec = scale * (Decimal(10) ** (-decimals))
                claims.append(_Claim(match.group(0).strip(), mag, prec, "scaled", None))
        except (InvalidOperation, ValueError):
            continue
    return claims


def _normalize_evidence(value: Any, unit: Optional[str]) -> Optional[_Evidence]:
    """Normalize an evidence value + unit into a comparable magnitude."""
    if value is None:
        return None
    try:
        base = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    unit_l = str(unit or "").strip().lower()

    if "%" in unit_l or "percent" in unit_l or "pct" in unit_l:
        return _Evidence(base, "percent", None)
    if unit_l in {"x", "ratio", "multiple", "mult"} or unit_l.endswith("_ratio"):
        return _Evidence(base, "ratio", None)

    scale = Decimal(1)
    for token in ("trillion", "billion", "million", "thousand", "bn", "mn", "tn"):
        if token in unit_l:
            scale = _SCALE[token]
            break
    currency = None
    if "usd" in unit_l or "$" in unit_l or "dollar" in unit_l:
        currency = "USD"
    elif "eur" in unit_l or "€" in unit_l:
        currency = "EUR"
    elif "gbp" in unit_l or "£" in unit_l:
        currency = "GBP"
    elif "jpy" in unit_l or "¥" in unit_l or "yen" in unit_l:
        currency = "JPY"

    magnitude = base * scale
    if currency:
        return _Evidence(magnitude, "currency", currency)
    return _Evidence(magnitude, "scaled" if scale != 1 else "number", None)


def _compatible(claim: _Claim, ev: _Evidence) -> bool:
    if claim.kind == "percent":
        return ev.kind in {"percent", "ratio"}
    if claim.kind == "ratio":
        return ev.kind == "ratio"
    if claim.kind in {"currency", "scaled"}:
        if ev.kind not in {"currency", "scaled", "number"}:
            return False
        if claim.kind == "currency" and claim.currency and ev.currency \
                and claim.currency != ev.currency:
            return False
        return True
    return False


def _value_matches(claim: _Claim, ev: _Evidence) -> bool:
    half = claim.precision / Decimal(2)
    if claim.kind == "percent" and ev.kind == "ratio":
        # A percentage may cite a fractional ratio (0.70 vs "70%").
        return abs(claim.magnitude / Decimal(100) - ev.magnitude) <= half / Decimal(100)
    return abs(claim.magnitude - ev.magnitude) <= half


# ── Citation parsing ──────────────────────────────────────────────────────

def _classify_bracket(content: str) -> tuple[Optional[str], str]:
    """Classify one ``[...]`` token: evidence | legacy | malformed | None."""
    stripped = content.strip()
    if re.fullmatch(r"E\d+", stripped):
        return "evidence", stripped
    if _LEGACY_RE.match(stripped):
        return "legacy", _LEGACY_RE.match(stripped).group(1)
    if stripped and stripped[0] in "Ee":
        rest = stripped[1:]
        if rest == "" or (rest and rest[0].isdigit()):
            # "[E]" or "[E12x]" — a botched evidence id, never a real citation.
            return "malformed", stripped
    return None, stripped


def _parse_citations(answer: str, by_id: dict) -> list[CitationRecord]:
    """Parse every citation in the answer into a deduplicated record list."""
    records: list[CitationRecord] = []
    seen: set = set()
    for match in _BRACKET_RE.finditer(answer):
        kind, payload = _classify_bracket(match.group(1))
        if kind == "evidence":
            key = ("e", payload)
            if key in seen:
                continue
            seen.add(key)
            item = by_id.get(payload)
            if item is not None:
                records.append(CitationRecord(
                    evidence_id=payload,
                    source_type=item.source_type,
                    ticker=item.ticker,
                    metric=item.metric,
                    period=item.period,
                    source_url=item.source_url,
                    support_status="supported",
                    item_type=item.item_type,
                    event_type=item.event_type,
                    authority_tier=item.authority_tier,
                    source=item.source or item.source_type,
                    date_semantics=dict(item.date_semantics),
                    canonical_security=item.canonical_security,
                    coverage_tier=item.coverage_tier,
                    source_category=item.source_category,
                    provider=item.provider,
                    publisher=item.publisher,
                ))
            else:
                records.append(CitationRecord(
                    evidence_id=payload, source_type=None, ticker=None,
                    metric=None, period=None, source_url=None,
                    support_status="missing"))
        elif kind == "legacy":
            parts = [p.strip() for p in payload.split("/")]
            source_type = parts[0] if parts and parts[0] else None
            ticker = parts[1] if len(parts) > 1 and parts[1] else None
            key = ("l", source_type, ticker)
            if key in seen:
                continue
            seen.add(key)
            records.append(CitationRecord(
                evidence_id=None, source_type=source_type, ticker=ticker,
                metric=None, period=None, source_url=None,
                support_status="supported"))
        elif kind == "malformed":
            key = ("m", payload)
            if key in seen:
                continue
            seen.add(key)
            records.append(CitationRecord(
                evidence_id=payload, source_type=None, ticker=None,
                metric=None, period=None, source_url=None,
                support_status="malformed"))
    return records


# ── Sentence / claim assembly ─────────────────────────────────────────────

def _split_sentences(answer: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", answer or "")
    return [p for p in pieces if p and p.strip()]


def _sentence_citations(sentence: str, by_id: dict) -> tuple[list[EvidenceItem], list[str]]:
    """Resolve the ledger items a sentence's citations point at.

    Returns ``(cited_items, evidence_ids)`` where ``cited_items`` includes
    both ``[E#]`` items that resolve and ledger items matched by a legacy
    ``[Source: type/ticker]`` label (the compatibility window).
    """
    items: list[EvidenceItem] = []
    e_ids: list[str] = []
    legacy: list[tuple[Optional[str], Optional[str]]] = []
    for match in _BRACKET_RE.finditer(sentence):
        kind, payload = _classify_bracket(match.group(1))
        if kind == "evidence":
            e_ids.append(payload)
            item = by_id.get(payload)
            if item is not None:
                items.append(item)
        elif kind == "legacy":
            parts = [p.strip().lower() for p in payload.split("/")]
            legacy.append((parts[0] if parts else None,
                           parts[1] if len(parts) > 1 else None))
    if legacy:
        for item in by_id.values():
            st = str(item.source_type or "").lower()
            tk = str(item.ticker or "").lower()
            for src, tick in legacy:
                if src and src in st and (not tick or tick == tk) and item not in items:
                    items.append(item)
    return items, e_ids


def _is_example(sentence: str) -> bool:
    low = sentence.lower()
    return any(marker in low for marker in _EXAMPLE_MARKERS)


_REASON_BUCKET = {
    "unit_mismatch": "unit",
    "period_mismatch": "period",
    "entity_mismatch": "entity",
    "value_mismatch": "value",
    "no_matching_evidence": "value",
    "uncited": "value",
}


def _match_against_items(
    claim: _Claim,
    cited_items: list[EvidenceItem],
    calculations: Optional[list[dict]],
    sentence_years: set,
    sentence_entities: set,
) -> Optional[tuple[str, Optional[str], Optional[str]]]:
    """Return ``(status, evidence_id, reason)`` or ``None`` if nothing matched."""
    best: Optional[tuple[str, Optional[str], Optional[str]]] = None
    for item in cited_items:
        ev = _normalize_evidence(item.value, item.unit)
        if ev is None:
            continue
        if not _compatible(claim, ev):
            best = best or ("unsupported", item.evidence_id, "unit_mismatch")
            continue
        if not _value_matches(claim, ev):
            best = best or ("unsupported", item.evidence_id, "value_mismatch")
            continue
        item_years = set(_YEAR_RE.findall(str(item.period or "")))
        if item_years and sentence_years and item_years.isdisjoint(sentence_years):
            best = best or ("unsupported", item.evidence_id, "period_mismatch")
            continue
        entities = {str(e).upper() for e in (item.entities or ()) if e}
        if entities and sentence_entities and entities.isdisjoint(sentence_entities):
            best = best or ("unsupported", item.evidence_id, "entity_mismatch")
            continue
        return "supported", item.evidence_id, None
    for calc in calculations or []:
        cv = calc.get("result") if "result" in calc else calc.get("value")
        ev = _normalize_evidence(cv, calc.get("unit"))
        if ev and _compatible(claim, ev) and _value_matches(claim, ev):
            return "supported", str(calc.get("id") or "calculation"), "calculation"
    return best


def validate_answer(
    answer: str,
    ledger,
    *,
    calculations: Optional[list[dict]] = None,
    require_evidence_ids: bool = False,
) -> AnswerValidation:
    """Validate an answer's numeric claims and citations against the ledger.

    ``ledger`` may be a list of :class:`EvidenceItem`, a list of plain dicts,
    or an ``{evidence_id: item}`` mapping. ``calculations`` is an optional list
    of recorded deterministic calculation dicts (``result``/``value``, ``unit``,
    ``id``). Never raises for well-formed input — malformed rows are skipped.
    """
    from .stream_events import current_emitter

    emitter = current_emitter()
    started_at = perf_counter()
    if emitter is not None:
        emitter.stage("validate", "started")
    items = _as_items(ledger)
    by_id = {it.evidence_id: it for it in items if it.evidence_id}
    all_tickers = {str(it.ticker).upper() for it in items if it.ticker}

    citation_records = _parse_citations(answer or "", by_id)
    supported_citations = sum(
        1 for c in citation_records if c.support_status == "supported")
    citation_support_rate = (
        round(supported_citations / len(citation_records), 4)
        if citation_records else None
    )

    claims: list[NumericClaim] = []
    mismatch = {"unit": 0, "period": 0, "entity": 0, "value": 0}
    for idx, sentence in enumerate(_split_sentences(answer or "")):
        if _is_example(sentence):
            continue
        cited_items, e_ids = _sentence_citations(sentence, by_id)
        sentence_years = set(_YEAR_RE.findall(sentence))
        sentence_entities = {t for t in _UPPER_TOKEN_RE.findall(sentence)
                             if t in all_tickers}
        cleaned = _BRACKET_RE.sub(" ", sentence)
        for numeric in _extract_numbers(cleaned):
            status, matched_id, reason = _classify_claim(
                numeric, cited_items, e_ids, items, calculations,
                sentence_years, sentence_entities)
            if status == "unsupported":
                mismatch[_REASON_BUCKET.get(reason or "value", "value")] += 1
            claims.append(NumericClaim(
                text=numeric.text, sentence_index=idx, kind=numeric.kind,
                citations=tuple(e_ids), status=status,
                matched_evidence_id=matched_id, reason=reason))

    n_sup = sum(1 for c in claims if c.status == "supported")
    n_unsup = sum(1 for c in claims if c.status == "unsupported")
    n_amb = sum(1 for c in claims if c.status == "ambiguous")

    citations_missing = sum(1 for c in citation_records if c.support_status == "missing")
    citations_malformed = sum(
        1 for c in citation_records if c.support_status == "malformed")
    violations = n_unsup > 0 or citations_missing > 0 or citations_malformed > 0
    if violations:
        status = "unsupported"
    elif not claims and not citation_records:
        status = "no_claims"
    else:
        status = "supported"

    result = AnswerValidation(
        status=status,
        citation_support_rate=citation_support_rate,
        numeric_claims_supported=n_sup,
        numeric_claims_unsupported=n_unsup,
        numeric_claims_ambiguous=n_amb,
        citations=tuple(citation_records),
        claims=tuple(claims),
        mismatch_counts=mismatch,
    )
    if emitter is not None:
        emitter.stage(
            "validate", "completed",
            elapsed_ms=(perf_counter() - started_at) * 1000,
            reason=result.status,
        )
    return result


def _classify_claim(
    numeric: _Claim,
    cited_items: list[EvidenceItem],
    e_ids: list[str],
    all_items: list[EvidenceItem],
    calculations: Optional[list[dict]],
    sentence_years: set,
    sentence_entities: set,
) -> tuple[str, Optional[str], Optional[str]]:
    """Return the ``(status, matched_id, reason)`` for one number."""
    if cited_items:
        result = _match_against_items(
            numeric, cited_items, calculations, sentence_years, sentence_entities)
        if result is not None:
            return result
        return "unsupported", None, "no_matching_evidence"
    # A [E#] citation was written but does not resolve to any ledger item.
    if e_ids:
        return "unsupported", None, "no_matching_evidence"
    # Uncited: an ambiguous match if the value is present anywhere in the ledger.
    for item in all_items:
        ev = _normalize_evidence(item.value, item.unit)
        if ev and _compatible(numeric, ev) and _value_matches(numeric, ev):
            return "ambiguous", item.evidence_id, "uncited_but_present"
    for calc in calculations or []:
        cv = calc.get("result") if "result" in calc else calc.get("value")
        ev = _normalize_evidence(cv, calc.get("unit"))
        if ev and _compatible(numeric, ev) and _value_matches(numeric, ev):
            return "ambiguous", str(calc.get("id") or "calculation"), "uncited_but_present"
    return "unsupported", None, "uncited"


def _as_items(ledger) -> list[EvidenceItem]:
    """Coerce a ledger argument into a list of :class:`EvidenceItem`."""
    if ledger is None:
        return []
    raw = ledger.values() if isinstance(ledger, dict) else ledger
    items: list[EvidenceItem] = []
    for entry in raw:
        if isinstance(entry, EvidenceItem):
            items.append(entry)
        elif isinstance(entry, dict):
            ticker = entry.get("ticker")
            entities = entry.get("entities")
            if not entities and ticker:
                entities = (str(ticker).upper(),)
            items.append(EvidenceItem(
                kind=entry.get("kind", "fact"),
                evidence_id=entry.get("evidence_id", ""),
                ticker=ticker,
                entities=tuple(entities or ()),
                metric=entry.get("metric"),
                value=entry.get("value"),
                unit=entry.get("unit"),
                period=entry.get("period"),
                source_type=entry.get("source_type") or entry.get("source"),
                source_url=entry.get("source_url"),
            ))
    return items

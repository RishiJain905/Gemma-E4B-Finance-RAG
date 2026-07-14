"""src/sec/event_classifier.py
Deterministic, explainable SEC filing event classification and term extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence


RULE_VERSION = "sec-events-v1"
_TEXT_LIMIT = 250_000


@dataclass(frozen=True)
class ClassifiedSECEvent:
    """One deterministic event signal emitted from SEC filing evidence."""

    event_type: str
    rule_version: str
    classification_reason: str
    amount: float | None = None
    currency: str | None = None
    security_type: str | None = None
    maturity: str | None = None
    rate: float | None = None
    source_document_types: tuple[str, ...] = ()


class SECEventClassifier:
    """Apply bounded form, item, exhibit, and text rules without model calls."""

    _EVENT_TYPES = (
        "debt_raise", "equity_raise", "convertible_offering",
        "shelf_registration", "prospectus_update", "acquisition", "divestiture",
        "earnings_release", "guidance_change", "leadership_change", "auditor_change",
        "buyback", "dividend_change", "insider_transaction",
        "beneficial_ownership_change",
    )
    _DEBT = re.compile(
        r"\b(?:senior|subordinated|secured|unsecured|convertible)?\s*"
        r"(?:notes?|debentures?|bonds?|debt securities)\b",
        re.IGNORECASE,
    )
    _EQUITY = re.compile(
        r"\b(?:common stock|ordinary shares?|preferred stock|equity securities|"
        r"public offering|at-the-market offering)\b",
        re.IGNORECASE,
    )
    _CONVERTIBLE = re.compile(
        r"\bconvertible\s+(?:senior\s+|subordinated\s+)?(?:notes?|debentures?|securities)\b",
        re.IGNORECASE,
    )

    def rule_inventory(self) -> tuple[str, ...]:
        """Return the complete versioned label inventory."""
        return self._EVENT_TYPES

    def classify(
        self,
        filing: Mapping[str, object],
        documents: Sequence[Mapping[str, object]] = (),
    ) -> list[ClassifiedSECEvent]:
        """Return zero or more deterministic events for one SEC accession."""
        form = self._normalize_form(filing.get("form") or filing.get("filing_type"))
        items = self._items(filing)
        evidence = self._evidence(filing, documents)
        text = "\n".join(value for _, value in evidence)[:_TEXT_LIMIT]
        document_types = tuple(dict.fromkeys(kind for kind, _ in evidence))
        matches: dict[str, str] = {}

        def emit(event_type: str, reason: str) -> None:
            matches.setdefault(event_type, reason)

        capital_form = form in {
            "S-1", "S-1/A", "S-3", "S-3/A", "S-3ASR", "S-3ASR/A",
            "424B2", "424B3", "424B5", "FWP",
        }
        offering_signal = form in {"424B2", "424B3", "424B5", "FWP"} or bool(
            re.search(
                r"\b(?:we are offering|initial public offering|firm commitment offering|"
                r"underwritten offering|offering of)\b",
                text,
                re.I,
            )
        )
        if (capital_form and offering_signal and self._DEBT.search(text)) or "2.03" in items:
            emit("debt_raise", self._reason(form, items, "debt-security terms or Item 2.03"))
        if (capital_form and offering_signal and self._EQUITY.search(text)) or "3.02" in items:
            emit("equity_raise", self._reason(form, items, "equity-offering terms or Item 3.02"))
        if capital_form and offering_signal and self._CONVERTIBLE.search(text):
            emit("convertible_offering", self._reason(form, items, "explicit convertible security terms"))
        if form in {"S-3", "S-3/A", "S-3ASR", "S-3ASR/A"}:
            emit("shelf_registration", self._reason(form, items, "shelf registration form"))
        if form in {"424B2", "424B3", "424B5", "FWP"}:
            emit("prospectus_update", self._reason(form, items, "prospectus or free-writing-prospectus form"))

        if "2.01" in items and re.search(r"\b(?:acquir(?:e|ed|ing)|acquisition|merger)\b", text, re.I):
            emit("acquisition", self._reason(form, items, "Item 2.01 acquisition language"))
        if "2.01" in items and re.search(
            r"\b(?:completed\s+(?:the\s+)?disposition|divestiture|divested|sold)\b",
            text,
            re.I,
        ):
            emit("divestiture", self._reason(form, items, "Item 2.01 disposition language"))
        if "2.02" in items or (
            "EX-99.1" in document_types
            and re.search(r"\b(?:earnings|financial results|results of operations)\b", text, re.I)
        ):
            emit("earnings_release", self._reason(form, items, "Item 2.02 or EX-99.1 earnings release"))
        if items.intersection({"2.02", "7.01", "8.01"}) and re.search(
            r"\b(?:guidance|outlook|forecast)\b", text, re.I,
        ) and re.search(r"\b(?:raise[sd]?|lower(?:ed|s)?|revis(?:e|ed)|withdrawn?|updated?)\b", text, re.I):
            emit("guidance_change", self._reason(form, items, "explicit guidance change"))
        if "5.02" in items:
            emit("leadership_change", self._reason(form, items, "Item 5.02 leadership disclosure"))
        if "4.01" in items:
            emit("auditor_change", self._reason(form, items, "Item 4.01 auditor disclosure"))
        if items.intersection({"7.01", "8.01"}) and re.search(
            r"\b(?:share repurchase|stock repurchase|buyback)\b", text, re.I,
        ):
            emit("buyback", self._reason(form, items, "repurchase language in material disclosure"))
        if re.search(r"\bdividend\b", text, re.I) and re.search(
            r"\b(?:increase[sd]?|reduce[sd]?|decrease[sd]?|suspend(?:ed|s)?|declar(?:e|ed))\b",
            text,
            re.I,
        ):
            emit("dividend_change", self._reason(form, items, "explicit dividend action"))
        if form in {"3", "3/A", "4", "4/A", "5", "5/A"}:
            emit("insider_transaction", self._reason(form, items, "Section 16 ownership form"))
        if form in {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}:
            emit("beneficial_ownership_change", self._reason(form, items, "Schedule 13 ownership form"))

        financing = self._financing_terms(text) if matches.keys() & {
            "debt_raise", "equity_raise", "convertible_offering",
        } else {}
        return [
            ClassifiedSECEvent(
                event_type=event_type,
                rule_version=RULE_VERSION,
                classification_reason=matches[event_type],
                amount=financing.get("amount"),
                currency=financing.get("currency"),
                security_type=financing.get("security_type"),
                maturity=financing.get("maturity"),
                rate=financing.get("rate"),
                source_document_types=document_types,
            )
            for event_type in self._EVENT_TYPES
            if event_type in matches
        ]

    @staticmethod
    def _normalize_form(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().upper())

    @staticmethod
    def _items(filing: Mapping[str, object]) -> set[str]:
        raw = filing.get("items") or filing.get("item_numbers") or ()
        if isinstance(raw, str):
            raw = re.findall(r"\b\d+\.\d{2}\b", raw)
        items = {str(item).strip() for item in raw if str(item).strip()}
        text = str(filing.get("text") or filing.get("primary_text") or "")[:_TEXT_LIMIT]
        items.update(re.findall(r"(?i)\bitem\s+(\d+\.\d{2})\b", text))
        return items

    @staticmethod
    def _evidence(
        filing: Mapping[str, object], documents: Sequence[Mapping[str, object]],
    ) -> list[tuple[str, str]]:
        evidence: list[tuple[str, str]] = []
        primary = str(filing.get("text") or filing.get("primary_text") or "")[:_TEXT_LIMIT]
        if primary:
            evidence.append(("PRIMARY", primary))
        for document in documents:
            text = str(document.get("text") or "")[:_TEXT_LIMIT]
            if not text:
                continue
            kind = str(document.get("document_type") or document.get("type") or "EXHIBIT").upper()
            if kind == "PRIMARY" and any(existing_kind == "PRIMARY" for existing_kind, _ in evidence):
                continue
            evidence.append((kind, text))
        return evidence

    @staticmethod
    def _reason(form: str, items: set[str], signal: str) -> str:
        item_text = f"; items={','.join(sorted(items))}" if items else ""
        return f"rule={RULE_VERSION}; form={form or 'UNKNOWN'}{item_text}; signal={signal}"

    def _financing_terms(self, text: str) -> dict[str, object]:
        result: dict[str, object] = {}
        amount = re.search(
            r"\b(?:aggregate\s+principal\s+amount|offering\s+(?:amount|size)|"
            r"gross\s+proceeds|we\s+are\s+offering)\b.{0,80}?"
            r"(?P<currency>US\$|\$|USD\s*)\s*(?P<number>\d+(?:\.\d+)?)\s*"
            r"(?P<scale>billion|million|thousand)\b",
            text,
            re.I | re.S,
        )
        if amount:
            multiplier = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
            result["amount"] = float(amount.group("number")) * multiplier[amount.group("scale").lower()]
            result["currency"] = "USD"
        rate = re.search(
            r"(?:\b(?:bear(?:ing)?\s+)?interest(?:\s+rate)?\s+(?:of|at|is)\s*"
            r"(?P<prefixed>\d{1,2}(?:\.\d{1,4})?)\s*%|"
            r"\b(?P<suffixed>\d{1,2}(?:\.\d{1,4})?)\s*%\s*(?:per annum|per year))",
            text,
            re.I,
        )
        if rate:
            result["rate"] = float(rate.group("prefixed") or rate.group("suffixed"))
        maturity = re.search(r"\b(?:due|maturing(?:\s+on)?)\s+(?:[A-Z][a-z]+\s+\d{1,2},\s+)?(20\d{2})\b", text, re.I)
        if maturity:
            result["maturity"] = maturity.group(1)
        security = self._CONVERTIBLE.search(text) or self._DEBT.search(text) or self._EQUITY.search(text)
        if security:
            result["security_type"] = re.sub(r"\s+", " ", security.group(0).strip().lower())
        return result

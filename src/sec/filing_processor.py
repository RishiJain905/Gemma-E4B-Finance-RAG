"""
src/sec/filing_processor.py
Filing processing pipeline — orchestrator that chains:
  Discover (EDGAR) → Download (EDGAR) → Parse (TraceAlchemy) → Store (hybrid)

Usage:
    processor = FilingProcessor()
    processor.process_core_tickers()   # Process all unprocessed filings
    processor.process_ticker("NVDA")   # Process just NVDA's filings
"""

import logging
import json
import re
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Optional

from .edgar_fetcher import SECEdgarFilingFetcher
from .event_classifier import RULE_VERSION, SECEventClassifier
from .filing_parser import TraceAlchemyFilingParser
from .filing_sections import split_filing_sections
from src.storage.store import Store
from src.ingestion.normalization import NORMALIZATION_VERSION, content_hash
from src.ingestion.records import EventRecord, NarrativeRecord

logger = logging.getLogger(__name__)


class FilingProcessor:
    """Orchestrates the SEC filing pipeline.

    Chains:
        1. SECEdgarFilingFetcher — discover + download filings
        2. TraceAlchemyFilingParser — extract structured facts
        3. Store.process_filing — save to SQLite + ChromaDB

    Usage:
        processor = FilingProcessor()

        # Full pipeline: discover new filings, then process all unprocessed
        processor.discover_and_process_all()

        # Or step by step:
        processor.discover_new_filings()
        processor.process_pending_filings()
    """

    def __init__(
        self,
        store: Optional[Store] = None,
        fetcher: Optional[SECEdgarFilingFetcher] = None,
        parser: Optional[TraceAlchemyFilingParser] = None,
        sec_config: Optional[dict] = None,
        parsed_dir: Optional[Path] = None,
    ):
        self.store = store or Store()
        self.sec_config = sec_config or self._load_sec_config()
        self.fetcher = fetcher or SECEdgarFilingFetcher(
            store=self.store, sec_config=self.sec_config,
        )
        self.parser = parser or TraceAlchemyFilingParser()
        self.event_classifier = SECEventClassifier()
        self.index_filing_text = bool(self.sec_config.get("index_filing_text", False))
        self.max_sections_per_filing = min(
            max(int(self.sec_config.get("max_sections_per_filing", 200)), 1), 500,
        )
        self.max_section_chars = min(
            max(int(self.sec_config.get("max_section_chars", 2_000_000)), 1),
            10_000_000,
        )
        self.index_forms = {
            str(form).upper()
            for form in self.sec_config.get("index_forms", ["10-K", "10-Q", "8-K"])
        }
        self.index_event_filings = bool(self.sec_config.get("index_event_filings", True))
        raw_db_path = getattr(self.store.sqlite, "db_path", None)
        db_path = Path(raw_db_path) if isinstance(raw_db_path, (str, Path)) else Path("data/finance.db")
        self.parsed_dir = (
            Path(parsed_dir) if parsed_dir is not None else db_path.parent / "sec" / "parsed"
        )
        self._index_run_counts = self._empty_index_counts()

    @staticmethod
    def _empty_index_counts() -> dict[str, int]:
        return {
            "sections_written": 0,
            "chunks_written": 0,
            "replacements": 0,
            "skipped": 0,
            "index_pending": 0,
        }

    @staticmethod
    def _load_sec_config() -> dict:
        """Load filing-text rollout controls with safe built-in defaults."""
        defaults = {
            "index_filing_text": False,
            "max_sections_per_filing": 200,
            "max_section_chars": 2_000_000,
            "index_forms": ["10-K", "10-Q", "8-K"],
            "index_event_filings": True,
        }
        config_path = Path(__file__).parents[2] / "configs" / "sec.yaml"
        if not config_path.exists():
            return defaults
        try:
            import yaml

            with open(config_path, encoding="utf-8") as config_file:
                loaded = yaml.safe_load(config_file) or {}
            defaults.update(loaded.get("sec", loaded))
        except Exception as error:  # noqa: BLE001 - retain safe defaults
            logger.warning("Could not load SEC filing config: %s", error)
        return defaults

    def _persist_parsed_artifact(self, accession: str, text: str) -> Path:
        """Write the parsed filing artifact before attempting vector indexing."""
        safe_accession = "".join(
            char for char in accession if char.isalnum() or char in "-_"
        )
        if not safe_accession:
            raise ValueError("Filing accession cannot produce a safe parsed path")
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        path = self.parsed_dir / f"{safe_accession}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    # ── Discovery Phase ───────────────────────────────

    def discover_new_filings(self, ticker: str) -> int:
        """Discover and register new filings for a single ticker.

        Returns:
            Number of newly registered filings.
        """
        return self.fetcher.register_discovered_filings(ticker)

    def discover_all_core_tickers(self) -> dict[str, int]:
        """Discover and register new filings for all core tickers.

        Returns:
            Dict of {ticker: new_filing_count}
        """
        return self.fetcher.register_all_core_tickers()

    # ── Processing Phase ──────────────────────────────

    def process_pending_filings(self, limit: int = 10) -> dict:
        """Process all unprocessed filings from the filings table.

        For each unprocessed filing:
          1. Check that the filing text is available (download if needed)
          2. Run TraceAlchemy parsing
          3. Store via Store.process_filing()

        Args:
            limit: Max filings to process in one batch

        Returns:
            {processed: N, failed: N, errors: [...]}
        """
        self._index_run_counts = self._empty_index_counts()
        unprocessed = self.store.sqlite.get_unprocessed_filings(limit=limit)
        if not unprocessed:
            logger.info("No unprocessed filings found")
            result = {"processed": 0, "failed": 0, "errors": []}
            if self.index_filing_text:
                result.update(self._index_run_counts)
            return result

        logger.info("Processing %d pending filings...", len(unprocessed))
        result = {"processed": 0, "failed": 0, "errors": []}

        for filing in unprocessed:
            try:
                success = self._process_single_filing(filing)
                if success:
                    result["processed"] += 1
                else:
                    result["failed"] += 1
                    result["errors"].append(
                        f"{filing['ticker']} {filing['filing_type']} "
                        f"({filing['accession']}): processing returned no facts"
                    )
            except Exception as e:
                result["failed"] += 1
                result["errors"].append(
                    f"{filing['ticker']} {filing['filing_type']} "
                    f"({filing['accession']}): {e}"
                )
                logger.exception("Failed to process filing %s", filing["accession"])

        logger.info(
            "Filing processing complete: %d processed, %d failed",
            result["processed"], result["failed"],
        )
        if self.index_filing_text:
            result.update(self._index_run_counts)
        return result

    def process_ticker(self, ticker: str, limit: int = 5) -> dict:
        """Process unprocessed filings for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Max filings to process

        Returns:
            {processed: N, failed: N, errors: [...]}
        """
        unprocessed = self.store.sqlite.get_unprocessed_filings(limit=limit)
        ticker_filings = [f for f in unprocessed if f.get("ticker", "").upper() == ticker.upper()]

        if not ticker_filings:
            logger.info("No unprocessed filings for %s", ticker)
            return {"processed": 0, "failed": 0, "errors": []}

        result = {"processed": 0, "failed": 0, "errors": []}
        for filing in ticker_filings:
            try:
                success = self._process_single_filing(filing)
                if success:
                    result["processed"] += 1
                else:
                    result["failed"] += 1
            except Exception as e:
                result["failed"] += 1
                result["errors"].append(str(e))

        return result

    # ── Combined Pipeline ─────────────────────────────

    def discover_and_process_all(self) -> dict:
        """Run the full pipeline: discover new filings, then process them.

        Returns:
            {
                "discovery": {ticker: new_count},
                "processing": {processed: N, failed: N, errors: [...]}
            }
        """
        # Phase 1: Discover
        discovery_result = self.discover_all_core_tickers()

        # Phase 2: Process
        processing_result = self.process_pending_filings(limit=50)

        return {
            "discovery": discovery_result,
            "processing": processing_result,
        }

    def discover_and_process_ticker(self, ticker: str) -> dict:
        """Discover + process for a single ticker."""
        new_count = self.discover_new_filings(ticker)
        process_result = self.process_ticker(ticker)
        return {
            "ticker": ticker,
            "new_filings_discovered": new_count,
            "processing": process_result,
        }

    # ── Internal ──────────────────────────────────────

    def _process_single_filing(self, filing: dict) -> bool:
        """Process one filing end-to-end.

        Steps:
          1. Download the filing text from EDGAR
          2. Run TraceAlchemy parser to extract facts
          3. Store via Store.process_filing()
          4. Log completion

        Returns:
            True if at least one fact was extracted and stored
        """
        accession = filing.get("accession", "unknown")
        ticker = filing.get("ticker", "")
        filing_type = filing.get("filing_type", "")
        period = filing.get("period", "")

        if str(filing.get("discovery_scope") or "deep").lower() == "broad":
            return self._process_event_filing(filing)

        logger.info(
            "Processing filing: %s %s %s (%s)",
            ticker, filing_type, period, accession,
        )

        # Step 1: Download
        text = self.fetcher.download_filing_text(filing)
        if not text:
            logger.warning("No text downloaded for filing %s, marking as error", accession)
            self.store.sqlite.mark_cache_stale(
                ticker, f"sec_{filing_type}_text",
                error=f"Download failed for {accession}",
            )
            return False

        # Step 2: Parse
        facts = self.parser.extract_facts_from_filing(
            ticker=ticker,
            filing_type=filing_type,
            filing_text=text,
            period=period,
        )

        if not facts:
            logger.warning(
                "No facts extracted from %s %s (%s)",
                ticker, filing_type, accession,
            )
            return False

        filing_record = {
            **filing,
            "source_type": f"sec_{filing_type.lower()}",
        }
        if not self.index_filing_text or filing_type.upper() not in self.index_forms:
            # Rollout disabled/form excluded: preserve the legacy path exactly.
            self.store.process_filing(
                filing_record=filing_record,
                extracted_text=text,
                extracted_facts=facts,
            )
            logger.info(
                "Successfully processed %s %s (%s): %d facts stored",
                ticker, filing_type, accession, len(facts),
            )
            return True

        started = time.monotonic()
        artifact_path = self._persist_parsed_artifact(accession, text)
        candidates = split_filing_sections(
            text, {**filing, "file_path": str(artifact_path)},
        )
        skipped = max(0, len(candidates) - self.max_sections_per_filing)
        usable_sections = []
        for section in candidates[: self.max_sections_per_filing]:
            if len(section.text) > self.max_section_chars:
                skipped += 1
                logger.warning(
                    "Skipping oversized SEC section %s (%d chars; cap %d)",
                    section.document_id, len(section.text), self.max_section_chars,
                )
                continue
            usable_sections.append(section)

        if not usable_sections:
            reason = "No usable filing sections after validation"
            self.store.sqlite.mark_filing_index_pending(
                accession, file_path=str(artifact_path), error=reason,
            )
            logger.warning("SEC filing %s index pending: %s", accession, reason)
            self._index_run_counts["skipped"] += skipped
            self._index_run_counts["index_pending"] += 1
            return False

        # Save structured facts without the legacy whole-document vector or
        # parsed mark. The final mark follows verified section indexing.
        self.store.process_filing(
            filing_record=filing_record,
            extracted_text=text,
            extracted_facts=facts,
            index_document=False,
            mark_parsed=False,
        )

        try:
            counts = self.store.add_filing_sections(usable_sections)
            if counts["sections_written"] != len(usable_sections):
                raise RuntimeError(
                    f"Expected {len(usable_sections)} section parents, "
                    f"stored {counts['sections_written']}"
                )
            indexed_total = self.store.count_filing_sections(accession)
            if indexed_total < len(usable_sections):
                raise RuntimeError(
                    f"Expected at least {len(usable_sections)} indexed sections, "
                    f"found {indexed_total}"
                )
        except Exception as error:  # noqa: BLE001 - retryable per-filing isolation
            self.store.sqlite.mark_filing_index_pending(
                accession, file_path=str(artifact_path), error=str(error),
            )
            logger.error(
                "SEC filing %s index pending after %.2fs: %s",
                accession, time.monotonic() - started, error,
            )
            self._index_run_counts["skipped"] += skipped
            self._index_run_counts["index_pending"] += 1
            return False

        self.store.sqlite.mark_filing_parsed(
            accession,
            embedding_id=f"sec:{accession}",
            file_path=str(artifact_path),
            section_count=counts["sections_written"],
            chunk_count=counts["chunks_written"],
        )
        for key in ("sections_written", "chunks_written", "replacements"):
            self._index_run_counts[key] += counts[key]
        self._index_run_counts["skipped"] += skipped + counts["skipped"]
        logger.info(
            "Indexed SEC filing %s: %d sections, %d chunks, %d replacements, "
            "%d skips in %.2fs",
            accession, counts["sections_written"], counts["chunks_written"],
            counts["replacements"], skipped + counts["skipped"],
            time.monotonic() - started,
        )
        return True

    def _process_event_filing(self, filing: dict) -> bool:
        """Persist selected broad filing evidence and deterministic events."""
        accession = str(filing.get("accession") or "").strip()
        ticker = str(filing.get("ticker") or "").strip().upper()
        form = str(filing.get("filing_type") or filing.get("form") or "").strip().upper()
        documents = self.fetcher.download_relevant_documents(filing)
        if not documents:
            logger.warning("No selected SEC documents downloaded for %s", accession)
            return False
        primary = next(
            (document for document in documents if document.get("document_type") == "PRIMARY"),
            documents[0],
        )
        items = self._json_list(filing.get("items_json") or filing.get("items"))
        primary_text = str(primary.get("text") or "")
        items = sorted(set(items) | set(
            re.findall(r"(?i)\bitem\s+(\d+\.\d{2})\b", primary_text)
        ))
        context = {
            **filing,
            "form": form,
            "text": primary_text,
            "items": items,
        }
        events = self.event_classifier.classify(context, documents)
        security = self.store.resolve_security(ticker)
        if not security:
            logger.error("Registered broad SEC filing %s has no canonical security", accession)
            return False
        security_id = str(security["security_id"])
        memberships = self.store.list_memberships(security_id=security_id, active=True)
        index_codes = tuple(sorted({str(row["index_code"]) for row in memberships}))
        now = datetime.now(timezone.utc).isoformat()
        filing_date = str(filing.get("filing_date") or "")
        published_at = f"{filing_date}T00:00:00Z" if filing_date else None
        source_url = str(primary.get("source_url") or filing.get("source_url") or "")
        body = "\n\n".join(
            f"[{document.get('document_type', 'EXHIBIT')}]\n{document.get('text', '')}"
            for document in documents
            if document.get("text")
        )[:1_000_000]
        item_id = f"sec:{accession}"
        narrative = NarrativeRecord(
            corpus_item_id=item_id,
            source_name="sec",
            source_category="regulatory_filing",
            provider_record_id=accession,
            original_publisher="U.S. Securities and Exchange Commission",
            item_type="sec_filing",
            event_type=events[0].event_type if len(events) == 1 else None,
            title=f"{ticker} {form} filed {filing_date}".strip(),
            body=body,
            published_at=published_at,
            observed_at=now,
            accessed_at=now,
            ingested_at=now,
            source_url=source_url,
            canonical_url=source_url,
            license_label="sec_public_filing",
            normalization_version=NORMALIZATION_VERSION,
            content_hash=content_hash(body),
            document_family=item_id,
            security_ids=(security_id,),
            tickers=(ticker,),
            index_codes=index_codes,
            sectors=(str(security["sector"]),) if security.get("sector") else (),
            metadata={
                "accession": accession,
                "form": form,
                "filing_item": ",".join(items),
                "exhibit": ",".join(
                    str(document.get("document_type") or "") for document in documents[1:]
                ),
            },
            indexing_status="pending" if self.index_event_filings else "not_applicable",
            evidence_authority="direct_sec",
        )
        narrative_result = self.store.upsert_narrative(narrative)
        if narrative_result.get("indexing_status") == "error":
            logger.warning("SEC event narrative indexing remains retryable for %s", accession)
            return False
        linked_item_id = str(narrative_result["corpus_item_id"])

        for event in events:
            stable_key = f"{accession}:{event.event_type}:{event.rule_version}"
            event_metadata = {"accession": accession, "form": form}
            if items:
                event_metadata["filing_item"] = ",".join(items)
            if event.source_document_types:
                event_metadata["exhibit"] = ",".join(event.source_document_types)
            if event.security_type:
                event_metadata["security_type"] = event.security_type
            if event.maturity:
                event_metadata["maturity"] = event.maturity
            self.store.upsert_event(EventRecord(
                event_id=f"sec:{stable_key}",
                event_type=event.event_type,
                effective_at=published_at,
                announced_at=published_at,
                status="announced",
                security_ids=(security_id,),
                source_corpus_item_ids=(linked_item_id,),
                source_name="sec",
                source_category="regulatory_filing",
                provider_record_id=stable_key,
                original_publisher="U.S. Securities and Exchange Commission",
                source_url=source_url,
                canonical_url=source_url,
                published_at=published_at,
                observed_at=now,
                accessed_at=now,
                ingested_at=now,
                license_label="sec_public_filing",
                normalization_version=NORMALIZATION_VERSION,
                amount=event.amount,
                currency=event.currency,
                rate=event.rate,
                classifier_version=RULE_VERSION,
                explanation=event.classification_reason,
                metadata=event_metadata,
                evidence_authority="direct_sec",
            ))

        if self.index_event_filings:
            sections = split_filing_sections(body, filing)
            if sections:
                try:
                    self.store.add_filing_sections(sections[: self.max_sections_per_filing])
                except Exception:  # noqa: BLE001 - narrative remains durable and replayable
                    logger.warning("SEC event section indexing failed for %s", accession, exc_info=True)
                    return False
        self.store.sqlite.mark_filing_parsed(accession, embedding_id=linked_item_id)
        return True

    @staticmethod
    def _json_list(value: object) -> list[str]:
        """Decode a stored JSON list while accepting already-normalized lists."""
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                return [value]
        return []

    # ── Status ─────────────────────────────────────────

    def status_report(self) -> dict:
        """Report on filing pipeline health.

        Returns:
            {
                "total_unprocessed": N,
                "total_processed": N,
                "filings_by_ticker": {ticker: {unprocessed: N, parsed: N}},
            }
        """
        all_filings = []
        with self.store.sqlite._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, status, COUNT(*) as cnt, "
                "SUM(index_section_count) as section_count, "
                "SUM(index_chunk_count) as chunk_count FROM filings "
                "GROUP BY ticker, status ORDER BY ticker"
            ).fetchall()
            all_filings = [dict(r) for r in rows]

        # Build per-ticker summary
        ticker_summary = {}
        total_unprocessed = 0
        total_parsed = 0
        total_index_pending = 0
        indexed_sections = 0
        indexed_chunks = 0

        for row in all_filings:
            ticker = row["ticker"]
            status = row["status"]
            cnt = row["cnt"]
            indexed_sections += int(row.get("section_count") or 0)
            indexed_chunks += int(row.get("chunk_count") or 0)

            if ticker not in ticker_summary:
                ticker_summary[ticker] = {"unprocessed": 0, "parsed": 0}

            if status == "unprocessed":
                ticker_summary[ticker]["unprocessed"] = cnt
                total_unprocessed += cnt
            elif status == "parsed":
                ticker_summary[ticker]["parsed"] = cnt
                total_parsed += cnt
            elif status == "index_pending":
                ticker_summary[ticker]["index_pending"] = cnt
                total_index_pending += cnt

        return {
            "total_unprocessed": total_unprocessed,
            "total_index_pending": total_index_pending,
            "total_parsed": total_parsed,
            "indexed_sections": indexed_sections,
            "indexed_chunks": indexed_chunks,
            "filings_by_ticker": ticker_summary,
        }

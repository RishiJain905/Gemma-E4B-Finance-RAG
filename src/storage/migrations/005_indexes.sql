-- Phase 2.3 lookup, deduplication, retention, and aggregate-query indexes.
CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker ON fundamentals(ticker);
CREATE INDEX IF NOT EXISTS idx_fundamentals_security ON fundamentals(security_id);
CREATE INDEX IF NOT EXISTS idx_fundamentals_metric ON fundamentals(metric);
CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_metric ON fundamentals(ticker, metric);
CREATE INDEX IF NOT EXISTS idx_fundamentals_period ON fundamentals(period);
CREATE INDEX IF NOT EXISTS idx_sec_companyfacts_ticker_concept_period
    ON sec_companyfacts(ticker, concept, period_end);
CREATE INDEX IF NOT EXISTS idx_sec_companyfacts_security ON sec_companyfacts(security_id);
CREATE INDEX IF NOT EXISTS idx_sec_companyfacts_ticker_filed_at
    ON sec_companyfacts(ticker, filed_at);
CREATE INDEX IF NOT EXISTS idx_sec_companyfacts_accession ON sec_companyfacts(accession);
CREATE INDEX IF NOT EXISTS idx_sec_companyfacts_ticker_kind_period
    ON sec_companyfacts(ticker, period_kind, period_end);
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_security ON filings(security_id);
CREATE INDEX IF NOT EXISTS idx_filings_status ON filings(status);
CREATE INDEX IF NOT EXISTS idx_cache_meta_status ON cache_meta(status);
CREATE INDEX IF NOT EXISTS idx_cache_meta_security ON cache_meta(security_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_run ON ingestion_log(run_id);
CREATE INDEX IF NOT EXISTS idx_securities_active_ticker ON securities(active, normalized_ticker);
CREATE INDEX IF NOT EXISTS idx_securities_cik ON securities(cik);
CREATE INDEX IF NOT EXISTS idx_securities_sector ON securities(sector);
CREATE INDEX IF NOT EXISTS idx_securities_industry ON securities(industry);
CREATE INDEX IF NOT EXISTS idx_securities_last_seen ON securities(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_security_aliases_lookup
    ON security_aliases(normalized_alias, provider, valid_from, valid_to);
CREATE UNIQUE INDEX IF NOT EXISTS idx_security_aliases_active_global
    ON security_aliases(normalized_alias) WHERE valid_to IS NULL AND provider IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_security_aliases_active_provider
    ON security_aliases(normalized_alias, provider)
    WHERE valid_to IS NULL AND provider IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memberships_active_index
    ON security_memberships(index_code, active, security_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memberships_one_active
    ON security_memberships(security_id, index_code) WHERE active = 1;
CREATE INDEX IF NOT EXISTS idx_universe_errors_run ON universe_errors(run_id);
CREATE INDEX IF NOT EXISTS idx_identity_errors_status
    ON identity_reconciliation_errors(status, issue_type, stage);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_items_provider_identity
    ON corpus_items(source, provider_record_id)
    WHERE provider_record_id IS NOT NULL AND provider_record_id <> '';
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_items_url_identity
    ON corpus_items(source, canonical_url, published_at)
    WHERE provider_record_id IS NULL AND canonical_url IS NOT NULL AND published_at IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_items_hash_identity
    ON corpus_items(source, content_hash)
    WHERE provider_record_id IS NULL AND canonical_url IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_items_document_family
    ON corpus_items(document_family_id);
CREATE INDEX IF NOT EXISTS idx_corpus_items_canonical_url ON corpus_items(canonical_url, published_at);
CREATE INDEX IF NOT EXISTS idx_corpus_items_content_hash ON corpus_items(content_hash);
CREATE INDEX IF NOT EXISTS idx_corpus_items_headline_window
    ON corpus_items(syndicated_key) WHERE syndicated_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_corpus_items_indexing_status
    ON corpus_items(indexing_status, updated_at);
CREATE INDEX IF NOT EXISTS idx_corpus_items_retention
    ON corpus_items(item_type, is_tombstone, indexing_status, published_at);
CREATE INDEX IF NOT EXISTS idx_corpus_items_source_category ON corpus_items(source_category);
CREATE INDEX IF NOT EXISTS idx_corpus_items_source ON corpus_items(source);
CREATE INDEX IF NOT EXISTS idx_corpus_items_item_type ON corpus_items(item_type);
CREATE INDEX IF NOT EXISTS idx_corpus_items_event_type
    ON corpus_items(event_type) WHERE event_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_corpus_item_sources_source
    ON corpus_item_sources(source_name, provider_record_id);
CREATE INDEX IF NOT EXISTS idx_corpus_item_securities_security
    ON corpus_item_securities(security_id, corpus_item_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_observations_provider
    ON corpus_observations(source_name, metric_id, provider_record_id, vintage_at)
    WHERE provider_record_id IS NOT NULL AND provider_record_id <> '';
CREATE INDEX IF NOT EXISTS idx_corpus_observations_metric_period
    ON corpus_observations(metric_id, period_end, vintage_at);
CREATE INDEX IF NOT EXISTS idx_corpus_observations_market_key
    ON corpus_observations(source_name, metric_id, period_end, tickers_json);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_events_provider
    ON corpus_events(source_name, provider_record_id)
    WHERE provider_record_id IS NOT NULL AND provider_record_id <> '';
CREATE INDEX IF NOT EXISTS idx_corpus_events_type_effective
    ON corpus_events(event_type, effective_at);
CREATE INDEX IF NOT EXISTS idx_source_cursors_status
    ON source_cursors(source, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_scheduler_runs_started ON scheduler_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_scheduler_run_sources_source
    ON scheduler_run_sources(source, ended_at DESC);
CREATE INDEX IF NOT EXISTS idx_bootstrap_partitions_status
    ON bootstrap_partitions(source, status, run_id);


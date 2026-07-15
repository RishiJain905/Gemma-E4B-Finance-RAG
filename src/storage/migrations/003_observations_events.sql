-- Phase 2.3 normalized structured observations and events/actions.
CREATE TABLE IF NOT EXISTS corpus_observations (
    observation_id TEXT PRIMARY KEY,
    metric_id TEXT NOT NULL,
    series_id TEXT,
    value_text TEXT NOT NULL,
    value_numeric REAL,
    unit TEXT NOT NULL,
    frequency TEXT NOT NULL,
    period_start TEXT,
    period_end TEXT NOT NULL,
    vintage_at TEXT,
    as_of_at TEXT,
    scope TEXT NOT NULL CHECK (scope IN ('security', 'sector', 'global')),
    tickers_json TEXT NOT NULL DEFAULT '[]',
    sector TEXT,
    source_name TEXT NOT NULL,
    source_category TEXT NOT NULL,
    provider_record_id TEXT,
    original_publisher TEXT,
    source_url TEXT NOT NULL,
    canonical_url TEXT,
    published_at TEXT,
    observed_at TEXT,
    accessed_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    license_label TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    evidence_authority TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS observation_securities (
    observation_id TEXT NOT NULL REFERENCES corpus_observations(observation_id) ON DELETE CASCADE,
    security_id TEXT NOT NULL REFERENCES securities(security_id),
    PRIMARY KEY (observation_id, security_id)
);

CREATE TABLE IF NOT EXISTS corpus_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    effective_at TEXT,
    announced_at TEXT,
    status TEXT NOT NULL,
    amount REAL,
    currency TEXT,
    rate REAL,
    ratio REAL,
    action_date TEXT,
    classifier_version TEXT,
    explanation TEXT CHECK (explanation IS NULL OR length(explanation) <= 4000),
    source_name TEXT NOT NULL,
    source_category TEXT NOT NULL,
    provider_record_id TEXT,
    original_publisher TEXT,
    source_url TEXT NOT NULL,
    canonical_url TEXT,
    published_at TEXT,
    observed_at TEXT,
    accessed_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    license_label TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    evidence_authority TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS event_securities (
    event_id TEXT NOT NULL REFERENCES corpus_events(event_id) ON DELETE CASCADE,
    security_id TEXT NOT NULL REFERENCES securities(security_id),
    PRIMARY KEY (event_id, security_id)
);

CREATE TABLE IF NOT EXISTS event_corpus_items (
    event_id TEXT NOT NULL REFERENCES corpus_events(event_id) ON DELETE CASCADE,
    corpus_item_id TEXT NOT NULL REFERENCES corpus_items(corpus_item_id),
    PRIMARY KEY (event_id, corpus_item_id)
);


-- Phase 2.3.7 follow-up: allow 'absent' daily-index dates.
-- EDGAR never publishes a daily index for market holidays and its S3 answers
-- 403 for the missing key forever; those dates must be durably skippable
-- instead of retried as rate limits (live incident: Juneteenth 2026-06-19
-- pinned the sec_filings discovery cursor for days). SQLite cannot widen a
-- CHECK constraint in place, so the table is rebuilt; it holds one row per
-- processed date, so the copy is trivial.
ALTER TABLE sec_daily_indexes RENAME TO sec_daily_indexes_legacy;

CREATE TABLE sec_daily_indexes (
    index_date TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processed', 'absent')),
    registered_count INTEGER NOT NULL DEFAULT 0,
    processed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO sec_daily_indexes SELECT * FROM sec_daily_indexes_legacy;

DROP TABLE sec_daily_indexes_legacy;

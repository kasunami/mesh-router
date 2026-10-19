BEGIN;

-- Historical last_heartbeat_at values may have been written at consume time.
-- Keep a clean source-event clock for freshness decisions and do not backfill
-- it from the legacy column: a fresh validated heartbeat must populate it.
ALTER TABLE mw_hosts
  ADD COLUMN IF NOT EXISTS last_heartbeat_event_at timestamptz;

COMMENT ON COLUMN mw_hosts.last_heartbeat_event_at IS
  'Latest validated worker heartbeat event timestamp; sole MW freshness clock.';

COMMIT;

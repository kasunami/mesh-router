BEGIN;

ALTER TABLE router_leases
  ADD COLUMN IF NOT EXISTS last_heartbeat_at timestamptz NULL;

UPDATE router_leases
SET last_heartbeat_at = COALESCE(last_heartbeat_at, acquired_at, now())
WHERE last_heartbeat_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_router_leases_lane_heartbeat
  ON router_leases(lane_id, state, last_heartbeat_at);

COMMIT;

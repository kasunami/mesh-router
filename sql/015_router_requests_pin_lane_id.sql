BEGIN;

ALTER TABLE router_requests
  ADD COLUMN IF NOT EXISTS pin_lane_id text NULL;

CREATE INDEX IF NOT EXISTS idx_router_requests_pin_lane_id
  ON router_requests(pin_lane_id, state, updated_at DESC);

COMMIT;

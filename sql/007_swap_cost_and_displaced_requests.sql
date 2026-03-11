BEGIN;

ALTER TABLE model_tuning_profiles
  ADD COLUMN IF NOT EXISTS cost_tier text NOT NULL DEFAULT 'standard',
  ADD COLUMN IF NOT EXISTS disables_sibling_lanes boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS exclusive_host_resources boolean NOT NULL DEFAULT false;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'model_tuning_profiles_cost_tier_check'
  ) THEN
    ALTER TABLE model_tuning_profiles
      ADD CONSTRAINT model_tuning_profiles_cost_tier_check
      CHECK (cost_tier IN ('standard', 'high', 'exclusive'));
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS swap_displaced_requests (
  displaced_request_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  swap_event_id uuid NULL,
  original_lease_id uuid NULL REFERENCES router_leases(lease_id) ON DELETE SET NULL,
  original_lane_id uuid NULL REFERENCES lanes(lane_id) ON DELETE SET NULL,
  replacement_lane_id uuid NULL REFERENCES lanes(lane_id) ON DELETE SET NULL,
  model_id uuid NULL REFERENCES models(model_id) ON DELETE SET NULL,
  route text NOT NULL,
  request_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  status text NOT NULL DEFAULT 'captured',
  handoff_attempted_at timestamptz NULL,
  handoff_completed_at timestamptz NULL,
  result_payload jsonb NULL,
  error_message text NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_swap_displaced_requests_model_created
  ON swap_displaced_requests(model_id, created_at DESC);

COMMIT;

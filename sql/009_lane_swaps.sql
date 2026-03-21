BEGIN;

CREATE TABLE IF NOT EXISTS lane_swaps (
  swap_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  requested_model_name citext NOT NULL,
  resolved_model_name citext NULL,
  source_mode text NULL,
  state text NOT NULL DEFAULT 'queued',
  terminal boolean NOT NULL DEFAULT false,
  error_message text NULL,
  details jsonb NOT NULL DEFAULT '{}'::jsonb,
  started_at timestamptz NOT NULL DEFAULT now(),
  last_event_at timestamptz NULL,
  completed_at timestamptz NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lane_swaps_lane_updated
  ON lane_swaps(lane_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_lane_swaps_active
  ON lane_swaps(lane_id, terminal, updated_at DESC);

CREATE TABLE IF NOT EXISTS lane_swap_events (
  event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  swap_id uuid NOT NULL REFERENCES lane_swaps(swap_id) ON DELETE CASCADE,
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  event_type text NOT NULL,
  state text NOT NULL,
  message text NULL,
  details jsonb NOT NULL DEFAULT '{}'::jsonb,
  error_message text NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lane_swap_events_swap_time
  ON lane_swap_events(swap_id, created_at DESC);

COMMIT;

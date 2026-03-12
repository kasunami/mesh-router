BEGIN;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'request_state') THEN
    CREATE TYPE request_state AS ENUM ('queued', 'acquired', 'running', 'released', 'failed', 'expired', 'canceled');
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS router_requests (
  request_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  route text NOT NULL,
  state request_state NOT NULL DEFAULT 'queued',
  owner text NOT NULL DEFAULT 'mesh-router',
  job_type text NOT NULL DEFAULT 'generic',
  app_name text NULL,
  client_request_id text NULL,
  requested_model_name citext NULL,
  downstream_model_name citext NULL,
  model_id uuid NULL REFERENCES models(model_id) ON DELETE SET NULL,
  lane_id uuid NULL REFERENCES lanes(lane_id) ON DELETE SET NULL,
  lease_id uuid NULL REFERENCES router_leases(lease_id) ON DELETE SET NULL,
  worker_id text NULL,
  base_url text NULL,
  pin_worker text NULL,
  pin_base_url text NULL,
  pin_lane_type text NULL,
  cancel_requested boolean NOT NULL DEFAULT false,
  cancel_requested_at timestamptz NULL,
  cancel_reason text NULL,
  request_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  result_payload jsonb NULL,
  error_kind text NULL,
  error_message text NULL,
  queued_at timestamptz NOT NULL DEFAULT now(),
  acquired_at timestamptz NULL,
  started_at timestamptz NULL,
  last_heartbeat_at timestamptz NULL,
  expires_at timestamptz NULL,
  released_at timestamptz NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_router_requests_state
  ON router_requests(state, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_router_requests_lease
  ON router_requests(lease_id);

CREATE INDEX IF NOT EXISTS idx_router_requests_lane_state
  ON router_requests(lane_id, state, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_router_requests_requested_model
  ON router_requests(requested_model_name, updated_at DESC);

COMMIT;

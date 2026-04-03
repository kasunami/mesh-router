BEGIN;

-- MW desired/actual state tables (mirrored from Projects/mesh-worker/sql/001_mw_state_model.sql)
-- This migration is intentionally idempotent (CREATE TABLE IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS mw_hosts (
  host_id text PRIMARY KEY,
  host_name text NOT NULL,
  platform text NOT NULL,
  service_manager text NOT NULL,
  config_version integer NOT NULL,
  config_hash text,
  startup_profile text,
  desired_profile text,
  actual_profile text,
  desired_generation bigint NOT NULL DEFAULT 0,
  last_transition_request_id uuid,
  last_transition_status text,
  last_transition_started_at timestamptz,
  last_transition_finished_at timestamptz,
  last_heartbeat_at timestamptz,
  last_state_at timestamptz,
  grpc_endpoint text,
  kafka_client_id text,
  control_mode text NOT NULL DEFAULT 'legacy',
  status text NOT NULL DEFAULT 'unknown',
  notes jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mw_services (
  host_id text NOT NULL REFERENCES mw_hosts(host_id) ON DELETE CASCADE,
  service_id text NOT NULL,
  manager_name text NOT NULL,
  backend_type text NOT NULL,
  kind text NOT NULL,
  desired_state text NOT NULL DEFAULT 'stopped',
  actual_state text NOT NULL DEFAULT 'unknown',
  listen_host text,
  listen_port integer,
  health_status text NOT NULL DEFAULT 'unknown',
  last_health_at timestamptz,
  last_error text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (host_id, service_id)
);

CREATE TABLE IF NOT EXISTS mw_lanes (
  host_id text NOT NULL REFERENCES mw_hosts(host_id) ON DELETE CASCADE,
  lane_id text NOT NULL,
  lane_name text,
  lane_type text NOT NULL,
  backend_type text NOT NULL,
  service_id text NOT NULL,
  resource_class text,
  desired_generation bigint,
  desired_profile text,
  actual_profile text,
  desired_model text,
  actual_model text,
  last_loaded_model text,
  desired_state text NOT NULL DEFAULT 'offline',
  actual_state text NOT NULL DEFAULT 'unknown',
  health_status text NOT NULL DEFAULT 'unknown',
  last_healthy_at timestamptz,
  last_transition_request_id uuid,
  last_transition_status text,
  suspension_reason text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (host_id, lane_id),
  FOREIGN KEY (host_id, service_id) REFERENCES mw_services(host_id, service_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS mw_profiles (
  host_id text NOT NULL REFERENCES mw_hosts(host_id) ON DELETE CASCADE,
  profile_id text NOT NULL,
  description text,
  services_to_start jsonb NOT NULL DEFAULT '[]'::jsonb,
  services_to_stop jsonb NOT NULL DEFAULT '[]'::jsonb,
  lane_models jsonb NOT NULL DEFAULT '{}'::jsonb,
  is_startup_default boolean NOT NULL DEFAULT false,
  is_recovery_default boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (host_id, profile_id)
);

CREATE TABLE IF NOT EXISTS mw_transitions (
  request_id uuid PRIMARY KEY,
  host_id text NOT NULL REFERENCES mw_hosts(host_id) ON DELETE CASCADE,
  transition_type text NOT NULL,
  requested_profile text,
  resolved_profile text,
  requested_lane_id text,
  requested_service_id text,
  requested_model text,
  resolved_model text,
  desired_generation bigint,
  status text NOT NULL,
  current_phase text,
  source text NOT NULL,
  trigger_reason text,
  error_kind text,
  error_message text,
  recovery_applied boolean NOT NULL DEFAULT false,
  recovery_profile text,
  started_at timestamptz,
  completed_at timestamptz,
  deadline_at timestamptz,
  details jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mw_transition_events (
  id bigserial PRIMARY KEY,
  request_id uuid NOT NULL REFERENCES mw_transitions(request_id) ON DELETE CASCADE,
  host_id text NOT NULL,
  event_type text NOT NULL,
  phase text,
  status text,
  sequence_no integer,
  message text,
  error_message text,
  details jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mw_active_requests (
  request_id uuid PRIMARY KEY,
  host_id text NOT NULL REFERENCES mw_hosts(host_id) ON DELETE CASCADE,
  lane_id text,
  profile_id text,
  route_type text NOT NULL,
  model_name text,
  status text NOT NULL,
  stream_transport text NOT NULL DEFAULT 'grpc',
  started_at timestamptz NOT NULL DEFAULT now(),
  deadline_at timestamptz,
  cancelled_at timestamptz,
  completed_at timestamptz,
  usage_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
  details jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS mw_heartbeats (
  id bigserial PRIMARY KEY,
  host_id text NOT NULL REFERENCES mw_hosts(host_id) ON DELETE CASCADE,
  heartbeat_at timestamptz NOT NULL,
  agent_version text,
  grpc_listening boolean,
  kafka_connected boolean,
  details jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_mw_services_host ON mw_services(host_id);
CREATE INDEX IF NOT EXISTS idx_mw_lanes_host ON mw_lanes(host_id);
CREATE INDEX IF NOT EXISTS idx_mw_transitions_host ON mw_transitions(host_id);
CREATE INDEX IF NOT EXISTS idx_mw_transitions_status ON mw_transitions(status);
CREATE INDEX IF NOT EXISTS idx_mw_transition_events_request ON mw_transition_events(request_id, sequence_no);
CREATE INDEX IF NOT EXISTS idx_mw_active_requests_host ON mw_active_requests(host_id);
CREATE INDEX IF NOT EXISTS idx_mw_active_requests_status ON mw_active_requests(status);
CREATE INDEX IF NOT EXISTS idx_mw_heartbeats_host_time ON mw_heartbeats(host_id, heartbeat_at DESC);

COMMIT;


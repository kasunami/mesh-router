BEGIN;

ALTER TABLE lanes
  ADD COLUMN IF NOT EXISTS memory_kind text NULL,
  ADD COLUMN IF NOT EXISTS usable_memory_bytes bigint NULL,
  ADD COLUMN IF NOT EXISTS runtime_overhead_bytes bigint NULL,
  ADD COLUMN IF NOT EXISTS reserved_headroom_bytes bigint NULL;

UPDATE lanes
SET reserved_headroom_bytes = 1073741824
WHERE reserved_headroom_bytes IS NULL;

ALTER TABLE host_model_artifacts
  ADD COLUMN IF NOT EXISTS storage_scope text NOT NULL DEFAULT 'local',
  ADD COLUMN IF NOT EXISTS storage_provider text NULL,
  ADD COLUMN IF NOT EXISTS format text NULL,
  ADD COLUMN IF NOT EXISTS sha256 text NULL;

CREATE INDEX IF NOT EXISTS idx_host_model_artifacts_host_path ON host_model_artifacts(host_id, local_path);
CREATE INDEX IF NOT EXISTS idx_host_model_artifacts_host_model ON host_model_artifacts(host_id, model_id);
CREATE INDEX IF NOT EXISTS idx_host_model_artifacts_scope_provider ON host_model_artifacts(storage_scope, storage_provider);

CREATE TABLE IF NOT EXISTS lane_model_viability (
  viability_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  artifact_id uuid NULL REFERENCES host_model_artifacts(artifact_id) ON DELETE SET NULL,
  source_locality text NOT NULL DEFAULT 'local',
  fits_memory boolean NOT NULL DEFAULT false,
  projected_free_bytes bigint NULL,
  required_memory_bytes bigint NULL,
  tps_estimate numeric NULL,
  tps_source text NULL,
  is_viable boolean NOT NULL,
  reason text NULL,
  last_checked_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (lane_id, model_id, source_locality)
);
CREATE INDEX IF NOT EXISTS idx_lane_model_viability_lookup ON lane_model_viability(lane_id, model_id, source_locality);

CREATE TABLE IF NOT EXISTS lane_model_swap_history (
  swap_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  artifact_id uuid NULL REFERENCES host_model_artifacts(artifact_id) ON DELETE SET NULL,
  source_mode text NOT NULL DEFAULT 'local',
  swapped_at timestamptz NOT NULL DEFAULT now(),
  copy_time_ms bigint NULL,
  load_time_ms bigint NULL,
  duration_ms bigint NULL,
  success boolean NOT NULL DEFAULT true,
  error_kind text NULL,
  error_message text NULL
);
CREATE INDEX IF NOT EXISTS idx_lane_model_swap_history_recent ON lane_model_swap_history(lane_id, model_id, swapped_at DESC);

CREATE TABLE IF NOT EXISTS lane_model_usage (
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  request_count bigint NOT NULL DEFAULT 0,
  last_used_at timestamptz NULL,
  last_swap_at timestamptz NULL,
  rolling_24h_count bigint NOT NULL DEFAULT 0,
  rolling_7d_count bigint NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (lane_id, model_id)
);
CREATE INDEX IF NOT EXISTS idx_lane_model_usage_recent ON lane_model_usage(last_used_at DESC);

COMMIT;

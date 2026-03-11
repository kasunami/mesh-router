BEGIN;

CREATE TABLE IF NOT EXISTS model_tuning_profiles (
  tuning_profile_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  host_id uuid NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  lane_id uuid NULL REFERENCES lanes(lane_id) ON DELETE SET NULL,
  storage_scheme text NOT NULL,
  settings jsonb NOT NULL DEFAULT '{}'::jsonb,
  prompt_tps numeric NULL,
  generation_tps numeric NULL,
  avg_total_latency_s numeric NULL,
  score numeric NULL,
  evaluation_count int NOT NULL DEFAULT 1,
  source_run_tag text NULL,
  notes text NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (host_id, model_id, storage_scheme)
);

CREATE INDEX IF NOT EXISTS idx_model_tuning_profiles_host_model
  ON model_tuning_profiles(host_id, model_id, storage_scheme);

CREATE INDEX IF NOT EXISTS idx_model_tuning_profiles_updated_at
  ON model_tuning_profiles(updated_at DESC);

COMMIT;

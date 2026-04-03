-- MW performance observations registry (durable, MB-aligned)
-- Intended DB: MW state DB (typically ai_mesh / MeshBrain DB via MESH_ROUTER_MW_STATE_DATABASE_URL)

CREATE TABLE IF NOT EXISTS mw_perf_observations (
  perf_observation_id BIGSERIAL PRIMARY KEY,
  observed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  host_id             TEXT NOT NULL,
  lane_id             TEXT NOT NULL,
  model_name          TEXT NOT NULL,
  backend_type        TEXT,
  lane_type           TEXT,
  modality            TEXT NOT NULL DEFAULT 'chat', -- chat|embeddings|images

  prompt_tokens       INTEGER,
  generated_tokens    INTEGER,
  first_token_ms      DOUBLE PRECISION,
  decode_tps          DOUBLE PRECISION,
  total_ms            DOUBLE PRECISION,
  was_cold            BOOLEAN,

  ok                  BOOLEAN NOT NULL DEFAULT TRUE,
  error_kind          TEXT,
  error_message       TEXT,

  metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_mw_perf_obs_lookup
  ON mw_perf_observations (host_id, lane_id, model_name, modality, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_mw_perf_obs_observed_at
  ON mw_perf_observations (observed_at DESC);


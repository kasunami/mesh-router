BEGIN;

-- Map canonical model names to downstream model IDs on a per-lane basis.
-- Example: request model_name='qwen3.5-9b' but downstream MLX expects '/Users/.../Qwen3.5-9B-6bit'.
CREATE TABLE IF NOT EXISTS lane_model_aliases (
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  downstream_model_name text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (lane_id, model_id)
);

COMMIT;


BEGIN;

ALTER TABLE models
  ADD COLUMN IF NOT EXISTS tags text[] NOT NULL DEFAULT '{}'::text[];

CREATE INDEX IF NOT EXISTS idx_models_tags_gin
  ON models
  USING GIN(tags);

ALTER TABLE lanes
  ADD COLUMN IF NOT EXISTS default_model_name citext NULL;

UPDATE lanes
SET default_model_name = current_model_name
WHERE default_model_name IS NULL
  AND current_model_name IS NOT NULL;

COMMIT;

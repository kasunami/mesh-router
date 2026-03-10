BEGIN;

ALTER TABLE lane_model_metrics
  ADD COLUMN IF NOT EXISTS request_latency_ms int NULL,
  ADD COLUMN IF NOT EXISTS prompt_tokens int NULL,
  ADD COLUMN IF NOT EXISTS completion_tokens int NULL;

COMMIT;


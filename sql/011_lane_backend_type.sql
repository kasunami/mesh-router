BEGIN;

ALTER TABLE lanes
  ADD COLUMN IF NOT EXISTS backend_type text NOT NULL DEFAULT 'llama';

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'lanes_backend_type_check'
  ) THEN
    ALTER TABLE lanes
      ADD CONSTRAINT lanes_backend_type_check CHECK (backend_type IN ('llama', 'sd', 'bitnet', 'mlx'));
  END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_lanes_backend_type
  ON lanes(backend_type);

COMMIT;

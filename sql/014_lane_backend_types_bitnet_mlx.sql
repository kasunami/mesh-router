-- Allow all backend types that MR can route and MW can publish.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'lanes_backend_type_check'
  ) THEN
    ALTER TABLE lanes DROP CONSTRAINT lanes_backend_type_check;
  END IF;

  ALTER TABLE lanes
    ADD CONSTRAINT lanes_backend_type_check CHECK (backend_type IN ('llama', 'sd', 'bitnet', 'mlx'));
END$$;

-- mesh_router database bootstrap
-- Target DB: mesh_router (separate from ai_mesh)
--
-- Design goals:
-- - Inventory: hosts, lanes, model artifacts/sources
-- - Routing inputs: lane readiness, loaded model, leases, health probes
-- - Metrics: load time, tokens/sec, error rates (per lane+model)
-- - Constraints: dualboot mutual-exclusion groups, disk space, memory budgets

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;

-- ---- Enums ----

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'lane_type') THEN
    CREATE TYPE lane_type AS ENUM ('cpu', 'gpu', 'mlx', 'router', 'other');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'lane_status') THEN
    CREATE TYPE lane_status AS ENUM ('ready', 'busy', 'suspended', 'offline', 'error');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'host_status') THEN
    CREATE TYPE host_status AS ENUM ('ready', 'degraded', 'offline', 'unknown');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'lease_state') THEN
    CREATE TYPE lease_state AS ENUM ('active', 'released', 'failed', 'expired');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'probe_kind') THEN
    CREATE TYPE probe_kind AS ENUM (
      'tcp',
      'http_health',
      'openai_models',
      'openai_chat_smoke',
      'disk_free',
      'gpu_info',
      'cpu_info',
      'ram_info'
    );
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_format') THEN
    CREATE TYPE model_format AS ENUM ('gguf', 'mlx', 'safetensors', 'other');
  END IF;
END$$;

-- ---- Core inventory ----

CREATE TABLE IF NOT EXISTS hosts (
  host_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  host_name citext UNIQUE NOT NULL,
  status host_status NOT NULL DEFAULT 'unknown',
  notes text NULL,

  -- Canonical access/management info
  mgmt_ssh_host text NULL,          -- e.g. worker-a.example
  mgmt_ssh_user text NULL,          -- e.g. mesh
  docs text NULL,                   -- short ops notes/instructions

  -- Capacity snapshots (best-effort, updated by probes)
  cpu_model text NULL,
  cpu_cores int NULL,
  cpu_threads int NULL,
  ram_total_bytes bigint NULL,
  ram_ai_budget_bytes bigint NULL,  -- expected usable RAM for AI work

  gpu_model text NULL,
  vram_total_bytes bigint NULL,
  vram_ai_budget_bytes bigint NULL, -- expected usable VRAM for AI work

  disk_total_bytes bigint NULL,
  disk_free_bytes bigint NULL,

  -- Model storage roots
  model_store_paths jsonb NOT NULL DEFAULT '[]'::jsonb, -- list of directories / mounts

  -- Runtime/services inventory
  ai_services jsonb NOT NULL DEFAULT '[]'::jsonb, -- e.g. [{"name":"llama.cpp","howto":"systemd llama-gpu"}]

  last_seen_at timestamptz NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dualboot_groups (
  dualboot_group_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  group_name citext UNIQUE NOT NULL,
  notes text NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS host_dualboot_members (
  host_id uuid NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
  dualboot_group_id uuid NOT NULL REFERENCES dualboot_groups(dualboot_group_id) ON DELETE CASCADE,
  PRIMARY KEY (host_id, dualboot_group_id)
);

-- A "lane" is a callable endpoint sharing a host's resources.
CREATE TABLE IF NOT EXISTS lanes (
  lane_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  host_id uuid NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
  lane_name citext NOT NULL,        -- e.g. "gpu", "cpu", "mlx", "router"
  lane_type lane_type NOT NULL,

  base_url text NOT NULL,           -- e.g. http://worker-a.example:11434
  openai_path_prefix text NOT NULL DEFAULT '', -- optional prefix if not /v1

  status lane_status NOT NULL DEFAULT 'offline',
  suspension_reason text NULL,
  current_model_name citext NULL,   -- what we believe is loaded
  in_use boolean NOT NULL DEFAULT false,

  -- Resource budgets for this lane (may be < host budgets if shared)
  ram_budget_bytes bigint NULL,
  vram_budget_bytes bigint NULL,

  -- Downstream auth / access instructions (stored, but secrets must live elsewhere)
  proxy_auth_mode text NULL,        -- e.g. "lease_bearer", "static_bearer_env"
  proxy_auth_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,

  last_probe_at timestamptz NULL,
  last_ok_at timestamptz NULL,
  last_error text NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),

  UNIQUE (host_id, lane_name),
  UNIQUE (base_url)
);

CREATE INDEX IF NOT EXISTS idx_lanes_host ON lanes(host_id);
CREATE INDEX IF NOT EXISTS idx_lanes_status ON lanes(status);

-- ---- Models ----

CREATE TABLE IF NOT EXISTS models (
  model_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name citext UNIQUE NOT NULL,  -- exact call name (no aliases)
  family citext NULL,                 -- e.g. qwen3.5
  params_b numeric NULL,              -- e.g. 9, 27
  quant text NULL,                    -- e.g. Q4_K_M, Q6
  format model_format NOT NULL DEFAULT 'other',
  size_bytes bigint NULL,             -- on-disk size
  context_default int NULL,
  notes text NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Where a model can be fetched from (archive store, HF, local path, etc.)
CREATE TABLE IF NOT EXISTS model_sources (
  source_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  source_kind text NOT NULL,        -- e.g. "model_archive", "huggingface", "local_path"
  source_ref text NOT NULL,         -- e.g. "model-archive://qwen3.5-9b/Q4_K_M" or "hf://org/repo/file"
  sha256 text NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (model_id, source_kind, source_ref)
);

-- Model presence on a host (disk location) and whether it's eligible to be used.
CREATE TABLE IF NOT EXISTS host_model_artifacts (
  artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  host_id uuid NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  local_path text NOT NULL,
  size_bytes bigint NULL,
  present boolean NOT NULL DEFAULT true,
  eligible boolean NOT NULL DEFAULT true,
  last_verified_at timestamptz NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (host_id, model_id, local_path)
);

CREATE INDEX IF NOT EXISTS idx_host_model_artifacts_host ON host_model_artifacts(host_id);
CREATE INDEX IF NOT EXISTS idx_host_model_artifacts_model ON host_model_artifacts(model_id);

-- Lane compatibility + policy decisions (fit, offload allowed, etc.)
CREATE TABLE IF NOT EXISTS lane_model_policy (
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,

  -- eligibility from router's POV
  allowed boolean NOT NULL DEFAULT true,
  disallow_reason text NULL,

  -- capacity assumptions used for placement decisions
  required_ram_bytes bigint NULL,
  required_vram_bytes bigint NULL,
  allow_cpu_gpu_split boolean NOT NULL DEFAULT false, -- allow partial GPU offload + system RAM
  max_ctx int NULL,

  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (lane_id, model_id)
);

-- ---- Probing & metrics ----

CREATE TABLE IF NOT EXISTS lane_probes (
  probe_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  kind probe_kind NOT NULL,
  ok boolean NOT NULL,
  status_code int NULL,
  latency_ms int NULL,
  details jsonb NOT NULL DEFAULT '{}'::jsonb,
  error text NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lane_probes_lane_time ON lane_probes(lane_id, created_at DESC);

-- Benchmarks and runtime observations per lane+model (includes load time and error rate inputs)
CREATE TABLE IF NOT EXISTS lane_model_metrics (
  metric_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,

  -- Measurements
  load_time_ms int NULL,         -- cold start / swap time
  tps numeric NULL,              -- tokens per second
  ttft_ms int NULL,              -- time to first token
  context_tokens int NULL,
  success boolean NOT NULL DEFAULT true,
  error_kind text NULL,
  error_message text NULL,

  run_tag text NULL,             -- e.g. "meshbench:qwen3.5-9b:smoke"
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lane_model_metrics_lane_model_time ON lane_model_metrics(lane_id, model_id, created_at DESC);

-- ---- Leases (router-level) ----

-- Router-level leases (separate from MeshBench), for future migration away from MeshBench.
CREATE TABLE IF NOT EXISTS router_leases (
  lease_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  lane_id uuid NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES models(model_id) ON DELETE RESTRICT,
  owner text NOT NULL,
  job_type text NOT NULL DEFAULT 'generic',
  state lease_state NOT NULL DEFAULT 'active',
  acquired_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL,
  released_at timestamptz NULL,
  details jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_router_leases_lane_state ON router_leases(lane_id, state);

-- ---- Audit/event log ----

CREATE TABLE IF NOT EXISTS events (
  event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kind text NOT NULL,
  lane_id uuid NULL REFERENCES lanes(lane_id) ON DELETE SET NULL,
  model_id uuid NULL REFERENCES models(model_id) ON DELETE SET NULL,
  host_id uuid NULL REFERENCES hosts(host_id) ON DELETE SET NULL,
  message text NULL,
  details jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_time ON events(created_at DESC);

COMMIT;

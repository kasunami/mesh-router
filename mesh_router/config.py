from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MESH_ROUTER_", extra="ignore")

    # Postgres (dedicated DB)
    database_url: str = "postgresql://username:password@localhost:5432/mesh_router"

    # MeshBench lease gate + proxy
    meshbench_base_url: str = "http://localhost:8787"
    meshbench_sync_enabled: bool = False

    # Lease tokens (validated by worker gateways). Only mesh-router should know this secret.
    # Tokens are JWT-like (HS256) with an exp claim; worker gateways call back to mesh-router
    # to validate, so the secret does not need to exist on worker nodes.
    lease_token_secret: str = "replace-with-random-secret"
    allow_dev_secrets: bool = False

    # Router behavior
    default_lease_ttl_seconds: int = 600
    default_lease_heartbeat_interval_seconds: int = 15
    default_lease_stale_seconds: int = 45
    default_owner: str = "mesh-router"
    default_job_type: str = "openai_proxy"
    deployment_revision: str = os.getenv("MESH_ROUTER_DEPLOYMENT_REVISION", "dev")

    # Sync behavior
    sync_interval_seconds: int = 30

    # Health probing (mesh-router-native; avoids hard dependency on MeshBench for availability)
    probe_interval_seconds: int = 15

    # Swap auth token — must match WORKER_STATIC_BEARER_TOKENS on each worker gateway
    swap_auth_token: str = "replace-with-worker-swap-token"
    # Must exceed worker-side WORKER_SWITCH_TIMEOUT_S so the router does not time out first.
    swap_proxy_timeout_seconds: int = 240
    # Public callback URL reachable by worker nodes for swap progress events.
    router_public_base_url: str = "http://localhost:4010"

    # MeshWorker control plane
    mw_kafka_bootstrap_servers: str = (
        os.getenv("KAFKA_BROKER")
        or f"{os.getenv('KAFKA_ADDRESS', '127.0.0.1')}:{os.getenv('KAFKA_PORT', '9092')}"
    )
    mw_kafka_commands_topic: str = "mr.worker.commands"
    mw_kafka_responses_topic: str = "mr.worker.responses"
    mw_kafka_state_topic: str = "mr.worker.state"
    mw_kafka_heartbeats_topic: str = "mr.worker.heartbeats"
    mw_kafka_cancels_topic: str = "mr.worker.cancels"
    mw_kafka_dead_letter_topic: str = "mr.worker.dlq"
    mw_kafka_client_id: str = "mesh-router"
    mw_kafka_consumer_group: str = "mesh-router-mw"
    mw_command_timeout_seconds: int = 30
    mw_control_enabled: bool = True
    mw_grpc_default_port: int = 50061

    # Optional separate DB for MW state (recommended: ai_mesh / MeshBrain DB).
    # When unset, MW consumer falls back to `database_url`.
    mw_state_database_url: str | None = os.getenv("MESH_ROUTER_MW_STATE_DATABASE_URL") or None

    # Ephemeral MW runtime-state cache. Postgres remains audit/fallback; Redis is the
    # preferred source for fast-changing active lane/backend/model truth.
    runtime_state_redis_url: str | None = (
        os.getenv("MESH_ROUTER_RUNTIME_STATE_REDIS_URL") or os.getenv("MESH_ROUTER_REDIS_URL") or None
    )
    runtime_state_ttl_seconds: int = 90

    # Placement gating for pilot cutovers. When enabled, prefer MW-managed lanes when multiple
    # candidates are eligible (reversible rollouts without per-request pinning).
    placement_prefer_mw_lanes: bool = False

    # Routing policy: treat some hosts as opportunistic/preemptible by default.
    # Requests can explicitly opt into opportunistic routing; otherwise MR prefers stable hosts.
    opportunistic_hosts: str = ""

    # Perf observation ingestion guard. When set, POST /api/perf/observations requires
    # X-Mesh-Internal-Token to match this value.
    internal_ingest_token: str | None = None

    # Automatic perf observation ingestion from real traffic (best-effort, never blocks responses).
    # Writes to mw_perf_observations in the MW state DB.
    perf_auto_observe_enabled: bool = True
    perf_auto_observe_sample_rate: float = 1.0
    perf_auto_observe_min_elapsed_ms: int = 50
    perf_auto_observe_max_total_ms: int = 600_000

    # Optional requestor-facing debug headers. When enabled, MR includes perf expectation
    # metadata (if available) in responses and SSE headers.
    route_debug_headers_enabled: bool = False

    # VLM routing (llama.cpp router-style backends)
    #
    # When enabled, mesh-router seeds a dedicated VLM lane that points at a llama.cpp router
    # service (supports POST /models/load) and can accept OpenAI-style multimodal chat payloads.
    vlm_seed_enabled: bool = False
    # Optional: seed multiple VLM lanes.
    # JSON array of objects like:
    #   [{"host_ref":"worker-a","lane_name":"vlm-worker-a","base_url":"http://worker-a.example:21437","llama_router":false}]
    # When set, takes precedence over vlm_lane_host_ref/vlm_lane_name/vlm_lane_base_url.
    vlm_lane_specs_json: str | None = None
    vlm_lane_host_ref: str = "model-router"
    vlm_lane_name: str = "vlm-router"
    vlm_lane_base_url: str = "http://llama-vision-router.example:4012"
    vlm_declared_models: str = "qwen3.5-9b-vlm,Qwen3.5-9B-VLM-Q4_K_M"

    # If true, a chat request containing image parts can be remapped from a text-only model
    # request (e.g. Qwen3.5-9B) to the default VLM alias.
    vlm_remap_text_model_requests: bool = True
    vlm_default_model: str = "qwen3.5-9b-vlm"

    # Reasoning-model output budgeting. Some backends count hidden reasoning tokens against
    # the same max_tokens field as user-visible answer tokens, so MR over-allocates backend
    # tokens and filters hidden reasoning from normal OpenAI responses.
    reasoning_model_patterns: str = "qwen3.5,qwen3-5,qwen3"
    reasoning_budget_tokens: int = 1024
    reasoning_min_visible_tokens: int = 256
    reasoning_max_backend_tokens: int = 4096
    expose_reasoning_content: bool = False


settings = Settings()


def validate_runtime_settings(current: Settings = settings) -> None:
    """Fail closed on known placeholder secrets outside explicit development mode."""
    if current.allow_dev_secrets:
        return
    placeholders = {
        "lease_token_secret": "replace-with-random-secret",
        "swap_auth_token": "replace-with-worker-swap-token",
    }
    invalid = [
        name
        for name, placeholder in placeholders.items()
        if str(getattr(current, name, "") or "").strip() == placeholder
    ]
    if invalid:
        joined = ", ".join(sorted(invalid))
        raise RuntimeError(
            f"refusing to start with placeholder runtime secret(s): {joined}; "
            "set real MESH_ROUTER_* values or MESH_ROUTER_ALLOW_DEV_SECRETS=1 for local development"
        )

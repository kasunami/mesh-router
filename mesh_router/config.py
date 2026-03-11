from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MESH_ROUTER_", extra="ignore")

    # Postgres (dedicated DB)
    database_url: str = "postgresql://username:password@localhost:5432/mesh_router"

    # MeshBench lease gate + proxy
    meshbench_base_url: str = "http://localhost:8787"

    # Lease tokens (validated by worker gateways). Only mesh-router should know this secret.
    # Tokens are JWT-like (HS256) with an exp claim; worker gateways call back to mesh-router
    # to validate, so the secret does not need to exist on worker nodes.
    lease_token_secret: str = "replace-with-random-secret"

    # Router behavior
    default_lease_ttl_seconds: int = 600
    default_lease_heartbeat_interval_seconds: int = 15
    default_lease_stale_seconds: int = 45
    default_owner: str = "mesh-router"
    default_job_type: str = "openai_proxy"

    # Sync behavior
    sync_interval_seconds: int = 30

    # Health probing (mesh-router-native; avoids hard dependency on MeshBench for availability)
    probe_interval_seconds: int = 15

    # Swap auth token — must match WORKER_STATIC_BEARER_TOKENS on each worker gateway
    swap_auth_token: str = "replace-with-worker-swap-token"
    # Must exceed worker-side WORKER_SWITCH_TIMEOUT_S so the router does not time out first.
    swap_proxy_timeout_seconds: int = 240


settings = Settings()

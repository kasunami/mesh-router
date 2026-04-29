from __future__ import annotations

from datetime import UTC, datetime

from mesh_router.runtime_state import RuntimeStateStore


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = value
        self.ttls[key] = ttl

    def mget(self, keys: list[str]) -> list[str | None]:
        return [self.values.get(key) for key in keys]


def test_runtime_state_store_writes_lane_facts_with_service_port_and_eta_metadata() -> None:
    redis = FakeRedis()
    store = RuntimeStateStore(redis)  # type: ignore[arg-type]
    now = datetime(2026, 4, 15, 12, 0, tzinfo=UTC)

    store.write_host_snapshot(
        host_id="static-deskix",
        snapshot={
            "actual_profile": "split_default",
            "service_states": [
                {
                    "service_id": "llama-gpu",
                    "backend_type": "llama.cpp",
                    "listen_port": 21434,
                    "actual_state": "running",
                    "health_status": "healthy",
                }
            ],
            "lane_states": [
                {
                    "lane_id": "gpu",
                    "lane_type": "gpu",
                    "backend_type": "llama.cpp",
                    "service_id": "llama-gpu",
                    "desired_model": "qwen3.5-9b",
                    "actual_model": "qwen3.5-9b",
                    "actual_state": "running",
                    "health_status": "healthy",
                    "current_backend_type": "llama",
                    "backend_swap_eta_ms": 0,
                    "eta_source": "mw_state_snapshot",
                    "eta_complete": True,
                }
            ],
        },
        observed_at=now,
        ttl_seconds=90,
    )

    facts = store.get_lane_facts([("static-deskix", "gpu")])

    fact = facts[("static-deskix", "gpu")]
    assert fact["last_heartbeat_at"] == now
    assert fact["actual_model"] == "qwen3.5-9b"
    assert fact["backend_type"] == "llama.cpp"
    assert fact["listen_port"] == 21434
    assert fact["metadata"]["current_backend_type"] == "llama"
    assert fact["metadata"]["backend_swap_eta_ms"] == 0
    assert fact["metadata"]["eta_complete"] is True
    assert redis.ttls[store.lane_key("static-deskix", "gpu")] == 90


def test_runtime_state_store_can_label_response_snapshots() -> None:
    redis = FakeRedis()
    store = RuntimeStateStore(redis)  # type: ignore[arg-type]
    now = datetime(2026, 4, 15, 12, 1, tzinfo=UTC)

    store.write_host_snapshot(
        host_id="static-deskix",
        snapshot={
            "service_states": [],
            "lane_states": [
                {
                    "lane_id": "gpu",
                    "backend_type": "llama.cpp",
                    "actual_model": "qwen3.5-9b",
                    "actual_state": "running",
                    "health_status": "healthy",
                }
            ],
        },
        observed_at=now,
        ttl_seconds=90,
        source="mw_response_snapshot",
    )

    fact = store.get_lane_facts([("static-deskix", "gpu")])[("static-deskix", "gpu")]
    assert fact["metadata"]["source"] == "mw_response_snapshot"


def test_runtime_state_store_maps_top_level_validated_candidates_to_lanes() -> None:
    redis = FakeRedis()
    store = RuntimeStateStore(redis)  # type: ignore[arg-type]
    now = datetime(2026, 4, 15, 12, 2, tzinfo=UTC)

    store.write_host_snapshot(
        host_id="static-deskix",
        snapshot={
            "service_states": [],
            "validated_candidates": [
                {"canonical_id": "Qwen3.5-9B-Q4_K_M.gguf", "lane_ids": ["gpu"]},
                {"canonical_id": "falcon3-10b", "lane_ids": ["cpu"]},
            ],
            "lane_states": [
                {"lane_id": "gpu", "actual_state": "running", "health_status": "healthy"},
            ],
        },
        observed_at=now,
        ttl_seconds=90,
    )

    fact = store.get_lane_facts([("static-deskix", "gpu")])[("static-deskix", "gpu")]
    assert fact["validated_candidates"] == [{"canonical_id": "Qwen3.5-9B-Q4_K_M.gguf", "lane_ids": ["gpu"]}]

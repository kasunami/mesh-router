from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# OpenAI-ish request/response models (minimal subset we need)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, alias="max_tokens")
    stream: bool | None = None

    # Router hints (non-OpenAI; accepted via extra body)
    mesh_pin_worker: str | None = None
    mesh_pin_base_url: str | None = None
    mesh_pin_lane_type: str | None = None
    mesh_allow_swap: bool | None = None

    extra_body: dict[str, Any] | None = None

    model_config = {"extra": "allow", "populate_by_name": True}


class ArtifactItem(BaseModel):
    name: str
    path: str
    format: str | None = None
    size_bytes: int | None = None
    mtime: float | None = None
    checksum: str | None = None
    metadata: dict[str, Any] | None = None


class HostInventoryScanRequest(BaseModel):
    host_id: str
    root_path: str | None = None
    artifacts: list[ArtifactItem]
    host_facts: dict[str, Any] | None = None
    scan_details: dict[str, Any] | None = None


class ArchiveInventoryScanRequest(BaseModel):
    archive_id: str
    provider: str | None = None
    root_path: str | None = None
    artifacts: list[ArtifactItem]
    scan_details: dict[str, Any] | None = None


class LaneModelCandidate(BaseModel):
    model_name: str
    locality: Literal["local", "remote", "unverified"]
    artifact_path: str | None = None
    artifact_host: str | None = None
    artifact_provider: str | None = None
    estimated_tps: float | None = None
    estimated_swap_ms: int | None = None
    swap_strategy: str | None = None
    reason: str | None = None
    size_bytes: int | None = None
    required_memory_bytes: int | None = None
    projected_free_bytes: int | None = None


class LaneCapabilityResponse(BaseModel):
    lane_id: str
    capabilities: list[str]
    supported_models: list[str]
    current_model: str | None = None
    local_viable_models: list[LaneModelCandidate] = Field(default_factory=list)
    remote_viable_models: list[LaneModelCandidate] = Field(default_factory=list)
    unverified_models: list[LaneModelCandidate] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class SwapPreflightResponse(BaseModel):
    lane_id: str
    model_name: str
    ok: bool
    source_mode: Literal["local", "remote_direct", "remote_copy_then_load"] | None = None
    artifact_path: str | None = None
    artifact_host: str | None = None
    artifact_provider: str | None = None
    estimated_swap_ms: int | None = None
    swap_strategy: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "mesh-router"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class ModelTuningProfileUpsertRequest(BaseModel):
    host_ref: str
    model_name: str
    storage_scheme: Literal["ram", "vram", "both"]
    settings: dict[str, Any]
    cost_tier: Literal["standard", "high", "exclusive"] = "standard"
    disables_sibling_lanes: bool = False
    exclusive_host_resources: bool = False
    prompt_tps: float | None = None
    generation_tps: float | None = None
    avg_total_latency_s: float | None = None
    score: float | None = None
    evaluation_count: int | None = 1
    lane_ref: str | None = None
    source_run_tag: str | None = None
    notes: str | None = None


class ModelTuningProfileResponse(BaseModel):
    tuning_profile_id: str
    host_id: str
    host_name: str
    model_id: str
    model_name: str
    lane_id: str | None = None
    lane_name: str | None = None
    lane_type: str | None = None
    storage_scheme: Literal["ram", "vram", "both"]
    settings: dict[str, Any]
    cost_tier: Literal["standard", "high", "exclusive"]
    disables_sibling_lanes: bool
    exclusive_host_resources: bool
    prompt_tps: float | None = None
    generation_tps: float | None = None
    avg_total_latency_s: float | None = None
    score: float | None = None
    evaluation_count: int
    source_run_tag: str | None = None
    notes: str | None = None
    created_at: str
    updated_at: str


class RouterRequestSubmitRequest(BaseModel):
    route: Literal["chat", "embeddings"]
    payload: dict[str, Any]
    owner: str | None = None
    job_type: str | None = None
    app_name: str | None = None
    client_request_id: str | None = None
    wait: bool = False


class RouterRequestCancelRequest(BaseModel):
    reason: str | None = None

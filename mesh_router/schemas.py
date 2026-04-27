from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# OpenAI-ish request/response models (minimal subset we need)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # OpenAI multimodal messages use `content` as a list of parts:
    # [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"..."}}]
    # Keep this permissive so mesh-router can proxy multimodal requests to
    # downstream servers (e.g. llama.cpp vision) without enforcing a specific schema.
    content: str | list[dict[str, Any]] | list[str] | None = None
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
    mesh_pin_lane_id: str | None = None
    mesh_allow_swap: bool | None = None

    extra_body: dict[str, Any] | None = None

    model_config = {"extra": "allow", "populate_by_name": True}


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: int | None = 1
    size: str | None = "1024x1024"
    response_format: Literal["url", "b64_json"] | None = "b64_json"
    quality: str | None = None

    mesh_pin_worker: str | None = None
    mesh_pin_base_url: str | None = None
    mesh_pin_lane_type: str | None = None
    mesh_pin_lane_id: str | None = None
    mesh_allow_swap: bool | None = None

    model_config = {"extra": "allow"}


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
    tags: list[str] = Field(default_factory=list)
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
    max_context_tokens: int | None = None


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
    tags: list[str] = Field(default_factory=list)
    object: str = "model"
    owned_by: str = "mesh-router"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class ModelTagsUpdateRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)
    mode: Literal["replace", "add", "remove"] = "replace"


class ModelTagsResponse(BaseModel):
    model_id: str
    model_name: str
    tags: list[str] = Field(default_factory=list)


class ModelTuningProfileUpsertRequest(BaseModel):
    host_ref: str
    model_name: str
    storage_scheme: Literal["ram", "vram", "both", "archive"]
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
    storage_scheme: Literal["ram", "vram", "both", "archive"]
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
    route: Literal["chat", "embeddings", "images"]
    payload: dict[str, Any]
    owner: str | None = None
    job_type: str | None = None
    app_name: str | None = None
    client_request_id: str | None = None
    wait: bool = False


class RouterRequestCancelRequest(BaseModel):
    reason: str | None = None


class LaneSwapEventRequest(BaseModel):
    event_type: str
    state: str
    message: str | None = None
    details: dict[str, Any] | None = None
    current_model_name: str | None = None
    error_message: str | None = None


class LaneSwapResponse(BaseModel):
    swap_id: str
    lane_id: str
    host_name: str
    requested_model_name: str
    resolved_model_name: str | None = None
    state: str
    terminal: bool
    source_mode: str | None = None
    error_message: str | None = None
    details: dict[str, Any]
    started_at: str
    last_event_at: str | None = None
    completed_at: str | None = None
    updated_at: str


class RestoreSplitModeRequest(BaseModel):
    cpu_model_name: str | None = None
    gpu_model_name: str | None = None
    mlx_model_name: str | None = None


class RestoreSplitModeResponse(BaseModel):
    host_id: str
    host_name: str
    actions: list[dict[str, Any]] = Field(default_factory=list)


class InventoryLane(BaseModel):
    lane_id: str
    lane_name: str
    host_id: str
    host_name: str
    lane_type: str | None = None
    backend_type: str | None = None
    base_url: str | None = None
    status: str
    effective_status: str | None = None
    readiness_reason: str | None = None
    current_model_name: str | None = None
    proxy_auth_metadata: dict[str, Any] | None = None

    # Capability hints (best-effort)
    local_viable_models: list[LaneModelCandidate] = Field(default_factory=list)
    remote_viable_models: list[LaneModelCandidate] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class InventoryHost(BaseModel):
    host_id: str
    host_name: str
    lanes: list[InventoryLane] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    policy: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class InventoryResponse(BaseModel):
    items: list[InventoryHost] = Field(default_factory=list)


class PerfObservationIngestRequest(BaseModel):
    host_id: str
    lane_id: str
    model_name: str
    backend_type: str | None = None
    lane_type: str | None = None
    modality: Literal["chat", "embeddings", "images"] = "chat"

    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    first_token_ms: float | None = None
    decode_tps: float | None = None
    total_ms: float | None = None
    was_cold: bool | None = None
    ok: bool = True
    error_kind: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class PerfExpectationItem(BaseModel):
    host_id: str
    lane_id: str
    model_name: str
    modality: str
    updated_at: str
    sample_count: int
    first_token_ms_p50: float | None = None
    decode_tps_p50: float | None = None
    total_ms_p50: float | None = None
    staleness_s: float | None = None
    source: Literal["observations"] = "observations"


class PerfExpectationResponse(BaseModel):
    items: list[PerfExpectationItem] = Field(default_factory=list)


class RouteResolveRequest(BaseModel):
    modality: Literal["chat", "embeddings", "images"] = "chat"
    model: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Explicit targeting mode (optional)
    host_name: str | None = None
    lane_id: str | None = None

    # Policy gates
    allow_opportunistic: bool = False


class RouteResolveResponse(BaseModel):
    ok: bool
    reason: str | None = None
    choice: dict[str, Any] | None = None
    perf: PerfExpectationItem | None = None
    candidates_considered: int | None = None


class MWCommandRequest(BaseModel):
    host_id: str
    message_type: Literal[
        "activate_profile",
        "load_model",
        "start_service",
        "stop_service",
        "restart_service",
        "health_probe",
        "unload_service",
        "unload_lane",
        "cancel_request",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)
    wait: bool = True
    timeout_seconds: int | None = None
    request_id: str | None = None


class MWCommandResponse(BaseModel):
    ok: bool
    pending: bool = False
    host_id: str
    request_id: str
    message_type: str
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    warning: str | None = None
    timeout_seconds: int | None = None
    response: dict[str, Any] | None = None


class MWCommandStatusResponse(BaseModel):
    found: bool
    request_id: str
    host_id: str | None = None
    status: str | None = None
    transition_type: str | None = None
    current_phase: str | None = None
    ok: bool | None = None
    error_kind: str | None = None
    error_message: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str | None = None

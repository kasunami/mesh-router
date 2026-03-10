from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

# Constants from requirements
MIN_FREE_MEMORY_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
MIN_VIABLE_TPS = 0.09


class ViabilityLaneInfo(BaseModel):
    lane_id: str
    lane_type: str  # 'cpu', 'gpu', 'mlx', etc.
    ram_budget_bytes: Optional[int] = None
    vram_budget_bytes: Optional[int] = None
    current_model_name: Optional[str] = None
    # Add host-level info if lane budget is missing
    host_ram_budget_bytes: Optional[int] = None
    host_vram_budget_bytes: Optional[int] = None


class ViabilityModelInfo(BaseModel):
    model_id: str
    model_name: str
    size_bytes: Optional[int] = None
    required_ram_bytes: Optional[int] = None
    required_vram_bytes: Optional[int] = None
    estimated_tps: Optional[float] = None


class ViabilityResult(BaseModel):
    is_viable: bool
    reason: Optional[str] = None
    projected_free_ram_bytes: Optional[int] = None
    projected_free_vram_bytes: Optional[int] = None
    estimated_tps: Optional[float] = None


class SwapEstimation(BaseModel):
    estimated_ms: int
    strategy: str  # 'historical', 'heuristic', 'instant'


def check_viability(lane: ViabilityLaneInfo, model: ViabilityModelInfo) -> ViabilityResult:
    """
    Core viability logic:
    1. Memory fit: at least 1 GiB must remain after loading the model.
    2. TPS: expected TPS must be > 0.09.
    """
    # 1. Memory fit rule
    # We use the budget from the lane if available, otherwise fallback to host budget.
    ram_limit = lane.ram_budget_bytes or lane.host_ram_budget_bytes or 0
    vram_limit = lane.vram_budget_bytes or lane.host_vram_budget_bytes or 0

    # Required memory: if not explicitly set in policy, fallback to model size_bytes as a heuristic
    req_ram = model.required_ram_bytes or 0
    req_vram = model.required_vram_bytes or 0

    if not req_ram and not req_vram and model.size_bytes:
        # Heuristic: if we don't know the exact requirements, assume it needs its size in the primary memory for its type
        if lane.lane_type == "gpu":
            req_vram = model.size_bytes
        else:
            req_ram = model.size_bytes

    projected_free_ram = ram_limit - req_ram
    projected_free_vram = vram_limit - req_vram

    # Check if we have enough memory and maintain 1 GiB headroom
    # For GPU lanes, we primarily care about VRAM, but RAM might also be a factor depending on the runner.
    # For now, we apply the 1GiB rule to the primary memory type for the lane.

    if lane.lane_type == "gpu":
        if projected_free_vram < MIN_FREE_MEMORY_BYTES:
            return ViabilityResult(
                is_viable=False,
                reason=f"Insufficient VRAM headroom: {projected_free_vram / 1024**3:.2f} GiB < 1 GiB",
                projected_free_ram_bytes=projected_free_ram,
                projected_free_vram_bytes=projected_free_vram,
            )
    elif lane.lane_type == "mlx":
        # MLX uses unified memory, so we check RAM
        if projected_free_ram < MIN_FREE_MEMORY_BYTES:
            return ViabilityResult(
                is_viable=False,
                reason=f"Insufficient unified RAM headroom: {projected_free_ram / 1024**3:.2f} GiB < 1 GiB",
                projected_free_ram_bytes=projected_free_ram,
                projected_free_vram_bytes=projected_free_vram,
            )
    else:  # cpu or other
        if projected_free_ram < MIN_FREE_MEMORY_BYTES:
            return ViabilityResult(
                is_viable=False,
                reason=f"Insufficient RAM headroom: {projected_free_ram / 1024**3:.2f} GiB < 1 GiB",
                projected_free_ram_bytes=projected_free_ram,
                projected_free_vram_bytes=projected_free_vram,
            )

    # 2. TPS rule
    if model.estimated_tps is None:
        return ViabilityResult(
            is_viable=False,
            reason="Unknown TPS (not viable by default)",
            projected_free_ram_bytes=projected_free_ram,
            projected_free_vram_bytes=projected_free_vram,
            estimated_tps=None,
        )

    if model.estimated_tps <= MIN_VIABLE_TPS:
        return ViabilityResult(
            is_viable=False,
            reason=f"Low TPS: {model.estimated_tps} <= {MIN_VIABLE_TPS}",
            projected_free_ram_bytes=projected_free_ram,
            projected_free_vram_bytes=projected_free_vram,
            estimated_tps=model.estimated_tps,
        )

    return ViabilityResult(
        is_viable=True,
        projected_free_ram_bytes=projected_free_ram,
        projected_free_vram_bytes=projected_free_vram,
        estimated_tps=model.estimated_tps,
    )


def estimate_swap_time(
    lane: ViabilityLaneInfo,
    model: ViabilityModelInfo,
    historical_avg_ms: Optional[int] = None,
) -> SwapEstimation:
    """
    Estimate the time to swap/load a model on a lane.
    Strategy:
    1. If same model is already loaded, 0ms (instant).
    2. If historical average is provided, use it.
    3. Heuristic fallback: size_bytes / estimated bandwidth.
    """
    if lane.current_model_name == model.model_name:
        return SwapEstimation(estimated_ms=0, strategy="instant")

    if historical_avg_ms is not None:
        return SwapEstimation(estimated_ms=historical_avg_ms, strategy="historical")

    # Heuristic fallback
    # Bandwidth assumptions (very rough):
    # GPU/NVMe: 500 MB/s
    # CPU/HDD: 50 MB/s
    # MLX/Unified: 200 MB/s
    size_bytes = model.size_bytes or 0
    if not size_bytes:
        # If no size, guess 5 seconds
        return SwapEstimation(estimated_ms=5000, strategy="heuristic")

    bandwidth_bps = 50 * 1024 * 1024  # Default 50 MB/s
    if lane.lane_type == "gpu":
        bandwidth_bps = 500 * 1024 * 1024
    elif lane.lane_type == "mlx":
        bandwidth_bps = 200 * 1024 * 1024

    estimated_ms = int((size_bytes / bandwidth_bps) * 1000)

    # Add overhead (e.g., 1 second for process start)
    estimated_ms += 1000

    return SwapEstimation(estimated_ms=estimated_ms, strategy="heuristic")

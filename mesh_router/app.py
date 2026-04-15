from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import threading
import uuid
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from .config import settings
from .db import db, mw_state_db
from .inventory import fetch_lane_inventory, group_inventory_by_host
from .mw_overlay import (
    _candidate_mw_binding,
    apply_mw_effective_status,
    _normalize_router_backend_type,
)
from .perf_registry import get_expectation, insert_observation
from .route_resolver import resolve_route
from .router import LanePlacementError, pick_lane_for_model
from .schemas import (
    ArchiveInventoryScanRequest,
    ChatCompletionRequest,
    ImageGenerationRequest,
    HostInventoryScanRequest,
    LaneModelCandidate,
    LaneCapabilityResponse,
    LaneSwapEventRequest,
    LaneSwapResponse,
    ModelTagsResponse,
    ModelTagsUpdateRequest,
    ModelTuningProfileResponse,
    ModelTuningProfileUpsertRequest,
    ModelInfo,
    MWCommandRequest,
    MWCommandResponse,
    MWCommandStatusResponse,
    ModelsResponse,
    RestoreSplitModeRequest,
    RestoreSplitModeResponse,
    RouterRequestCancelRequest,
    RouterRequestSubmitRequest,
    SwapPreflightResponse,
    InventoryResponse,
    PerfExpectationResponse,
    PerfObservationIngestRequest,
    RouteResolveRequest,
    RouteResolveResponse,
)
from .tokens import sign_token, verify_token
from .viability import ViabilityLaneInfo, ViabilityModelInfo, check_viability, estimate_swap_time
from .logging_config import setup_logging
from .mw_control import MWControlError, MeshWorkerCommandClient
from .mw_grpc import MwGrpcClient, MwGrpcClientError, MwGrpcTarget

# Configure standardized logging
setup_logging(service_name="mesh-router")
logger = logging.getLogger(__name__)

app = FastAPI(title="mesh-router", version="0.1.0")


@lru_cache(maxsize=1)
def _mw_client() -> MeshWorkerCommandClient:
    return MeshWorkerCommandClient.from_settings()


def _normalize_mw_host_id(host_name: str) -> str:
    return (host_name or "").strip().lower().replace(" ", "-")


def _infer_mw_lane_id_for_row(row: dict[str, Any]) -> str | None:
    lane_name = str(row.get("lane_name") or "").strip()
    if lane_name:
        return lane_name
    lane_type = str(row.get("lane_type") or "").strip().lower()
    if lane_type in {"cpu", "gpu", "combined", "mlx"}:
        return lane_type
    return None


def _backend_compatibility_reason(
    *,
    model_name: str,
    tags: list[str] | None,
    backend_type: str | None,
    lane_type: str | None,
) -> str | None:
    normalized_backend = _normalize_router_backend_type(backend_type)
    normalized_lane_type = str(lane_type or "").strip().lower()
    normalized_tags = set(_normalized_model_tags(tags))
    lowered_model = str(model_name or "").strip().lower()

    is_flux = (
        lowered_model.startswith("flux")
        or "flux" in normalized_tags
        or "stable-diffusion" in normalized_tags
        or "image" in normalized_tags
    )
    if is_flux:
        if normalized_backend != "sd":
            return "model requires stable-diffusion backend"
        return None

    is_bitnet = (
        "bitnet" in normalized_tags
        or "bitnet" in lowered_model
        or "1.58bit" in lowered_model
    )
    if is_bitnet:
        if normalized_backend != "bitnet":
            return "model requires bitnet backend"
        if normalized_lane_type != "cpu":
            return "bitnet models are cpu-only"
        return None

    if normalized_backend in {"sd", "bitnet"}:
        return f"model is incompatible with backend {normalized_backend}"
    return None


def _should_include_candidate_for_capabilities(*, mw_authoritative: bool, source_locality: str) -> bool:
    if not mw_authoritative:
        return True
    return source_locality == "local"


def _mw_runtime_candidate_tags(*, lane_row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    backend = _normalize_router_backend_type(lane_row.get("backend_type"))
    lane_type = str(lane_row.get("lane_type") or "").strip().lower()
    if backend:
        tags.append(backend)
    if lane_type:
        tags.append(lane_type)
    if backend == "bitnet":
        tags.append("bitnet")
    return tags


def _path_matches_local_model_root(*, artifact_path: str | None, local_model_root: str | None) -> bool:
    if not local_model_root:
        return True
    path_value = str(artifact_path or "").strip()
    root_value = str(local_model_root or "").strip()
    if not path_value or not root_value:
        return False

    normalized_path = path_value.rstrip("/")
    normalized_root = root_value.rstrip("/")
    return normalized_path == normalized_root or normalized_path.startswith(f"{normalized_root}/")


def _prune_lane_model_viability_outside_local_root(
    cur: Any,
    *,
    lane_id: str,
    local_model_root: str | None,
) -> None:
    if not local_model_root:
        return
    root_value = str(local_model_root).rstrip("/")
    root_prefix = f"{root_value}/%"
    cur.execute(
        """
        DELETE FROM lane_model_viability lmv
        USING host_model_artifacts hma
        WHERE lmv.lane_id=%s
          AND lmv.source_locality='local'
          AND lmv.artifact_id=hma.artifact_id
          AND NOT (hma.local_path=%s OR hma.local_path LIKE %s)
        """,
        (lane_id, root_value, root_prefix),
    )


def _maybe_record_perf_observation(
    *,
    host_name: str | None,
    lane_id: str | None,
    model_name: str | None,
    modality: str,
    backend_type: str | None = None,
    lane_type: str | None = None,
    elapsed_ms: int | None,
    first_token_ms: float | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    decode_tps: float | None,
    was_cold: bool | None = None,
    ok: bool,
    error_kind: str | None,
    error_message: str | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Best-effort perf observation ingestion from real traffic.

    - Never raises (observations must not fail requests).
    - Requires mw_perf_observations to exist in MW state DB.
    """

    if not settings.perf_auto_observe_enabled:
        return
    try:
        sample_rate = float(settings.perf_auto_observe_sample_rate)
    except Exception:
        sample_rate = 1.0
    if sample_rate < 1.0 and random.random() > max(0.0, sample_rate):
        return
    if not host_name or not lane_id or not model_name:
        return
    if elapsed_ms is None:
        return
    if elapsed_ms < int(settings.perf_auto_observe_min_elapsed_ms):
        return
    try:
        if elapsed_ms > int(settings.perf_auto_observe_max_total_ms):
            return
    except Exception:
        pass
    if not ok and (error_kind or "") == "canceled":
        # Avoid polluting the registry with user-initiated cancels.
        return

    md = dict(metadata or {})
    md["source"] = "auto_observe"
    try:
        with mw_state_db.connect() as conn:
            with conn.cursor() as cur:
                insert_observation(
                    cur=cur,
                    obs={
                        "host_id": host_name,
                        "lane_id": lane_id,
                        "model_name": model_name,
                        "backend_type": backend_type or ("sd" if modality == "images" else "llama"),
                        "lane_type": lane_type,
                        "modality": modality,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": completion_tokens,
                        "first_token_ms": float(first_token_ms) if first_token_ms is not None else None,
                        "decode_tps": float(decode_tps) if decode_tps is not None else None,
                        "total_ms": float(elapsed_ms),
                        "was_cold": was_cold,
                        "ok": bool(ok),
                        "error_kind": error_kind,
                        "error_message": error_message,
                        "metadata": md,
                    },
                )
            conn.commit()
    except Exception:
        logger.exception(
            "Failed to auto-ingest perf observation",
            extra={"host_name": host_name, "lane_id": lane_id, "model_name": model_name, "modality": modality},
        )


def _maybe_add_perf_expectation_headers(
    *,
    headers: dict[str, str],
    host_id: str | None,
    lane_id: str | None,
    model_name: str | None,
    modality: str,
) -> None:
    """
    Best-effort requestor-facing perf expectation headers.

    This is intentionally optional/off by default to avoid adding DB overhead
    or leaking internal details unintentionally.
    """

    if not settings.route_debug_headers_enabled:
        return
    if not host_id or not lane_id or not model_name:
        return
    try:
        with mw_state_db.connect() as conn:
            with conn.cursor() as cur:
                exp = get_expectation(
                    cur=cur,
                    host_id=str(host_id),
                    lane_id=str(lane_id),
                    model_name=str(model_name),
                    modality=str(modality),
                )
        if not exp:
            return
        headers["X-Mesh-Perf-Sample-Count"] = str(int(exp.sample_count))
        headers["X-Mesh-Perf-Updated-At"] = exp.updated_at.isoformat()
        if exp.first_token_ms_p50 is not None:
            headers["X-Mesh-Perf-FirstTokenMs-P50"] = str(float(exp.first_token_ms_p50))
        if exp.decode_tps_p50 is not None:
            headers["X-Mesh-Perf-DecodeTps-P50"] = str(float(exp.decode_tps_p50))
        if exp.total_ms_p50 is not None:
            headers["X-Mesh-Perf-TotalMs-P50"] = str(float(exp.total_ms_p50))
    except Exception:
        return


def _mw_target_for_lane(*, cur: Any, lane_id: str) -> MwGrpcTarget | None:
    """
    Returns a MW gRPC target when the lane is explicitly marked as MW-managed.

    Convention (lane.proxy_auth_metadata):
    - control_plane: "mw"
    - mw_host_id: optional override (default derived from host_name)
    - mw_lane_id: optional override (default lane_name)
    - mw_grpc_port: optional override (default settings.mw_grpc_default_port)
    """
    cur.execute(
        """
        SELECT
          l.lane_id,
          l.lane_name,
          l.lane_type,
          l.backend_type,
          l.base_url,
          l.proxy_auth_metadata,
          h.host_name
        FROM lanes l
        JOIN hosts h ON h.host_id = l.host_id
        WHERE l.lane_id=%s
        """,
        (lane_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    meta = row.get("proxy_auth_metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    binding = _candidate_mw_binding(row)
    if binding is None:
        return None
    inferred_host_id, inferred_lane_id, inferred = binding
    if inferred:
        mw_host_seen = False
        mw_lane_seen = False
        fallback_lane_id: str | None = None
        try:
            with mw_state_db.connect() as conn:
                with conn.cursor() as mw_cur:
                    mw_cur.execute(
                        """
                        SELECT EXISTS(SELECT 1 FROM mw_hosts WHERE host_id=%s) AS host_exists,
                               EXISTS(SELECT 1 FROM mw_lanes WHERE host_id=%s AND lane_id=%s) AS lane_exists
                        """,
                        (inferred_host_id, inferred_host_id, inferred_lane_id),
                    )
                    seen = mw_cur.fetchone() or {}
                    mw_host_seen = bool(seen.get("host_exists"))
                    mw_lane_seen = bool(seen.get("lane_exists"))
                    if mw_host_seen and not mw_lane_seen:
                        mw_cur.execute(
                            """
                            SELECT lane_id, lane_type, backend_type, service_id
                            FROM mw_lanes
                            WHERE host_id=%s
                            ORDER BY lane_id
                            """,
                            (inferred_host_id,),
                        )
                        candidate_rows = list(mw_cur.fetchall() or [])
                        if len(candidate_rows) == 1:
                            fallback_lane_id = str(candidate_rows[0].get("lane_id") or "").strip() or None
                        else:
                            desired_lane_type = str(row.get("lane_type") or "").strip().lower()
                            desired_backend_type = str(row.get("backend_type") or "").strip().lower()
                            preferred = None
                            for candidate in candidate_rows:
                                candidate_lane_type = str(candidate.get("lane_type") or "").strip().lower()
                                candidate_backend_type = str(candidate.get("backend_type") or "").strip().lower()
                                if desired_lane_type and candidate_lane_type == desired_lane_type:
                                    preferred = candidate
                                    break
                                if desired_backend_type and candidate_backend_type == desired_backend_type:
                                    preferred = candidate
                            if preferred is not None:
                                fallback_lane_id = str(preferred.get("lane_id") or "").strip() or None
        except Exception as exc:
            logger.warning(
                "MW state lookup failed for inferred lane binding host=%s lane=%s: %s",
                inferred_host_id,
                inferred_lane_id,
                exc,
            )
        if not mw_lane_seen and not mw_host_seen:
            return None
        meta = {
            "control_plane": "mw",
            "mw_host_id": inferred_host_id,
            "mw_lane_id": fallback_lane_id or inferred_lane_id,
            "mw_inferred": True,
        }
    base_url = str(row.get("base_url") or "")
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        return None
    port = int(meta.get("mw_grpc_port") or settings.mw_grpc_default_port)
    endpoint = f"{host}:{port}"
    host_id = str(meta.get("mw_host_id") or _normalize_mw_host_id(str(row.get("host_name") or "")))
    lane_name = str(row.get("lane_name") or "").strip()
    lane_id_str = str(meta.get("mw_lane_id") or lane_name or _infer_mw_lane_id_for_row(row) or "")
    if not host_id or not lane_id_str:
        return None
    return MwGrpcTarget(endpoint=endpoint, host_id=host_id, lane_id=lane_id_str)


def _mw_effective_lane_row_for_capabilities(*, lane_row: dict[str, Any], host_row: dict[str, Any] | None) -> dict[str, Any]:
    candidate = dict(lane_row)
    candidate["host_name"] = str((host_row or {}).get("host_name") or candidate.get("host_name") or "")
    apply_mw_effective_status(
        [candidate],
        mw_state_db=mw_state_db,
        stale_seconds=settings.default_lease_stale_seconds,
    )
    return candidate


def _configure_tracing(app: FastAPI) -> bool:
    """Best-effort OTEL setup for FastAPI when explicitly enabled."""
    enabled = os.getenv("MESH_ROUTER_OTEL_ENABLED", "false").lower() in {"1", "true", "yes"}
    if not enabled:
        return False

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        logger.warning(
            "MESH_ROUTER_OTEL_ENABLED=true but OTEL_EXPORTER_OTLP_ENDPOINT is unset; tracing disabled"
        )
        return False

    insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() in {"1", "true", "yes"}
    service_name = os.getenv("OTEL_SERVICE_NAME", "mesh-router").strip() or "mesh-router"

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("OpenTelemetry tracing enabled for %s -> %s", service_name, endpoint)
        return True
    except Exception as exc:
        logger.warning("Failed to initialize OpenTelemetry tracing: %s", exc)
        return False


_configure_tracing(app)

ARCHIVE_PROVIDERS = {"packhub", "packhub02"}
REQUEST_TERMINAL_STATES = {"released", "failed", "expired", "canceled"}
SWAP_TERMINAL_STATES = {"ready", "failed", "canceled"}
IMAGE_DEFAULT_WIDTH = 1024
IMAGE_DEFAULT_HEIGHT = 1024


def _bytes_from_gib(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value) * (1024**3))
    except Exception:
        return None


def _normalize_model_format(value: str | None) -> str:
    normalized = (value or "other").strip().lower()
    if normalized in {"gguf", "mlx", "safetensors"}:
        return normalized
    return "other"


def _model_lookup_keys(model_name: str | None) -> set[str]:
    raw = (model_name or "").strip()
    if not raw:
        return set()

    def _add_variant(keys: set[str], value: str) -> None:
        cleaned = value.strip().lower()
        if cleaned:
            keys.add(cleaned)

    keys = {raw.lower()}
    stem = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    keys.add(stem.lower())

    lowered_stem = stem.lower()
    if lowered_stem.endswith((".gguf", ".safetensors", ".bin")):
        stem_no_ext = stem.rsplit(".", 1)[0]
        keys.add(stem_no_ext.lower())
    else:
        stem_no_ext = stem

    normalized = stem_no_ext.lower().replace("_", "-").replace(":", "-")
    keys.add(normalized)

    dequantized = re.sub(r"[-_.]q\d+(?:[-_.]k(?:[-_.][a-z0-9]+)?)?$", "", normalized)
    _add_variant(keys, dequantized)
    debitted = re.sub(r"[-_.](?:\d+(?:\.\d+)?bit|fp8)$", "", normalized)
    _add_variant(keys, debitted)
    if dequantized:
        dequantized_debitted = re.sub(r"[-_.](?:\d+(?:\.\d+)?bit|fp8)$", "", dequantized)
        _add_variant(keys, dequantized_debitted)
    for value in list(keys):
        stripped = re.sub(r"[-_.](?:instruct|instruction|chat|it)$", "", value)
        _add_variant(keys, stripped)

    out = {key for key in keys if key}
    out |= _family_size_tags_from_keys(out)
    return out


def _normalize_model_tag(tag: str | None) -> str | None:
    raw = (tag or "").strip().lower()
    if not raw:
        return None
    return raw.replace("_", "-")


def _normalized_model_tags(tags: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags or []:
        value = _normalize_model_tag(tag)
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _family_size_tags_from_keys(keys: set[str]) -> set[str]:
    tags: set[str] = set()
    for key in keys:
        for family, pattern in (
            ("qwen3.5", r"qwen3\.5[-_:]?(\d+(?:\.\d+)?)(b|m)\b"),
            ("falcon3", r"falcon3[-_:]?(\d+(?:\.\d+)?)(b|m)\b"),
            ("lfm2.5", r"lfm2\.5[-_:]?(\d+(?:\.\d+)?)(b|m)\b"),
            ("gemma4", r"gemma4[-_:]?(\d+(?:\.\d+)?)(b|m)\b"),
        ):
            match = re.search(pattern, key)
            if match:
                size = f"{match.group(1)}{match.group(2).lower()}"
                tags.add(f"{family}:{size}")
                tags.add(f"{family}-{size}")
    return tags


def _inferred_model_tags(model_name: str | None) -> list[str]:
    return sorted(_family_size_tags_from_keys(_model_lookup_keys(model_name)))


def _model_tags_with_inferred(model_name: str | None, tags: list[str] | None) -> list[str]:
    return _normalized_model_tags(list(tags or []) + _inferred_model_tags(model_name))


def _is_exact_model_request(model_name: str | None) -> bool:
    raw = (model_name or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if "/" in raw or "\\" in raw:
        return True
    if lowered.endswith((".gguf", ".safetensors", ".bin")):
        return True
    if re.search(r"(?:^|[-_.:])q\d+(?:[-_.]k(?:[-_.][a-z0-9]+)?)?(?:$|[-_.])", lowered):
        return True
    if re.search(r"(?:^|[-_.:])(?:\d+bit|fp8)(?:$|[-_.])", lowered):
        return True
    return False


def _model_request_matches_candidate(
    requested_model_name: str,
    candidate_model_name: str,
    candidate_tags: list[str] | None = None,
) -> bool:
    if _is_exact_model_request(requested_model_name):
        # Also match against the basename so that lanes storing full local paths
        # (e.g. /Users/kasunami/models/Qwen3.5-9B-6bit) match a bare name request.
        path_parts = re.split(r"[\\/]+", candidate_model_name)
        candidate_stem = path_parts[-1] if path_parts else candidate_model_name
        candidate_parent = path_parts[-2] if len(path_parts) >= 2 else ""
        if (
            candidate_model_name == requested_model_name
            or candidate_stem == requested_model_name
            or candidate_parent == requested_model_name
        ):
            return True
    request_keys = _model_lookup_keys(requested_model_name)
    if request_keys & _model_lookup_keys(candidate_model_name):
        return True
    return bool(request_keys & set(_model_tags_with_inferred(candidate_model_name, candidate_tags)))


def _resolve_swap_candidate(
    capabilities: LaneCapabilityResponse,
    requested_model_name: str,
) -> tuple[LaneModelCandidate | None, str | None]:
    groups = (
        ("viable", capabilities.local_viable_models + capabilities.remote_viable_models),
        ("unverified", capabilities.unverified_models),
    )

    for group_name, candidates in groups:
        for candidate in candidates:
            if _model_request_matches_candidate(
                requested_model_name,
                candidate.model_name,
                candidate.tags,
            ):
                return candidate, group_name

    return None, None


def _estimate_text_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, (len(value) + 3) // 4)
    if isinstance(value, list):
        return sum(_estimate_text_tokens(item) for item in value) + max(0, len(value) * 4)
    if isinstance(value, dict):
        total = 0
        for key in ("content", "text", "input", "prompt", "name", "tool_call_id"):
            if key in value:
                total += _estimate_text_tokens(value.get(key))
        if value.get("tool_calls"):
            total += _estimate_text_tokens(value.get("tool_calls"))
        return total
    return _estimate_text_tokens(str(value))


def _estimate_request_context_tokens(*, route: str, payload: dict[str, Any]) -> int | None:
    if route == "images":
        return None

    prompt_tokens = 0
    if route == "chat":
        messages = payload.get("messages") or []
        prompt_tokens = sum(_estimate_text_tokens(message) for message in messages) + max(32, len(messages) * 8)
    elif route == "embeddings":
        prompt_tokens = _estimate_text_tokens(payload.get("input"))
    else:
        prompt_tokens = _estimate_text_tokens(payload)

    max_tokens = payload.get("max_tokens")
    try:
        completion_budget = max(0, int(max_tokens)) if max_tokens is not None else 0
    except Exception:
        completion_budget = 0

    total = prompt_tokens + completion_budget
    return total if total > 0 else None


def _resolve_host_id(cur, host_ref: str, *, create: bool = False) -> tuple[str, str]:
    cur.execute(
        """
        SELECT host_id, host_name
        FROM hosts
        WHERE host_id::text=%s OR host_name::text=%s
        """,
        (host_ref, host_ref),
    )
    row = cur.fetchone()
    if row:
        return str(row["host_id"]), str(row["host_name"])
    if not create:
        raise HTTPException(status_code=404, detail=f"host not found: {host_ref}")

    cur.execute(
        """
        INSERT INTO hosts (host_name, status, notes)
        VALUES (%s, 'unknown', %s)
        RETURNING host_id, host_name
        """,
        (host_ref, "Auto-created from inventory scan"),
    )
    row = cur.fetchone()
    return str(row["host_id"]), str(row["host_name"])


def _ensure_model(cur, *, model_name: str, model_format: str | None, size_bytes: int | None) -> str:
    cur.execute(
        """
        INSERT INTO models (model_name, format, size_bytes)
        VALUES (%s, %s::model_format, %s)
        ON CONFLICT (model_name) DO UPDATE
        SET format = COALESCE(models.format, EXCLUDED.format),
            size_bytes = COALESCE(EXCLUDED.size_bytes, models.size_bytes),
            updated_at = now()
        RETURNING model_id
        """,
        (model_name, _normalize_model_format(model_format), size_bytes),
    )
    return str(cur.fetchone()["model_id"])


def _resolve_model_ref(cur, model_ref: str) -> tuple[str, str, list[str]]:
    cur.execute(
        """
        SELECT model_id, model_name, tags
        FROM models
        WHERE model_id::text=%s OR model_name::text=%s
        LIMIT 1
        """,
        (model_ref, model_ref),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"model not found: {model_ref}")
    return str(row["model_id"]), str(row["model_name"]), _normalized_model_tags(row.get("tags") or [])


def _resolve_lane_ref(cur, lane_ref: str | None) -> tuple[str | None, str | None, str | None]:
    if not lane_ref:
        return None, None, None
    cur.execute(
        """
        SELECT lane_id, lane_name, lane_type
        FROM lanes
        WHERE lane_id::text=%s OR lane_name::text=%s OR base_url=%s
        LIMIT 1
        """,
        (lane_ref, lane_ref, lane_ref),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"lane not found: {lane_ref}")
    return str(row["lane_id"]), str(row.get("lane_name") or ""), str(row.get("lane_type") or "")


def _row_to_tuning_profile(row: dict[str, Any]) -> ModelTuningProfileResponse:
    return ModelTuningProfileResponse(
        tuning_profile_id=str(row["tuning_profile_id"]),
        host_id=str(row["host_id"]),
        host_name=str(row["host_name"]),
        model_id=str(row["model_id"]),
        model_name=str(row["model_name"]),
        lane_id=str(row["lane_id"]) if row.get("lane_id") else None,
        lane_name=str(row["lane_name"]) if row.get("lane_name") else None,
        lane_type=str(row["lane_type"]) if row.get("lane_type") else None,
        storage_scheme=str(row["storage_scheme"]),
        settings=dict(row.get("settings") or {}),
        cost_tier=str(row.get("cost_tier") or "standard"),
        disables_sibling_lanes=bool(row.get("disables_sibling_lanes") or False),
        exclusive_host_resources=bool(row.get("exclusive_host_resources") or False),
        prompt_tps=float(row["prompt_tps"]) if row.get("prompt_tps") is not None else None,
        generation_tps=float(row["generation_tps"]) if row.get("generation_tps") is not None else None,
        avg_total_latency_s=float(row["avg_total_latency_s"]) if row.get("avg_total_latency_s") is not None else None,
        score=float(row["score"]) if row.get("score") is not None else None,
        evaluation_count=int(row.get("evaluation_count") or 1),
        source_run_tag=str(row["source_run_tag"]) if row.get("source_run_tag") else None,
        notes=str(row["notes"]) if row.get("notes") else None,
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )


def _upsert_model_tuning_profile(cur, req: ModelTuningProfileUpsertRequest) -> ModelTuningProfileResponse:
    host_id, _ = _resolve_host_id(cur, req.host_ref, create=False)
    model_id = _ensure_model(cur, model_name=req.model_name, model_format=None, size_bytes=None)
    lane_id, _, _ = _resolve_lane_ref(cur, req.lane_ref)

    cur.execute(
        """
        INSERT INTO model_tuning_profiles (
          host_id, model_id, lane_id, storage_scheme, settings,
          cost_tier, disables_sibling_lanes, exclusive_host_resources,
          prompt_tps, generation_tps, avg_total_latency_s, score,
          evaluation_count, source_run_tag, notes, updated_at
        )
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (host_id, model_id, storage_scheme)
        DO UPDATE SET
          lane_id = EXCLUDED.lane_id,
          settings = EXCLUDED.settings,
          cost_tier = EXCLUDED.cost_tier,
          disables_sibling_lanes = EXCLUDED.disables_sibling_lanes,
          exclusive_host_resources = EXCLUDED.exclusive_host_resources,
          prompt_tps = EXCLUDED.prompt_tps,
          generation_tps = EXCLUDED.generation_tps,
          avg_total_latency_s = EXCLUDED.avg_total_latency_s,
          score = EXCLUDED.score,
          evaluation_count = EXCLUDED.evaluation_count,
          source_run_tag = EXCLUDED.source_run_tag,
          notes = EXCLUDED.notes,
          updated_at = now()
        RETURNING tuning_profile_id
        """,
        (
            host_id,
            model_id,
            lane_id,
            req.storage_scheme,
            Jsonb(req.settings),
            req.cost_tier,
            req.disables_sibling_lanes,
            req.exclusive_host_resources,
            req.prompt_tps,
            req.generation_tps,
            req.avg_total_latency_s,
            req.score,
            req.evaluation_count or 1,
            req.source_run_tag,
            req.notes,
        ),
    )
    tuning_profile_id = str(cur.fetchone()["tuning_profile_id"])
    cur.execute(
        """
        SELECT
          tp.tuning_profile_id,
          tp.host_id,
          h.host_name,
          tp.model_id,
          m.model_name,
          tp.lane_id,
          l.lane_name,
          l.lane_type,
          tp.storage_scheme,
          tp.settings,
          tp.cost_tier,
          tp.disables_sibling_lanes,
          tp.exclusive_host_resources,
          tp.prompt_tps,
          tp.generation_tps,
          tp.avg_total_latency_s,
          tp.score,
          tp.evaluation_count,
          tp.source_run_tag,
          tp.notes,
          tp.created_at,
          tp.updated_at
        FROM model_tuning_profiles tp
        JOIN hosts h ON h.host_id = tp.host_id
        JOIN models m ON m.model_id = tp.model_id
        LEFT JOIN lanes l ON l.lane_id = tp.lane_id
        WHERE tp.tuning_profile_id = %s
        """,
        (tuning_profile_id,),
    )
    return _row_to_tuning_profile(cur.fetchone())


def _update_host_inventory_metadata(
    cur,
    *,
    host_id: str,
    root_path: str | None,
    host_facts: dict[str, Any] | None,
    status: str = "ready",
) -> None:
    host_facts = host_facts or {}
    update_parts = [
        "status=%s",
        "last_seen_at=now()",
        "updated_at=now()",
    ]
    params: list[Any] = [status]

    if root_path:
        update_parts.append(
            """
            model_store_paths=(
              SELECT COALESCE(jsonb_agg(DISTINCT value), '[]'::jsonb)
              FROM jsonb_array_elements_text(COALESCE(hosts.model_store_paths, '[]'::jsonb) || to_jsonb(ARRAY[%s]::text[]))
            )
            """
        )
        params.append(root_path)

    hostname = host_facts.get("hostname")
    if hostname:
        update_parts.append("notes = COALESCE(notes, %s)")
        params.append(f"Inventory-reported hostname: {hostname}")

    ram_total_bytes = host_facts.get("total_ram_bytes") or _bytes_from_gib(host_facts.get("total_ram_gb"))
    if ram_total_bytes is not None:
        update_parts.append("ram_total_bytes=%s")
        params.append(ram_total_bytes)

    available_ram_bytes = host_facts.get("available_ram_bytes") or _bytes_from_gib(host_facts.get("available_ram_gb"))
    if available_ram_bytes is not None:
        update_parts.append("ram_ai_budget_bytes=%s")
        params.append(available_ram_bytes)

    params.append(host_id)
    cur.execute(f"UPDATE hosts SET {', '.join(update_parts)} WHERE host_id=%s", tuple(params))


def _ingest_artifacts(
    cur,
    *,
    host_id: str,
    artifacts: list[Any],
    storage_scope: str,
    storage_provider: str | None,
    root_path: str | None = None,
) -> list[dict[str, Any]]:
    ingested: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for artifact in artifacts:
        model_name = artifact.name.strip()
        local_path = artifact.path.strip()
        if not model_name or not local_path:
            continue
        seen_paths.add(local_path)
        model_id = _ensure_model(
            cur,
            model_name=model_name,
            model_format=artifact.format,
            size_bytes=artifact.size_bytes,
        )
        cur.execute(
            """
            INSERT INTO host_model_artifacts (
              host_id, model_id, local_path, size_bytes,
              present, last_verified_at, updated_at,
              storage_scope, storage_provider, format, sha256
            )
            VALUES (%s, %s, %s, %s, true, now(), now(), %s, %s, %s, %s)
            ON CONFLICT (host_id, model_id, local_path) DO UPDATE
            SET size_bytes = COALESCE(EXCLUDED.size_bytes, host_model_artifacts.size_bytes),
                present = true,
                last_verified_at = now(),
                updated_at = now(),
                storage_scope = EXCLUDED.storage_scope,
                storage_provider = EXCLUDED.storage_provider,
                format = COALESCE(EXCLUDED.format, host_model_artifacts.format),
                sha256 = COALESCE(EXCLUDED.sha256, host_model_artifacts.sha256)
            RETURNING artifact_id
            """,
            (
                host_id,
                model_id,
                local_path,
                artifact.size_bytes,
                storage_scope,
                storage_provider,
                artifact.format,
                artifact.checksum,
            ),
        )
        artifact_id = str(cur.fetchone()["artifact_id"])
        ingested.append(
            {
                "artifact_id": artifact_id,
                "model_id": model_id,
                "model_name": model_name,
                "local_path": local_path,
                "size_bytes": artifact.size_bytes,
                "storage_scope": storage_scope,
                "storage_provider": storage_provider,
            }
        )

    if root_path:
        root_raw = str(root_path).rstrip("/")
        root_prefix = f"{root_raw}/%"
        if seen_paths:
            cur.execute(
                """
                UPDATE host_model_artifacts
                SET present = false, updated_at = now()
                WHERE host_id=%s
                  AND storage_scope=%s
                  AND (local_path=%s OR local_path LIKE %s)
                  AND local_path <> ALL(%s)
                """,
                (host_id, storage_scope, root_raw, root_prefix, list(seen_paths)),
            )
        else:
            cur.execute(
                """
                UPDATE host_model_artifacts
                SET present = false, updated_at = now()
                WHERE host_id=%s
                  AND storage_scope=%s
                  AND (local_path=%s OR local_path LIKE %s)
                """,
                (host_id, storage_scope, root_raw, root_prefix),
            )
    return ingested


def _latest_tps(cur, lane_id: str, model_id: str) -> float | None:
    cur.execute(
        """
        SELECT tps
        FROM lane_model_metrics
        WHERE lane_id=%s AND model_id=%s AND success=true AND tps IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (lane_id, model_id),
    )
    row = cur.fetchone()
    return float(row["tps"]) if row and row.get("tps") is not None else None


def _historical_swap_ms(cur, lane_id: str, model_id: str, source_mode: str) -> int | None:
    cur.execute(
        """
        SELECT AVG(duration_ms)::bigint AS avg_ms
        FROM lane_model_swap_history
        WHERE lane_id=%s AND model_id=%s AND source_mode=%s AND success=true AND duration_ms IS NOT NULL
        """,
        (lane_id, model_id, source_mode),
    )
    row = cur.fetchone()
    return int(row["avg_ms"]) if row and row.get("avg_ms") is not None else None


def _tuning_profile_metrics(
    cur,
    *,
    host_id: str,
    lane_id: str,
    model_name: str,
) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT
          tp.prompt_tps,
          tp.generation_tps,
          tp.avg_total_latency_s,
          tp.settings,
          tp.storage_scheme,
          tp.updated_at
        FROM model_tuning_profiles tp
        JOIN models m ON m.model_id = tp.model_id
        WHERE tp.host_id = %s
          AND m.model_name = %s
          AND (tp.lane_id = %s OR tp.lane_id IS NULL)
        ORDER BY CASE WHEN tp.lane_id = %s THEN 0 ELSE 1 END, tp.updated_at DESC
        LIMIT 1
        """,
        (host_id, model_name, lane_id, lane_id),
    )
    return cur.fetchone()


def _resolve_lane(cur, lane_ref: str) -> dict[str, Any]:
    cur.execute(
        """
        SELECT lane_id, host_id, lane_name, lane_type, backend_type, base_url, current_model_name,
               default_model_name,
               ram_budget_bytes, vram_budget_bytes, usable_memory_bytes,
               runtime_overhead_bytes, reserved_headroom_bytes
        FROM lanes
        WHERE lane_id::text=%s OR lane_name::text=%s OR base_url=%s
        """,
        (lane_ref, lane_ref, lane_ref),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="lane not found")
    return row


def _resolve_host(cur, host_id: str) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT host_id, host_name, mgmt_ssh_host, mgmt_ssh_user, model_store_paths,
               ram_total_bytes, ram_ai_budget_bytes, vram_total_bytes, vram_ai_budget_bytes
        FROM hosts
        WHERE host_id=%s
        """,
        (host_id,),
    )
    return cur.fetchone()


def _local_model_root(host_row: dict[str, Any] | None, lane_row: dict[str, Any] | None = None) -> str | None:
    if not host_row:
        return None
    paths = host_row.get("model_store_paths") or []
    if isinstance(paths, list) and paths:
        normalized = [path.strip() for path in paths if isinstance(path, str) and path.strip()]
        if not normalized:
            return None
        lane_type = str((lane_row or {}).get("lane_type") or "").lower()
        backend_type = str((lane_row or {}).get("backend_type") or "").lower()
        if lane_type == "gpu" and backend_type == "sd":
            for path in normalized:
                lowered = path.lower()
                if "image-model" in lowered or "/image" in lowered:
                    return path
        for path in normalized:
            lowered = path.lower()
            if "image-model" in lowered:
                continue
            return path
        return normalized[0]
    return None


def _upsert_usage(cur, *, lane_id: str, model_id: str, used_at: datetime, swap_at: datetime | None = None) -> None:
    cur.execute(
        """
        INSERT INTO lane_model_usage (
          lane_id, model_id, request_count, last_used_at, last_swap_at,
          rolling_24h_count, rolling_7d_count, updated_at
        )
        VALUES (%s, %s, 1, %s, %s, 1, 1, now())
        ON CONFLICT (lane_id, model_id) DO UPDATE
        SET request_count = lane_model_usage.request_count + 1,
            last_used_at = EXCLUDED.last_used_at,
            last_swap_at = COALESCE(EXCLUDED.last_swap_at, lane_model_usage.last_swap_at),
            rolling_24h_count = CASE
              WHEN lane_model_usage.last_used_at IS NOT NULL AND lane_model_usage.last_used_at >= now() - interval '24 hours'
                THEN lane_model_usage.rolling_24h_count + 1
              ELSE 1
            END,
            rolling_7d_count = CASE
              WHEN lane_model_usage.last_used_at IS NOT NULL AND lane_model_usage.last_used_at >= now() - interval '7 days'
                THEN lane_model_usage.rolling_7d_count + 1
              ELSE 1
            END,
            updated_at = now()
        """,
        (lane_id, model_id, used_at, swap_at),
    )


def _build_lane_capability_payload(cur, lane_ref: str) -> tuple[dict[str, Any], LaneCapabilityResponse]:
    lane_row = _resolve_lane(cur, lane_ref)
    resolved_lane_id = str(lane_row["lane_id"])
    resolved_host_id = str(lane_row["host_id"])
    mw_target = _mw_target_for_lane(cur=cur, lane_id=resolved_lane_id)
    mw_authoritative = mw_target is not None
    host_row = _resolve_host(cur, resolved_host_id)
    lane_row = _mw_effective_lane_row_for_capabilities(lane_row=lane_row, host_row=host_row)
    lane_type = str(lane_row["lane_type"])
    local_model_root = _local_model_root(host_row, lane_row)
    if mw_authoritative:
        _prune_lane_model_viability_outside_local_root(
            cur,
            lane_id=resolved_lane_id,
            local_model_root=local_model_root,
        )
    cur.execute(
        """
        SELECT
          hma.artifact_id,
          hma.host_id,
          h.host_name,
          hma.storage_scope,
          hma.storage_provider,
          hma.local_path,
          hma.size_bytes,
          hma.present,
          m.model_id,
          m.model_name,
          m.tags,
          p.required_ram_bytes,
          p.required_vram_bytes,
          p.allowed,
          p.max_ctx
        FROM host_model_artifacts hma
        JOIN models m ON m.model_id=hma.model_id
        JOIN hosts h ON h.host_id=hma.host_id
        LEFT JOIN lane_model_policy p ON p.lane_id=%s AND p.model_id=hma.model_id
        WHERE hma.present=true
          AND (hma.host_id=%s OR hma.storage_scope='archive')
          AND (p.allowed IS DISTINCT FROM false)
        ORDER BY m.model_name, hma.storage_scope, h.host_name
        """,
        (resolved_lane_id, resolved_host_id),
    )
    artifact_rows = cur.fetchall()

    lane_info = ViabilityLaneInfo(
        lane_id=resolved_lane_id,
        lane_type=lane_type,
        ram_budget_bytes=lane_row.get("usable_memory_bytes") or lane_row.get("ram_budget_bytes"),
        vram_budget_bytes=lane_row.get("usable_memory_bytes") if lane_type == "gpu" else lane_row.get("vram_budget_bytes"),
        current_model_name=str(lane_row.get("current_model_name") or "") or None,
        host_ram_budget_bytes=((host_row.get("ram_ai_budget_bytes") or host_row.get("ram_total_bytes")) if host_row else None),
        host_vram_budget_bytes=((host_row.get("vram_ai_budget_bytes") or host_row.get("vram_total_bytes")) if host_row else None),
    )
    runtime_overhead = int(lane_row.get("runtime_overhead_bytes") or 0)
    reserved_headroom = int(lane_row.get("reserved_headroom_bytes") or (1024**3))

    candidates_by_model: dict[str, LaneModelCandidate] = {}
    local_viable: list[LaneModelCandidate] = []
    remote_viable: list[LaneModelCandidate] = []
    unverified: list[LaneModelCandidate] = []

    for row in artifact_rows:
        model_id = str(row["model_id"])
        model_name = str(row["model_name"])
        artifact_id = str(row["artifact_id"])
        size_bytes = int(row["size_bytes"]) if row.get("size_bytes") is not None else None
        compatibility_reason = _backend_compatibility_reason(
            model_name=model_name,
            tags=row.get("tags") or [],
            backend_type=lane_row.get("backend_type"),
            lane_type=lane_type,
        )
        source_locality = "local" if str(row["host_id"]) == resolved_host_id else "remote"
        if not _should_include_candidate_for_capabilities(
            mw_authoritative=mw_authoritative,
            source_locality=source_locality,
        ):
            continue
        if source_locality == "local" and not _path_matches_local_model_root(
            artifact_path=row.get("local_path"),
            local_model_root=local_model_root,
        ):
            continue
        if compatibility_reason:
            cur.execute(
                """
                INSERT INTO lane_model_viability (
                  lane_id, model_id, artifact_id, source_locality,
                  fits_memory, projected_free_bytes, required_memory_bytes,
                  tps_estimate, tps_source, is_viable, reason, last_checked_at
                )
                VALUES (%s, %s, %s, %s, false, NULL, NULL, NULL, 'unknown', false, %s, now())
                ON CONFLICT (lane_id, model_id, source_locality) DO UPDATE
                SET artifact_id = EXCLUDED.artifact_id,
                    fits_memory = EXCLUDED.fits_memory,
                    projected_free_bytes = EXCLUDED.projected_free_bytes,
                    required_memory_bytes = EXCLUDED.required_memory_bytes,
                    tps_estimate = EXCLUDED.tps_estimate,
                    tps_source = EXCLUDED.tps_source,
                    is_viable = EXCLUDED.is_viable,
                    reason = EXCLUDED.reason,
                    last_checked_at = now()
                """,
                (
                    resolved_lane_id,
                    model_id,
                    artifact_id,
                    source_locality,
                    compatibility_reason,
                ),
            )
            continue
        tuning_profile = _tuning_profile_metrics(
            cur,
            host_id=resolved_host_id,
            lane_id=resolved_lane_id,
            model_name=model_name,
        )
        required_memory_bytes = int(
            row["required_vram_bytes"] if lane_type == "gpu" else row["required_ram_bytes"]
        ) if (row.get("required_vram_bytes") is not None or row.get("required_ram_bytes") is not None) else None
        tps_estimate = _latest_tps(cur, resolved_lane_id, model_id)
        if tps_estimate is None and tuning_profile:
            fallback_tps = tuning_profile.get("generation_tps") or tuning_profile.get("prompt_tps")
            if fallback_tps is not None:
                tps_estimate = float(fallback_tps)
        target_context_tokens = int(row["max_ctx"]) if row.get("max_ctx") is not None else None
        if tuning_profile and isinstance(tuning_profile.get("settings"), dict):
            ctx_value = tuning_profile["settings"].get("ctx_size")
            try:
                if target_context_tokens is None and ctx_value is not None:
                    target_context_tokens = int(ctx_value)
            except Exception:
                if target_context_tokens is None:
                    target_context_tokens = None
        model_lane_info = lane_info.model_copy(
            update={
                "target_context_tokens": target_context_tokens,
            }
        )
        model_info = ViabilityModelInfo(
            model_id=model_id,
            model_name=model_name,
            size_bytes=size_bytes,
            required_ram_bytes=required_memory_bytes if lane_type != "gpu" else None,
            required_vram_bytes=required_memory_bytes if lane_type == "gpu" else None,
            estimated_tps=tps_estimate,
        )
        viability = check_viability(model_lane_info, model_info)
        projected_free_bytes = viability.projected_free_vram_bytes if lane_type == "gpu" else viability.projected_free_ram_bytes
        if projected_free_bytes is not None:
            projected_free_bytes -= runtime_overhead
        fits_memory = projected_free_bytes is not None and projected_free_bytes >= reserved_headroom
        if fits_memory and viability.is_viable:
            locality = source_locality
        elif fits_memory and viability.estimated_tps is None:
            locality = "unverified"
        else:
            locality = None

        source_mode = "local" if source_locality == "local" else "remote_copy_then_load"
        swap_estimate = estimate_swap_time(
            lane_info,
            model_info,
            historical_avg_ms=_historical_swap_ms(cur, resolved_lane_id, model_id, source_mode),
        )
        cur.execute(
            """
            INSERT INTO lane_model_viability (
              lane_id, model_id, artifact_id, source_locality,
              fits_memory, projected_free_bytes, required_memory_bytes,
              tps_estimate, tps_source, is_viable, reason, last_checked_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (lane_id, model_id, source_locality) DO UPDATE
            SET artifact_id = EXCLUDED.artifact_id,
                fits_memory = EXCLUDED.fits_memory,
                projected_free_bytes = EXCLUDED.projected_free_bytes,
                required_memory_bytes = EXCLUDED.required_memory_bytes,
                tps_estimate = EXCLUDED.tps_estimate,
                tps_source = EXCLUDED.tps_source,
                is_viable = EXCLUDED.is_viable,
                reason = EXCLUDED.reason,
                last_checked_at = now()
            """,
            (
                resolved_lane_id,
                model_id,
                artifact_id,
                source_locality,
                fits_memory,
                projected_free_bytes,
                required_memory_bytes,
                viability.estimated_tps,
                "measured" if viability.estimated_tps is not None else "unknown",
                locality in {"local", "remote"},
                None if locality in {"local", "remote"} else viability.reason,
            ),
        )
        if locality is None:
            continue
        candidate = LaneModelCandidate(
            model_name=model_name,
            tags=_model_tags_with_inferred(model_name, row.get("tags") or []),
            locality=locality,
            artifact_path=str(row["local_path"]),
            artifact_host=str(row["host_name"]),
            artifact_provider=str(row["storage_provider"] or ""),
            estimated_tps=viability.estimated_tps,
            estimated_swap_ms=swap_estimate.estimated_ms,
            swap_strategy=swap_estimate.strategy,
            reason=viability.reason if not fits_memory else None,
            size_bytes=size_bytes,
            required_memory_bytes=required_memory_bytes,
            projected_free_bytes=projected_free_bytes,
            max_context_tokens=target_context_tokens,
        )
        previous = candidates_by_model.get(model_name)
        if previous is not None:
            current_is_local_artifact = str(row["host_id"]) == resolved_host_id
            previous_is_local_artifact = previous.artifact_host == str(host_row["host_name"]) if host_row else False
            current_score = (
                0 if current_is_local_artifact else 1,
                0 if locality == "local" else 1 if locality == "remote" else 2,
                candidate.estimated_swap_ms or 10**12,
            )
            previous_score = (
                0 if previous_is_local_artifact else 1,
                0 if previous.locality == "local" else 1 if previous.locality == "remote" else 2,
                previous.estimated_swap_ms or 10**12,
            )
            if previous_score <= current_score:
                continue
        candidates_by_model[model_name] = candidate

    supported_models = sorted(candidates_by_model.keys())

    current_model_name = str(lane_row.get("current_model_name") or "").strip()
    if mw_authoritative and current_model_name and current_model_name not in candidates_by_model:
        runtime_tags = _mw_runtime_candidate_tags(lane_row=lane_row)
        compatibility_reason = _backend_compatibility_reason(
            model_name=current_model_name,
            tags=runtime_tags,
            backend_type=lane_row.get("backend_type"),
            lane_type=lane_type,
        )
        if compatibility_reason is None:
            current_model_max_ctx = None
            cur.execute(
                """
                SELECT p.max_ctx
                FROM models m
                LEFT JOIN lane_model_policy p ON p.lane_id=%s AND p.model_id=m.model_id
                WHERE m.model_name=%s
                LIMIT 1
                """,
                (resolved_lane_id, current_model_name),
            )
            current_policy = cur.fetchone() or {}
            try:
                if current_policy.get("max_ctx") is not None:
                    current_model_max_ctx = int(current_policy["max_ctx"])
            except Exception:
                current_model_max_ctx = None
            candidates_by_model[current_model_name] = LaneModelCandidate(
                model_name=current_model_name,
                tags=runtime_tags,
                locality="local",
                artifact_path=None,
                artifact_host=str(host_row["host_name"]) if host_row else None,
                artifact_provider="mw_runtime",
                estimated_tps=None,
                estimated_swap_ms=0,
                swap_strategy="already_loaded",
                reason=None,
                size_bytes=None,
                required_memory_bytes=None,
                projected_free_bytes=None,
                max_context_tokens=current_model_max_ctx,
            )
            supported_models = sorted(candidates_by_model.keys())

    for candidate in sorted(candidates_by_model.values(), key=lambda item: item.model_name.lower()):
        if candidate.locality == "local":
            local_viable.append(candidate)
        elif candidate.locality == "remote":
            remote_viable.append(candidate)
        else:
            unverified.append(candidate)

    normalized_backend = _normalize_router_backend_type(lane_row.get("backend_type"))
    if normalized_backend == "sd":
        capabilities = ["images", "inference"]
    else:
        capabilities = ["chat"]
        if lane_type in ("cpu", "gpu", "mlx"):
            capabilities.append("inference")

    metadata = {
        "lane_type": lane_type,
        "backend_type": normalized_backend or lane_row.get("backend_type"),
        "lane_name": str(lane_row["lane_name"]),
        "host_name": str(host_row["host_name"]) if host_row else None,
        "mw_authoritative": mw_authoritative,
        "memory_summary": {
            "usable_memory_bytes": lane_row.get("usable_memory_bytes"),
            "ram_budget_bytes": lane_row.get("ram_budget_bytes"),
            "vram_budget_bytes": lane_row.get("vram_budget_bytes"),
            "runtime_overhead_bytes": runtime_overhead,
            "reserved_headroom_bytes": reserved_headroom,
        },
        "archive_providers": sorted(ARCHIVE_PROVIDERS),
        "local_model_root": local_model_root,
    }

    response = LaneCapabilityResponse(
        lane_id=resolved_lane_id,
        capabilities=capabilities,
        supported_models=supported_models,
        current_model=str(lane_row.get("current_model_name") or "") or None,
        local_viable_models=local_viable,
        remote_viable_models=remote_viable,
        unverified_models=unverified,
        metadata=_strip_nones(metadata),
    )
    return {
        "lane": lane_row,
        "host": host_row,
        "lane_info": lane_info,
        "local_model_root": local_model_root,
    }, response

def _strip_nones(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if v is None:
                continue
            vv = _strip_nones(v)
            # Drop empty dicts/lists introduced by stripping.
            if vv == {} or vv == []:
                continue
            out[k] = vv
        return out
    if isinstance(value, list):
        return [_strip_nones(v) for v in value if v is not None]
    return value


def _parse_image_size(size: str | None) -> tuple[int, int]:
    raw = str(size or "").strip().lower()
    if not raw:
        return IMAGE_DEFAULT_WIDTH, IMAGE_DEFAULT_HEIGHT
    match = re.fullmatch(r"(\d{1,5})x(\d{1,5})", raw)
    if not match:
        raise HTTPException(status_code=400, detail="size must be WIDTHxHEIGHT")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="size must be positive")
    return width, height


def _normalize_image_request(raw_payload: dict[str, Any]) -> dict[str, Any]:
    try:
        req = ImageGenerationRequest.model_validate(raw_payload or {})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    width, height = _parse_image_size(req.size)
    return {
        "route": "images",
        "requested_model_name": req.model,
        "pin_worker": req.mesh_pin_worker,
        "pin_base_url": req.mesh_pin_base_url,
        "pin_lane_type": req.mesh_pin_lane_type,
        "pin_lane_id": req.mesh_pin_lane_id,
        "response_format": req.response_format or "b64_json",
        "request_payload": {
            "prompt": req.prompt,
            "width": width,
            "height": height,
            "batch_count": max(1, min(int(req.n or 1), 16)),
            "seed": -1,
            "cfg_scale": 1.0,
            "sample_steps": 4,
        },
    }


def _translate_sd_response_to_openai(
    *,
    response_payload: dict[str, Any],
    response_format: str,
) -> dict[str, Any]:
    images = response_payload.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError("sd.cpp response missing images")

    data: list[dict[str, str]] = []
    for item in images:
        if isinstance(item, dict):
            image_b64 = str(item.get("b64_json") or item.get("image") or item.get("data") or "").strip()
            image_url = str(item.get("url") or "").strip()
        else:
            image_b64 = str(item or "").strip()
            image_url = ""

        if response_format == "url":
            if image_url:
                data.append({"url": image_url})
            elif image_b64:
                data.append({"url": f"data:image/png;base64,{image_b64}"})
            else:
                raise RuntimeError("sd.cpp response image item missing url/image data")
            continue

        if image_b64:
            data.append({"b64_json": image_b64})
        elif image_url.startswith("data:image/") and "," in image_url:
            data.append({"b64_json": image_url.split(",", 1)[1]})
        else:
            raise RuntimeError("sd.cpp response image item missing base64 data")

    return {"created": int(time.time()), "data": data}


def _downstream_payload(req: ChatCompletionRequest) -> dict[str, Any]:
    raw = req.model_dump(by_alias=True)
    extra_body = raw.pop("extra_body", None) or {}
    # Remove router-only hint fields.
    for k in list(raw.keys()):
        if k.startswith("mesh_"):
            raw.pop(k, None)
    if isinstance(extra_body, dict):
        for key, value in extra_body.items():
            raw[key] = value
    return _strip_nones(raw)


def _resolve_lane_downstream_alias(cur, *, lane_id: str, model_id: str) -> str | None:
    cur.execute(
        """
        SELECT downstream_model_name
        FROM lane_model_aliases
        WHERE lane_id=%s AND model_id=%s
        """,
        (lane_id, model_id),
    )
    alias_row = cur.fetchone()
    if alias_row and alias_row.get("downstream_model_name"):
        return str(alias_row["downstream_model_name"])

    lane_row = _resolve_lane(cur, lane_id)
    host_row = _resolve_host(cur, str(lane_row["host_id"])) if lane_row else None
    local_model_root = _local_model_root(host_row, lane_row)
    host_id = str((lane_row or {}).get("host_id") or "").strip()
    if not host_id:
        return None
    cur.execute(
        """
        SELECT local_path
        FROM host_model_artifacts
        WHERE host_id=%s AND model_id=%s AND present=true
        ORDER BY char_length(local_path) ASC
        """,
        (host_id, model_id),
    )
    for row in cur.fetchall() or []:
        local_path = str((row or {}).get("local_path") or "").strip()
        if not local_path:
            continue
        if local_model_root and not _path_matches_local_model_root(
            artifact_path=local_path,
            local_model_root=local_model_root,
        ):
            continue
        return local_path
    return None


def _resolve_downstream_model_for_lane(
    cur,
    *,
    lane_id: str,
    requested_model_name: str,
    model_id: str | None,
) -> str:
    cur.execute(
        """
        SELECT l.current_model_name, m.tags AS current_model_tags
        FROM lanes l
        LEFT JOIN models m ON m.model_name = l.current_model_name
        WHERE l.lane_id=%s
        """,
        (lane_id,),
    )
    lane_row = cur.fetchone()
    current_model_name = str((lane_row or {}).get("current_model_name") or "").strip()
    current_model_tags = _model_tags_with_inferred(current_model_name, (lane_row or {}).get("current_model_tags") or [])
    if current_model_name and _model_request_matches_candidate(
        requested_model_name,
        current_model_name,
        current_model_tags,
    ):
        cur.execute("SELECT model_id FROM models WHERE model_name=%s", (current_model_name,))
        current_model_row = cur.fetchone()
        if current_model_row and current_model_row.get("model_id"):
            downstream_model = _resolve_lane_downstream_alias(
                cur,
                lane_id=lane_id,
                model_id=str(current_model_row["model_id"]),
            )
            if downstream_model:
                return downstream_model
        return current_model_name

    if model_id:
        downstream_model = _resolve_lane_downstream_alias(cur, lane_id=lane_id, model_id=model_id)
        if downstream_model:
            return downstream_model

    return requested_model_name


def _lane_gateway_healthy(
    base_url: str,
    *,
    host_id: str | None = None,
    lane_id: str | None = None,
) -> bool:
    if settings.mw_control_enabled and host_id and lane_id:
        try:
            result = _mw_client().send_command(
                host_id=host_id,
                message_type="health_probe",
                payload={"lane_id": lane_id},
                wait=True,
                timeout_seconds=10,
            )
            return bool(result.get("ok"))
        except Exception:
            pass
    health_urls = ("/health", "/healthz", "/readyz", "/livez")
    try:
        with httpx.Client(timeout=5.0) as client:
            for path in health_urls:
                try:
                    resp = client.get(f"{base_url.rstrip('/')}{path}")
                    if resp.status_code == 200:
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/health/liveliness")
def health_liveliness() -> dict[str, Any]:
    # Compatibility endpoint (LiteLLM-style health paths are used by some clients).
    return {"ok": True}


@app.get("/health/readiness")
def health_readiness() -> dict[str, Any]:
    return {"ok": True}


@app.post("/api/mw/commands", response_model=MWCommandResponse)
def api_mw_command(req: MWCommandRequest) -> MWCommandResponse:
    payload = dict(req.payload or {})
    if req.message_type == "load_model":
        # Operator safety / back-compat: accept model_id as an alias for model_name.
        if "model_name" not in payload and "model_id" in payload:
            payload["model_name"] = payload.get("model_id")
        if not str(payload.get("model_name") or "").strip():
            raise HTTPException(status_code=400, detail="load_model requires payload.model_name (or model_id alias)")
    try:
        result = _mw_client().send_command(
            host_id=req.host_id,
            message_type=req.message_type,
            payload=payload,
            request_id=req.request_id,
            wait=req.wait,
            timeout_seconds=req.timeout_seconds,
        )
    except MWControlError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    resp = MWCommandResponse(
        ok=bool(result.get("ok", False)),
        pending=bool(result.get("pending", False)),
        host_id=req.host_id,
        request_id=str(result.get("request_id") or req.request_id or ""),
        message_type=req.message_type,
        result=dict(result.get("result") or {}),
        error=result.get("error"),
        warning=str(result.get("warning")) if result.get("warning") else None,
        timeout_seconds=int(result.get("timeout_seconds")) if result.get("timeout_seconds") is not None else None,
        response=dict(result.get("response") or {}) if result.get("response") else None,
    )
    if resp.pending:
        # Avoid false-negative operational behavior: the command was delivered but not yet observed as terminal.
        return JSONResponse(
            status_code=202,
            content=resp.model_dump(),
            headers={
                # Operator-friendly: point clients at the status endpoint for follow-up polling.
                "Location": f"/api/mw/commands/{resp.request_id}",
                "Retry-After": "2",
            },
        )
    return resp


@app.get("/api/mw/commands/{request_id}", response_model=MWCommandStatusResponse)
def api_mw_command_status(request_id: str) -> MWCommandStatusResponse:
    rid = str(request_id).strip()
    if not rid:
        raise HTTPException(status_code=400, detail="missing request_id")
    try:
        # MW state can live in a separate DB (`MESH_ROUTER_MW_STATE_DATABASE_URL`).
        # Always query the MW state DB handle here so that "202 pending + poll" is operationally safe.
        with mw_state_db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      request_id,
                      host_id,
                      transition_type,
                      status,
                      current_phase,
                      error_kind,
                      error_message,
                      started_at,
                      completed_at,
                      updated_at
                    FROM mw_transitions
                    WHERE request_id=%s::uuid
                    LIMIT 1
                    """,
                    (rid,),
                )
                row = cur.fetchone()
    except Exception as exc:
        # MW tables may not exist in every environment; keep response stable.
        raise HTTPException(status_code=503, detail=f"mw_transitions unavailable: {exc}") from exc

    if not row:
        return MWCommandStatusResponse(found=False, request_id=rid)

    status = str(row.get("status") or "")
    ok: bool | None = None
    if status in {"completed"}:
        ok = True
    if status in {"failed", "rejected", "cancelled"}:
        ok = False

    return MWCommandStatusResponse(
        found=True,
        request_id=str(row["request_id"]),
        host_id=str(row.get("host_id") or ""),
        status=status or None,
        transition_type=str(row.get("transition_type") or "") or None,
        current_phase=str(row.get("current_phase") or "") or None,
        ok=ok,
        error_kind=str(row.get("error_kind") or "") or None,
        error_message=str(row.get("error_message") or "") or None,
        started_at=row["started_at"].isoformat() if row.get("started_at") else None,
        completed_at=row["completed_at"].isoformat() if row.get("completed_at") else None,
        updated_at=row["updated_at"].isoformat() if row.get("updated_at") else None,
    )


@app.post("/api/mw/hosts/{host_id}/health-probe", response_model=MWCommandResponse)
def api_mw_health_probe(host_id: str, service_id: str | None = None, lane_id: str | None = None) -> MWCommandResponse:
    payload: dict[str, Any] = {}
    if service_id:
        payload["service_id"] = service_id
    if lane_id:
        payload["lane_id"] = lane_id
    return api_mw_command(
        MWCommandRequest(
            host_id=host_id,
            message_type="health_probe",
            payload=payload,
            wait=True,
        )
    )


@app.get("/api/lanes")
def api_lanes() -> dict[str, Any]:
    # Note: MW state (mw_*) may live in a separate DB (`MESH_ROUTER_MW_STATE_DATABASE_URL`).
    # We intentionally compute MW lane readiness/current-model in Python to avoid cross-DB joins.
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  l.lane_id,
                  h.host_id,
                  h.host_name,
                  l.lane_name,
                  l.lane_type,
                  l.backend_type,
                  l.base_url,
                  l.status,
                  l.current_model_name,
                  l.ram_budget_bytes,
                  l.vram_budget_bytes,
                  l.proxy_auth_mode,
                  l.proxy_auth_metadata,
                  l.suspension_reason,
                  l.last_probe_at,
                  l.last_ok_at,
                  l.created_at,
                  l.updated_at
                FROM lanes l
                JOIN hosts h ON h.host_id = l.host_id
                ORDER BY h.host_name, l.lane_name, l.base_url
                """
            )
            rows = cur.fetchall()

    items = [
        {
            "lane_id": str(row["lane_id"]),
            "host_id": str(row["host_id"]),
            "host_name": str(row["host_name"]),
            "lane_name": str(row["lane_name"]),
            "lane_type": str(row["lane_type"]),
            "backend_type": str(row.get("backend_type") or "llama"),
            "base_url": str(row["base_url"]),
            "status": str(row["status"]),
            "current_model_name": str(row["current_model_name"]) if row.get("current_model_name") else None,
            "ram_budget_bytes": int(row["ram_budget_bytes"]) if row.get("ram_budget_bytes") is not None else None,
            "vram_budget_bytes": int(row["vram_budget_bytes"]) if row.get("vram_budget_bytes") is not None else None,
            "proxy_auth_mode": str(row["proxy_auth_mode"]) if row.get("proxy_auth_mode") else None,
            "proxy_auth_metadata": dict(row.get("proxy_auth_metadata") or {}),
            "suspension_reason": str(row["suspension_reason"]) if row.get("suspension_reason") else None,
            "current_backend_type": row.get("current_backend_type"),
            "desired_model_name": row.get("desired_model_name"),
            "desired_backend_type": row.get("desired_backend_type"),
            "backend_swap_eta_ms": row.get("backend_swap_eta_ms"),
            "model_swap_eta_ms": row.get("model_swap_eta_ms"),
            "total_swap_eta_ms": row.get("total_swap_eta_ms"),
            "eta_source": row.get("eta_source"),
            "eta_complete": row.get("eta_complete"),
            "last_probe_at": row["last_probe_at"].isoformat() if row.get("last_probe_at") else None,
            "last_ok_at": row["last_ok_at"].isoformat() if row.get("last_ok_at") else None,
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }
        for row in rows
    ]

    apply_mw_effective_status(
        items,
        mw_state_db=mw_state_db,
        stale_seconds=settings.default_lease_stale_seconds,
    )
    for item in items:
        if item.get("effective_status"):
            item["status"] = str(item["effective_status"])
        if "readiness_reason" not in item:
            item["readiness_reason"] = None

    return {"items": items}


@app.get("/api/inventory", response_model=InventoryResponse)
def api_inventory() -> InventoryResponse:
    """
    Capability/inventory plane (host → lanes → supported models), with MW-derived effective status/model overlay
    for MW-managed lanes.
    """
    with db.connect() as conn:
        with conn.cursor() as cur:
            rows = fetch_lane_inventory(cur=cur)

    hosts = group_inventory_by_host(rows)
    items: list[dict[str, Any]] = []
    for host in hosts:
        lanes_out: list[dict[str, Any]] = []
        for lane in host.get("lanes") or []:
            lane_id = str(lane.get("lane_id") or "")
            with db.connect() as conn:
                with conn.cursor() as cur:
                    _, capability_payload = _build_lane_capability_payload(cur, lane_id)
            local = [candidate.model_dump() for candidate in capability_payload.local_viable_models]
            remote = [candidate.model_dump() for candidate in capability_payload.remote_viable_models]
            lane_out = {
                "lane_id": lane_id,
                "lane_name": str(lane.get("lane_name") or ""),
                "host_id": str(lane.get("host_id") or ""),
                "host_name": str(lane.get("host_name") or ""),
                "lane_type": str(lane.get("lane_type") or "") or None,
                "backend_type": str(lane.get("backend_type") or "") or None,
                "base_url": str(lane.get("base_url") or "") or None,
                "status": str(lane.get("effective_status") or lane.get("status") or ""),
                "effective_status": str(lane.get("effective_status") or "") or None,
                "readiness_reason": str(lane.get("readiness_reason") or "") or None,
                "current_model_name": lane.get("current_model_name"),
                "current_backend_type": lane.get("current_backend_type"),
                "desired_model_name": lane.get("desired_model_name"),
                "desired_backend_type": lane.get("desired_backend_type"),
                "backend_swap_eta_ms": lane.get("backend_swap_eta_ms"),
                "model_swap_eta_ms": lane.get("model_swap_eta_ms"),
                "total_swap_eta_ms": lane.get("total_swap_eta_ms"),
                "eta_source": lane.get("eta_source"),
                "eta_complete": lane.get("eta_complete"),
                "proxy_auth_metadata": dict(lane.get("proxy_auth_metadata") or {}),
                "capabilities": list(capability_payload.capabilities or []),
                "supported_models": list(capability_payload.supported_models or []),
                "local_viable_models": local,
                "remote_viable_models": remote,
                "unverified_models": [candidate.model_dump() for candidate in capability_payload.unverified_models],
                "capability_metadata": dict(capability_payload.metadata or {}),
            }
            lanes_out.append(lane_out)
        items.append(
            {
                "host_id": str(host.get("host_id") or ""),
                "host_name": str(host.get("host_name") or ""),
                "lanes": lanes_out,
                "tags": list(host.get("tags") or []),
                "policy": dict(host.get("policy") or {}),
            }
        )
    return InventoryResponse(items=items)


@app.get("/api/perf/expectations", response_model=PerfExpectationResponse)
def api_perf_expectations(
    host_id: str,
    lane_id: str,
    model_name: str,
    modality: Literal["chat", "embeddings", "images"] = "chat",
) -> PerfExpectationResponse:
    try:
        with mw_state_db.connect() as conn:
            with conn.cursor() as cur:
                exp = get_expectation(cur=cur, host_id=host_id, lane_id=lane_id, model_name=model_name, modality=modality)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"mw_state_db unavailable: {exc}") from exc
    if not exp:
        return PerfExpectationResponse(items=[])
    now = datetime.now(tz=UTC)
    staleness_s = (now - exp.updated_at).total_seconds()
    return PerfExpectationResponse(
        items=[
            {
                "host_id": exp.host_id,
                "lane_id": exp.lane_id,
                "model_name": exp.model_name,
                "modality": exp.modality,
                "updated_at": exp.updated_at.isoformat(),
                "sample_count": exp.sample_count,
                "first_token_ms_p50": exp.first_token_ms_p50,
                "decode_tps_p50": exp.decode_tps_p50,
                "total_ms_p50": exp.total_ms_p50,
                "staleness_s": staleness_s,
                "source": "observations",
            }
        ]
    )


@app.post("/api/perf/observations")
def api_perf_observations(
    req: PerfObservationIngestRequest,
    x_mesh_internal_token: str | None = Header(default=None, alias="x-mesh-internal-token"),
) -> dict[str, Any]:
    token = settings.internal_ingest_token
    if token and (x_mesh_internal_token or "") != token:
        raise HTTPException(status_code=403, detail="missing or invalid x-mesh-internal-token")
    try:
        with mw_state_db.connect() as conn:
            with conn.cursor() as cur:
                insert_observation(cur=cur, obs=req.model_dump())
            conn.commit()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"failed to ingest observation: {exc}") from exc
    return {"ok": True}


@app.post("/api/routes/resolve", response_model=RouteResolveResponse)
def api_routes_resolve(req: RouteResolveRequest) -> RouteResolveResponse:
    choice, perf, reason, candidates_considered = resolve_route(
        model=req.model,
        modality=req.modality,
        tags=req.tags,
        host_name=req.host_name,
        lane_id=req.lane_id,
        allow_opportunistic=req.allow_opportunistic,
    )
    if not choice:
        return RouteResolveResponse(ok=False, reason=reason, candidates_considered=candidates_considered)
    return RouteResolveResponse(
        ok=True,
        choice=choice,
        perf=perf,
        candidates_considered=candidates_considered,
    )


class LaneUpsertRequest(BaseModel):
    # NOTE: This model must be defined before it is referenced by `api_lane_upsert`.
    # With `from __future__ import annotations`, FastAPI may cache a ForwardRef at
    # route registration time; leaving this definition below the endpoint can
    # break OpenAPI generation (/openapi.json) under Pydantic v2.
    host_ref: str
    lane_name: str
    lane_type: Literal["cpu", "gpu", "mlx", "router", "other"]
    backend_type: Literal["llama", "sd"] = "llama"
    base_url: str
    status: Literal["ready", "busy", "suspended", "offline", "error"] = "offline"
    ram_budget_bytes: int | None = None
    vram_budget_bytes: int | None = None
    proxy_auth_mode: str | None = None
    proxy_auth_metadata: dict[str, Any] | None = None


@app.post("/api/lanes")
def api_lane_upsert(req: LaneUpsertRequest) -> dict[str, Any]:
    with db.connect() as conn:
        with conn.cursor() as cur:
            host_id, host_name = _resolve_host_id(cur, req.host_ref, create=True)
            cur.execute(
                """
                INSERT INTO lanes (
                  host_id,
                  lane_name,
                  lane_type,
                  backend_type,
                  base_url,
                  status,
                  ram_budget_bytes,
                  vram_budget_bytes,
                  proxy_auth_mode,
                  proxy_auth_metadata,
                  updated_at
                )
                VALUES (
                  %s,
                  %s,
                  %s::lane_type,
                  %s,
                  %s,
                  %s::lane_status,
                  %s,
                  %s,
                  %s,
                  %s::jsonb,
                  now()
                )
                ON CONFLICT (base_url)
                DO UPDATE SET
                  host_id = EXCLUDED.host_id,
                  lane_name = EXCLUDED.lane_name,
                  lane_type = EXCLUDED.lane_type,
                  backend_type = EXCLUDED.backend_type,
                  status = EXCLUDED.status,
                  ram_budget_bytes = EXCLUDED.ram_budget_bytes,
                  vram_budget_bytes = EXCLUDED.vram_budget_bytes,
                  proxy_auth_mode = EXCLUDED.proxy_auth_mode,
                  proxy_auth_metadata = EXCLUDED.proxy_auth_metadata,
                  updated_at = now()
                RETURNING lane_id, lane_name, lane_type, backend_type, base_url, status, ram_budget_bytes, vram_budget_bytes
                """,
                (
                    host_id,
                    req.lane_name,
                    req.lane_type,
                    req.backend_type,
                    req.base_url,
                    req.status,
                    req.ram_budget_bytes,
                    req.vram_budget_bytes,
                    req.proxy_auth_mode,
                    Jsonb(req.proxy_auth_metadata or {}),
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return {
        "ok": True,
        "lane_id": str(row["lane_id"]),
        "host_id": host_id,
        "host_name": host_name,
        "lane_name": str(row["lane_name"]),
        "lane_type": str(row["lane_type"]),
        "backend_type": str(row.get("backend_type") or "llama"),
        "base_url": str(row["base_url"]),
        "status": str(row["status"]),
        "ram_budget_bytes": int(row["ram_budget_bytes"]) if row.get("ram_budget_bytes") is not None else None,
        "vram_budget_bytes": int(row["vram_budget_bytes"]) if row.get("vram_budget_bytes") is not None else None,
    }


@app.get("/v1/models")
def v1_models() -> dict[str, Any]:
    def _is_canonical(model: str) -> bool:
        m = (model or "").strip()
        if not m:
            return False
        # Avoid downstream aliases like filesystem paths or URLs.
        if "/" in m or "\\" in m:
            return False
        if "://" in m:
            return False
        if len(m) > 128:
            return False
        return True

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT model_name, tags FROM models ORDER BY model_name")
            rows = cur.fetchall()
    data = [
        ModelInfo(
            id=str(r["model_name"]),
            tags=_normalized_model_tags(r.get("tags") or []),
        )
        for r in rows
        if _is_canonical(str(r["model_name"]))
    ]
    resp = ModelsResponse(data=data)
    return resp.model_dump()


@app.post("/api/models/{model_ref}/tags", response_model=ModelTagsResponse)
def api_model_tags(model_ref: str, body: ModelTagsUpdateRequest) -> ModelTagsResponse:
    tags = _normalized_model_tags(body.tags)
    with db.connect() as conn:
        with conn.cursor() as cur:
            model_id, model_name, existing_tags = _resolve_model_ref(cur, model_ref)
            existing = list(existing_tags)
            if body.mode == "replace":
                updated_tags = tags
            elif body.mode == "add":
                updated_tags = _normalized_model_tags(existing + tags)
            else:
                remove = set(tags)
                updated_tags = [tag for tag in existing if tag not in remove]
            cur.execute(
                """
                UPDATE models
                SET tags=%s::text[],
                    updated_at=now()
                WHERE model_id=%s
                RETURNING model_id, model_name, tags
                """,
                (updated_tags, model_id),
            )
            row = cur.fetchone()
        conn.commit()
    return ModelTagsResponse(
        model_id=str(row["model_id"]),
        model_name=str(row["model_name"]),
        tags=_normalized_model_tags(row.get("tags") or []),
    )


@app.post("/api/hosts/{host_ref}/restore-split-mode", response_model=RestoreSplitModeResponse)
def api_restore_split_mode(host_ref: str, body: RestoreSplitModeRequest) -> RestoreSplitModeResponse:
    with db.connect() as conn:
        with conn.cursor() as cur:
            host_id, host_name = _resolve_host_id(cur, host_ref, create=False)
        conn.commit()
    overrides = {
        lane_type: model_name.strip()
        for lane_type, model_name in {
            "cpu": body.cpu_model_name,
            "gpu": body.gpu_model_name,
            "mlx": body.mlx_model_name,
        }.items()
        if model_name and model_name.strip()
    }
    actions = _restore_split_mode_for_host(
        host_id=host_id,
        host_name=host_name,
        overrides=overrides,
    )
    return RestoreSplitModeResponse(
        host_id=host_id,
        host_name=host_name,
        actions=actions,
    )


def _bearer_token(req: Request) -> str:
    auth = (req.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    return ""


def _cleanup_expired_router_requests(cur) -> None:
    cur.execute(
        """
        UPDATE router_requests
        SET state='expired',
            error_kind=COALESCE(error_kind, 'stale'),
            error_message=COALESCE(error_message, 'request heartbeat expired'),
            released_at=COALESCE(released_at, now()),
            updated_at=now()
        WHERE state IN ('queued', 'acquired', 'running')
          AND (
            (expires_at IS NOT NULL AND expires_at <= now())
            OR COALESCE(last_heartbeat_at, started_at, acquired_at, queued_at) < now() - (%s * interval '1 second')
          )
        """,
        (settings.default_lease_stale_seconds,),
    )


def _create_router_request(
    *,
    route: str,
    request_payload: dict[str, Any],
    owner: str,
    job_type: str,
    requested_model_name: str | None,
    app_name: str | None = None,
    client_request_id: str | None = None,
    pin_worker: str | None = None,
    pin_base_url: str | None = None,
    pin_lane_type: str | None = None,
) -> str:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO router_requests (
                  route,
                  state,
                  owner,
                  job_type,
                  app_name,
                  client_request_id,
                  requested_model_name,
                  request_payload,
                  pin_worker,
                  pin_base_url,
                  pin_lane_type
                )
                VALUES (%s, 'queued', %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                RETURNING request_id
                """,
                (
                    route,
                    owner,
                    job_type,
                    app_name,
                    client_request_id,
                    requested_model_name,
                    Jsonb(request_payload),
                    pin_worker,
                    pin_base_url,
                    pin_lane_type,
                ),
            )
            request_id = str(cur.fetchone()["request_id"])
        conn.commit()
    return request_id


def _touch_router_request(*, request_id: str, state: str | None = None, **fields: Any) -> None:
    allowed_fields = {
        "lease_id",
        "lane_id",
        "model_id",
        "worker_id",
        "base_url",
        "requested_model_name",
        "downstream_model_name",
        "result_payload",
        "error_kind",
        "error_message",
        "cancel_requested",
        "cancel_requested_at",
        "cancel_reason",
        "expires_at",
        "released_at",
        "last_heartbeat_at",
        "app_name",
        "client_request_id",
    }
    set_parts: list[str] = []
    params: list[Any] = []
    if state is not None:
        set_parts.append("state=%s::request_state")
        params.append(state)
        if state == "acquired":
            set_parts.append("acquired_at=COALESCE(acquired_at, now())")
        if state == "running":
            set_parts.append("started_at=COALESCE(started_at, now())")
        if state in REQUEST_TERMINAL_STATES:
            set_parts.append("released_at=COALESCE(released_at, now())")
    for key, value in fields.items():
        if key not in allowed_fields:
            continue
        if key == "result_payload":
            set_parts.append(f"{key}=%s::jsonb")
            params.append(Jsonb(value) if value is not None else None)
            continue
        if key == "released_at":
            # Finalizers may explicitly pass released_at while terminal states also
            # auto-populate it. Avoid duplicate column assignments.
            if any(part.split("=")[0] == "released_at" for part in set_parts):
                continue
        set_parts.append(f"{key}=%s")
        params.append(value)
    set_parts.append("updated_at=now()")
    with db.connect() as conn:
        with conn.cursor() as cur:
            _cleanup_expired_router_requests(cur)
            cur.execute(
                f"""
                UPDATE router_requests
                SET {", ".join(set_parts)}
                WHERE request_id=%s
                """,
                tuple(params + [request_id]),
            )
        conn.commit()


def _request_cancel_requested(request_id: str) -> bool:
    with db.connect() as conn:
        with conn.cursor() as cur:
            _cleanup_expired_router_requests(cur)
            cur.execute("SELECT cancel_requested FROM router_requests WHERE request_id=%s", (request_id,))
            row = cur.fetchone()
    return bool((row or {}).get("cancel_requested"))


def _fetch_router_request(request_id: str) -> dict[str, Any] | None:
    with db.connect() as conn:
        with conn.cursor() as cur:
            _cleanup_expired_router_leases(cur)
            _cleanup_expired_router_requests(cur)
            cur.execute(
                """
                SELECT
                  rr.request_id,
                  rr.route,
                  rr.state,
                  rr.owner,
                  rr.job_type,
                  rr.app_name,
                  rr.client_request_id,
                  rr.requested_model_name,
                  rr.downstream_model_name,
                  rr.model_id,
                  rr.lane_id,
                  rr.lease_id,
                  rr.worker_id,
                  rr.base_url,
                  rr.pin_worker,
                  rr.pin_base_url,
                  rr.pin_lane_type,
                  rr.cancel_requested,
                  rr.cancel_requested_at,
                  rr.cancel_reason,
                  rr.request_payload,
                  rr.result_payload,
                  rr.error_kind,
                  rr.error_message,
                  rr.queued_at,
                  rr.acquired_at,
                  rr.started_at,
                  rr.last_heartbeat_at,
                  rr.expires_at,
                  rr.released_at,
                  rr.created_at,
                  rr.updated_at,
                  m.model_name,
                  m.context_default,
                  cmp.max_ctx AS lane_max_ctx,
                  l.lane_name,
                  l.lane_type,
                  l.status AS lane_status,
                  h.host_name,
                  rl.state AS lease_state
                FROM router_requests rr
                LEFT JOIN models m ON m.model_id = rr.model_id
                LEFT JOIN lane_model_policy cmp ON cmp.lane_id = rr.lane_id AND cmp.model_id = rr.model_id
                LEFT JOIN lanes l ON l.lane_id = rr.lane_id
                LEFT JOIN hosts h ON h.host_id = l.host_id
                LEFT JOIN router_leases rl ON rl.lease_id = rr.lease_id
                WHERE rr.request_id=%s
                """,
                (request_id,),
            )
            row = cur.fetchone()
    return row


def _serialize_router_request(row: dict[str, Any]) -> dict[str, Any]:
    state = str(row.get("state") or "").lower()
    last_progress_at = (
        row.get("last_heartbeat_at")
        or row.get("started_at")
        or row.get("acquired_at")
        or row.get("queued_at")
    )
    return {
        "request_id": str(row["request_id"]),
        "route": str(row.get("route") or ""),
        "state": state,
        "terminal": state in REQUEST_TERMINAL_STATES,
        "owner": str(row.get("owner") or ""),
        "job_type": str(row.get("job_type") or ""),
        "app_name": row.get("app_name"),
        "client_request_id": row.get("client_request_id"),
        "lane_id": str(row["lane_id"]) if row.get("lane_id") else None,
        "lane_name": str(row.get("lane_name") or "") or None,
        "lane_type": str(row.get("lane_type") or "") or None,
        "lane_status": str(row.get("lane_status") or "") or None,
        "host": str(row.get("host_name") or "") or None,
        "lease_id": str(row["lease_id"]) if row.get("lease_id") else None,
        "lease_state": str(row.get("lease_state") or "") or None,
        "worker_id": row.get("worker_id"),
        "base_url": row.get("base_url"),
        "requested_model_name": row.get("requested_model_name"),
        "downstream_model": row.get("downstream_model_name"),
        "model_name": row.get("model_name") or row.get("requested_model_name"),
        "cancel_requested": bool(row.get("cancel_requested")),
        "cancel_requested_at": row["cancel_requested_at"].isoformat() if row.get("cancel_requested_at") else None,
        "cancel_reason": row.get("cancel_reason"),
        "error_kind": row.get("error_kind"),
        "error_message": row.get("error_message"),
        "queued_at": row["queued_at"].isoformat() if row.get("queued_at") else None,
        "acquired_at": row["acquired_at"].isoformat() if row.get("acquired_at") else None,
        "started_at": row["started_at"].isoformat() if row.get("started_at") else None,
        "last_heartbeat_at": row["last_heartbeat_at"].isoformat() if row.get("last_heartbeat_at") else None,
        "expires_at": row["expires_at"].isoformat() if row.get("expires_at") else None,
        "released_at": row["released_at"].isoformat() if row.get("released_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        "last_progress_at": last_progress_at.isoformat() if last_progress_at else None,
        "result_available": row.get("result_payload") is not None,
        "result_payload": row.get("result_payload"),
    }


def _router_request_health(row: dict[str, Any]) -> dict[str, Any]:
    state = str(row.get("state") or "").lower()
    now = datetime.now(UTC)
    heartbeat_at = row.get("last_heartbeat_at") or row.get("started_at") or row.get("acquired_at") or row.get("queued_at")
    age_s = None
    if heartbeat_at is not None:
        age_s = max(0.0, (now - heartbeat_at).total_seconds())
    heartbeat_fresh = age_s is None or age_s <= float(settings.default_lease_stale_seconds)
    return {
        "request_id": str(row["request_id"]),
        "state": state,
        "terminal": state in REQUEST_TERMINAL_STATES,
        "healthy": (state == "released") or (state not in REQUEST_TERMINAL_STATES and heartbeat_fresh and not bool(row.get("cancel_requested"))),
        "lease_active": str(row.get("lease_state") or "").lower() == "active",
        "heartbeat_fresh": heartbeat_fresh,
        "heartbeat_age_seconds": age_s,
        "stale_after_seconds": int(settings.default_lease_stale_seconds),
        "cancel_requested": bool(row.get("cancel_requested")),
        "last_progress_at": heartbeat_at.isoformat() if heartbeat_at else None,
        "expires_at": row["expires_at"].isoformat() if row.get("expires_at") else None,
    }


def _serialize_lane_swap(row: dict[str, Any]) -> LaneSwapResponse:
    return LaneSwapResponse(
        swap_id=str(row["swap_id"]),
        lane_id=str(row["lane_id"]),
        host_name=str(row.get("host_name") or ""),
        requested_model_name=str(row.get("requested_model_name") or ""),
        resolved_model_name=str(row.get("resolved_model_name") or "") or None,
        state=str(row.get("state") or ""),
        terminal=bool(row.get("terminal")),
        source_mode=str(row.get("source_mode") or "") or None,
        error_message=str(row.get("error_message") or "") or None,
        details=dict(row.get("details") or {}),
        started_at=row["started_at"].isoformat(),
        last_event_at=row["last_event_at"].isoformat() if row.get("last_event_at") else None,
        completed_at=row["completed_at"].isoformat() if row.get("completed_at") else None,
        updated_at=row["updated_at"].isoformat(),
    )


def _fetch_lane_swap(cur, swap_id: str) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT
          ls.swap_id,
          ls.lane_id,
          ls.requested_model_name,
          ls.resolved_model_name,
          ls.source_mode,
          ls.state,
          ls.terminal,
          ls.error_message,
          ls.details,
          ls.started_at,
          ls.last_event_at,
          ls.completed_at,
          ls.updated_at,
          h.host_name
        FROM lane_swaps ls
        JOIN lanes l ON l.lane_id = ls.lane_id
        JOIN hosts h ON h.host_id = l.host_id
        WHERE ls.swap_id=%s
        """,
        (swap_id,),
    )
    return cur.fetchone()


def _create_lane_swap(
    cur,
    *,
    lane_id: str,
    requested_model_name: str,
    resolved_model_name: str | None,
    source_mode: str | None,
    details: dict[str, Any] | None = None,
) -> str:
    cur.execute(
        """
        INSERT INTO lane_swaps (
          lane_id, requested_model_name, resolved_model_name, source_mode,
          state, terminal, details, started_at, last_event_at, updated_at
        )
        VALUES (%s, %s, %s, %s, 'queued', false, %s::jsonb, now(), now(), now())
        RETURNING swap_id
        """,
        (
            lane_id,
            requested_model_name,
            resolved_model_name,
            source_mode,
            Jsonb(details or {}),
        ),
    )
    swap_id = str(cur.fetchone()["swap_id"])
    cur.execute(
        """
        INSERT INTO lane_swap_events (
          swap_id, lane_id, event_type, state, message, details
        )
        VALUES (%s, %s, 'router_enqueued', 'queued', %s, %s::jsonb)
        """,
        (
            swap_id,
            lane_id,
            f"swap requested for {resolved_model_name or requested_model_name}",
            Jsonb(details or {}),
        ),
    )
    cur.execute(
        """
        UPDATE lanes
        SET status='suspended',
            suspension_reason=%s,
            updated_at=now()
        WHERE lane_id=%s
        """,
        (f"swap:{swap_id}:queued", lane_id),
    )
    return swap_id


def _record_lane_swap_event(
    cur,
    *,
    swap_id: str,
    event_type: str,
    state: str,
    message: str | None = None,
    details: dict[str, Any] | None = None,
    current_model_name: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    swap_row = _fetch_lane_swap(cur, swap_id)
    if not swap_row:
        raise HTTPException(status_code=404, detail="swap not found")

    lane_id = str(swap_row["lane_id"])
    payload = dict(details or {})
    terminal = state in SWAP_TERMINAL_STATES
    now = datetime.now(UTC)

    cur.execute(
        """
        INSERT INTO lane_swap_events (
          swap_id, lane_id, event_type, state, message, details, error_message
        )
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
        """,
        (swap_id, lane_id, event_type, state, message, Jsonb(payload), error_message),
    )
    cur.execute(
        """
        UPDATE lane_swaps
        SET state=%s,
            terminal=%s,
            details=COALESCE(details, '{}'::jsonb) || %s::jsonb,
            error_message=COALESCE(%s, error_message),
            last_event_at=%s,
            completed_at=CASE WHEN %s THEN %s ELSE completed_at END,
            updated_at=%s
        WHERE swap_id=%s
        """,
        (
            state,
            terminal,
            Jsonb(payload),
            error_message,
            now,
            terminal,
            now,
            now,
            swap_id,
        ),
    )

    if current_model_name:
        cur.execute(
            "UPDATE lanes SET current_model_name=%s, updated_at=now() WHERE lane_id=%s",
            (current_model_name, lane_id),
        )

    if state == "ready":
        cur.execute(
            """
            UPDATE lanes
            SET status='ready',
                suspension_reason=NULL,
                updated_at=now()
            WHERE lane_id=%s
            """,
            (lane_id,),
        )
    elif state in {"queued", "stopping_siblings", "copying", "restarting", "loading"}:
        cur.execute(
            """
            UPDATE lanes
            SET status='suspended',
                suspension_reason=%s,
                updated_at=now()
            WHERE lane_id=%s
            """,
            (f"swap:{swap_id}:{state}", lane_id),
        )
    elif state in {"failed", "canceled"}:
        cur.execute(
            """
            UPDATE lanes
            SET status='error',
                suspension_reason=%s,
                updated_at=now()
            WHERE lane_id=%s
            """,
            (f"swap:{swap_id}:{state}", lane_id),
        )

    row = _fetch_lane_swap(cur, swap_id)
    if not row:
        raise HTTPException(status_code=404, detail="swap disappeared")
    return row


def _load_swap_tuning_profile(cur, *, host_id: str, lane_id: str, model_name: str) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT
          tp.tuning_profile_id,
          tp.host_id,
          tp.model_id,
          tp.lane_id,
          tp.storage_scheme,
          tp.settings,
          tp.cost_tier,
          tp.disables_sibling_lanes,
          tp.exclusive_host_resources,
          tp.prompt_tps,
          tp.generation_tps,
          tp.avg_total_latency_s,
          tp.score,
          tp.evaluation_count,
          tp.source_run_tag,
          tp.notes
        FROM model_tuning_profiles tp
        JOIN models m ON m.model_id = tp.model_id
        WHERE tp.host_id = %s
          AND m.model_name = %s
          AND (tp.lane_id = %s OR tp.lane_id IS NULL)
        ORDER BY CASE WHEN tp.lane_id = %s THEN 0 ELSE 1 END, tp.updated_at DESC
        LIMIT 1
        """,
        (host_id, model_name, lane_id, lane_id),
    )
    return cur.fetchone()


def _list_host_sibling_lanes(cur, *, host_id: str, lane_id: str) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT lane_id, lane_name, lane_type, base_url, status, suspension_reason, current_model_name, default_model_name
        FROM lanes
        WHERE host_id=%s AND lane_id<>%s
        ORDER BY lane_name
        """,
        (host_id, lane_id),
    )
    return list(cur.fetchall())


def _list_host_lanes(cur, *, host_id: str) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT lane_id, lane_name, lane_type, base_url, status, suspension_reason, current_model_name, default_model_name
        FROM lanes
        WHERE host_id=%s
        ORDER BY lane_name
        """,
        (host_id,),
    )
    return list(cur.fetchall())


def _lane_split_slot(lane_row: dict[str, Any]) -> str | None:
    lane_name = str(lane_row.get("lane_name") or "").strip().lower()
    lane_type = str(lane_row.get("lane_type") or "").strip().lower()
    if lane_name in {"cpu", "gpu", "mlx"}:
        return lane_name
    if not lane_name and lane_type in {"cpu", "gpu", "mlx"}:
        return lane_type
    return None


def _desired_restore_model_name(lane_row: dict[str, Any], overrides: dict[str, str]) -> str | None:
    slot = _lane_split_slot(lane_row)
    if not slot:
        return None
    requested = (overrides.get(slot) or "").strip()
    if requested:
        return requested
    default_model_name = str(lane_row.get("default_model_name") or "").strip()
    if default_model_name:
        return default_model_name
    current_model_name = str(lane_row.get("current_model_name") or "").strip()
    return current_model_name or None


def _list_active_router_leases(cur, lane_ids: list[str]) -> list[dict[str, Any]]:
    if not lane_ids:
        return []
    _cleanup_expired_router_leases(cur)
    cur.execute(
        """
        SELECT
          rl.lease_id,
          rl.lane_id,
          rl.model_id,
          m.model_name,
          rl.owner,
          rl.job_type,
          rl.state,
          rl.acquired_at,
          rl.last_heartbeat_at,
          rl.expires_at,
          rl.details
        FROM router_leases rl
        JOIN models m ON m.model_id = rl.model_id
        WHERE rl.lane_id = ANY(%s::uuid[])
          AND rl.state = 'active'
          AND rl.expires_at > now()
        ORDER BY rl.acquired_at
        """,
        (lane_ids,),
    )
    return list(cur.fetchall())


def _summarize_active_leases(active_leases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for lease in active_leases:
        details = dict(lease.get("details") or {})
        summary.append(
            {
                "lease_id": str(lease["lease_id"]),
                "lane_id": str(lease["lane_id"]),
                "model_name": str(lease.get("model_name") or ""),
                "owner": str(lease.get("owner") or ""),
                "job_type": str(lease.get("job_type") or ""),
                "route": str(details.get("route") or "chat"),
                "acquired_at": lease["acquired_at"].isoformat() if lease.get("acquired_at") else None,
                "last_heartbeat_at": lease["last_heartbeat_at"].isoformat() if lease.get("last_heartbeat_at") else None,
                "expires_at": lease["expires_at"].isoformat() if lease.get("expires_at") else None,
            }
        )
    return summary


def _set_lane_suspension(cur, *, lane_id: str, suspended: bool, reason: str) -> None:
    if suspended:
        cur.execute(
            """
            UPDATE lanes
            SET status='offline', suspension_reason=%s, updated_at=now()
            WHERE lane_id=%s
            """,
            (reason, lane_id),
        )
        return
    cur.execute(
        """
        UPDATE lanes
        SET status=CASE
              WHEN status IN ('offline', 'suspended') AND COALESCE(suspension_reason, '')=%s THEN 'ready'
              ELSE status
            END,
            suspension_reason=CASE
              WHEN COALESCE(suspension_reason, '')=%s THEN NULL
              ELSE suspension_reason
            END,
            updated_at=now()
        WHERE lane_id=%s
        """,
        (reason, reason, lane_id),
    )


def _display_lane_status(
    *,
    raw_status: str,
    suspension_reason: str | None,
    active_swap: dict[str, Any] | None,
) -> str:
    if active_swap:
        return str(active_swap.get("state") or raw_status)
    if (suspension_reason or "").strip():
        return "suspended"
    if raw_status == "suspended":
        return "offline"
    return raw_status


def _mark_leases_canceled_for_swap(
    cur,
    *,
    leases: list[dict[str, Any]],
    reason: str,
) -> None:
    for lease in leases:
        details = dict(lease.get("details") or {})
        details["swap_canceled"] = True
        details["swap_cancel_reason"] = reason
        details["swap_canceled_at"] = datetime.now(UTC).isoformat()
        cur.execute(
            """
            UPDATE router_leases
            SET state='failed',
                released_at=now(),
                details=%s::jsonb
            WHERE lease_id=%s AND state='active'
            """,
            (Jsonb(details), lease["lease_id"]),
        )


def _record_displaced_request(
    cur,
    *,
    lease: dict[str, Any],
    replacement_lane_id: str | None,
    status: str,
    result_payload: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    details = dict(lease.get("details") or {})
    cur.execute(
        """
        INSERT INTO swap_displaced_requests (
          original_lease_id,
          original_lane_id,
          replacement_lane_id,
          model_id,
          route,
          request_payload,
          status,
          handoff_attempted_at,
          handoff_completed_at,
          result_payload,
          error_message,
          updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, now(), %s, %s::jsonb, %s, now())
        """,
        (
            lease["lease_id"],
            lease["lane_id"],
            replacement_lane_id,
            lease["model_id"],
            str(details.get("route") or "chat"),
            Jsonb(dict(details.get("request_payload") or {})),
            status,
            datetime.now(UTC) if status in {"rerouted", "reroute_failed", "no_candidate", "unsupported"} else None,
            Jsonb(result_payload or {}) if result_payload is not None else None,
            error_message,
        ),
    )


def _reroute_displaced_lease(
    *,
    lease: dict[str, Any],
    excluded_lane_ids: set[str],
) -> dict[str, Any]:
    details = dict(lease.get("details") or {})
    route = str(details.get("route") or "chat")
    request_payload = dict(details.get("request_payload") or {})
    model_name = str(lease.get("model_name") or "").strip()
    if route not in {"chat", "embeddings", "images"} or not request_payload or not model_name:
        return {"status": "unsupported", "reason": "lease has no replayable request payload"}

    try:
        choice = pick_lane_for_model(
            model=model_name,
            backend_type=("sd" if route == "images" else None),
            pin_lane_type=request_payload.get("mesh_pin_lane_type"),
            exclude_lane_ids=excluded_lane_ids,
        )
    except Exception as exc:
        return {"status": "no_candidate", "reason": str(exc)}

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT model_id FROM models WHERE model_name=%s", (model_name,))
            model_row = cur.fetchone()
            if not model_row:
                return {"status": "reroute_failed", "reason": f"model not registered: {model_name}"}
            model_id = str(model_row["model_id"])
            downstream_model = _resolve_downstream_model_for_lane(
                cur,
                lane_id=choice.lane_id,
                requested_model_name=model_name,
                model_id=model_id,
            )
        conn.commit()

    lease_id, expires_at = _acquire_router_lease(
        lane_id=choice.lane_id,
        model_id=model_id,
        owner=f"{settings.default_owner}:swap-reroute",
        job_type=f"{route}-reroute",
        ttl_seconds=settings.default_lease_ttl_seconds,
        details={
            "worker_id": choice.worker_id,
            "base_url": choice.base_url,
            "downstream_model": downstream_model,
            "route": route,
            "request_payload": request_payload,
            "rerouted_from_lease_id": str(lease["lease_id"]),
        },
    )
    token = sign_token(
        {
            "lease_id": lease_id,
            "lane_id": choice.lane_id,
            "worker_id": choice.worker_id,
            "base_url": choice.base_url,
            "model": downstream_model,
            "owner": f"{settings.default_owner}:swap-reroute",
            "exp": int(expires_at.timestamp()),
        }
    )
    payload = dict(request_payload)
    payload["model"] = downstream_model
    try:
        with httpx.Client(timeout=float(max(30, settings.default_lease_ttl_seconds))) as client:
            endpoint = "/v1/chat/completions" if route == "chat" else "/v1/embeddings" if route == "embeddings" else "/sdapi/v1/txt2img"
            response = client.post(
                f"{choice.base_url.rstrip('/')}{endpoint}",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text}
        if response.status_code >= 400:
            _release_router_lease(lease_id=lease_id, ok=False)
            return {
                "status": "reroute_failed",
                "replacement_lane_id": choice.lane_id,
                "reason": f"worker proxy http_{response.status_code}",
                "result_payload": body,
            }
        result_payload = body
        if route == "images" and isinstance(body, dict):
            result_payload = _translate_sd_response_to_openai(
                response_payload=body,
                response_format="b64_json",
            )
        _release_router_lease(lease_id=lease_id, ok=True)
        return {
            "status": "rerouted",
            "replacement_lane_id": choice.lane_id,
            "replacement_base_url": choice.base_url,
            "result_payload": result_payload,
        }
    except Exception as exc:
        _release_router_lease(lease_id=lease_id, ok=False)
        return {
            "status": "reroute_failed",
            "replacement_lane_id": choice.lane_id,
            "reason": str(exc),
        }


def _call_lane_service_action(
    *,
    base_url: str,
    action: Literal["start", "stop", "restart"],
    host_id: str | None = None,
    lane_id: str | None = None,
) -> dict[str, Any]:
    if settings.mw_control_enabled and host_id and lane_id:
        message_type = {
            "start": "start_service",
            "stop": "stop_service",
            "restart": "restart_service",
        }[action]
        try:
            result = _mw_client().send_command(
                host_id=host_id,
                message_type=message_type,
                payload={"lane_id": lane_id},
                wait=True,
                timeout_seconds=max(30, settings.mw_command_timeout_seconds),
            )
            if not result.get("ok"):
                raise HTTPException(status_code=502, detail=result.get("error") or result.get("response") or result)
            return dict(result.get("result") or {})
        except HTTPException:
            raise
        except Exception:
            pass
    with httpx.Client(timeout=float(max(180, settings.swap_proxy_timeout_seconds))) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/admin/service-action",
            json={"action": action},
            headers={"Authorization": f"Bearer {settings.swap_auth_token}"},
        )
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=body)
    return body


def _call_lane_swap_gateway(*, base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=float(max(180, settings.swap_proxy_timeout_seconds))) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/swap-model",
            json=payload,
            headers={"Authorization": f"Bearer {settings.swap_auth_token}"},
        )
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=body)
    return body


def _build_swap_gateway_payload(
    *,
    lane_state: dict[str, Any],
    preflight: SwapPreflightResponse,
    artifact_row: dict[str, Any],
    requested_model_name: str,
    swap_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    source_mode = preflight.source_mode or "local"
    exact_downstream_model = Path(str(preflight.artifact_path)).name if preflight.artifact_path else preflight.model_name
    payload: dict[str, Any] = {
        "model_name": preflight.model_name,
        "model_path": preflight.artifact_path,
        "model_alias": exact_downstream_model,
        "source_mode": source_mode,
        "swap_id": swap_id,
        "swap_callback_url": f"{settings.router_public_base_url.rstrip('/')}/api/lane-swaps/{swap_id}/events" if swap_id else None,
        "swap_callback_token": settings.swap_auth_token if swap_id else None,
    }
    if source_mode == "remote_copy_then_load":
        local_model_root = lane_state.get("local_model_root")
        if not local_model_root:
            raise HTTPException(status_code=409, detail="worker has no configured local model root")
        remote_user = artifact_row.get("mgmt_ssh_user")
        remote_host = artifact_row.get("mgmt_ssh_host")
        if remote_user and remote_host:
            payload["copy_source"] = f"{remote_user}@{remote_host}:{preflight.artifact_path}"
        else:
            payload["copy_source"] = preflight.artifact_path
        payload["copy_destination"] = f"{str(local_model_root).rstrip('/')}/{preflight.model_name}"
    return source_mode, payload


def _resolve_swap_execution(
    cur,
    *,
    lane_id: str,
    model_name: str,
    allow_unverified: bool = False,
) -> tuple[dict[str, Any], SwapPreflightResponse, dict[str, Any], str]:
    lane_state, preflight = _swap_preflight(
        cur,
        lane_id,
        model_name,
        allow_unverified=allow_unverified,
    )
    if not preflight.ok:
        raise HTTPException(status_code=409, detail=preflight.reason or "swap preflight failed")
    cur.execute(
        """
        SELECT hma.artifact_id, hma.model_id, hma.host_id, hma.local_path, h.host_name, h.mgmt_ssh_host, h.mgmt_ssh_user
        FROM host_model_artifacts hma
        JOIN models m ON m.model_id=hma.model_id
        JOIN hosts h ON h.host_id=hma.host_id
        WHERE m.model_name=%s AND hma.local_path=%s
        LIMIT 1
        """,
        (preflight.model_name, preflight.artifact_path),
    )
    artifact_row = cur.fetchone()
    if not artifact_row:
        raise HTTPException(status_code=404, detail="artifact not found for swap")
    source_mode, _ = _build_swap_gateway_payload(
        lane_state=lane_state,
        preflight=preflight,
        artifact_row=artifact_row,
        requested_model_name=model_name,
        swap_id=None,
    )
    return lane_state, preflight, artifact_row, source_mode


def _restore_split_mode_for_host(
    *,
    host_id: str,
    host_name: str,
    overrides: dict[str, str],
    skip_lane_id: str | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    with db.connect() as conn:
        with conn.cursor() as cur:
            lanes = _list_host_lanes(cur, host_id=host_id)
        conn.commit()

    non_split_lanes = [lane for lane in lanes if _lane_split_slot(lane) is None]
    split_lanes = [lane for lane in lanes if _lane_split_slot(lane) is not None]

    for lane in non_split_lanes:
        if skip_lane_id and str(lane["lane_id"]) == skip_lane_id:
            continue
        try:
            result = _call_lane_service_action(
                base_url=str(lane["base_url"]),
                action="stop",
                host_id=host_id,
                lane_id=str(lane["lane_id"]),
            )
            with db.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE lanes
                        SET status='offline',
                            suspension_reason=NULL,
                            updated_at=now()
                        WHERE lane_id=%s
                        """,
                        (str(lane["lane_id"]),),
                    )
                conn.commit()
            actions.append(
                {
                    "lane_id": str(lane["lane_id"]),
                    "lane_name": str(lane.get("lane_name") or ""),
                    "lane_type": str(lane.get("lane_type") or ""),
                    "action": "stop",
                    "ok": True,
                    "details": result,
                }
            )
        except Exception as exc:
            actions.append(
                {
                    "lane_id": str(lane["lane_id"]),
                    "lane_name": str(lane.get("lane_name") or ""),
                    "lane_type": str(lane.get("lane_type") or ""),
                    "action": "stop",
                    "ok": False,
                    "error": str(exc),
                }
            )

    for lane in split_lanes:
        lane_id = str(lane["lane_id"])
        lane_slot = _lane_split_slot(lane) or ""
        desired_model_name = _desired_restore_model_name(lane, overrides)
        if lane_id == skip_lane_id:
            actions.append(
                {
                    "lane_id": lane_id,
                    "lane_name": str(lane.get("lane_name") or ""),
                    "lane_type": str(lane.get("lane_type") or ""),
                    "action": "preserve",
                    "ok": True,
                    "model_name": desired_model_name,
                }
            )
            continue
        if not desired_model_name:
            actions.append(
                {
                    "lane_id": lane_id,
                    "lane_name": str(lane.get("lane_name") or ""),
                    "lane_type": str(lane.get("lane_type") or ""),
                    "action": "swap-model",
                    "ok": False,
                    "error": f"no desired model configured for {lane_slot} lane",
                }
            )
            continue
        try:
            with db.connect() as conn:
                with conn.cursor() as cur:
                    lane_state, preflight, artifact_row, source_mode = _resolve_swap_execution(
                        cur,
                        lane_id=lane_id,
                        model_name=desired_model_name,
                    )
                conn.commit()
            _, payload = _build_swap_gateway_payload(
                lane_state=lane_state,
                preflight=preflight,
                artifact_row=artifact_row,
                requested_model_name=desired_model_name,
            )
            result = _call_lane_swap_gateway(base_url=str(lane["base_url"]), payload=payload)
            with db.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE lanes
                        SET current_model_name=%s,
                            status='ready',
                            suspension_reason=NULL,
                            updated_at=now()
                        WHERE lane_id=%s
                        """,
                        (preflight.model_name, lane_id),
                    )
                    _upsert_usage(
                        cur,
                        lane_id=lane_id,
                        model_id=str(artifact_row["model_id"]),
                        used_at=datetime.now(UTC),
                        swap_at=datetime.now(UTC),
                    )
                conn.commit()
            actions.append(
                {
                    "lane_id": lane_id,
                    "lane_name": str(lane.get("lane_name") or ""),
                    "lane_type": str(lane.get("lane_type") or ""),
                    "action": "swap-model",
                    "ok": True,
                    "requested_model_name": desired_model_name,
                    "resolved_model_name": preflight.model_name,
                    "source_mode": source_mode,
                    "details": result,
                }
            )
        except Exception as exc:
            actions.append(
                {
                    "lane_id": lane_id,
                    "lane_name": str(lane.get("lane_name") or ""),
                    "lane_type": str(lane.get("lane_type") or ""),
                    "action": "swap-model",
                    "ok": False,
                    "requested_model_name": desired_model_name,
                    "error": str(exc),
                }
            )
    return actions


def _cleanup_expired_router_leases(cur) -> None:
    cur.execute(
        """
        UPDATE router_leases
        SET state='expired'
        WHERE state='active'
          AND COALESCE(last_heartbeat_at, acquired_at) < now() - (%s * interval '1 second')
        """,
        (settings.default_lease_stale_seconds,),
    )


def _acquire_router_lease(
    *,
    lane_id: str,
    model_id: str,
    owner: str,
    job_type: str,
    ttl_seconds: int,
    details: dict[str, Any],
) -> tuple[str, datetime]:
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=max(30, int(ttl_seconds)))
    with db.connect() as conn:
        with conn.cursor() as cur:
            _cleanup_expired_router_leases(cur)
            cur.execute(
                """
                SELECT 1
                FROM router_leases
                WHERE lane_id=%s AND state='active' AND expires_at > now()
                LIMIT 1
                """,
                (lane_id,),
            )
            if cur.fetchone():
                raise RuntimeError("lane busy")
            cur.execute(
                """
                INSERT INTO router_leases (lane_id, model_id, owner, job_type, state, acquired_at, last_heartbeat_at, expires_at, details)
                VALUES (%s, %s, %s, %s, 'active', %s, %s, %s, %s::jsonb)
                RETURNING lease_id
                """,
                (lane_id, model_id, owner, job_type, now, now, expires_at, Jsonb(details)),
            )
            lease_id = str(cur.fetchone()["lease_id"])
        conn.commit()
    return lease_id, expires_at


def _heartbeat_router_lease(*, lease_id: str) -> None:
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=max(30, int(settings.default_lease_stale_seconds)))
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE router_leases
                SET last_heartbeat_at=%s, expires_at=%s
                WHERE lease_id=%s AND state='active'
                """,
                (now, expires_at, lease_id),
            )
        conn.commit()


def _release_router_lease(*, lease_id: str, ok: bool) -> None:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE router_leases
                SET state=%s, released_at=now()
                WHERE lease_id=%s AND state='active'
                """,
                ("released" if ok else "failed", lease_id),
            )
        conn.commit()


@app.post("/api/router-leases/validate")
def api_router_lease_validate(
    req: Request,
    body: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Validate a router-issued lease token. Intended for worker-token-gateway.py."""
    token = _bearer_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    expected_model = None
    try:
        expected_model = str((body or {}).get("model") or "").strip() or None
    except Exception:
        expected_model = None
    try:
        claims = verify_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

    lease_id = str(claims.get("lease_id") or "").strip()
    lane_id = str(claims.get("lane_id") or "").strip()
    claim_model = str(claims.get("model") or "").strip() or None
    if not lease_id or not lane_id:
        raise HTTPException(status_code=401, detail="invalid token claims")
    if expected_model and claim_model and expected_model != claim_model:
        raise HTTPException(status_code=401, detail="model mismatch")

    with db.connect() as conn:
        with conn.cursor() as cur:
            _cleanup_expired_router_leases(cur)
            cur.execute(
                """
                SELECT lease_id, lane_id, state, expires_at
                FROM router_leases
                WHERE lease_id=%s AND lane_id=%s
                """,
                (lease_id, lane_id),
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="lease not found")
    if str(row["state"]) != "active":
        raise HTTPException(status_code=401, detail="lease not active")
    try:
        _heartbeat_router_lease(lease_id=lease_id)
    except Exception:
        pass
    return {"ok": True, **claims}


@app.get("/api/lanes/{lane_id}/lease-status")
def api_lane_lease_status(lane_id: str) -> dict[str, Any]:
    """Return current router lease state for a lane.

    This is intended for clients like Mesh Computer that need to wait on the
    router's source-of-truth lease/heartbeat state instead of applying their
    own blind retry timing against pinned lanes.
    """
    with db.connect() as conn:
        with conn.cursor() as cur:
            _cleanup_expired_router_leases(cur)
            active = _list_active_router_leases(cur, [lane_id])
            cur.execute(
                """
                SELECT l.lane_id, l.lane_name, l.lane_type, l.base_url, l.status, l.suspension_reason, l.current_model_name,
                       h.host_name,
                       proxy_auth_metadata, backend_type
                FROM lanes l
                JOIN hosts h ON h.host_id = l.host_id
                WHERE l.lane_id=%s
                """,
                (lane_id,),
            )
            lane = cur.fetchone()
    if not lane:
        raise HTTPException(status_code=404, detail="lane not found")
    lane_row = dict(lane)
    apply_mw_effective_status(
        [lane_row],
        mw_state_db=mw_state_db,
        stale_seconds=settings.default_lease_stale_seconds,
    )

    active_summary = _summarize_active_leases(active)
    latest = active_summary[-1] if active_summary else None
    return {
        "lane_id": str(lane_row["lane_id"]),
        "lane_name": str(lane_row.get("lane_name") or ""),
        "lane_type": str(lane_row.get("lane_type") or ""),
        "base_url": str(lane_row.get("base_url") or ""),
        "lane_status": str(lane_row.get("effective_status") or lane_row.get("status") or ""),
        "suspension_reason": lane_row.get("suspension_reason"),
        "current_model": lane_row.get("current_model_name"),
        "lease_active": bool(active_summary),
        "active_lease_count": len(active_summary),
        "active_leases": active_summary,
        "latest_lease": latest,
        "heartbeat_interval_seconds": int(settings.default_lease_heartbeat_interval_seconds),
        "stale_after_seconds": int(settings.default_lease_stale_seconds),
    }


@app.get("/api/router-requests/{request_id}")
def api_router_request_status(request_id: str) -> dict[str, Any]:
    row = _fetch_router_request(request_id)
    if not row:
        raise HTTPException(status_code=404, detail="request not found")
    return _serialize_router_request(row)


@app.get("/api/router-requests/{request_id}/health")
def api_router_request_health(request_id: str) -> dict[str, Any]:
    row = _fetch_router_request(request_id)
    if not row:
        raise HTTPException(status_code=404, detail="request not found")
    return _router_request_health(row)


def _normalize_route_request(*, route: str, raw_payload: dict[str, Any]) -> dict[str, Any]:
    route_name = str(route or "").strip().lower()
    if route_name == "chat":
        try:
            req = ChatCompletionRequest.model_validate(raw_payload or {})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "route": "chat",
            "requested_model_name": req.model,
            "pin_worker": req.mesh_pin_worker,
            "pin_base_url": req.mesh_pin_base_url,
            "pin_lane_type": req.mesh_pin_lane_type,
            "pin_lane_id": req.mesh_pin_lane_id,
            "request_payload": _downstream_payload(req),
        }
    if route_name == "embeddings":
        payload = dict(raw_payload or {})
        model_name = str(payload.get("model") or "").strip()
        if not model_name:
            raise HTTPException(status_code=400, detail="model is required")
        
        # Compatibility mapping: some clients use 'prompt' instead of 'input' for embeddings.
        if "prompt" in payload and "input" not in payload:
            payload["input"] = payload.pop("prompt")

        pin_worker = payload.get("mesh_pin_worker")
        pin_base_url = payload.get("mesh_pin_base_url")
        pin_lane_type = payload.get("mesh_pin_lane_type")
        pin_lane_id = payload.get("mesh_pin_lane_id")
        for key in list(payload.keys()):
            if key.startswith("mesh_"):
                payload.pop(key, None)
        return {
            "route": "embeddings",
            "requested_model_name": model_name,
            "pin_worker": pin_worker,
            "pin_base_url": pin_base_url,
            "pin_lane_type": pin_lane_type,
            "pin_lane_id": pin_lane_id,
            "request_payload": _strip_nones(payload),
        }
    if route_name == "images":
        return _normalize_image_request(raw_payload)
    raise HTTPException(status_code=400, detail=f"unsupported route: {route}")


async def _collect_mw_chat_completion(
    *,
    target: MwGrpcTarget,
    request_id: str,
    model: str,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    content_parts: list[str] = []
    usage_prompt: int | None = None
    usage_completion: int | None = None
    usage_total: int | None = None
    finish_reason: str | None = None

    async for event in MwGrpcClient().stream_chat(
        target=target,
        request_id=request_id,
        model=model,
        messages=[item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in request_payload.get("messages") or []],
        temperature=request_payload.get("temperature"),
        max_tokens=request_payload.get("max_tokens"),
        deadline_unix_ms=None,
    ):
        if str(event.event_type or "") in {"failed", "cancelled"}:
            code = str(event.error_code or "mw_error")
            message = str(event.error_message or "MW chat request failed")
            raise RuntimeError(f"{code}: {message}")
        try:
            if event.usage and getattr(event.usage, "prompt_tokens", None) is not None:
                usage_prompt = int(event.usage.prompt_tokens)
            if event.usage and getattr(event.usage, "completion_tokens", None) is not None:
                usage_completion = int(event.usage.completion_tokens)
            if event.usage and getattr(event.usage, "total_tokens", None) is not None:
                usage_total = int(event.usage.total_tokens)
        except Exception:
            pass
        if str(getattr(event, "finish_reason", "") or ""):
            finish_reason = str(event.finish_reason)

        raw = bytes(event.raw_backend_payload or b"").strip()
        if raw == b"[DONE]":
            break
        if raw:
            try:
                item = json.loads(raw.decode("utf-8"))
                choice = ((item or {}).get("choices") or [{}])[0]
                delta = (choice.get("delta") or {}).get("content")
                message = (choice.get("message") or {}).get("content")
                text = delta if delta is not None else message
                if text:
                    content_parts.append(str(text))
                if choice.get("finish_reason"):
                    finish_reason = str(choice.get("finish_reason"))
                usage = (item or {}).get("usage") or {}
                if usage.get("prompt_tokens") is not None:
                    usage_prompt = int(usage.get("prompt_tokens"))
                if usage.get("completion_tokens") is not None:
                    usage_completion = int(usage.get("completion_tokens"))
                if usage.get("total_tokens") is not None:
                    usage_total = int(usage.get("total_tokens"))
            except Exception:
                content_parts.append(raw.decode("utf-8", errors="replace"))
        elif getattr(event, "text_delta", ""):
            content_parts.append(str(getattr(event, "text_delta", "")))
        if str(event.event_type or "") in {"completed"}:
            break

    if usage_total is None and (usage_prompt is not None or usage_completion is not None):
        usage_total = int(usage_prompt or 0) + int(usage_completion or 0)

    response: dict[str, Any] = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish_reason or "stop",
            }
        ],
    }
    if usage_prompt is not None or usage_completion is not None or usage_total is not None:
        response["usage"] = {
            "prompt_tokens": int(usage_prompt or 0),
            "completion_tokens": int(usage_completion or 0),
            "total_tokens": int(usage_total or 0),
        }
    return response


def _execute_router_request(
    *,
    request_id: str,
    route: str,
    raw_payload: dict[str, Any],
    owner: str,
    job_type: str,
) -> dict[str, Any]:
    normalized = _normalize_route_request(route=route, raw_payload=raw_payload)
    model_name = str(normalized["requested_model_name"])
    pin_worker = normalized.get("pin_worker")
    pin_base_url = normalized.get("pin_base_url")
    pin_lane_type = normalized.get("pin_lane_type")
    pin_lane_id = normalized.get("pin_lane_id")
    request_payload = dict(normalized["request_payload"])
    response_format = str(normalized.get("response_format") or "b64_json")
    if route == "images":
        backend_type = "sd"
    elif route == "chat":
        backend_type = None
    else:
        backend_type = "llama"
    request_context_tokens = _estimate_request_context_tokens(route=route, payload=request_payload)
    downstream_model = model_name

    _touch_router_request(
        request_id=request_id,
        requested_model_name=model_name,
    )
    if _request_cancel_requested(request_id):
        _touch_router_request(
            request_id=request_id,
            state="canceled",
            error_kind="canceled",
            error_message="request canceled before acquisition",
        )
        return {}

    lease_id: str | None = None
    lane_id: str | None = None
    model_id: str | None = None
    choice = None
    resp_data: dict[str, Any] | None = None
    ok = False
    final_state = "failed"
    err_kind: str | None = None
    err_msg: str | None = None
    tps: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None if route == "chat" else 0
    started = time.time()
    did_swap = False

    try:
        _acquire_excluded: set[str] = set()
        for _acquire_attempt in range(3):
            try:
                choice = pick_lane_for_model(
                    model=model_name,
                    backend_type=backend_type,
                    request_context_tokens=request_context_tokens,
                    pin_worker=pin_worker,
                    pin_base_url=pin_base_url,
                    pin_lane_type=pin_lane_type,
                    pin_lane_id=pin_lane_id,
                    exclude_lane_ids=_acquire_excluded or None,
                )
            except LanePlacementError as exc:
                err_kind = "routing_error"
                err_msg = str(exc)
                raise
            except Exception as exc:
                err_kind = "routing_error"
                err_msg = str(exc)
                raise

            if choice and not _model_request_matches_candidate(model_name, choice.current_model_name or ""):
                # Swap needed — trigger swap and wait
                try:
                    # api_lane_swap_model is a synchronous blocking function
                    api_lane_swap_model(
                        choice.lane_id,
                        SwapModelRequest(
                            model_name=model_name,
                            swap_urgency="wait"
                        )
                    )
                    did_swap = True
                except Exception as swap_err:
                    logger.warning("Auto-swap failed for lane %s model %s: %s", choice.lane_id, model_name, swap_err)
                    raise RuntimeError(f"no READY lanes available serving {model_name} and auto-swap failed: {swap_err}")

            downstream_model = model_name
            with db.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO models (model_name, format)
                        VALUES (%s, 'other'::model_format)
                        ON CONFLICT (model_name) DO UPDATE SET updated_at=now()
                        """,
                        (model_name,),
                    )
                    cur.execute("SELECT model_id FROM models WHERE model_name=%s", (model_name,))
                    model_id = str(cur.fetchone()["model_id"])
                    lane_id = choice.lane_id or None
                    if lane_id is not None:
                        downstream_model = _resolve_downstream_model_for_lane(
                            cur,
                            lane_id=str(lane_id),
                            requested_model_name=model_name,
                            model_id=str(model_id),
                        )
                conn.commit()

            request_payload["model"] = downstream_model
            try:
                lease_id, expires_at = _acquire_router_lease(
                    lane_id=choice.lane_id,
                    model_id=str(model_id),
                    owner=owner,
                    job_type=job_type,
                    ttl_seconds=settings.default_lease_ttl_seconds,
                    details={
                        "worker_id": choice.worker_id,
                        "base_url": choice.base_url,
                        "downstream_model": downstream_model,
                        "route": route,
                        "request_id": request_id,
                        "request_payload": request_payload,
                    },
                )
                break  # lease acquired — exit retry loop
            except RuntimeError as _busy_err:
                if "lane busy" in str(_busy_err) and _acquire_attempt < 2:
                    logger.info("Lane %s busy, retrying pick with exclusion (attempt %d)", choice.lane_id, _acquire_attempt + 1)
                    _acquire_excluded.add(choice.lane_id)
                    continue
                raise
        _touch_router_request(
            request_id=request_id,
            state="acquired",
            lease_id=lease_id,
            lane_id=choice.lane_id,
            model_id=model_id,
            worker_id=choice.worker_id,
            base_url=choice.base_url,
            downstream_model_name=downstream_model,
            expires_at=expires_at,
            last_heartbeat_at=datetime.now(UTC),
        )

        if _request_cancel_requested(request_id):
            final_state = "canceled"
            err_kind = "canceled"
            err_msg = "request canceled after acquisition"
            return {}

        token = sign_token(
            {
                "lease_id": lease_id,
                "lane_id": choice.lane_id,
                "worker_id": choice.worker_id,
                "base_url": choice.base_url,
                "model": downstream_model,
                "owner": owner,
                "exp": int(expires_at.timestamp()),
            }
        )
        _touch_router_request(
            request_id=request_id,
            state="running",
            last_heartbeat_at=datetime.now(UTC),
            expires_at=expires_at,
        )

        stop_heartbeat = threading.Event()
        heartbeat_error: dict[str, str] = {}

        def _heartbeat_loop() -> None:
            interval = max(5, int(settings.default_lease_heartbeat_interval_seconds))
            while not stop_heartbeat.wait(interval):
                try:
                    _heartbeat_router_lease(lease_id=lease_id)
                    _touch_router_request(
                        request_id=request_id,
                        last_heartbeat_at=datetime.now(UTC),
                        expires_at=datetime.now(UTC) + timedelta(seconds=max(30, int(settings.default_lease_stale_seconds))),
                    )
                except Exception as exc:
                    heartbeat_error["error"] = str(exc)
                    break

        heartbeat_thread = threading.Thread(target=_heartbeat_loop, name=f"request-heartbeat-{request_id}", daemon=True)
        heartbeat_thread.start()
        try:
            endpoint = "/v1/chat/completions" if route == "chat" else "/v1/embeddings" if route == "embeddings" else "/sdapi/v1/txt2img"
            request_timeout = 300.0 if route == "images" else float(max(30, settings.default_lease_ttl_seconds))
            mw_target: MwGrpcTarget | None = None
            if route == "chat" and settings.mw_control_enabled and lane_id:
                try:
                    with db.connect() as conn:
                        with conn.cursor() as cur:
                            mw_target = _mw_target_for_lane(cur=cur, lane_id=str(lane_id))
                except Exception:
                    mw_target = None
            if mw_target is not None:
                if not _model_request_matches_candidate(model_name, choice.current_model_name or ""):
                    try:
                        result = _mw_client().send_command(
                            host_id=mw_target.host_id,
                            message_type="load_model",
                            payload={"lane_id": mw_target.lane_id, "model_name": model_name},
                            wait=True,
                            timeout_seconds=max(30, settings.mw_command_timeout_seconds),
                        )
                        if not bool(result.get("ok", False)):
                            raise RuntimeError(str(result.get("error") or "MW load_model failed"))
                    except Exception as exc:
                        raise RuntimeError(f"MW pre-chat load_model failed: {exc}") from exc
                resp_data = asyncio.run(
                    _collect_mw_chat_completion(
                        target=mw_target,
                        request_id=request_id,
                        model=downstream_model,
                        request_payload=request_payload,
                    )
                )
                downstream_status_code = 200
            else:
                with httpx.Client(timeout=request_timeout) as client:
                    downstream_response = client.post(
                        f"{choice.base_url.rstrip('/')}{endpoint}",
                        json=request_payload,
                        headers={"Authorization": f"Bearer {token}"},
                    )
                downstream_status_code = downstream_response.status_code
                try:
                    resp_data = downstream_response.json()
                except Exception:
                    resp_data = {"raw": downstream_response.text}
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=2)
        if heartbeat_error:
            err_kind = "heartbeat_error"
            err_msg = heartbeat_error["error"]
            raise RuntimeError(f"request heartbeat failed: {heartbeat_error['error']}")

        if downstream_status_code >= 400:
            err_kind = "proxy_error"
            err_msg = f"worker proxy http_{downstream_status_code}: {resp_data}"
            raise RuntimeError(err_msg)

        if route == "images":
            resp_data = _translate_sd_response_to_openai(
                response_payload=resp_data if isinstance(resp_data, dict) else {},
                response_format=response_format,
            )

        try:
            usage = (resp_data or {}).get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage.get("prompt_tokens"))
            elif route == "embeddings" and usage.get("total_tokens") is not None:
                prompt_tokens = int(usage.get("total_tokens"))
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage.get("completion_tokens"))
        except Exception:
            pass
        try:
            timings = (((resp_data or {}).get("timings") or {}) if isinstance(resp_data, dict) else {})
            if timings.get("predicted_per_second") is not None:
                tps = float(timings["predicted_per_second"])
        except Exception:
            pass
        if route == "embeddings" and prompt_tokens:
            elapsed_ms = max(1, int((time.time() - started) * 1000))
            tps = round((float(prompt_tokens) * 1000.0) / float(elapsed_ms), 6)

        ok = True
        final_state = "released"
        return resp_data or {}
    except Exception as exc:
        if final_state != "canceled":
            final_state = "failed"
        if err_kind is None:
            err_kind = "request_error"
        if err_msg is None:
            err_msg = str(exc)
        raise
    finally:
        elapsed_ms = int((time.time() - started) * 1000)
        try:
            _maybe_record_perf_observation(
                host_name=str(choice.worker_id) if choice else None,
                lane_id=str(lane_id) if lane_id else None,
                model_name=str(downstream_model) if downstream_model else None,
                modality=route,
                backend_type=str(choice.backend_type) if choice else None,
                lane_type=str(choice.lane_type) if choice else None,
                elapsed_ms=elapsed_ms,
                first_token_ms=None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                decode_tps=tps,
                was_cold=did_swap if route in {"chat", "embeddings"} else None,
                ok=ok,
                error_kind=None if ok else err_kind,
                error_message=None if ok else err_msg,
                metadata={
                    "request_id": request_id,
                    "route": route,
                    "downstream_model": downstream_model,
                    "actual_model": choice.current_model_name if choice else None,
                    "requested_model": model_name,
                    "pin_worker": pin_worker,
                    "pin_base_url": pin_base_url,
                    "pin_lane_type": pin_lane_type,
                    "pin_lane_id": pin_lane_id,
                },
            )
        except Exception:
            pass
        try:
            with db.connect() as conn:
                with conn.cursor() as cur:
                    if lane_id is not None and model_id is not None:
                        cur.execute(
                            """
                            INSERT INTO lane_model_metrics (
                              lane_id, model_id,
                              load_time_ms, request_latency_ms,
                              tps, prompt_tokens, completion_tokens,
                              success, error_kind, error_message, run_tag
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                lane_id,
                                model_id,
                                None,
                                elapsed_ms,
                                tps,
                                prompt_tokens,
                                completion_tokens,
                                ok,
                                err_kind,
                                err_msg,
                                f"mesh-router:{route}",
                            ),
                        )
                        if ok:
                            _upsert_usage(
                                cur,
                                lane_id=str(lane_id),
                                model_id=str(model_id),
                                used_at=datetime.now(UTC),
                            )
                conn.commit()
        except Exception:
            logger.exception("Failed to record request metrics", extra={"request_id": request_id})
        try:
            if lease_id is not None:
                _release_router_lease(lease_id=lease_id, ok=ok)
        except Exception:
            logger.exception("Failed to release request lease", extra={"request_id": request_id, "lease_id": lease_id})
        try:
            _touch_router_request(
                request_id=request_id,
                state=final_state,
                lease_id=lease_id,
                lane_id=lane_id,
                model_id=model_id,
                result_payload=resp_data if ok else None,
                error_kind=None if ok else err_kind,
                error_message=None if ok else err_msg,
                last_heartbeat_at=datetime.now(UTC),
                released_at=datetime.now(UTC),
            )
        except Exception:
            logger.exception("Failed to finalize router request state", extra={"request_id": request_id})


def _execute_router_request_async(*, request_id: str, route: str, raw_payload: dict[str, Any], owner: str, job_type: str) -> None:
    try:
        _execute_router_request(
            request_id=request_id,
            route=route,
            raw_payload=raw_payload,
            owner=owner,
            job_type=job_type,
        )
    except Exception:
        logger.exception("Asynchronous router request failed", extra={"request_id": request_id, "route": route})


def _execute_router_request_streaming(
    *,
    request_id: str,
    route: str,
    raw_payload: dict[str, Any],
    owner: str,
    job_type: str,
) -> StreamingResponse:
    if route != "chat":
        raise HTTPException(status_code=400, detail="streaming is only supported for chat route")

    normalized = _normalize_route_request(route=route, raw_payload=raw_payload)
    model_name = str(normalized["requested_model_name"])
    pin_worker = normalized.get("pin_worker")
    pin_base_url = normalized.get("pin_base_url")
    pin_lane_type = normalized.get("pin_lane_type")
    pin_lane_id = normalized.get("pin_lane_id")
    request_payload = dict(normalized["request_payload"])
    request_payload["stream"] = True
    request_context_tokens = _estimate_request_context_tokens(route=route, payload=request_payload)

    _touch_router_request(
        request_id=request_id,
        requested_model_name=model_name,
    )
    if _request_cancel_requested(request_id):
        _touch_router_request(
            request_id=request_id,
            state="canceled",
            error_kind="canceled",
            error_message="request canceled before acquisition",
        )
        return StreamingResponse(iter([b"data: [DONE]\n\n"]), media_type="text/event-stream")

    started = time.time()
    lease_id: str | None = None
    lane_id: str | None = None
    model_id: str | None = None
    choice = None

    try:
        try:
            choice = pick_lane_for_model(
                model=model_name,
                backend_type="llama",
                request_context_tokens=request_context_tokens,
                pin_worker=pin_worker,
                pin_base_url=pin_base_url,
                pin_lane_type=pin_lane_type,
                pin_lane_id=pin_lane_id,
            )
        except LanePlacementError as exc:
            raise HTTPException(
                status_code=int(getattr(exc, "status_code", 503)),
                detail=str(exc),
                headers={"X-Mesh-Request-Id": request_id},
            ) from exc

        downstream_model = model_name
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO models (model_name, format)
                    VALUES (%s, 'other'::model_format)
                    ON CONFLICT (model_name) DO UPDATE SET updated_at=now()
                    """,
                    (model_name,),
                )
                cur.execute("SELECT model_id FROM models WHERE model_name=%s", (model_name,))
                model_id = str(cur.fetchone()["model_id"])
                lane_id = choice.lane_id or None
                if lane_id is not None:
                    downstream_model = _resolve_downstream_model_for_lane(
                        cur,
                        lane_id=str(lane_id),
                        requested_model_name=model_name,
                        model_id=str(model_id),
                    )
            conn.commit()

        request_payload["model"] = downstream_model
        lease_id, expires_at = _acquire_router_lease(
            lane_id=choice.lane_id,
            model_id=str(model_id),
            owner=owner,
            job_type=job_type,
            ttl_seconds=settings.default_lease_ttl_seconds,
            details={
                "worker_id": choice.worker_id,
                "base_url": choice.base_url,
                "downstream_model": downstream_model,
                "route": route,
                "request_id": request_id,
                "request_payload": request_payload,
            },
        )
        _touch_router_request(
            request_id=request_id,
            state="acquired",
            lease_id=lease_id,
            lane_id=choice.lane_id,
            model_id=model_id,
            worker_id=choice.worker_id,
            base_url=choice.base_url,
            downstream_model_name=downstream_model,
            expires_at=expires_at,
            last_heartbeat_at=datetime.now(UTC),
        )

        token = sign_token(
            {
                "lease_id": lease_id,
                "lane_id": choice.lane_id,
                "worker_id": choice.worker_id,
                "base_url": choice.base_url,
                "model": downstream_model,
                "owner": owner,
                "exp": int(expires_at.timestamp()),
            }
        )

        headers = {
            "X-Mesh-Request-Id": request_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        if lane_id:
            headers["X-Mesh-Lane-Id"] = str(lane_id)
        if choice and choice.worker_id:
            headers["X-Mesh-Worker-Id"] = str(choice.worker_id)
        if downstream_model:
            headers["X-Mesh-Model-Name"] = str(downstream_model)

        mw_target: MwGrpcTarget | None = None
        if settings.mw_control_enabled and lane_id:
            try:
                with db.connect() as conn:
                    with conn.cursor() as cur:
                        mw_target = _mw_target_for_lane(cur=cur, lane_id=str(lane_id))
            except Exception:
                mw_target = None
        _maybe_add_perf_expectation_headers(
            headers=headers,
            host_id=str(getattr(mw_target, "host_id", "") or "") if mw_target is not None else (str(choice.worker_id) if choice else None),
            lane_id=str(lane_id) if lane_id else None,
            model_name=str(downstream_model) if downstream_model else None,
            modality="chat",
        )

        # Best-effort: ensure the requested model is loaded on MW-managed lanes before streaming.
        if mw_target is not None and settings.mw_control_enabled:
            if not _model_request_matches_candidate(model_name, choice.current_model_name or ""):
                try:
                    result = _mw_client().send_command(
                        host_id=mw_target.host_id,
                        message_type="load_model",
                        payload={"lane_id": mw_target.lane_id, "model_name": model_name},
                        wait=True,
                        timeout_seconds=max(30, settings.mw_command_timeout_seconds),
                    )
                    if not bool(result.get("ok", False)):
                        raise RuntimeError(str(result.get("error") or "MW load_model failed"))
                except Exception as exc:
                    raise RuntimeError(f"MW pre-stream load_model failed: {exc}") from exc

        stop_heartbeat = threading.Event()
        heartbeat_error: dict[str, Any] = {}

        def _heartbeat_loop() -> None:
            if lease_id is None:
                return
            while not stop_heartbeat.is_set():
                try:
                    _heartbeat_router_lease(lease_id=lease_id)
                    _touch_router_request(
                        request_id=request_id,
                        last_heartbeat_at=datetime.now(UTC),
                        expires_at=datetime.now(UTC) + timedelta(seconds=max(30, int(settings.default_lease_stale_seconds))),
                    )
                except Exception as exc:
                    heartbeat_error["error"] = str(exc)
                    break
                stop_heartbeat.wait(timeout=float(settings.default_lease_heartbeat_interval_seconds))

        heartbeat_thread = threading.Thread(target=_heartbeat_loop, name=f"request-heartbeat-{request_id}", daemon=True)
        heartbeat_thread.start()

        async def _event_stream() -> Any:
            ok = False
            err_kind: str | None = None
            err_msg: str | None = None
            done_sent = False
            first_token_at: float | None = None
            usage_prompt: int | None = None
            usage_completion: int | None = None
            try:
                if heartbeat_error:
                    raise RuntimeError(f"request heartbeat failed: {heartbeat_error.get('error')}")

                if mw_target is not None:
                    client = MwGrpcClient()
                    async for event in client.stream_chat(
                        target=mw_target,
                        request_id=request_id,
                        model=downstream_model,
                        messages=[item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in request_payload.get("messages") or []],
                        temperature=request_payload.get("temperature"),
                        max_tokens=request_payload.get("max_tokens"),
                        deadline_unix_ms=None,
                    ):
                        if _request_cancel_requested(request_id):
                            break
                        try:
                            if event.usage and getattr(event.usage, "prompt_tokens", None) is not None:
                                usage_prompt = int(event.usage.prompt_tokens)
                            if event.usage and getattr(event.usage, "completion_tokens", None) is not None:
                                usage_completion = int(event.usage.completion_tokens)
                        except Exception:
                            pass
                        raw = bytes(event.raw_backend_payload or b"")
                        if raw:
                            if raw.strip() == b"[DONE]":
                                done_sent = True
                                yield b"data: [DONE]\n\n"
                                break
                            if first_token_at is None:
                                first_token_at = time.time()
                            yield b"data: " + raw + b"\n\n"
                        if str(event.event_type or "") in {"completed"}:
                            break
                        if str(event.event_type or "") in {"failed", "cancelled"}:
                            err_kind = str(event.error_code or "mw_error")
                            err_msg = str(event.error_message or "mw stream failed")
                            break
                else:
                    endpoint = "/v1/chat/completions"
                    request_timeout = float(max(30, settings.default_lease_ttl_seconds))
                    async with httpx.AsyncClient(timeout=request_timeout) as client:
                        async with client.stream(
                            "POST",
                            f"{choice.base_url.rstrip('/')}{endpoint}",
                            json=request_payload,
                            headers={"Authorization": f"Bearer {token}"},
                        ) as downstream:
                            if downstream.status_code >= 400:
                                body = await downstream.aread()
                                raise RuntimeError(f"worker proxy http_{downstream.status_code}: {body[:2048]!r}")
                            async for chunk in downstream.aiter_bytes():
                                if _request_cancel_requested(request_id):
                                    break
                                if chunk:
                                    if b"data: [DONE]" in chunk:
                                        done_sent = True
                                    elif first_token_at is None:
                                        first_token_at = time.time()
                                    yield chunk
                ok = True
            except Exception as exc:
                if err_kind is None:
                    err_kind = "stream_error"
                if err_msg is None:
                    err_msg = str(exc)
                logger.warning("Streaming request failed: %s", exc, extra={"request_id": request_id})
                yield b"data: " + json.dumps({"error": {"message": err_msg, "type": err_kind}}).encode("utf-8") + b"\n\n"
            finally:
                stop_heartbeat.set()
                heartbeat_thread.join(timeout=2)
                try:
                    if not done_sent:
                        yield b"data: [DONE]\n\n"
                        done_sent = True
                except Exception:
                    pass

                canceled = bool(_request_cancel_requested(request_id))
                if canceled:
                    ok = False
                    if err_kind is None:
                        err_kind = "canceled"
                    if err_msg is None:
                        err_msg = "request canceled during streaming"
                final_state = "released" if ok else ("canceled" if canceled else "failed")
                elapsed_ms = int((time.time() - started) * 1000)
                first_token_ms: float | None = None
                decode_tps: float | None = None
                if first_token_at is not None:
                    first_token_ms = max(0.0, (first_token_at - started) * 1000.0)
                    if usage_completion is not None:
                        denom_s = max(0.001, (time.time() - first_token_at))
                        decode_tps = float(usage_completion) / float(denom_s)
                try:
                    _touch_router_request(
                        request_id=request_id,
                        state=final_state,
                        error_kind=None if ok else (err_kind or "stream_error"),
                        error_message=None if ok else (err_msg or "stream failed"),
                        released_at=datetime.now(UTC),
                        last_heartbeat_at=datetime.now(UTC),
                    )
                except Exception:
                    logger.exception("Failed to finalize streaming router request", extra={"request_id": request_id})
                try:
                    _maybe_record_perf_observation(
                        host_name=str(getattr(mw_target, "host_id", "") or "") if mw_target is not None else (str(choice.worker_id) if choice else None),
                        lane_id=str(lane_id) if lane_id else None,
                        model_name=str(downstream_model) if downstream_model else None,
                        modality=route,
                        backend_type=str(choice.backend_type) if choice else None,
                        lane_type=str(choice.lane_type) if choice else None,
                        elapsed_ms=elapsed_ms,
                        first_token_ms=first_token_ms,
                        prompt_tokens=usage_prompt,
                        completion_tokens=usage_completion,
                        decode_tps=decode_tps,
                        was_cold=None,
                        ok=ok,
                        error_kind=None if ok else (err_kind or "stream_error"),
                        error_message=None if ok else (err_msg or "stream failed"),
                        metadata={
                            "request_id": request_id,
                            "route": route,
                            "downstream_model": downstream_model,
                            "actual_model": choice.current_model_name if choice else None,
                            "requested_model": model_name,
                            "pin_worker": pin_worker,
                            "pin_base_url": pin_base_url,
                            "pin_lane_type": pin_lane_type,
                            "pin_lane_id": pin_lane_id,
                            "mw_managed": bool(mw_target is not None),
                            "mw_target_host_id": getattr(mw_target, "host_id", None) if mw_target is not None else None,
                            "mw_target_lane_id": getattr(mw_target, "lane_id", None) if mw_target is not None else None,
                        },
                    )
                except Exception:
                    pass
                try:
                    if lease_id is not None:
                        _release_router_lease(lease_id=lease_id, ok=ok)
                except Exception:
                    logger.exception("Failed to release streaming request lease", extra={"request_id": request_id})
                try:
                    with db.connect() as conn:
                        with conn.cursor() as cur:
                            if lane_id is not None and model_id is not None:
                                cur.execute(
                                    """
                                    INSERT INTO lane_model_metrics (
                                      lane_id, model_id,
                                      load_time_ms, request_latency_ms,
                                      tps, prompt_tokens, completion_tokens,
                                      success, error_kind, error_message, run_tag
                                    )
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    """,
                                    (
                                        lane_id,
                                        model_id,
                                        None,
                                        elapsed_ms,
                                        decode_tps,
                                        usage_prompt,
                                        usage_completion,
                                        ok,
                                        None if ok else (err_kind or "stream_error"),
                                        None if ok else (err_msg or "stream failed"),
                                        "mesh-router:chat:stream",
                                    ),
                                )
                        conn.commit()
                except Exception:
                    logger.exception("Failed to record streaming request metrics", extra={"request_id": request_id})

        return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=headers)
    except Exception as exc:
        if lease_id is not None:
            try:
                _release_router_lease(lease_id=lease_id, ok=False)
            except Exception:
                pass
        _touch_router_request(
            request_id=request_id,
            state="failed",
            error_kind="request_error",
            error_message=str(exc),
        )
        status_code = int(getattr(exc, "status_code")) if isinstance(exc, LanePlacementError) else 502
        headers = {"X-Mesh-Request-Id": request_id}
        if lane_id:
            headers["X-Mesh-Lane-Id"] = str(lane_id)
        if choice and getattr(choice, "worker_id", None):
            headers["X-Mesh-Worker-Id"] = str(choice.worker_id)
        raise HTTPException(status_code=status_code, detail=str(exc), headers=headers) from exc


@app.post("/api/router-requests")
def api_router_request_submit(req: RouterRequestSubmitRequest, response: Response) -> dict[str, Any]:
    normalized = _normalize_route_request(route=req.route, raw_payload=req.payload)
    request_id = _create_router_request(
        route=str(normalized["route"]),
        request_payload=dict(normalized["request_payload"]),
        owner=req.owner or settings.default_owner,
        job_type=req.job_type or settings.default_job_type,
        app_name=req.app_name,
        client_request_id=req.client_request_id,
        requested_model_name=str(normalized["requested_model_name"]),
        pin_worker=normalized.get("pin_worker"),
        pin_base_url=normalized.get("pin_base_url"),
        pin_lane_type=normalized.get("pin_lane_type"),
    )
    response.headers["X-Mesh-Request-Id"] = request_id
    if req.wait:
        try:
            result = _execute_router_request(
                request_id=request_id,
                route=str(normalized["route"]),
                raw_payload=dict(req.payload),
                owner=req.owner or settings.default_owner,
                job_type=req.job_type or settings.default_job_type,
            )
        except Exception:
            result = None
        row = _fetch_router_request(request_id)
        if not row:
            raise HTTPException(status_code=500, detail="request disappeared")
        return {"request": _serialize_router_request(row), "result": result}

    threading.Thread(
        target=_execute_router_request_async,
        kwargs={
            "request_id": request_id,
            "route": str(normalized["route"]),
            "raw_payload": dict(req.payload),
            "owner": req.owner or settings.default_owner,
            "job_type": req.job_type or settings.default_job_type,
        },
        name=f"router-request-{request_id}",
        daemon=True,
    ).start()
    response.status_code = 202
    row = _fetch_router_request(request_id)
    if not row:
        raise HTTPException(status_code=500, detail="request not created")
    return _serialize_router_request(row)


@app.post("/api/router-requests/{request_id}/cancel")
def api_router_request_cancel(request_id: str, body: RouterRequestCancelRequest) -> dict[str, Any]:
    row = _fetch_router_request(request_id)
    if not row:
        raise HTTPException(status_code=404, detail="request not found")
    state = str(row.get("state") or "").lower()
    cancel_reason = body.reason or "cancel requested"
    now = datetime.now(UTC)
    if state == "queued":
        _touch_router_request(
            request_id=request_id,
            state="canceled",
            cancel_requested=True,
            cancel_requested_at=now,
            cancel_reason=cancel_reason,
            error_kind="canceled",
            error_message=cancel_reason,
        )
    elif state not in REQUEST_TERMINAL_STATES:
        _touch_router_request(
            request_id=request_id,
            cancel_requested=True,
            cancel_requested_at=now,
            cancel_reason=cancel_reason,
        )
    row = _fetch_router_request(request_id)
    if not row:
        raise HTTPException(status_code=500, detail="request disappeared")
    return {
        "accepted": str(row.get("state") or "").lower() not in REQUEST_TERMINAL_STATES or str(row.get("state") or "").lower() == "canceled",
        "request": _serialize_router_request(row),
        "health": _router_request_health(row),
    }


@app.post("/v1/chat/completions")
def v1_chat_completions(
    req: ChatCompletionRequest,
    response: Response,
    x_mesh_pin_worker: str | None = Header(default=None),
    x_mesh_pin_base_url: str | None = Header(default=None),
    x_mesh_pin_lane_type: str | None = Header(default=None),
    x_mesh_pin_lane_id: str | None = Header(default=None),
) -> Any:
    raw_payload = req.model_dump(by_alias=True)
    if x_mesh_pin_worker is not None:
        raw_payload["mesh_pin_worker"] = x_mesh_pin_worker
    if x_mesh_pin_base_url is not None:
        raw_payload["mesh_pin_base_url"] = x_mesh_pin_base_url
    if x_mesh_pin_lane_type is not None:
        raw_payload["mesh_pin_lane_type"] = x_mesh_pin_lane_type
    if x_mesh_pin_lane_id is not None:
        raw_payload["mesh_pin_lane_id"] = x_mesh_pin_lane_id
    normalized = _normalize_route_request(route="chat", raw_payload=raw_payload)
    request_id = _create_router_request(
        route="chat",
        request_payload=dict(normalized["request_payload"]),
        owner=settings.default_owner,
        job_type=settings.default_job_type,
        requested_model_name=str(normalized["requested_model_name"]),
        pin_worker=normalized.get("pin_worker"),
        pin_base_url=normalized.get("pin_base_url"),
        pin_lane_type=normalized.get("pin_lane_type"),
    )
    response.headers["X-Mesh-Request-Id"] = request_id

    if bool((normalized.get("request_payload") or {}).get("stream")):
        return _execute_router_request_streaming(
            request_id=request_id,
            route="chat",
            raw_payload=raw_payload,
            owner=settings.default_owner,
            job_type=settings.default_job_type,
        )
    try:
        result = _execute_router_request(
            request_id=request_id,
            route="chat",
            raw_payload=raw_payload,
            owner=settings.default_owner,
            job_type=settings.default_job_type,
        )
        row = _fetch_router_request(request_id)
        if row and row.get("lane_id"):
            response.headers["X-Mesh-Lane-Id"] = str(row["lane_id"])
        if row and row.get("worker_id"):
            response.headers["X-Mesh-Worker-Id"] = str(row["worker_id"])
        if row:
            effective_model_name = (
                row.get("downstream_model_name")
                or row.get("model_name")
                or row.get("requested_model_name")
            )
            if effective_model_name:
                response.headers["X-Mesh-Model-Name"] = str(effective_model_name)
            _maybe_add_perf_expectation_headers(
                headers=response.headers,  # type: ignore[arg-type]
                host_id=str(row.get("worker_id") or ""),
                lane_id=str(row.get("lane_id") or ""),
                model_name=str(effective_model_name) if effective_model_name else None,
                modality="chat",
            )
            max_ctx = row.get("lane_max_ctx")
            if max_ctx is None:
                max_ctx = row.get("context_default")
            if max_ctx is not None:
                response.headers["X-Mesh-Model-Max-Ctx"] = str(max_ctx)
        return result
    except HTTPException:
        raise
    except HTTPException:
        raise
    except Exception as exc:
        row = _fetch_router_request(request_id)
        headers: dict[str, str] = {"X-Mesh-Request-Id": request_id}
        if row and row.get("lane_id"):
            headers["X-Mesh-Lane-Id"] = str(row["lane_id"])
        if row and row.get("worker_id"):
            headers["X-Mesh-Worker-Id"] = str(row["worker_id"])
        if isinstance(exc, HTTPException):
            merged = dict(exc.headers or {})
            merged.update(headers)
            raise HTTPException(status_code=exc.status_code, detail=exc.detail, headers=merged)
        status_code = (
            int(getattr(exc, "status_code"))
            if isinstance(exc, LanePlacementError)
            else (503 if "no READY lanes" in str(exc) or "lane busy" in str(exc) else 502)
        )
        raise HTTPException(status_code=status_code, detail=str(exc), headers=headers)


@app.post("/v1/embeddings")
def v1_embeddings(
    req: Request,
    response: Response,
    body: dict[str, Any] = Body(default_factory=dict),
    x_mesh_pin_worker: str | None = Header(default=None),
    x_mesh_pin_base_url: str | None = Header(default=None),
    x_mesh_pin_lane_type: str | None = Header(default=None),
    x_mesh_pin_lane_id: str | None = Header(default=None),
) -> dict[str, Any]:
    raw_payload = dict(body or {})
    if x_mesh_pin_worker is not None:
        raw_payload["mesh_pin_worker"] = x_mesh_pin_worker
    if x_mesh_pin_base_url is not None:
        raw_payload["mesh_pin_base_url"] = x_mesh_pin_base_url
    if x_mesh_pin_lane_type is not None:
        raw_payload["mesh_pin_lane_type"] = x_mesh_pin_lane_type
    if x_mesh_pin_lane_id is not None:
        raw_payload["mesh_pin_lane_id"] = x_mesh_pin_lane_id
    normalized = _normalize_route_request(route="embeddings", raw_payload=raw_payload)
    request_id = _create_router_request(
        route="embeddings",
        request_payload=dict(normalized["request_payload"]),
        owner=settings.default_owner,
        job_type=settings.default_job_type,
        requested_model_name=str(normalized["requested_model_name"]),
        pin_worker=normalized.get("pin_worker"),
        pin_base_url=normalized.get("pin_base_url"),
        pin_lane_type=normalized.get("pin_lane_type"),
    )
    response.headers["X-Mesh-Request-Id"] = request_id
    try:
        result = _execute_router_request(
            request_id=request_id,
            route="embeddings",
            raw_payload=raw_payload,
            owner=settings.default_owner,
            job_type=settings.default_job_type,
        )
        row = _fetch_router_request(request_id)
        if row and row.get("lane_id"):
            response.headers["X-Mesh-Lane-Id"] = str(row["lane_id"])
        if row and row.get("worker_id"):
            response.headers["X-Mesh-Worker-Id"] = str(row["worker_id"])
        return result
    except HTTPException:
        raise
    except Exception as exc:
        headers = {"X-Mesh-Request-Id": request_id}
        row = _fetch_router_request(request_id)
        if row and row.get("lane_id"):
            headers["X-Mesh-Lane-Id"] = str(row["lane_id"])
        if row and row.get("worker_id"):
            headers["X-Mesh-Worker-Id"] = str(row["worker_id"])
        if isinstance(exc, HTTPException):
            merged = dict(exc.headers or {})
            merged.update(headers)
            raise HTTPException(status_code=exc.status_code, detail=exc.detail, headers=merged)
        status_code = (
            int(getattr(exc, "status_code"))
            if isinstance(exc, LanePlacementError)
            else (503 if "no READY lanes" in str(exc) or "lane busy" in str(exc) else 502)
        )
        raise HTTPException(status_code=status_code, detail=str(exc), headers=headers)


@app.post("/v1/images/generations")
def v1_images_generations(
    req: ImageGenerationRequest,
    response: Response,
    x_mesh_pin_worker: str | None = Header(default=None),
    x_mesh_pin_base_url: str | None = Header(default=None),
    x_mesh_pin_lane_type: str | None = Header(default=None),
    x_mesh_pin_lane_id: str | None = Header(default=None),
) -> dict[str, Any]:
    raw_payload = req.model_dump()
    if x_mesh_pin_worker is not None:
        raw_payload["mesh_pin_worker"] = x_mesh_pin_worker
    if x_mesh_pin_base_url is not None:
        raw_payload["mesh_pin_base_url"] = x_mesh_pin_base_url
    if x_mesh_pin_lane_type is not None:
        raw_payload["mesh_pin_lane_type"] = x_mesh_pin_lane_type
    if x_mesh_pin_lane_id is not None:
        raw_payload["mesh_pin_lane_id"] = x_mesh_pin_lane_id
    normalized = _normalize_route_request(route="images", raw_payload=raw_payload)
    request_id = _create_router_request(
        route="images",
        request_payload=dict(normalized["request_payload"]),
        owner=settings.default_owner,
        job_type=settings.default_job_type,
        requested_model_name=str(normalized["requested_model_name"]),
        pin_worker=normalized.get("pin_worker"),
        pin_base_url=normalized.get("pin_base_url"),
        pin_lane_type=normalized.get("pin_lane_type"),
    )
    response.headers["X-Mesh-Request-Id"] = request_id
    try:
        result = _execute_router_request(
            request_id=request_id,
            route="images",
            raw_payload=raw_payload,
            owner=settings.default_owner,
            job_type=settings.default_job_type,
        )
        row = _fetch_router_request(request_id)
        if row and row.get("lane_id"):
            response.headers["X-Mesh-Lane-Id"] = str(row["lane_id"])
        if row and row.get("worker_id"):
            response.headers["X-Mesh-Worker-Id"] = str(row["worker_id"])
        return result
    except Exception as exc:
        row = _fetch_router_request(request_id)
        headers: dict[str, str] = {"X-Mesh-Request-Id": request_id}
        if row and row.get("lane_id"):
            headers["X-Mesh-Lane-Id"] = str(row["lane_id"])
        if row and row.get("worker_id"):
            headers["X-Mesh-Worker-Id"] = str(row["worker_id"])
        if isinstance(exc, HTTPException):
            merged = dict(exc.headers or {})
            merged.update(headers)
            raise HTTPException(status_code=exc.status_code, detail=exc.detail, headers=merged)
        status_code = (
            int(getattr(exc, "status_code"))
            if isinstance(exc, LanePlacementError)
            else (503 if "no READY lanes" in str(exc) or "lane busy" in str(exc) else 502)
        )
        raise HTTPException(status_code=status_code, detail=str(exc), headers=headers)


@app.post("/api/embeddings")
def legacy_embeddings(
    req: Request,
    response: Response,
    body: dict[str, Any] = Body(default_factory=dict),
    x_mesh_pin_worker: str | None = Header(default=None),
    x_mesh_pin_base_url: str | None = Header(default=None),
    x_mesh_pin_lane_type: str | None = Header(default=None),
    x_mesh_pin_lane_id: str | None = Header(default=None),
) -> dict[str, Any]:
    # Compatibility endpoint for legacy Ollama clients (like the current watchdog)
    # 1. Map 'prompt' to 'input' if needed (handled by _normalize_route_request)
    # 2. Return top-level 'embedding' key instead of OpenAI list
    raw_payload = dict(body or {})
    if x_mesh_pin_worker is not None:
        raw_payload["mesh_pin_worker"] = x_mesh_pin_worker
    if x_mesh_pin_base_url is not None:
        raw_payload["mesh_pin_base_url"] = x_mesh_pin_base_url
    if x_mesh_pin_lane_type is not None:
        raw_payload["mesh_pin_lane_type"] = x_mesh_pin_lane_type
    if x_mesh_pin_lane_id is not None:
        raw_payload["mesh_pin_lane_id"] = x_mesh_pin_lane_id
    normalized = _normalize_route_request(route="embeddings", raw_payload=raw_payload)
    request_id = _create_router_request(
        route="embeddings",
        request_payload=dict(normalized["request_payload"]),
        owner=settings.default_owner,
        job_type="legacy_proxy",
        requested_model_name=str(normalized["requested_model_name"]),
        pin_worker=normalized.get("pin_worker"),
        pin_base_url=normalized.get("pin_base_url"),
        pin_lane_type=normalized.get("pin_lane_type"),
    )
    response.headers["X-Mesh-Request-Id"] = request_id
    try:
        result = _execute_router_request(
            request_id=request_id,
            route="embeddings",
            raw_payload=raw_payload,
            owner=settings.default_owner,
            job_type="legacy_proxy",
        )
        if "data" in result and len(result["data"]) > 0:
            return {"embedding": result["data"][0]["embedding"]}
        return result
    except Exception as exc:
        row = _fetch_router_request(request_id)
        headers = {"X-Mesh-Request-Id": request_id}
        if row and row.get("worker_id"):
            headers["X-Mesh-Worker-Id"] = str(row["worker_id"])
        if isinstance(exc, HTTPException):
            merged = dict(exc.headers or {})
            merged.update(headers)
            raise HTTPException(status_code=exc.status_code, detail=exc.detail, headers=merged)
        status_code = (
            int(getattr(exc, "status_code"))
            if isinstance(exc, LanePlacementError)
            else (503 if "no READY lanes" in str(exc) or "lane busy" in str(exc) else 502)
        )
        raise HTTPException(status_code=status_code, detail=str(exc), headers=headers)


class SwapModelRequest(BaseModel):
    model_name: str
    allow_unverified: bool = False
    swap_urgency: Literal["wait", "cancel"] = "wait"
    wait_timeout_s: int = 1800


def _swap_cost_metadata(
    *,
    lane_state: dict[str, Any],
    model_name: str,
    tuning_profile: dict[str, Any] | None,
    sibling_lanes: list[dict[str, Any]],
    active_leases: list[dict[str, Any]],
) -> dict[str, Any]:
    return _strip_nones(
        {
            "host_id": str(lane_state["host"]["host_id"]) if lane_state.get("host") else None,
            "host_name": str(lane_state["host"]["host_name"]) if lane_state.get("host") else None,
            "storage_scheme": str(tuning_profile.get("storage_scheme") or "unknown") if tuning_profile else None,
            "cost_tier": str(tuning_profile.get("cost_tier") or "standard") if tuning_profile else "standard",
            "disables_sibling_lanes": bool(tuning_profile.get("disables_sibling_lanes") or False) if tuning_profile else False,
            "exclusive_host_resources": bool(tuning_profile.get("exclusive_host_resources") or False) if tuning_profile else False,
            "prompt_tps": float(tuning_profile["prompt_tps"]) if tuning_profile and tuning_profile.get("prompt_tps") is not None else None,
            "generation_tps": float(tuning_profile["generation_tps"]) if tuning_profile and tuning_profile.get("generation_tps") is not None else None,
            "avg_total_latency_s": float(tuning_profile["avg_total_latency_s"]) if tuning_profile and tuning_profile.get("avg_total_latency_s") is not None else None,
            "score": float(tuning_profile["score"]) if tuning_profile and tuning_profile.get("score") is not None else None,
            "affected_sibling_lanes": [
                {
                    "lane_id": str(row["lane_id"]),
                    "lane_name": str(row.get("lane_name") or ""),
                    "lane_type": str(row.get("lane_type") or ""),
                    "base_url": str(row.get("base_url") or ""),
                }
                for row in sibling_lanes
            ],
            "active_leases": _summarize_active_leases(active_leases),
            "model_name": model_name,
        }
    )


def _wait_for_no_active_leases(cur, *, lane_ids: list[str], timeout_s: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + max(1, int(timeout_s))
    while True:
        active = _list_active_router_leases(cur, lane_ids)
        if not active:
            return []
        if time.monotonic() >= deadline:
            return active
        time.sleep(2)


def _swap_preflight(
    cur,
    lane_ref: str,
    model_name: str,
    *,
    allow_unverified: bool = False,
) -> tuple[dict[str, Any], SwapPreflightResponse]:
    state, capabilities = _build_lane_capability_payload(cur, lane_ref)
    resolved_candidate, candidate_group = _resolve_swap_candidate(capabilities, model_name)
    resolved_model_name = resolved_candidate.model_name if resolved_candidate is not None else model_name
    tuning_profile = _load_swap_tuning_profile(
        cur,
        host_id=str(state["host"]["host_id"]),
        lane_id=str(state["lane"]["lane_id"]),
        model_name=resolved_model_name,
    )
    sibling_lanes = (
        _list_host_sibling_lanes(
            cur,
            host_id=str(state["host"]["host_id"]),
            lane_id=str(state["lane"]["lane_id"]),
        )
        if tuning_profile and bool(tuning_profile.get("disables_sibling_lanes") or tuning_profile.get("exclusive_host_resources"))
        else []
    )
    affected_lane_ids = [str(state["lane"]["lane_id"])] + [str(row["lane_id"]) for row in sibling_lanes]
    active_leases = _list_active_router_leases(cur, affected_lane_ids)
    if resolved_candidate is not None and candidate_group == "viable":
        source_mode = "local" if resolved_candidate.locality == "local" else "remote_copy_then_load"
        metadata: dict[str, Any] = {
            "capability_source": resolved_candidate.locality,
            "requested_model_name": model_name,
        }
        if resolved_candidate.model_name != model_name:
            metadata["resolved_model_name"] = resolved_candidate.model_name
        if resolved_candidate.locality == "remote":
            metadata["local_model_root"] = state["local_model_root"]
        metadata["swap_cost"] = _swap_cost_metadata(
            lane_state=state,
            model_name=resolved_model_name,
            tuning_profile=tuning_profile,
            sibling_lanes=sibling_lanes,
            active_leases=active_leases,
        )
        return state, SwapPreflightResponse(
            lane_id=capabilities.lane_id,
            model_name=resolved_model_name,
            ok=True,
            source_mode=source_mode,
            artifact_path=resolved_candidate.artifact_path,
            artifact_host=resolved_candidate.artifact_host,
            artifact_provider=resolved_candidate.artifact_provider or None,
            estimated_swap_ms=resolved_candidate.estimated_swap_ms,
            swap_strategy=resolved_candidate.swap_strategy,
            metadata=_strip_nones(metadata),
        )
    if resolved_candidate is not None and candidate_group == "unverified":
        if allow_unverified:
            source_mode = "local" if resolved_candidate.artifact_host == capabilities.metadata.get("host_name") else "remote_copy_then_load"
            return state, SwapPreflightResponse(
                lane_id=capabilities.lane_id,
                model_name=resolved_model_name,
                ok=True,
                source_mode=source_mode,
                artifact_path=resolved_candidate.artifact_path,
                artifact_host=resolved_candidate.artifact_host,
                artifact_provider=resolved_candidate.artifact_provider or None,
                estimated_swap_ms=resolved_candidate.estimated_swap_ms,
                swap_strategy=resolved_candidate.swap_strategy,
                reason="allow_unverified override applied",
                metadata=_strip_nones(
                    {
                        "override": "allow_unverified",
                        "requested_model_name": model_name,
                        "resolved_model_name": resolved_model_name if resolved_model_name != model_name else None,
                        "swap_cost": _swap_cost_metadata(
                            lane_state=state,
                            model_name=resolved_model_name,
                            tuning_profile=tuning_profile,
                            sibling_lanes=sibling_lanes,
                            active_leases=active_leases,
                        ),
                    }
                ),
            )
        return state, SwapPreflightResponse(
            lane_id=capabilities.lane_id,
            model_name=resolved_model_name,
            ok=False,
            artifact_path=resolved_candidate.artifact_path,
            artifact_host=resolved_candidate.artifact_host,
            artifact_provider=resolved_candidate.artifact_provider or None,
            estimated_swap_ms=resolved_candidate.estimated_swap_ms,
            swap_strategy=resolved_candidate.swap_strategy,
            reason=resolved_candidate.reason or "model is unverified for this lane",
            metadata=_strip_nones(
                {
                    "requested_model_name": model_name,
                    "resolved_model_name": resolved_model_name if resolved_model_name != model_name else None,
                    "swap_cost": _swap_cost_metadata(
                        lane_state=state,
                        model_name=resolved_model_name,
                        tuning_profile=tuning_profile,
                        sibling_lanes=sibling_lanes,
                        active_leases=active_leases,
                    ),
                }
            ),
        )
    return state, SwapPreflightResponse(
        lane_id=capabilities.lane_id,
        model_name=model_name,
        ok=False,
        reason="model is not viable for this lane",
        metadata={
            "swap_cost": _swap_cost_metadata(
                lane_state=state,
                model_name=model_name,
                tuning_profile=tuning_profile,
                sibling_lanes=sibling_lanes,
                active_leases=active_leases,
            )
        },
    )


@app.post("/api/lanes/{lane_id}/swap-preflight", response_model=SwapPreflightResponse)
def api_lane_swap_preflight(lane_id: str, req: SwapModelRequest) -> SwapPreflightResponse:
    with db.connect() as conn:
        with conn.cursor() as cur:
            _, response = _swap_preflight(cur, lane_id, req.model_name, allow_unverified=req.allow_unverified)
        conn.commit()
    return response


@app.post("/api/lanes/{lane_id}/swap-model")
def api_lane_swap_model(lane_id: str, req: SwapModelRequest) -> dict[str, Any]:
    """
    Ask the worker gateway for a specific lane to swap its loaded model.
    The gateway updates the mesh-llama-launch config and restarts the systemd service.
    On success, updates current_model_name in the DB.
    """
    started = time.time()
    artifact_row: dict[str, Any] | None = None
    lane_state: dict[str, Any] | None = None
    source_mode: str | None = None
    preflight: SwapPreflightResponse | None = None
    tuning_profile: dict[str, Any] | None = None
    sibling_lanes: list[dict[str, Any]] = []
    reroute_results: list[dict[str, Any]] = []
    active_before_action: list[dict[str, Any]] = []
    suspension_reason = "exclusive swap profile active"
    swap_id: str | None = None
    mw_target: MwGrpcTarget | None = None
    host_id: str | None = None
    host_name: str | None = None
    target_lane_slot: str | None = None
    with db.connect() as conn:
        with conn.cursor() as cur:
            lane_state, preflight = _swap_preflight(
                cur,
                lane_id,
                req.model_name,
                allow_unverified=req.allow_unverified,
            )
            if not preflight.ok:
                conn.commit()
                raise HTTPException(status_code=409, detail=preflight.reason or "swap preflight failed")
            host_id = str(lane_state["host"]["host_id"])
            host_name = str(lane_state["host"]["host_name"])
            target_lane_slot = _lane_split_slot(lane_state["lane"])
            tuning_profile = _load_swap_tuning_profile(
                cur,
                host_id=host_id,
                lane_id=str(lane_state["lane"]["lane_id"]),
                model_name=preflight.model_name,
            )
            sibling_lanes = _list_host_sibling_lanes(
                cur,
                host_id=host_id,
                lane_id=str(lane_state["lane"]["lane_id"]),
            )
            impacted_sibling_lanes = (
                sibling_lanes
                if tuning_profile and bool(tuning_profile.get("disables_sibling_lanes") or tuning_profile.get("exclusive_host_resources"))
                else []
            )
            impacted_sibling_lane_ids = {str(row["lane_id"]) for row in impacted_sibling_lanes}
            affected_lane_ids = [str(preflight.lane_id)] + [str(row["lane_id"]) for row in impacted_sibling_lanes]
            if req.swap_urgency == "wait":
                active_before_action = _wait_for_no_active_leases(
                    cur,
                    lane_ids=affected_lane_ids,
                    timeout_s=req.wait_timeout_s,
                )
                if active_before_action:
                    conn.commit()
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error": "timed out waiting for active runs to finish before swap",
                            "active_leases": _summarize_active_leases(active_before_action),
                            "swap_urgency": req.swap_urgency,
                        },
                    )
            else:
                active_before_action = _list_active_router_leases(cur, affected_lane_ids)
                excluded_lane_ids = set(affected_lane_ids)
                for lease in active_before_action:
                    reroute = _reroute_displaced_lease(lease=lease, excluded_lane_ids=excluded_lane_ids)
                    reroute_results.append(
                        {
                            "original_lease_id": str(lease["lease_id"]),
                            "original_lane_id": str(lease["lane_id"]),
                            **reroute,
                        }
                    )
                    _record_displaced_request(
                        cur,
                        lease=lease,
                        replacement_lane_id=reroute.get("replacement_lane_id"),
                        status=str(reroute.get("status") or "captured"),
                        result_payload=reroute.get("result_payload"),
                        error_message=reroute.get("reason"),
                    )
                if active_before_action:
                    _mark_leases_canceled_for_swap(
                        cur,
                        leases=active_before_action,
                        reason=f"swap canceled active runs for {req.model_name}",
                    )
            if impacted_sibling_lanes:
                for sibling in impacted_sibling_lanes:
                    _set_lane_suspension(
                        cur,
                        lane_id=str(sibling["lane_id"]),
                        suspended=True,
                        reason=suspension_reason,
                    )
            source_mode = preflight.source_mode or "local"
            cur.execute(
                """
                SELECT hma.artifact_id, hma.model_id, hma.host_id, hma.local_path, h.host_name, h.mgmt_ssh_host, h.mgmt_ssh_user
                FROM host_model_artifacts hma
                JOIN models m ON m.model_id=hma.model_id
                JOIN hosts h ON h.host_id=hma.host_id
                WHERE m.model_name=%s AND hma.local_path=%s
                LIMIT 1
                """,
                (preflight.model_name, preflight.artifact_path),
            )
            artifact_row = cur.fetchone()
            mw_target = _mw_target_for_lane(cur=cur, lane_id=str(preflight.lane_id))
            swap_id = _create_lane_swap(
                cur,
                lane_id=str(preflight.lane_id),
                requested_model_name=req.model_name,
                resolved_model_name=preflight.model_name,
                source_mode=source_mode,
                details=_strip_nones(
                    {
                        "artifact_path": preflight.artifact_path,
                        "artifact_host": preflight.artifact_host,
                        "artifact_provider": preflight.artifact_provider,
                        "swap_strategy": preflight.swap_strategy,
                        "estimated_swap_ms": preflight.estimated_swap_ms,
                    }
                ),
            )
        conn.commit()
    if not lane_state or not preflight or not artifact_row:
        raise HTTPException(status_code=404, detail="artifact not found for swap")

    base_url = str(lane_state["lane"]["base_url"])
    impacted_sibling_lanes = (
        sibling_lanes
        if tuning_profile and bool(tuning_profile.get("disables_sibling_lanes") or tuning_profile.get("exclusive_host_resources"))
        else []
    )
    sibling_service_actions: list[dict[str, Any]] = []
    if impacted_sibling_lanes:
        try:
            with db.connect() as conn:
                with conn.cursor() as cur:
                    _record_lane_swap_event(
                        cur,
                        swap_id=str(swap_id),
                        event_type="stopping_siblings",
                        state="stopping_siblings",
                        details={"siblings": [str(row["lane_id"]) for row in impacted_sibling_lanes]},
                    )
                conn.commit()
        except Exception:
            pass
        for sibling in impacted_sibling_lanes:
            try:
                action_result = _call_lane_service_action(
                    base_url=str(sibling["base_url"]),
                    action="stop",
                    host_id=host_id,
                    lane_id=str(sibling["lane_id"]),
                )
                sibling_service_actions.append(
                    {
                        "lane_id": str(sibling["lane_id"]),
                        "base_url": str(sibling["base_url"]),
                        "action": "stop",
                        "ok": True,
                        "details": action_result,
                    }
                )
            except Exception as exc:
                sibling_service_actions.append(
                    {
                        "lane_id": str(sibling["lane_id"]),
                        "base_url": str(sibling["base_url"]),
                        "action": "stop",
                        "ok": False,
                        "error": str(exc),
                    }
                )
                raise

    ok = False
    err_kind = None
    err_msg = None
    data: dict[str, Any] | None = None
    current_model_name = str(lane_state["lane"].get("current_model_name") or "").strip()
    try:
        if current_model_name and _model_request_matches_candidate(preflight.model_name, current_model_name):
            if _lane_gateway_healthy(
                base_url,
                host_id=str(lane_state["host"]["host_id"]),
                lane_id=str(preflight.lane_id),
            ):
                data = {
                    "ok": True,
                    "model_name": preflight.model_name,
                    "model_path": preflight.artifact_path,
                    "model_alias": preflight.model_name,
                    "copy_time_ms": 0,
                    "load_time_ms": 0,
                    "ready_time_ms": 0,
                    "health_path": "/health",
                    "noop": True,
                    "reason": "requested model already loaded and healthy",
                }
                ok = True
        if not ok:
            if mw_target is not None and settings.mw_control_enabled:
                result = _mw_client().send_command(
                    host_id=mw_target.host_id,
                    message_type="load_model",
                    payload={"lane_id": mw_target.lane_id, "model_name": preflight.model_name},
                    wait=True,
                    timeout_seconds=max(30, settings.mw_command_timeout_seconds),
                )
                if not bool(result.get("ok", False)):
                    raise HTTPException(status_code=409, detail=str(result.get("error") or "MW load_model failed"))
                data = {
                    "ok": True,
                    "model_name": preflight.model_name,
                    "model_path": preflight.artifact_path,
                    "model_alias": preflight.model_name,
                    "copy_time_ms": 0,
                    "load_time_ms": 0,
                    "ready_time_ms": 0,
                    "control_plane": "mw",
                }
            else:
                _, payload = _build_swap_gateway_payload(
                    lane_state=lane_state,
                    preflight=preflight,
                    artifact_row=artifact_row,
                    requested_model_name=req.model_name,
                    swap_id=swap_id,
                )
                data = _call_lane_swap_gateway(base_url=base_url, payload=payload)
            ok = True
    except HTTPException as exc_http:
        err_kind = "swap_http_error"
        err_msg = str(exc_http.detail) if exc_http.detail is not None else str(data)
        raise
    except Exception as e:
        err_kind = "swap_proxy_error"
        err_msg = str(e)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        duration_ms = int((time.time() - started) * 1000)
        try:
            with db.connect() as conn:
                with conn.cursor() as cur:
                    if ok and artifact_row is not None:
                        cur.execute(
                            """
                            UPDATE lanes
                            SET current_model_name=%s,
                                status='ready',
                                suspension_reason=NULL,
                                updated_at=now()
                            WHERE lane_id=%s
                            """,
                            (preflight.model_name, preflight.lane_id),
                        )
                        _upsert_usage(
                            cur,
                            lane_id=preflight.lane_id,
                            model_id=str(artifact_row["model_id"]),
                            used_at=datetime.now(UTC),
                            swap_at=datetime.now(UTC),
                        )
                    if artifact_row is not None:
                        cur.execute(
                            """
                            INSERT INTO lane_model_swap_history (
                              lane_id, model_id, artifact_id, source_mode,
                              copy_time_ms, load_time_ms, duration_ms,
                              success, error_kind, error_message
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                preflight.lane_id,
                                artifact_row["model_id"],
                                artifact_row["artifact_id"],
                                source_mode,
                                data.get("copy_time_ms") if isinstance(data, dict) else None,
                                data.get("load_time_ms") if isinstance(data, dict) else None,
                                data.get("ready_time_ms") if isinstance(data, dict) and data.get("ready_time_ms") is not None else duration_ms,
                                ok,
                                err_kind,
                                err_msg,
                            ),
                        )
                    if swap_id:
                        _record_lane_swap_event(
                            cur,
                            swap_id=str(swap_id),
                            event_type="router_success" if ok else "router_failure",
                            state="ready" if ok else "failed",
                            details=data if isinstance(data, dict) else {"raw": data},
                            current_model_name=preflight.model_name if ok else None,
                            error_message=err_msg,
                        )
                    if sibling_lanes:
                        for sibling in sibling_lanes:
                            _set_lane_suspension(
                                cur,
                                lane_id=str(sibling["lane_id"]),
                                suspended=bool(ok and str(sibling["lane_id"]) in impacted_sibling_lane_ids),
                                reason=suspension_reason,
                            )
                conn.commit()
        except Exception:
            pass
        if (
            ok
            and sibling_lanes
            and not impacted_sibling_lanes
            and host_id
            and host_name
            and target_lane_slot
            and any(_lane_split_slot(sibling) is None for sibling in sibling_lanes)
        ):
            split_restore_actions = _restore_split_mode_for_host(
                host_id=host_id,
                host_name=host_name,
                overrides={target_lane_slot: preflight.model_name},
                skip_lane_id=str(preflight.lane_id),
            )
            sibling_service_actions.extend(split_restore_actions)

    return {
        "ok": True,
        "lane_id": preflight.lane_id,
        "model_name": preflight.model_name,
        "requested_model_name": req.model_name,
        "source_mode": source_mode,
        "swap_urgency": req.swap_urgency,
        "cost_tier": str(tuning_profile.get("cost_tier") or "standard") if tuning_profile else "standard",
        "storage_scheme": str(tuning_profile.get("storage_scheme") or "unknown") if tuning_profile else None,
        "active_leases_handled": _summarize_active_leases(active_before_action),
        "reroute_results": reroute_results,
        "sibling_service_actions": sibling_service_actions,
        "details": data,
        "swap_id": swap_id,
    }


@app.get("/api/lane-swaps/{swap_id}", response_model=LaneSwapResponse)
def api_lane_swap_status(swap_id: str) -> LaneSwapResponse:
    with db.connect() as conn:
        with conn.cursor() as cur:
            row = _fetch_lane_swap(cur, swap_id)
        conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="swap not found")
    return _serialize_lane_swap(row)


@app.post("/api/lane-swaps/{swap_id}/events", response_model=LaneSwapResponse)
def api_lane_swap_event(swap_id: str, req: Request, body: LaneSwapEventRequest) -> LaneSwapResponse:
    token = _bearer_token(req)
    if token != settings.swap_auth_token:
        raise HTTPException(status_code=403, detail="unauthorized")
    with db.connect() as conn:
        with conn.cursor() as cur:
            row = _record_lane_swap_event(
                cur,
                swap_id=swap_id,
                event_type=body.event_type,
                state=body.state,
                message=body.message,
                details=body.details,
                current_model_name=body.current_model_name,
                error_message=body.error_message,
            )
        conn.commit()
    return _serialize_lane_swap(row)


@app.get("/mesh/inventory")
def mesh_inventory() -> dict[str, Any]:
    """Best-effort inventory view: hosts, lanes, and known per-lane model options."""
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  h.host_name,
                  h.status as host_status,
                  l.lane_id,
                  l.lane_type,
                  l.base_url,
                  l.status as lane_status,
                  l.suspension_reason,
                  l.current_model_name,
                  l.last_ok_at,
                  l.last_probe_at
                FROM lanes l
                JOIN hosts h ON h.host_id=l.host_id
                ORDER BY h.host_name, l.lane_type, l.base_url
                """
            )
            lanes = cur.fetchall()
            # Allowed/known models per lane (policy + aliases).
            cur.execute(
                """
                SELECT l.lane_id, m.model_name, p.max_ctx
                FROM lane_model_policy p
                JOIN lanes l ON l.lane_id=p.lane_id
                JOIN models m ON m.model_id=p.model_id
                WHERE p.allowed=true
                """
            )
            policy = cur.fetchall()
            cur.execute(
                """
                SELECT a.lane_id, m.model_name
                FROM lane_model_aliases a
                JOIN models m ON m.model_id=a.model_id
                """
            )
            aliases = cur.fetchall()
            cur.execute(
                """
                SELECT DISTINCT ON (ls.lane_id)
                  ls.swap_id,
                  ls.lane_id,
                  ls.state,
                  ls.terminal,
                  ls.details,
                  ls.requested_model_name,
                  ls.resolved_model_name,
                  ls.error_message,
                  ls.started_at,
                  ls.last_event_at,
                  ls.completed_at,
                  ls.updated_at
                FROM lane_swaps ls
                WHERE NOT ls.terminal
                ORDER BY ls.lane_id, ls.updated_at DESC
                """
            )
            active_swaps = cur.fetchall()

    models_by_lane: dict[str, set[str]] = {}
    model_context_by_lane: dict[str, dict[str, int | None]] = {}
    for r in policy:
        lane_key = str(r["lane_id"])
        model_name = str(r["model_name"])
        models_by_lane.setdefault(lane_key, set()).add(model_name)
        model_context_by_lane.setdefault(lane_key, {})[model_name] = int(r["max_ctx"]) if r.get("max_ctx") is not None else None
    for r in aliases:
        models_by_lane.setdefault(str(r["lane_id"]), set()).add(str(r["model_name"]))
    active_swaps_by_lane = {str(r["lane_id"]): r for r in active_swaps}
    active_swap_siblings: dict[str, dict[str, Any]] = {}
    for swap in active_swaps:
        details = dict(swap.get("details") or {})
        sibling_ids = details.get("siblings") or details.get("affected_sibling_lanes") or []
        for sibling_id in sibling_ids:
            sibling_key = str(sibling_id or "").strip()
            if sibling_key:
                active_swap_siblings[sibling_key] = swap

    out = []
    for r in lanes:
        lane_id = str(r["lane_id"])
        known = sorted(models_by_lane.get(lane_id, set()))
        cm = (r.get("current_model_name") or "").strip()
        if cm and cm not in known:
            known.append(cm)
        active_swap = active_swaps_by_lane.get(lane_id)
        sibling_swap = active_swap_siblings.get(lane_id)
        lane_status = _display_lane_status(
            raw_status=str(r["lane_status"]),
            suspension_reason=str(r.get("suspension_reason") or "") or None,
            active_swap=active_swap or sibling_swap,
        )
        if sibling_swap and not active_swap:
            lane_status = "suspended"
        out.append(
            {
                "host": str(r["host_name"]),
                "host_status": str(r["host_status"]),
                "lane_id": lane_id,
                "lane_type": str(r["lane_type"]),
                "base_url": str(r["base_url"]),
                "lane_status": lane_status,
                "raw_lane_status": str(r["lane_status"]),
                "suspension_reason": str(r.get("suspension_reason") or "") or None,
                "current_model": cm or None,
                "known_models": known,
                "known_models_detail": [
                    {
                        "model_name": model_name,
                        "max_ctx": model_context_by_lane.get(lane_id, {}).get(model_name),
                    }
                    for model_name in known
                ],
                "last_ok_at": r.get("last_ok_at").isoformat() if r.get("last_ok_at") else None,
                "last_probe_at": r.get("last_probe_at").isoformat() if r.get("last_probe_at") else None,
                "active_swap": (
                    {
                        "swap_id": str(active_swap["swap_id"]),
                        "state": str(active_swap.get("state") or ""),
                        "requested_model_name": str(active_swap.get("requested_model_name") or ""),
                        "resolved_model_name": str(active_swap.get("resolved_model_name") or "") or None,
                        "error_message": str(active_swap.get("error_message") or "") or None,
                        "started_at": active_swap["started_at"].isoformat() if active_swap.get("started_at") else None,
                        "last_event_at": active_swap["last_event_at"].isoformat() if active_swap.get("last_event_at") else None,
                        "completed_at": active_swap["completed_at"].isoformat() if active_swap.get("completed_at") else None,
                        "updated_at": active_swap["updated_at"].isoformat() if active_swap.get("updated_at") else None,
                    }
                    if active_swap
                    else None
                ),
                "blocked_by_swap": (
                    {
                        "swap_id": str(sibling_swap["swap_id"]),
                        "state": str(sibling_swap.get("state") or ""),
                        "requested_model_name": str(sibling_swap.get("requested_model_name") or ""),
                        "resolved_model_name": str(sibling_swap.get("resolved_model_name") or "") or None,
                        "started_at": sibling_swap["started_at"].isoformat() if sibling_swap.get("started_at") else None,
                        "last_event_at": sibling_swap["last_event_at"].isoformat() if sibling_swap.get("last_event_at") else None,
                    }
                    if sibling_swap and not active_swap
                    else None
                ),
            }
        )
    return {"lanes": out}


@app.post("/api/inventory/host-scan")
def api_inventory_host_scan(req: HostInventoryScanRequest) -> dict[str, Any]:
    """Persist a host-local model inventory scan."""
    with db.connect() as conn:
        with conn.cursor() as cur:
            host_id, host_name = _resolve_host_id(cur, req.host_id, create=True)
            _update_host_inventory_metadata(
                cur,
                host_id=host_id,
                root_path=req.root_path,
                host_facts=req.host_facts,
                status="ready",
            )
            ingested = _ingest_artifacts(
                cur,
                host_id=host_id,
                artifacts=req.artifacts,
                storage_scope="local",
                storage_provider="local",
                root_path=req.root_path,
            )
        conn.commit()
    return {
        "ok": True,
        "host_id": host_id,
        "host_name": host_name,
        "artifact_count": len(ingested),
        "status": "ingested",
    }


@app.post("/api/inventory/archive-scan")
def api_inventory_archive_scan(req: ArchiveInventoryScanRequest) -> dict[str, Any]:
    """Persist an archive model inventory scan for packhub-style servers."""
    provider = (req.provider or req.archive_id or "archive").strip().lower()
    with db.connect() as conn:
        with conn.cursor() as cur:
            host_id, host_name = _resolve_host_id(cur, req.archive_id, create=True)
            _update_host_inventory_metadata(
                cur,
                host_id=host_id,
                root_path=req.root_path,
                host_facts=None,
                status="ready",
            )
            ingested = _ingest_artifacts(
                cur,
                host_id=host_id,
                artifacts=req.artifacts,
                storage_scope="archive",
                storage_provider=provider,
                root_path=req.root_path,
            )
        conn.commit()
    return {
        "ok": True,
        "archive_id": host_id,
        "archive_name": host_name,
        "provider": provider,
        "artifact_count": len(ingested),
        "status": "ingested",
    }


@app.get("/api/lanes/{lane_id}/capabilities", response_model=LaneCapabilityResponse)
def api_lane_capabilities(lane_id: str) -> LaneCapabilityResponse:
    """Return viable local and remote models for a lane."""
    with db.connect() as conn:
        with conn.cursor() as cur:
            _, response = _build_lane_capability_payload(cur, lane_id)
        conn.commit()
    return response


@app.get("/api/model-tuning-profiles", response_model=list[ModelTuningProfileResponse])
def api_model_tuning_profiles(
    host_ref: str | None = None,
    model_name: str | None = None,
    storage_scheme: str | None = None,
) -> list[ModelTuningProfileResponse]:
    with db.connect() as conn:
        with conn.cursor() as cur:
            clauses = []
            params: list[Any] = []
            if host_ref:
                clauses.append("(h.host_id::text=%s OR h.host_name::text=%s)")
                params.extend([host_ref, host_ref])
            if model_name:
                clauses.append("m.model_name=%s")
                params.append(model_name)
            if storage_scheme:
                clauses.append("tp.storage_scheme=%s")
                params.append(storage_scheme)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur.execute(
                f"""
                SELECT
                  tp.tuning_profile_id,
                  tp.host_id,
                  h.host_name,
                  tp.model_id,
                  m.model_name,
                  tp.lane_id,
                  l.lane_name,
                  l.lane_type,
                  tp.storage_scheme,
                  tp.settings,
                  tp.cost_tier,
                  tp.disables_sibling_lanes,
                  tp.exclusive_host_resources,
                  tp.prompt_tps,
                  tp.generation_tps,
                  tp.avg_total_latency_s,
                  tp.score,
                  tp.evaluation_count,
                  tp.source_run_tag,
                  tp.notes,
                  tp.created_at,
                  tp.updated_at
                FROM model_tuning_profiles tp
                JOIN hosts h ON h.host_id = tp.host_id
                JOIN models m ON m.model_id = tp.model_id
                LEFT JOIN lanes l ON l.lane_id = tp.lane_id
                {where_sql}
                ORDER BY tp.updated_at DESC, h.host_name, m.model_name
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        conn.commit()
    return [_row_to_tuning_profile(row) for row in rows]


@app.post("/api/model-tuning-profiles", response_model=ModelTuningProfileResponse)
def api_upsert_model_tuning_profile(req: ModelTuningProfileUpsertRequest) -> ModelTuningProfileResponse:
    with db.connect() as conn:
        with conn.cursor() as cur:
            response = _upsert_model_tuning_profile(cur, req)
        conn.commit()
    return response

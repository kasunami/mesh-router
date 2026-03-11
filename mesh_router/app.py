from __future__ import annotations

import time
import threading
from typing import Any, Literal

from datetime import UTC, datetime, timedelta

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Request
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from .config import settings
from .db import db
from .router import pick_lane_for_model
from .schemas import (
    ArchiveInventoryScanRequest,
    ChatCompletionRequest,
    HostInventoryScanRequest,
    LaneModelCandidate,
    LaneCapabilityResponse,
    ModelTuningProfileResponse,
    ModelTuningProfileUpsertRequest,
    ModelInfo,
    ModelsResponse,
    SwapPreflightResponse,
)
from .tokens import sign_token, verify_token
from .viability import ViabilityLaneInfo, ViabilityModelInfo, check_viability, estimate_swap_time


app = FastAPI(title="mesh-router", version="0.1.0")

ARCHIVE_PROVIDERS = {"packhub", "packhub02"}


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

    if seen_paths:
        cur.execute(
            """
            UPDATE host_model_artifacts
            SET present = false, updated_at = now()
            WHERE host_id=%s
              AND storage_scope=%s
              AND local_path <> ALL(%s)
            """,
            (host_id, storage_scope, list(seen_paths)),
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
        SELECT lane_id, host_id, lane_name, lane_type, base_url, current_model_name,
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


def _local_model_root(host_row: dict[str, Any] | None) -> str | None:
    if not host_row:
        return None
    paths = host_row.get("model_store_paths") or []
    if isinstance(paths, list) and paths:
        for path in paths:
            if isinstance(path, str) and path.strip():
                return path.strip()
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
    lane_type = str(lane_row["lane_type"])
    host_row = _resolve_host(cur, resolved_host_id)
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
          p.required_ram_bytes,
          p.required_vram_bytes,
          p.allowed
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
        target_context_tokens = None
        if tuning_profile and isinstance(tuning_profile.get("settings"), dict):
            ctx_value = tuning_profile["settings"].get("ctx_size")
            try:
                if ctx_value is not None:
                    target_context_tokens = int(ctx_value)
            except Exception:
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
        source_locality = "local" if str(row["host_id"]) == resolved_host_id else "remote"
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
    for candidate in sorted(candidates_by_model.values(), key=lambda item: item.model_name.lower()):
        if candidate.locality == "local":
            local_viable.append(candidate)
        elif candidate.locality == "remote":
            remote_viable.append(candidate)
        else:
            unverified.append(candidate)

    capabilities = ["chat"]
    if lane_type in ("cpu", "gpu", "mlx"):
        capabilities.append("inference")

    metadata = {
        "lane_type": lane_type,
        "lane_name": str(lane_row["lane_name"]),
        "host_name": str(host_row["host_name"]) if host_row else None,
        "memory_summary": {
            "usable_memory_bytes": lane_row.get("usable_memory_bytes"),
            "ram_budget_bytes": lane_row.get("ram_budget_bytes"),
            "vram_budget_bytes": lane_row.get("vram_budget_bytes"),
            "runtime_overhead_bytes": runtime_overhead,
            "reserved_headroom_bytes": reserved_headroom,
        },
        "archive_providers": sorted(ARCHIVE_PROVIDERS),
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
        "local_model_root": _local_model_root(host_row),
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


def _downstream_payload(req: ChatCompletionRequest) -> dict[str, Any]:
    raw = req.model_dump(by_alias=True)
    # Remove router-only hint fields.
    for k in list(raw.keys()):
        if k.startswith("mesh_") or k in {"extra_body"}:
            raw.pop(k, None)
    return _strip_nones(raw)


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
            cur.execute("SELECT model_name FROM models ORDER BY model_name")
            rows = cur.fetchall()
    data = [ModelInfo(id=str(r["model_name"])) for r in rows if _is_canonical(str(r["model_name"]))]
    resp = ModelsResponse(data=data)
    return resp.model_dump()


def _bearer_token(req: Request) -> str:
    auth = (req.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    return ""


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
        SELECT lane_id, lane_name, lane_type, base_url, status, suspension_reason, current_model_name
        FROM lanes
        WHERE host_id=%s AND lane_id<>%s
        ORDER BY lane_name
        """,
        (host_id, lane_id),
    )
    return list(cur.fetchall())


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
              WHEN status='offline' AND COALESCE(suspension_reason, '')=%s THEN 'ready'
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
    if route not in {"chat", "embeddings"} or not request_payload or not model_name:
        return {"status": "unsupported", "reason": "lease has no replayable request payload"}

    try:
        choice = pick_lane_for_model(
            model=model_name,
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
            downstream_model = model_name
            cur.execute(
                """
                SELECT downstream_model_name
                FROM lane_model_aliases
                WHERE lane_id=%s AND model_id=%s
                """,
                (choice.lane_id, model_id),
            )
            alias_row = cur.fetchone()
            if alias_row and alias_row.get("downstream_model_name"):
                downstream_model = str(alias_row["downstream_model_name"])
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
            endpoint = "/v1/chat/completions" if route == "chat" else "/v1/embeddings"
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
        _release_router_lease(lease_id=lease_id, ok=True)
        return {
            "status": "rerouted",
            "replacement_lane_id": choice.lane_id,
            "replacement_base_url": choice.base_url,
            "result_payload": body,
        }
    except Exception as exc:
        _release_router_lease(lease_id=lease_id, ok=False)
        return {
            "status": "reroute_failed",
            "replacement_lane_id": choice.lane_id,
            "reason": str(exc),
        }


def _call_lane_service_action(*, base_url: str, action: Literal["start", "stop", "restart"]) -> dict[str, Any]:
    with httpx.Client(timeout=180.0) as client:
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


@app.post("/v1/chat/completions")
def v1_chat_completions(
    req: ChatCompletionRequest,
    x_mesh_pin_worker: str | None = Header(default=None),
    x_mesh_pin_base_url: str | None = Header(default=None),
) -> dict[str, Any]:
    # Pins: header overrides body.
    pin_worker = x_mesh_pin_worker or req.mesh_pin_worker
    pin_base_url = x_mesh_pin_base_url or req.mesh_pin_base_url
    pin_lane_type = req.mesh_pin_lane_type

    try:
        choice = pick_lane_for_model(
            model=req.model,
            pin_worker=pin_worker,
            pin_base_url=pin_base_url,
            pin_lane_type=pin_lane_type,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    lease = None
    started = time.time()

    # Ensure model exists in our DB for metrics and resolve any per-lane downstream alias.
    downstream_model = req.model
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO models (model_name, format)
                VALUES (%s, 'other'::model_format)
                ON CONFLICT (model_name) DO UPDATE SET updated_at=now()
                """,
                (req.model,),
            )
            cur.execute("SELECT model_id FROM models WHERE model_name=%s", (req.model,))
            model_id = cur.fetchone()["model_id"]
            lane_id = choice.lane_id or None

            if lane_id is not None:
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
                    downstream_model = str(alias_row["downstream_model_name"])
        conn.commit()

    ok = True
    err_kind = None
    err_msg = None
    resp_data: dict[str, Any] | None = None
    tps: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    try:
        payload = _downstream_payload(req)
        payload["model"] = downstream_model
        # Acquire a router lease (the only source of truth). The worker gateway will
        # call back to /api/router-leases/validate with this token.
        lease_id, expires_at = _acquire_router_lease(
            lane_id=choice.lane_id,
            model_id=str(model_id),
            owner=settings.default_owner,
            job_type=settings.default_job_type,
            ttl_seconds=settings.default_lease_ttl_seconds,
            details={
                "worker_id": choice.worker_id,
                "base_url": choice.base_url,
                "downstream_model": downstream_model,
                "route": "chat",
                "request_payload": payload,
            },
        )
        lease = {
            "lease_id": lease_id,
            "expires_at": expires_at.isoformat(),
        }
        token = sign_token(
            {
                "lease_id": lease_id,
                "lane_id": choice.lane_id,
                "worker_id": choice.worker_id,
                "base_url": choice.base_url,
                "model": downstream_model,
                "owner": settings.default_owner,
                "exp": int(expires_at.timestamp()),
            }
        )
        # Apply downstream model alias if needed (must match leased model for worker gateway validation).
        # Proxy directly to the worker gateway (11434/11435). Backends are bound to localhost.
        stop_heartbeat = threading.Event()
        heartbeat_error: dict[str, str] = {}

        def _heartbeat_loop() -> None:
            interval = max(5, int(settings.default_lease_heartbeat_interval_seconds))
            while not stop_heartbeat.wait(interval):
                try:
                    _heartbeat_router_lease(lease_id=lease_id)
                except Exception as exc:
                    heartbeat_error["error"] = str(exc)
                    break

        heartbeat_thread = threading.Thread(target=_heartbeat_loop, name=f"lease-heartbeat-{lease_id}", daemon=True)
        heartbeat_thread.start()
        try:
            with httpx.Client(timeout=float(max(30, settings.default_lease_ttl_seconds))) as client:
                r = client.post(
                    f"{choice.base_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=2)
        if heartbeat_error:
            raise RuntimeError(f"lease heartbeat failed: {heartbeat_error['error']}")
        try:
            resp_data = r.json()
        except Exception:
            resp_data = {"raw": r.text}
        if r.status_code >= 400:
            raise RuntimeError(f"worker proxy http_{r.status_code}: {resp_data}")

        # Best-effort parse of llama.cpp timings/usage when available.
        try:
            usage = (resp_data or {}).get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens")) if usage.get("prompt_tokens") is not None else None
            completion_tokens = int(usage.get("completion_tokens")) if usage.get("completion_tokens") is not None else None
        except Exception:
            pass
        try:
            timings = (((resp_data or {}).get("timings") or {}) if isinstance(resp_data, dict) else {})
            if timings.get("predicted_per_second") is not None:
                tps = float(timings["predicted_per_second"])
        except Exception:
            pass

        return resp_data
    except Exception as e:
        ok = False
        err_kind = "proxy_error"
        err_msg = str(e)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        elapsed_ms = int((time.time() - started) * 1000)
        # Metrics: best-effort insert.
        try:
            with db.connect() as conn:
                with conn.cursor() as cur:
                    if lane_id is not None:
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
                                "mesh-router:chat",
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
            pass
        try:
            if lease is not None:
                _release_router_lease(lease_id=str(lease["lease_id"]), ok=ok)
        except Exception:
            pass


@app.post("/v1/embeddings")
def v1_embeddings(
    req: Request,
    body: dict[str, Any] = Body(default_factory=dict),
    x_mesh_pin_worker: str | None = Header(default=None),
    x_mesh_pin_base_url: str | None = Header(default=None),
) -> dict[str, Any]:
    model_name = str((body or {}).get("model") or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")
    pin_worker = x_mesh_pin_worker or body.get("mesh_pin_worker")
    pin_base_url = x_mesh_pin_base_url or body.get("mesh_pin_base_url")
    pin_lane_type = body.get("mesh_pin_lane_type")

    try:
        choice = pick_lane_for_model(
            model=model_name,
            pin_worker=pin_worker,
            pin_base_url=pin_base_url,
            pin_lane_type=pin_lane_type,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    lease = None
    started = time.time()
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
            model_id = cur.fetchone()["model_id"]
            lane_id = choice.lane_id or None
            if lane_id is not None:
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
                    downstream_model = str(alias_row["downstream_model_name"])
        conn.commit()

    ok = True
    err_kind = None
    err_msg = None
    resp_data: dict[str, Any] | None = None
    tps: float | None = None
    prompt_tokens: int | None = None
    completion_tokens = 0
    try:
        payload = dict(body or {})
        payload["model"] = downstream_model
        for k in list(payload.keys()):
            if k.startswith("mesh_"):
                payload.pop(k, None)
        lease_id, expires_at = _acquire_router_lease(
            lane_id=choice.lane_id,
            model_id=str(model_id),
            owner=settings.default_owner,
            job_type=settings.default_job_type,
            ttl_seconds=settings.default_lease_ttl_seconds,
            details={
                "worker_id": choice.worker_id,
                "base_url": choice.base_url,
                "downstream_model": downstream_model,
                "route": "embeddings",
                "request_payload": payload,
            },
        )
        lease = {
            "lease_id": lease_id,
            "expires_at": expires_at.isoformat(),
        }
        token = sign_token(
            {
                "lease_id": lease_id,
                "lane_id": choice.lane_id,
                "worker_id": choice.worker_id,
                "base_url": choice.base_url,
                "model": downstream_model,
                "owner": settings.default_owner,
                "exp": int(expires_at.timestamp()),
            }
        )
        stop_heartbeat = threading.Event()
        heartbeat_error: dict[str, str] = {}

        def _heartbeat_loop() -> None:
            interval = max(5, int(settings.default_lease_heartbeat_interval_seconds))
            while not stop_heartbeat.wait(interval):
                try:
                    _heartbeat_router_lease(lease_id=lease_id)
                except Exception as exc:
                    heartbeat_error["error"] = str(exc)
                    break

        heartbeat_thread = threading.Thread(target=_heartbeat_loop, name=f"lease-heartbeat-{lease_id}", daemon=True)
        heartbeat_thread.start()
        try:
            with httpx.Client(timeout=float(max(30, settings.default_lease_ttl_seconds))) as client:
                r = client.post(
                    f"{choice.base_url.rstrip('/')}/v1/embeddings",
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=2)
        if heartbeat_error:
            raise RuntimeError(f"lease heartbeat failed: {heartbeat_error['error']}")
        try:
            resp_data = r.json()
        except Exception:
            resp_data = {"raw": r.text}
        if r.status_code >= 400:
            raise RuntimeError(f"worker proxy http_{r.status_code}: {resp_data}")

        usage = (resp_data or {}).get("usage") or {}
        if usage.get("prompt_tokens") is not None:
            prompt_tokens = int(usage["prompt_tokens"])
        elif usage.get("total_tokens") is not None:
            prompt_tokens = int(usage["total_tokens"])
        elapsed_ms = max(1, int((time.time() - started) * 1000))
        if prompt_tokens:
            tps = round((float(prompt_tokens) * 1000.0) / float(elapsed_ms), 6)
        return resp_data
    except Exception as e:
        ok = False
        err_kind = "proxy_error"
        err_msg = str(e)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        elapsed_ms = int((time.time() - started) * 1000)
        try:
            with db.connect() as conn:
                with conn.cursor() as cur:
                    if lane_id is not None:
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
                                "mesh-router:embeddings",
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
            pass
        try:
            if lease is not None:
                _release_router_lease(lease_id=str(lease["lease_id"]), ok=ok)
        except Exception:
            pass


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
    tuning_profile = _load_swap_tuning_profile(
        cur,
        host_id=str(state["host"]["host_id"]),
        lane_id=str(state["lane"]["lane_id"]),
        model_name=model_name,
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
    for candidate in capabilities.local_viable_models + capabilities.remote_viable_models:
        if candidate.model_name == model_name:
            source_mode = "local" if candidate.locality == "local" else "remote_copy_then_load"
            metadata: dict[str, Any] = {"capability_source": candidate.locality}
            if candidate.locality == "remote":
                metadata["local_model_root"] = state["local_model_root"]
            metadata["swap_cost"] = _swap_cost_metadata(
                lane_state=state,
                model_name=model_name,
                tuning_profile=tuning_profile,
                sibling_lanes=sibling_lanes,
                active_leases=active_leases,
            )
            return state, SwapPreflightResponse(
                lane_id=capabilities.lane_id,
                model_name=model_name,
                ok=True,
                source_mode=source_mode,
                artifact_path=candidate.artifact_path,
                artifact_host=candidate.artifact_host,
                artifact_provider=candidate.artifact_provider or None,
                estimated_swap_ms=candidate.estimated_swap_ms,
                swap_strategy=candidate.swap_strategy,
                metadata=_strip_nones(metadata),
            )
    for candidate in capabilities.unverified_models:
        if candidate.model_name == model_name:
            if allow_unverified:
                source_mode = "local" if candidate.artifact_host == capabilities.metadata.get("host_name") else "remote_copy_then_load"
                return state, SwapPreflightResponse(
                    lane_id=capabilities.lane_id,
                    model_name=model_name,
                    ok=True,
                    source_mode=source_mode,
                    artifact_path=candidate.artifact_path,
                    artifact_host=candidate.artifact_host,
                    artifact_provider=candidate.artifact_provider or None,
                    estimated_swap_ms=candidate.estimated_swap_ms,
                    swap_strategy=candidate.swap_strategy,
                    reason="allow_unverified override applied",
                    metadata={
                        "override": "allow_unverified",
                        "swap_cost": _swap_cost_metadata(
                            lane_state=state,
                            model_name=model_name,
                            tuning_profile=tuning_profile,
                            sibling_lanes=sibling_lanes,
                            active_leases=active_leases,
                        ),
                    },
                )
            return state, SwapPreflightResponse(
                lane_id=capabilities.lane_id,
                model_name=model_name,
                ok=False,
                artifact_path=candidate.artifact_path,
                artifact_host=candidate.artifact_host,
                artifact_provider=candidate.artifact_provider or None,
                estimated_swap_ms=candidate.estimated_swap_ms,
                swap_strategy=candidate.swap_strategy,
                reason=candidate.reason or "model is unverified for this lane",
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
            tuning_profile = _load_swap_tuning_profile(
                cur,
                host_id=str(lane_state["host"]["host_id"]),
                lane_id=str(lane_state["lane"]["lane_id"]),
                model_name=req.model_name,
            )
            sibling_lanes = _list_host_sibling_lanes(
                cur,
                host_id=str(lane_state["host"]["host_id"]),
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
                (req.model_name, preflight.artifact_path),
            )
            artifact_row = cur.fetchone()
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
        for sibling in impacted_sibling_lanes:
            try:
                action_result = _call_lane_service_action(base_url=str(sibling["base_url"]), action="stop")
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

    payload: dict[str, Any] = {
        "model_name": req.model_name,
        "model_path": preflight.artifact_path,
        "model_alias": req.model_name,
        "source_mode": source_mode,
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
        payload["copy_destination"] = f"{str(local_model_root).rstrip('/')}/{req.model_name}"

    ok = False
    err_kind = None
    err_msg = None
    data: dict[str, Any] | None = None
    try:
        with httpx.Client(timeout=180.0) as client:
            r = client.post(
                f"{base_url.rstrip('/')}/swap-model",
                json=payload,
                headers={"Authorization": f"Bearer {settings.swap_auth_token}"},
            )
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text}
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=str(data))
            ok = True
    except HTTPException:
        err_kind = "swap_http_error"
        err_msg = str(data)
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
                            "UPDATE lanes SET current_model_name=%s, updated_at=now() WHERE lane_id=%s",
                            (req.model_name, preflight.lane_id),
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
        if ok and sibling_lanes and not impacted_sibling_lanes:
            for sibling in sibling_lanes:
                try:
                    action_result = _call_lane_service_action(base_url=str(sibling["base_url"]), action="start")
                    sibling_service_actions.append(
                        {
                            "lane_id": str(sibling["lane_id"]),
                            "base_url": str(sibling["base_url"]),
                            "action": "start",
                            "ok": True,
                            "details": action_result,
                        }
                    )
                except Exception as exc:
                    sibling_service_actions.append(
                        {
                            "lane_id": str(sibling["lane_id"]),
                            "base_url": str(sibling["base_url"]),
                            "action": "start",
                            "ok": False,
                            "error": str(exc),
                        }
                    )

    return {
        "ok": True,
        "lane_id": preflight.lane_id,
        "model_name": req.model_name,
        "source_mode": source_mode,
        "swap_urgency": req.swap_urgency,
        "cost_tier": str(tuning_profile.get("cost_tier") or "standard") if tuning_profile else "standard",
        "storage_scheme": str(tuning_profile.get("storage_scheme") or "unknown") if tuning_profile else None,
        "active_leases_handled": _summarize_active_leases(active_before_action),
        "reroute_results": reroute_results,
        "sibling_service_actions": sibling_service_actions,
        "details": data,
    }


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
                SELECT l.lane_id, m.model_name
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

    models_by_lane: dict[str, set[str]] = {}
    for r in policy + aliases:
        models_by_lane.setdefault(str(r["lane_id"]), set()).add(str(r["model_name"]))

    out = []
    for r in lanes:
        lane_id = str(r["lane_id"])
        known = sorted(models_by_lane.get(lane_id, set()))
        cm = (r.get("current_model_name") or "").strip()
        if cm and cm not in known:
            known.append(cm)
        out.append(
            {
                "host": str(r["host_name"]),
                "host_status": str(r["host_status"]),
                "lane_id": lane_id,
                "lane_type": str(r["lane_type"]),
                "base_url": str(r["base_url"]),
                "lane_status": str(r["lane_status"]),
                "current_model": cm or None,
                "known_models": known,
                "last_ok_at": r.get("last_ok_at").isoformat() if r.get("last_ok_at") else None,
                "last_probe_at": r.get("last_probe_at").isoformat() if r.get("last_probe_at") else None,
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

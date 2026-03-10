from __future__ import annotations

import time
from typing import Any

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
        required_memory_bytes = int(
            row["required_vram_bytes"] if lane_type == "gpu" else row["required_ram_bytes"]
        ) if (row.get("required_vram_bytes") is not None or row.get("required_ram_bytes") is not None) else None
        tps_estimate = _latest_tps(cur, resolved_lane_id, model_id)
        model_info = ViabilityModelInfo(
            model_id=model_id,
            model_name=model_name,
            size_bytes=size_bytes,
            required_ram_bytes=required_memory_bytes if lane_type != "gpu" else None,
            required_vram_bytes=required_memory_bytes if lane_type == "gpu" else None,
            estimated_tps=tps_estimate,
        )
        viability = check_viability(lane_info, model_info)
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
            current_score = (
                0 if locality == "local" else 1 if locality == "remote" else 2,
                candidate.estimated_swap_ms or 10**12,
            )
            previous_score = (
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


def _cleanup_expired_router_leases(cur) -> None:
    cur.execute(
        "UPDATE router_leases SET state='expired' WHERE state='active' AND expires_at < now()"
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
    expires_at = datetime.now(UTC) + timedelta(seconds=max(30, int(ttl_seconds)))
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
                INSERT INTO router_leases (lane_id, model_id, owner, job_type, state, expires_at, details)
                VALUES (%s, %s, %s, %s, 'active', %s, %s::jsonb)
                RETURNING lease_id
                """,
                (lane_id, model_id, owner, job_type, expires_at, Jsonb(details)),
            )
            lease_id = str(cur.fetchone()["lease_id"])
        conn.commit()
    return lease_id, expires_at


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
        payload = _downstream_payload(req)
        # Apply downstream model alias if needed (must match leased model for worker gateway validation).
        payload["model"] = downstream_model
        # Proxy directly to the worker gateway (11434/11435). Backends are bound to localhost.
        with httpx.Client(timeout=600.0) as client:
            r = client.post(
                f"{choice.base_url.rstrip('/')}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
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


class SwapModelRequest(BaseModel):
    model_name: str
    allow_unverified: bool = False


def _swap_preflight(
    cur,
    lane_ref: str,
    model_name: str,
    *,
    allow_unverified: bool = False,
) -> tuple[dict[str, Any], SwapPreflightResponse]:
    state, capabilities = _build_lane_capability_payload(cur, lane_ref)
    for candidate in capabilities.local_viable_models + capabilities.remote_viable_models:
        if candidate.model_name == model_name:
            source_mode = "local" if candidate.locality == "local" else "remote_copy_then_load"
            metadata: dict[str, Any] = {"capability_source": candidate.locality}
            if candidate.locality == "remote":
                metadata["local_model_root"] = state["local_model_root"]
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
                    metadata={"override": "allow_unverified"},
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
            )
    return state, SwapPreflightResponse(
        lane_id=capabilities.lane_id,
        model_name=model_name,
        ok=False,
        reason="model is not viable for this lane",
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
                conn.commit()
        except Exception:
            pass

    return {
        "ok": True,
        "lane_id": preflight.lane_id,
        "model_name": req.model_name,
        "source_mode": source_mode,
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

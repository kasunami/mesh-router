from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def _normalize_router_backend_type(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"llama.cpp", "llama"}:
        return "llama"
    if raw in {"bitnet.cpp", "bitnet"}:
        return "bitnet"
    if raw in {"stable-diffusion.cpp", "sd", "stable-diffusion"}:
        return "sd"
    if raw == "mlx":
        return "mlx"
    return raw


def _normalize_mw_host_id(host_name: str | None) -> str:
    return str(host_name or "").strip().lower().replace(" ", "-")


def _candidate_mw_binding(row: dict[str, Any]) -> tuple[str, str, bool] | None:
    pam = row.get("proxy_auth_metadata") or {}
    if isinstance(pam, dict) and str(pam.get("control_plane") or "").strip().lower() == "mw":
        host_id = str(pam.get("mw_host_id") or "").strip() or _normalize_mw_host_id(str(row.get("host_name") or ""))
        lane_id = str(pam.get("mw_lane_id") or "").strip()
        if not lane_id:
            lane_name = str(row.get("lane_name") or "").strip()
            lane_type = str(row.get("lane_type") or "").strip().lower()
            lane_id = lane_name or (lane_type if lane_type in {"cpu", "gpu", "combined", "mlx"} else "")
        if host_id and lane_id:
            return host_id, lane_id, False
        return None

    # Legacy MR rows may not be explicitly tagged MW-managed even though MW publishes state for the host/lane.
    host_id = _normalize_mw_host_id(str(row.get("host_name") or ""))
    lane_name = str(row.get("lane_name") or "").strip()
    lane_type = str(row.get("lane_type") or "").strip().lower()
    inferred_lane_id = lane_name or (lane_type if lane_type in {"cpu", "gpu", "combined", "mlx"} else "")
    if host_id and inferred_lane_id:
        return host_id, inferred_lane_id, True
    return None


def _mw_effective_status_and_reason(
    fact: dict[str, Any],
    *,
    stale_cutoff: datetime,
) -> tuple[str, str | None]:
    hb = fact.get("last_heartbeat_at")
    actual_state = str(fact.get("actual_state") or "").strip()
    health_status = str(fact.get("health_status") or "").strip()

    if not fact:
        return "offline", "mw_state_missing"
    if not hb or not isinstance(hb, datetime):
        return "offline", "stale_heartbeat"
    if hb < stale_cutoff:
        return "offline", "stale_heartbeat"
    if actual_state not in {"running", "ready"}:
        return "offline", "not_running"
    if health_status != "healthy":
        return "offline", "unhealthy"
    return "ready", None


def apply_mw_effective_status(
    rows: list[dict[str, Any]],
    *,
    mw_state_db: Any,
    stale_seconds: int,
) -> None:
    """
    Enrich lane rows (from the router DB) with MW-derived readiness + current model when lanes are
    MW-managed and MW state lives in a separate DB.

    This is a *best-effort overlay*:
    - If MW state DB is unavailable, rows are marked offline with a reason.
    - Rows are modified in place with:
      - effective_status: "ready" | "offline" (when MW-managed lanes can be interpreted)
      - readiness_reason: machine-readable exclusion reason when not ready
      - current_model_name: set to MW `actual_model` when present (for more truthful inventory)
    """
    mw_pairs: list[tuple[str, str]] = []
    for row in rows:
        binding = _candidate_mw_binding(row)
        if binding is not None:
            mw_pairs.append((binding[0], binding[1]))
    if not mw_pairs:
        return

    facts: dict[tuple[str, str], dict[str, Any]] = {}
    explicit_bindings: set[tuple[str, str]] = set()
    for row in rows:
        binding = _candidate_mw_binding(row)
        if binding is None:
            continue
        if not binding[2]:
            explicit_bindings.add((binding[0], binding[1]))
    try:
        with mw_state_db.connect() as conn:
            with conn.cursor() as cur:
                values_sql = ",".join(["(%s,%s)"] * len(mw_pairs))
                params: list[Any] = []
                for host_id, lane_id in mw_pairs:
                    params.extend([host_id, lane_id])
                cur.execute(
                    f"""
                    WITH wanted(host_id, lane_id) AS (VALUES {values_sql})
                    SELECT
                      w.host_id,
                      w.lane_id,
                      mh.last_heartbeat_at,
                      ml.actual_state,
                      ml.health_status,
                      ml.actual_model,
                      ml.backend_type
                    FROM wanted w
                    LEFT JOIN mw_hosts mh ON mh.host_id = w.host_id
                    LEFT JOIN mw_lanes ml ON ml.host_id = w.host_id AND ml.lane_id = w.lane_id
                    """,
                    tuple(params),
                )
                for r in cur.fetchall():
                    key = (str(r["host_id"]), str(r["lane_id"]))
                    facts[key] = dict(r)
    except Exception:
        for row in rows:
            binding = _candidate_mw_binding(row)
            if binding is None:
                continue
            if binding[2]:
                continue
            row["effective_status"] = "offline"
            row["readiness_reason"] = "mw_state_unavailable"
        return

    now = datetime.now(tz=UTC)
    stale_cutoff = now - timedelta(seconds=int(stale_seconds))
    for row in rows:
        binding = _candidate_mw_binding(row)
        if binding is None:
            continue
        host_id, lane_id, inferred = binding
        f = facts.get((host_id, lane_id)) or {}
        if inferred and not any(
            f.get(key) is not None
            for key in ("last_heartbeat_at", "actual_state", "health_status", "actual_model", "backend_type")
        ):
            continue
        effective_status, readiness_reason = _mw_effective_status_and_reason(
            f,
            stale_cutoff=stale_cutoff,
        )
        row_backend = _normalize_router_backend_type(str(row.get("backend_type") or ""))
        fact_backend = _normalize_router_backend_type(str(f.get("backend_type") or ""))
        shared_explicit_binding = inferred and (host_id, lane_id) in explicit_bindings
        if shared_explicit_binding and row_backend and fact_backend and row_backend != fact_backend:
            row["effective_status"] = "offline"
            row["readiness_reason"] = "backend_mismatch"
            continue
        if not inferred and row_backend and fact_backend and row_backend != fact_backend:
            row["effective_status"] = "offline"
            row["readiness_reason"] = "backend_mismatch"
            continue
        row["effective_status"] = effective_status
        row["readiness_reason"] = readiness_reason

        if f.get("actual_model"):
            row["current_model_name"] = f.get("actual_model")
        if f.get("backend_type"):
            row["backend_type"] = _normalize_router_backend_type(str(f.get("backend_type") or ""))

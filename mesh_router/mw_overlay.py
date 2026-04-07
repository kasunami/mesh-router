from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


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
        pam = row.get("proxy_auth_metadata") or {}
        if not isinstance(pam, dict):
            continue
        if str(pam.get("control_plane") or "") != "mw":
            continue
        host_id = str(pam.get("mw_host_id") or "").strip()
        lane_id = str(pam.get("mw_lane_id") or "").strip()
        if host_id and lane_id:
            mw_pairs.append((host_id, lane_id))
    if not mw_pairs:
        return

    facts: dict[tuple[str, str], dict[str, Any]] = {}
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
                      ml.actual_model
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
            pam = row.get("proxy_auth_metadata") or {}
            if not isinstance(pam, dict):
                continue
            if str(pam.get("control_plane") or "") != "mw":
                continue
            row["effective_status"] = "offline"
            row["readiness_reason"] = "mw_state_unavailable"
        return

    now = datetime.now(tz=UTC)
    stale_cutoff = now - timedelta(seconds=int(stale_seconds))
    for row in rows:
        pam = row.get("proxy_auth_metadata") or {}
        if not isinstance(pam, dict):
            continue
        if str(pam.get("control_plane") or "") != "mw":
            continue
        host_id = str(pam.get("mw_host_id") or "").strip()
        lane_id = str(pam.get("mw_lane_id") or "").strip()
        if not host_id or not lane_id:
            continue
        f = facts.get((host_id, lane_id)) or {}
        effective_status, readiness_reason = _mw_effective_status_and_reason(
            f,
            stale_cutoff=stale_cutoff,
        )
        row["effective_status"] = effective_status
        row["readiness_reason"] = readiness_reason

        if f.get("actual_model"):
            row["current_model_name"] = f.get("actual_model")

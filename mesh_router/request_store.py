from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from psycopg.types.json import Jsonb

from .config import settings
from .db import db
from .lease_store import cleanup_expired_router_leases


REQUEST_TERMINAL_STATES = {"released", "failed", "expired", "canceled"}


def cleanup_expired_router_requests(cur: Any) -> None:
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


def create_router_request(
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


def touch_router_request(*, request_id: str, state: str | None = None, **fields: Any) -> None:
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
            if any(part.split("=")[0] == "released_at" for part in set_parts):
                continue
        set_parts.append(f"{key}=%s")
        params.append(value)
    set_parts.append("updated_at=now()")
    with db.connect() as conn:
        with conn.cursor() as cur:
            cleanup_expired_router_requests(cur)
            cur.execute(
                f"""
                UPDATE router_requests
                SET {", ".join(set_parts)}
                WHERE request_id=%s
                """,
                tuple(params + [request_id]),
            )
        conn.commit()


def request_cancel_requested(request_id: str) -> bool:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cleanup_expired_router_requests(cur)
            cur.execute("SELECT cancel_requested FROM router_requests WHERE request_id=%s", (request_id,))
            row = cur.fetchone()
    return bool((row or {}).get("cancel_requested"))


def fetch_router_request(request_id: str) -> dict[str, Any] | None:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cleanup_expired_router_leases(cur)
            cleanup_expired_router_requests(cur)
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


def serialize_router_request(row: dict[str, Any]) -> dict[str, Any]:
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


def router_request_health(row: dict[str, Any]) -> dict[str, Any]:
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

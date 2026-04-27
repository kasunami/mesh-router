from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg.types.json import Jsonb

from .config import settings
from .db import db


def cleanup_expired_router_leases(cur: Any) -> None:
    cur.execute(
        """
        UPDATE router_leases
        SET state='expired'
        WHERE state='active'
          AND COALESCE(last_heartbeat_at, acquired_at) < now() - (%s * interval '1 second')
        """,
        (settings.default_lease_stale_seconds,),
    )


def list_active_router_leases(cur: Any, lane_ids: list[str]) -> list[dict[str, Any]]:
    if not lane_ids:
        return []
    cleanup_expired_router_leases(cur)
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


def acquire_router_lease(
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
            cleanup_expired_router_leases(cur)
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


def heartbeat_router_lease(*, lease_id: str) -> None:
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


def release_router_lease(*, lease_id: str, ok: bool) -> None:
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

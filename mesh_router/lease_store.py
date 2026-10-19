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
            # Serialize reservation across siblings and qualified dispatch.
            # A qualified request holds the matching session lock for its
            # entire load/inference lifetime; legacy callers fail closed.
            cur.execute("SELECT host_id FROM lanes WHERE lane_id=%s", (lane_id,))
            host = cur.fetchone()
            if not host:
                raise RuntimeError("lane missing")
            cur.execute("SELECT pg_try_advisory_xact_lock(hashtextextended(%s,0)) AS locked",
                        ("mr-host:" + str(host["host_id"]),))
            if not cur.fetchone()["locked"]:
                raise RuntimeError("lane busy: host reserved")
            cleanup_expired_router_leases(cur)
            cur.execute(
                """
                SELECT 1
                FROM router_leases rl JOIN lanes l USING(lane_id)
                WHERE l.host_id=%s AND rl.state='active' AND rl.expires_at > now()
                LIMIT 1
                """,
                (str(host["host_id"]),),
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
            cur.execute(
                """UPDATE router_requests rr
                   SET last_heartbeat_at=rl.last_heartbeat_at,
                       expires_at=rl.expires_at, updated_at=now()
                   FROM router_leases rl
                   WHERE rr.lease_id=rl.lease_id AND rl.lease_id=%s
                     AND rl.job_type='qualified_chat' AND rl.state='active'
                     AND rr.state IN ('acquired','running')""", (lease_id,))
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
            cur.execute(
                """UPDATE router_requests rr
                   SET state=CASE WHEN rr.cancel_requested THEN 'canceled'::request_state
                                  ELSE rl.state::text::request_state END,
                       released_at=rl.released_at, updated_at=now()
                   FROM router_leases rl
                   WHERE rr.lease_id=rl.lease_id AND rl.lease_id=%s
                     AND rl.job_type='qualified_chat'
                     AND rl.state IN ('released','failed')
                     AND rr.state IN ('acquired','running')""", (lease_id,))
        conn.commit()


def require_idle_control_host(mw_host_id: str) -> None:
    """Reject disruptive control commands while any sibling has a live lease.

    This is an admission check, not a replacement for the worker's own lifecycle
    serialization. Internal qualified loading already owns its host reservation.
    """
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT 1 FROM router_leases rl
                JOIN lanes l USING(lane_id) JOIN hosts h USING(host_id)
                WHERE lower(h.host_name::text)=lower(%s)
                  AND rl.state='active' AND rl.expires_at>now() LIMIT 1""",
                (mw_host_id,))
            if cur.fetchone():
                raise RuntimeError("host busy: active router lease protects inference")

from __future__ import annotations

import time
from typing import Any

import httpx

from .config import settings
from .db import db, q


def _probe_lane_health(base_url: str) -> tuple[bool, int | None, float, str | None]:
    """Probe the worker gateway health endpoint (no auth).
    Returns (ok, status_code, latency_ms, error_msg)
    """
    url = f"{base_url.rstrip('/')}/healthz"
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=3.5) as c:
            r = c.get(url)
            # Fallback for standalone Ollama which doesn't have /healthz
            if r.status_code == 404:
                r = c.get(f"{base_url.rstrip('/')}/api/tags")
        
        latency_ms = (time.perf_counter() - start) * 1000
        if r.status_code == 200:
            return True, r.status_code, latency_ms, None
        return False, r.status_code, latency_ms, f"http_{r.status_code}"
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, None, latency_ms, str(e)


def _probe_lane_model(base_url: str) -> str | None:
    """Try to detect currently loaded model on the lane."""
    try:
        # After confirming base_url is healthy, sync current_model_name from Ollama/llama.cpp
        with httpx.Client(timeout=3.0) as client:
            # Try Ollama-style /api/tags first, then OpenAI-style /v1/models
            for probe_path in ["/api/tags", "/v1/models"]:
                try:
                    r = client.get(f"{base_url.rstrip('/')}{probe_path}")
                    if r.status_code == 200:
                        data = r.json()
                        # Ollama: {"models": [{"name": "qwen2.5-9b:latest",...}]}
                        # llama.cpp: {"data": [{"id": "qwen2.5-9b",...}]}
                        models = data.get("models") or data.get("data") or []
                        if models:
                            loaded = (models[0].get("name") or models[0].get("id") or "").split(":")[0]
                            if loaded:
                                return loaded
                        break
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _enforce_dualboot_mutual_exclusion(cur: Any) -> None:
    """
    Enforces dual-boot mutual exclusion by ensuring only one host per
    dualboot_group_id is marked as 'ready'.

    This function queries the database to identify hosts that are currently
    marked as 'ready' and have been seen recently (within the last 2 minutes).
    It selects a single 'active' host per dualboot_group_id, prioritizing
    the one with the most recent 'last_seen_at' timestamp.

    All other hosts within the same dualboot_group_id that are not the
    selected 'active' host will be updated to 'offline' status and their
    associated lanes will be set to 'suspended'. This mechanism prevents
    conflicting operations in dual-boot configurations. For example, if
    'packpup1' and 'pupix1' are configured with the same `dualboot_group_id`
    in the `host_dualboot_members` table, this function ensures only one
    can be 'ready' at a time.

    This is a best-effort mechanism and relies on accurate host status
    reporting and timely updates, as well as correct database configuration
    of `host_dualboot_members` for the intended dualboot pairs.
    """
    # 1. Identify groups that have an active (ready) host.
    # We use DISTINCT ON to pick one 'winner' per group (most recent last_seen_at).
    # 2. Update all other members of those groups to offline.
    # 3. Suspend lanes for those other members.
    cur.execute("""
        WITH group_winners AS (
            SELECT DISTINCT ON (hdbm.dualboot_group_id)
                hdbm.dualboot_group_id,
                h.host_id,
                h.host_name
            FROM hosts h
            JOIN host_dualboot_members hdbm ON h.host_id = hdbm.host_id
            WHERE h.status = 'ready'
            AND h.last_seen_at > now() - interval '2 minutes'
            ORDER BY hdbm.dualboot_group_id, h.last_seen_at DESC
        ),
        to_disable AS (
            SELECT hdbm.host_id, gw.host_name as active_host_name
            FROM host_dualboot_members hdbm
            JOIN group_winners gw ON hdbm.dualboot_group_id = gw.dualboot_group_id
            WHERE hdbm.host_id != gw.host_id
        ),
        update_hosts AS (
            UPDATE hosts h
            SET status = 'offline',
                notes = 'dualboot_other_side_active: ' || td.active_host_name,
                updated_at = now()
            FROM to_disable td
            WHERE h.host_id = td.host_id
            AND h.status != 'offline'
        )
        UPDATE lanes l
        SET status = 'suspended',
            suspension_reason = 'dualboot_other_side_active: ' || td.active_host_name,
            updated_at = now()
        FROM to_disable td
        WHERE l.host_id = td.host_id
        AND l.status != 'offline';
    """)


def probe_once() -> None:
    with db.connect() as conn:
        with conn.cursor() as cur:
            lanes = q(
                cur,
                """
                SELECT lane_id, host_id, base_url, status
                FROM lanes
                WHERE lane_type != 'router'
                ORDER BY base_url
                """,
                (),
            )

            for lane in lanes:
                lane_id = lane["lane_id"]
                host_id = lane["host_id"]
                base_url = str(lane["base_url"])
                
                ok, code, latency, err = _probe_lane_health(base_url)
                
                # Mapping probe results to lane status:
                # - If probe is successful (HTTP 200 OK), set lane status to 'ready'.
                # - If probe is successful but returns a non-200 status code (e.g., 503 Service Unavailable),
                #   it means the host is reachable but the service is not ready, so set lane status to 'suspended'.
                # - If the probe results in a network error or timeout (HTTP status code is None),
                #   it means the host is unreachable, so set lane status to 'offline'.
                if ok:
                    new_status = "ready"
                elif code is not None:
                    new_status = "suspended"
                else:
                    new_status = "offline"

                # Log to lane_probes
                cur.execute(
                    """
                    INSERT INTO lane_probes (lane_id, kind, ok, status_code, latency_ms, error)
                    VALUES (%s, 'http_health'::probe_kind, %s, %s, %s, %s)
                    """,
                    (lane_id, ok, code, int(latency), (err or "")[:240] if not ok else None),
                )
                
                # Update lane status
                cur.execute(
                    """
                    UPDATE lanes
                    SET status=CASE
                          WHEN COALESCE(suspension_reason, '') <> '' THEN status
                          ELSE %s::lane_status
                        END,
                        last_probe_at=now(),
                        last_ok_at=CASE WHEN %s THEN now() ELSE last_ok_at END,
                        last_error=%s,
                        updated_at=now()
                    WHERE lane_id=%s
                    """,
                    (new_status, ok, (err or "probe_failed")[:240] if not ok else None, lane_id),
                )
                
                # If ready, update host status too
                if ok:
                    cur.execute(
                        """
                        UPDATE hosts 
                        SET status='ready', last_seen_at=now(), updated_at=now()
                        WHERE host_id=%s
                        """,
                        (host_id,),
                    )
                    
                    # Also detect currently loaded model
                    loaded_model = _probe_lane_model(base_url)
                    if loaded_model:
                        cur.execute(
                            "UPDATE lanes SET current_model_name=%s WHERE lane_id=%s",
                            (loaded_model, lane_id),
                        )
            
            # After all lanes probed, enforce dual-boot mutual exclusion
            _enforce_dualboot_mutual_exclusion(cur)
            
        conn.commit()


def run_forever() -> None:
    while True:
        try:
            probe_once()
        except Exception:
            # best-effort; detailed logging can be added later
            pass
        time.sleep(max(5, int(settings.probe_interval_seconds)))

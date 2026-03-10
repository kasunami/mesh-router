from __future__ import annotations

import time
from typing import Any

import httpx
from psycopg.types.json import Jsonb

from .config import settings
from .db import db


def _is_canonical_model_name(model: str) -> bool:
    m = (model or "").strip()
    if not m:
        return False
    # Avoid filesystem paths or URLs as model IDs in our canonical catalog.
    if "/" in m or "\\" in m:
        return False
    if "://" in m:
        return False
    if len(m) > 128:
        return False
    return True


def _upsert_host_and_lane(item: dict[str, Any]) -> None:
    worker_id = str(item.get("worker_id") or "").strip()
    base_url = str(item.get("base_url") or "").strip()
    lane_type = str(item.get("lane_type") or "other").strip().lower()
    status = str(item.get("status") or "offline").strip().lower()
    current_model = item.get("current_model")
    md = item.get("metadata") or {}

    if not worker_id or not base_url:
        return

    # 1) host upsert (minimal)
    with db.connect() as conn:
        with conn.cursor() as cur:
            # Upsert model name if present and canonical (minimal; enriched later by model research/sync jobs).
            if current_model and _is_canonical_model_name(str(current_model)):
                cur.execute(
                    """
                    INSERT INTO models (model_name, format)
                    VALUES (%s, 'other'::model_format)
                    ON CONFLICT (model_name)
                    DO UPDATE SET updated_at=now()
                    """,
                    (str(current_model).strip(),),
                )

            cur.execute(
                """
                INSERT INTO hosts (host_name, status, last_seen_at)
                VALUES (%s, %s, now())
                ON CONFLICT (host_name)
                DO UPDATE SET status=EXCLUDED.status, last_seen_at=now(), updated_at=now()
                RETURNING host_id
                """,
                (worker_id, "ready" if status in ("ready", "busy") else ("offline" if status == "offline" else "unknown")),
            )
            host_id = cur.fetchone()["host_id"]

            # 2) lane upsert
            lane_name = lane_type  # keep simple for now ("cpu","gpu","mlx","router")
            cur.execute(
                """
                INSERT INTO lanes (
                  host_id, lane_name, lane_type, base_url, status,
                  current_model_name, proxy_auth_mode, proxy_auth_metadata,
                  last_probe_at, last_ok_at, updated_at
                )
                VALUES (%s, %s, %s::lane_type, %s, %s::lane_status,
                        %s, %s, %s::jsonb,
                        now(), CASE WHEN %s IN ('ready','busy') THEN now() ELSE NULL END, now())
                ON CONFLICT (base_url)
                DO UPDATE SET
                  host_id=EXCLUDED.host_id,
                  lane_name=EXCLUDED.lane_name,
                  lane_type=EXCLUDED.lane_type,
                  status=EXCLUDED.status,
                  current_model_name=EXCLUDED.current_model_name,
                  proxy_auth_mode=EXCLUDED.proxy_auth_mode,
                  proxy_auth_metadata=EXCLUDED.proxy_auth_metadata,
                  last_probe_at=now(),
                  last_ok_at=CASE WHEN EXCLUDED.status IN ('ready','busy') THEN now() ELSE lanes.last_ok_at END,
                  updated_at=now()
                """,
                (
                    host_id,
                    lane_name,
                    lane_type if lane_type in ("cpu", "gpu", "mlx", "router") else "other",
                    base_url,
                    status if status in ("ready", "busy", "suspended", "offline", "error") else "offline",
                    str(current_model).strip() if current_model else None,
                    str(md.get("proxy_auth_mode") or "").strip() or None,
                    Jsonb(md),
                    status,
                ),
            )
        conn.commit()


def sync_once() -> None:
    with httpx.Client(timeout=10.0) as client:
        r = client.get(f"{settings.meshbench_base_url.rstrip('/')}/api/worker-lanes")
        r.raise_for_status()
        payload = r.json()
    for item in payload.get("items", []):
        _upsert_host_and_lane(item)


def run_forever() -> None:
    while True:
        try:
            sync_once()
        except Exception:
            # Keep going; detailed logging will be added later.
            pass
        time.sleep(max(5, int(settings.sync_interval_seconds)))

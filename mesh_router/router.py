from __future__ import annotations

from dataclasses import dataclass
import re

from .db import db, q
from .config import settings


@dataclass(frozen=True)
class LaneChoice:
    lane_id: str
    worker_id: str
    base_url: str
    lane_type: str


def _model_lookup_keys(model_name: str | None) -> set[str]:
    raw = (model_name or "").strip()
    if not raw:
        return set()

    stem = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    lowered_stem = stem.lower()
    stem_no_ext = stem.rsplit(".", 1)[0] if lowered_stem.endswith((".gguf", ".safetensors", ".bin")) else stem
    normalized = stem_no_ext.lower().replace("_", "-").replace(":", "-")
    keys = {raw.lower(), stem.lower(), normalized}

    for value in list(keys):
        dequantized = re.sub(r"[-_.]q\d+(?:[-_.]k(?:[-_.][a-z0-9]+)?)?$", "", value)
        debitted = re.sub(r"[-_.](?:\d+bit|fp8)$", "", value)
        keys.add(dequantized)
        keys.add(debitted)
        keys.add(re.sub(r"[-_.](?:\d+bit|fp8)$", "", dequantized))

    return {key for key in keys if key}


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


def _model_matches_request(requested_model: str, candidate_model: str | None) -> bool:
    candidate = (candidate_model or "").strip()
    if not candidate:
        return False
    if _is_exact_model_request(requested_model):
        return candidate == requested_model
    return bool(_model_lookup_keys(requested_model) & _model_lookup_keys(candidate))


def pick_lane_for_model(
    *,
    model: str,
    pin_worker: str | None = None,
    pin_base_url: str | None = None,
    pin_lane_type: str | None = None,
    exclude_lane_ids: set[str] | None = None,
) -> LaneChoice:
    """
    Minimal placement logic:
    - If pinned (worker + base_url), use it.
    - Else prefer READY lanes whose current_model already matches.
    - Else any READY lane.

    This will be expanded to account for lane_model_policy, dualboot groups, disk/RAM budgets,
    load times, TPS, and error rates.
    """
    excluded = {lane_id for lane_id in (exclude_lane_ids or set()) if lane_id}

    if pin_worker and pin_base_url:
        with db.connect() as conn:
            with conn.cursor() as cur:
                rows = q(
                    cur,
                    """
                    SELECT l.lane_id, h.host_name, l.base_url, l.lane_type
                    FROM lanes l
                    JOIN hosts h ON h.host_id=l.host_id
                    WHERE h.host_name=%s AND l.base_url=%s
                      AND (%s::text[] IS NULL OR l.lane_id::text <> ALL(%s::text[]))
                    LIMIT 1
                    """,
                    (pin_worker, pin_base_url, list(excluded) or None, list(excluded) or None),
                )
        if not rows:
            raise RuntimeError("pinned lane not found")
        r0 = rows[0]
        return LaneChoice(
            lane_id=str(r0["lane_id"]),
            worker_id=str(r0["host_name"]),
            base_url=str(r0["base_url"]),
            lane_type=str(r0["lane_type"]),
        )

# These hosts are part of a dualboot group and rely on probe.py for mutual exclusion.
    preferred_hosts = ["Static-Deskix", "Static-Mobile-2", "pupix1", "tiffs-macbook"]

    if pin_worker and not pin_base_url:
        # Pick the best READY lane for that host.
        with db.connect() as conn:
            with conn.cursor() as cur:
                rows = q(
                    cur,
                    """
                    SELECT l.lane_id, h.host_name, l.base_url, l.lane_type, l.current_model_name
                    FROM lanes l
                    JOIN hosts h ON h.host_id = l.host_id
                    WHERE h.host_name=%s
                      AND l.status='ready'
                      AND NOT EXISTS (
                        SELECT 1 FROM router_leases rl
                        WHERE rl.lane_id = l.lane_id
                          AND rl.state = 'active'
                          AND COALESCE(rl.last_heartbeat_at, rl.acquired_at) > now() - (%s * interval '1 second')
                      )
                      AND (%s::text IS NULL OR l.lane_type::text = %s::text)
                      AND (%s::text[] IS NULL OR l.lane_id::text <> ALL(%s::text[]))
                    ORDER BY
                      CASE l.lane_type WHEN 'gpu' THEN 0 WHEN 'mlx' THEN 1 WHEN 'cpu' THEN 2 ELSE 9 END,
                      l.base_url ASC
                    LIMIT 20
                    """,
                    (
                        pin_worker,
                        settings.default_lease_stale_seconds,
                        pin_lane_type,
                        pin_lane_type,
                        list(excluded) or None,
                        list(excluded) or None,
                    ),
                )
        matched = [row for row in rows if _model_matches_request(model, row.get("current_model_name"))]
        if not matched:
            raise RuntimeError(f"no READY lanes for pinned worker serving requested model: {model}")
        r0 = matched[0]
        return LaneChoice(
            lane_id=str(r0["lane_id"]),
            worker_id=str(r0["host_name"]),
            base_url=str(r0["base_url"]),
            lane_type=str(r0["lane_type"]),
        )

    def _pick(allow_fallback_hosts: bool) -> LaneChoice | None:
        with db.connect() as conn:
            with conn.cursor() as cur:
                rows = q(
                    cur,
                    """
                    SELECT l.lane_id, h.host_name, l.base_url, l.lane_type, l.status, l.current_model_name
                    FROM lanes l
                    JOIN hosts h ON h.host_id = l.host_id
                    WHERE l.status = 'ready'
                      AND NOT EXISTS (
                        SELECT 1 FROM router_leases rl
                        WHERE rl.lane_id = l.lane_id
                          AND rl.state = 'active'
                          AND COALESCE(rl.last_heartbeat_at, rl.acquired_at) > now() - (%s * interval '1 second')
                      )
                      AND (%s OR h.host_name IN ('Static-Deskix','Static-Mobile-2','pupix1','tiffs-macbook'))
                      AND h.host_name NOT IN ('litellm-router')
                      AND (%s::text IS NULL OR l.lane_type::text = %s::text)
                      AND (%s::text[] IS NULL OR l.lane_id::text <> ALL(%s::text[]))
                    ORDER BY
                      -- Prefer primary worker hosts; 'pupix1' is in a dualboot group. Keep other hosts as last-resort.
                      CASE h.host_name
                        WHEN 'Static-Deskix' THEN 0
                        WHEN 'Static-Mobile-2' THEN 1
                        WHEN 'pupix1' THEN 2
                        WHEN 'tiffs-macbook' THEN 3
                        ELSE 999
                      END,
                      -- Prefer GPU for generation, then MLX, then CPU.
                      CASE l.lane_type WHEN 'gpu' THEN 0 WHEN 'mlx' THEN 1 WHEN 'cpu' THEN 2 ELSE 9 END,
                      h.host_name ASC
                    LIMIT 50
                    """,
                    (
                        settings.default_lease_stale_seconds,
                        allow_fallback_hosts,
                        pin_lane_type,
                        pin_lane_type,
                        list(excluded) or None,
                        list(excluded) or None,
                    ),
                )
        matched = [row for row in rows if _model_matches_request(model, row.get("current_model_name"))]
        if not matched:
            return None
        r0 = matched[0]
        return LaneChoice(
            lane_id=str(r0["lane_id"]),
            worker_id=str(r0["host_name"]),
            base_url=str(r0["base_url"]),
            lane_type=str(r0["lane_type"]),
        )

    chosen = _pick(allow_fallback_hosts=False) or _pick(allow_fallback_hosts=True)
    if not chosen:
        raise RuntimeError(f"no READY lanes available serving requested model: {model}")
    return chosen

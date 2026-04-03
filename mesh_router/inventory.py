from __future__ import annotations

from typing import Any

from .config import settings
from .db import mw_state_db
from .mw_overlay import apply_mw_effective_status


def fetch_lane_inventory(*, cur: Any) -> list[dict[str, Any]]:
    """
    Returns a lane inventory list from the router DB, augmented with MW-derived effective readiness/model
    for MW-managed lanes when MW state lives in a separate DB.
    """
    cur.execute(
        """
        SELECT
          l.lane_id,
          l.lane_name,
          l.lane_type,
          l.backend_type,
          l.base_url,
          l.status,
          l.proxy_auth_metadata,
          l.current_model_name,
          h.host_id,
          h.host_name,
          COALESCE((
            SELECT jsonb_agg(
              jsonb_build_object(
                'model_name', m.model_name,
                'tags', COALESCE(m.tags, '{}'::text[]),
                'max_ctx', p.max_ctx,
                'locality', lmv.source_locality
              )
              ORDER BY m.model_name
            )
            FROM lane_model_viability lmv
            JOIN models m ON m.model_id=lmv.model_id
            LEFT JOIN lane_model_policy p ON p.lane_id=l.lane_id AND p.model_id=lmv.model_id
            WHERE lmv.lane_id=l.lane_id AND lmv.is_viable=true
          ), '[]'::jsonb) as viable_models
        FROM lanes l
        JOIN hosts h ON h.host_id = l.host_id
        ORDER BY h.host_name ASC, l.lane_name ASC
        """
    )
    rows = [dict(r) for r in (cur.fetchall() or [])]
    apply_mw_effective_status(rows, mw_state_db=mw_state_db, stale_seconds=settings.default_lease_stale_seconds)
    return rows


def group_inventory_by_host(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_host: dict[str, dict[str, Any]] = {}
    for r in rows:
        host_id = str(r.get("host_id") or "")
        host_name = str(r.get("host_name") or "")
        if not host_id or not host_name:
            continue
        key = host_id
        if key not in by_host:
            by_host[key] = {"host_id": host_id, "host_name": host_name, "lanes": [], "tags": [], "policy": {}}
        by_host[key]["lanes"].append(r)

    # Policy tagging: stable vs opportunistic hosts
    opportunistic = {h.strip() for h in (settings.opportunistic_hosts or "").split(",") if h.strip()}
    for h in by_host.values():
        hn = str(h.get("host_name") or "")
        is_opp = hn in opportunistic
        h["policy"] = {"opportunistic": is_opp}
        if is_opp:
            h["tags"].append("opportunistic")
        else:
            h["tags"].append("stable")
    return list(by_host.values())


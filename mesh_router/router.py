from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from .db import db, mw_state_db, q
from .config import settings

_RECENT_PROXY_ERROR_COOLDOWN_S = 900


@dataclass(frozen=True)
class LaneChoice:
    lane_id: str
    worker_id: str
    base_url: str
    lane_type: str
    backend_type: str
    current_model_name: str | None = None
    current_model_max_ctx: int | None = None


class LanePlacementError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def _context_is_sufficient(required_tokens: int | None, max_ctx: int | None) -> bool:
    if required_tokens is None or required_tokens <= 0:
        return True
    if max_ctx is None or max_ctx <= 0:
        return True
    return required_tokens <= max_ctx


def _context_limit_message(*, model: str, required_tokens: int | None, max_available_ctx: int | None) -> str:
    if required_tokens and max_available_ctx:
        return (
            f"requested context ({required_tokens} tokens estimated) exceeds the maximum configured context "
            f"available for model {model} ({max_available_ctx} tokens)"
        )
    if required_tokens:
        return f"requested context ({required_tokens} tokens estimated) exceeds the maximum configured context available for model {model}"
    return f"requested context exceeds the maximum configured context available for model {model}"


def _normalize_model_tag(tag: str | None) -> str | None:
    raw = (tag or "").strip().lower()
    if not raw:
        return None
    return raw.replace("_", "-")


def _normalized_model_tags(tags: list[str] | None) -> set[str]:
    out: set[str] = set()
    for tag in tags or []:
        value = _normalize_model_tag(tag)
        if value:
            out.add(value)
    return out


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


def _model_matches_request(
    requested_model: str,
    candidate_model: str | None,
    candidate_tags: list[str] | None = None,
) -> bool:
    candidate = (candidate_model or "").strip()
    if not candidate:
        return False
    if _is_exact_model_request(requested_model):
        # Also match against the basename so that lanes storing full local paths
        # (e.g. /Users/kasunami/models/Qwen3.5-9B-6bit) match a bare name request.
        path_parts = re.split(r"[\\/]+", candidate)
        candidate_stem = path_parts[-1] if path_parts else candidate
        candidate_parent = path_parts[-2] if len(path_parts) >= 2 else ""
        return (
            candidate == requested_model
            or candidate_stem == requested_model
            or candidate_parent == requested_model
        )
    request_keys = _model_lookup_keys(requested_model)
    if request_keys & _model_lookup_keys(candidate):
        return True
    return bool(request_keys & _normalized_model_tags(candidate_tags))


def pick_lane_for_model(
    *,
    model: str,
    backend_type: str | None = None,
    request_context_tokens: int | None = None,
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

    def _apply_mw_effective_status(rows: list[dict[str, Any]]) -> None:
        """
        Enrich rows (from the router DB) with MW-derived readiness + current model when
        lanes are MW-managed and MW state lives in a separate DB.
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

        # Query MW state DB in one shot. If unavailable, leave rows unchanged.
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
            return

        stale_seconds = settings.default_lease_stale_seconds
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
            last_hb = f.get("last_heartbeat_at")
            actual_state = str(f.get("actual_state") or "")
            health_status = str(f.get("health_status") or "")
            ready = False
            try:
                if last_hb is not None:
                    # psycopg returns aware datetimes; compare via epoch seconds.
                    age_s = (datetime.now(tz=timezone.utc) - last_hb).total_seconds()
                    ready = age_s <= float(stale_seconds) and actual_state == "running" and health_status == "healthy"
            except Exception:
                ready = False
            row["effective_status"] = "ready" if ready else "offline"
            actual_model = str(f.get("actual_model") or "").strip()
            if actual_model:
                row["current_model_name"] = actual_model
                row["current_model_tags"] = []

    if pin_worker and pin_base_url:
        with db.connect() as conn:
            with conn.cursor() as cur:
                rows = q(
                    cur,
                    """
                    SELECT l.lane_id, h.host_name, l.base_url, l.lane_type, l.backend_type,
                           l.status,
                           l.proxy_auth_metadata,
                           l.current_model_name,
                           cmp.max_ctx AS current_model_max_ctx
                    FROM lanes l
                    JOIN hosts h ON h.host_id=l.host_id
                    LEFT JOIN models cm ON cm.model_name=l.current_model_name
                    LEFT JOIN lane_model_policy cmp ON cmp.lane_id=l.lane_id AND cmp.model_id=cm.model_id
                    WHERE h.host_name=%s AND l.base_url=%s
                      AND (%s::text IS NULL OR l.backend_type = %s::text)
                      AND (%s::text[] IS NULL OR l.lane_id::text <> ALL(%s::text[]))
                    LIMIT 1
                    """,
                    (
                        pin_worker,
                        pin_base_url,
                        backend_type,
                        backend_type,
                        list(excluded) or None,
                        list(excluded) or None,
                    ),
                )
        if not rows:
            raise RuntimeError("pinned lane not found")
        _apply_mw_effective_status(rows)
        r0 = rows[0]
        return LaneChoice(
            lane_id=str(r0["lane_id"]),
            worker_id=str(r0["host_name"]),
            base_url=str(r0["base_url"]),
            lane_type=str(r0["lane_type"]),
            backend_type=str(r0.get("backend_type") or "llama"),
            current_model_name=r0.get("current_model_name"),
            current_model_max_ctx=int(r0["current_model_max_ctx"]) if r0.get("current_model_max_ctx") is not None else None,
        )

    if pin_worker and not pin_base_url:
        # Pick the best READY lane for that host.
        with db.connect() as conn:
            with conn.cursor() as cur:
                rows = q(
                    cur,
                    """
                    SELECT l.lane_id, h.host_name, l.base_url, l.lane_type, l.backend_type,
                           l.status,
                           l.proxy_auth_metadata,
                           l.current_model_name,
                           m.tags AS current_model_tags,
                           cmp.max_ctx AS current_model_max_ctx
                    FROM lanes l
                    JOIN hosts h ON h.host_id = l.host_id
                    LEFT JOIN models m ON m.model_name = l.current_model_name
                    LEFT JOIN lane_model_policy cmp ON cmp.lane_id=l.lane_id AND cmp.model_id=m.model_id
                    WHERE h.host_name=%s
                      AND (l.status='ready' OR (l.proxy_auth_metadata->>'control_plane')='mw')
                      AND NOT EXISTS (
                        SELECT 1 FROM router_leases rl
                        WHERE rl.lane_id = l.lane_id
                          AND rl.state = 'active'
                          AND COALESCE(rl.last_heartbeat_at, rl.acquired_at) > now() - (%s * interval '1 second')
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM router_requests rr
                        WHERE rr.lane_id = l.lane_id
                          AND rr.error_kind = 'proxy_error'
                          AND COALESCE(rr.error_message, '') NOT ILIKE '%%exceeds the available context size%%'
                          AND rr.released_at > now() - (%s * interval '1 second')
                      )
                      AND (%s::text IS NULL OR l.lane_type::text = %s::text)
                      AND (%s::text IS NULL OR l.backend_type = %s::text)
                      AND (%s::text[] IS NULL OR l.lane_id::text <> ALL(%s::text[]))
                    ORDER BY
                      CASE l.lane_type WHEN 'gpu' THEN 0 WHEN 'mlx' THEN 1 WHEN 'cpu' THEN 2 ELSE 9 END,
                      l.base_url ASC
                    LIMIT 20
                    """,
                    (
                        pin_worker,
                        settings.default_lease_stale_seconds,
                        _RECENT_PROXY_ERROR_COOLDOWN_S,
                        pin_lane_type,
                        pin_lane_type,
                        backend_type,
                        backend_type,
                        list(excluded) or None,
                        list(excluded) or None,
                    ),
                )
        _apply_mw_effective_status(rows)
        # Defensive: even if the SQL layer changes, pinning must never route to a different host.
        rows = [r for r in rows if str(r.get("host_name") or "") == str(pin_worker)]

        def _status(row: dict) -> str:
            return str(row.get("effective_status") or row.get("status") or "ready")
        matched = [
            row
            for row in rows
            if _status(row) == "ready"
            and _model_matches_request(model, row.get("current_model_name"), row.get("current_model_tags") or [])
            and _context_is_sufficient(request_context_tokens, row.get("current_model_max_ctx"))
        ]
        context_mismatched = [
            row
            for row in rows
            if _status(row) == "ready"
            and _model_matches_request(model, row.get("current_model_name"), row.get("current_model_tags") or [])
            and not _context_is_sufficient(request_context_tokens, row.get("current_model_max_ctx"))
        ]
        if not matched and context_mismatched:
            max_available_ctx = max(
                int(row["current_model_max_ctx"])
                for row in context_mismatched
                if row.get("current_model_max_ctx") is not None
            )
            raise LanePlacementError(
                _context_limit_message(
                    model=model,
                    required_tokens=request_context_tokens,
                    max_available_ctx=max_available_ctx,
                ),
                status_code=422,
            )
        if not matched:
            # Pinned-worker semantics should be "place it on that host" even if the model isn't already loaded.
            # For MW-managed lanes, the caller will best-effort `load_model` before streaming.
            eligible = [
                row
                for row in rows
                if _status(row) == "ready" and _context_is_sufficient(request_context_tokens, row.get("current_model_max_ctx"))
            ]
            if not eligible:
                raise RuntimeError(f"no READY lanes for pinned worker: {pin_worker}")
            r0 = eligible[0]
        else:
            r0 = matched[0]
        return LaneChoice(
            lane_id=str(r0["lane_id"]),
            worker_id=str(r0["host_name"]),
            base_url=str(r0["base_url"]),
            lane_type=str(r0["lane_type"]),
            backend_type=str(r0.get("backend_type") or "llama"),
            current_model_name=r0.get("current_model_name"),
            current_model_max_ctx=int(r0["current_model_max_ctx"]) if r0.get("current_model_max_ctx") is not None else None,
        )

    def _pick() -> LaneChoice | None:
        with db.connect() as conn:
            with conn.cursor() as cur:
                rows = q(
                    cur,
                    """
	                    SELECT
	                      l.lane_id, h.host_name, l.base_url, l.lane_type, l.backend_type, l.status, l.suspension_reason,
	                      l.proxy_auth_metadata,
	                      l.status::text AS effective_status,
	                      l.current_model_name AS current_model_name,
	                      cmp.max_ctx AS current_model_max_ctx,
	                      current_m.tags AS current_model_tags,
                      COALESCE((
                        SELECT jsonb_agg(
                          jsonb_build_object(
                            'model_name', m.model_name,
                            'tags', COALESCE(m.tags, '{}'::text[]),
                            'max_ctx', p.max_ctx
                          )
                          ORDER BY m.model_name
                        )
                        FROM lane_model_viability lmv
                        JOIN models m ON m.model_id=lmv.model_id
                        LEFT JOIN lane_model_policy p ON p.lane_id=l.lane_id AND p.model_id=lmv.model_id
                        WHERE lmv.lane_id=l.lane_id AND lmv.is_viable=true AND lmv.source_locality='local'
                      ), '[]'::jsonb) as local_viable_models,
                      COALESCE((
                        SELECT jsonb_agg(
                          jsonb_build_object(
                            'model_name', m.model_name,
                            'tags', COALESCE(m.tags, '{}'::text[]),
                            'max_ctx', p.max_ctx
                          )
                          ORDER BY m.model_name
                        )
                        FROM lane_model_viability lmv
                        JOIN models m ON m.model_id=lmv.model_id
                        LEFT JOIN lane_model_policy p ON p.lane_id=l.lane_id AND p.model_id=lmv.model_id
                        WHERE lmv.lane_id=l.lane_id AND lmv.is_viable=true AND lmv.source_locality='remote'
                      ), '[]'::jsonb) as remote_viable_models
                    FROM lanes l
                    JOIN hosts h ON h.host_id = l.host_id
                    LEFT JOIN models current_m ON current_m.model_name = l.current_model_name
                    LEFT JOIN lane_model_policy cmp ON cmp.lane_id=l.lane_id AND cmp.model_id=current_m.model_id
                    WHERE (l.status IN ('ready', 'suspended') OR (l.proxy_auth_metadata->>'control_plane') = 'mw')
                      AND (%s::text IS NULL OR h.host_name = %s::text)
                      AND NOT EXISTS (
                        SELECT 1 FROM router_leases rl
                        WHERE rl.lane_id = l.lane_id
                          AND rl.state = 'active'
                          AND COALESCE(rl.last_heartbeat_at, rl.acquired_at) > now() - (%s * interval '1 second')
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM router_requests rr
                        WHERE rr.lane_id = l.lane_id
                          AND rr.error_kind = 'proxy_error'
                          AND COALESCE(rr.error_message, '') NOT ILIKE '%%exceeds the available context size%%'
                          AND rr.released_at > now() - (%s * interval '1 second')
                      )
                      AND h.host_name NOT IN ('litellm-router')
                      AND (%s::text IS NULL OR l.lane_type::text = %s::text)
                      AND (
                        CASE
                          WHEN %s::text IS NOT NULL THEN l.backend_type = %s::text
                          ELSE l.backend_type = 'llama' OR l.backend_type IS NULL
                        END
                      )
                      AND (%s::text[] IS NULL OR l.lane_id::text <> ALL(%s::text[]))
                    ORDER BY
                      -- Prefer GPU for generation, then MLX, then CPU.
                      CASE l.lane_type WHEN 'gpu' THEN 0 WHEN 'mlx' THEN 1 WHEN 'cpu' THEN 2 ELSE 9 END,
                      h.host_name ASC
                    LIMIT 50
                    """,
                    (
                        pin_worker,
                        pin_worker,
                        settings.default_lease_stale_seconds,
                        _RECENT_PROXY_ERROR_COOLDOWN_S,
                        pin_lane_type,
                        pin_lane_type,
                        backend_type,
                        backend_type,
                        list(excluded) or None,
                        list(excluded) or None,
                    ),
                )
        _apply_mw_effective_status(rows)
        if settings.placement_prefer_mw_lanes:
            def _is_mw(row: dict) -> bool:
                pam = row.get("proxy_auth_metadata") or {}
                return isinstance(pam, dict) and str(pam.get("control_plane") or "") == "mw"

            # Stable sort: keep prior preference ordering, but front-load MW-managed lanes.
            rows.sort(key=lambda r: 0 if _is_mw(r) else 1)
        if pin_worker:
            # Defensive: even if the SQL layer changes, pinning must never route to a different host.
            rows = [r for r in rows if str(r.get("host_name") or "") == pin_worker]

        def _status(row: dict) -> str:
            return str(row.get("effective_status") or row.get("status") or "")

        # Only 'ready' lanes are eligible for direct dispatch.
        matched = [
            row
            for row in rows
            if _status(row) == "ready"
            and _model_matches_request(model, row.get("current_model_name"), row.get("current_model_tags") or [])
            and _context_is_sufficient(request_context_tokens, row.get("current_model_max_ctx"))
        ]
        context_mismatched = [
            row
            for row in rows
            if _status(row) == "ready"
            and _model_matches_request(model, row.get("current_model_name"), row.get("current_model_tags") or [])
            and not _context_is_sufficient(request_context_tokens, row.get("current_model_max_ctx"))
        ]
        if matched:
            r0 = matched[0]
            return LaneChoice(
                lane_id=str(r0["lane_id"]),
                worker_id=str(r0["host_name"]),
                base_url=str(r0["base_url"]),
                lane_type=str(r0["lane_type"]),
                backend_type=str(r0.get("backend_type") or "llama"),
                current_model_name=r0.get("current_model_name"),
                current_model_max_ctx=int(r0["current_model_max_ctx"]) if r0.get("current_model_max_ctx") is not None else None,
            )
        if context_mismatched:
            max_available_ctx = max(
                int(row["current_model_max_ctx"])
                for row in context_mismatched
                if row.get("current_model_max_ctx") is not None
            )
            raise LanePlacementError(
                _context_limit_message(
                    model=model,
                    required_tokens=request_context_tokens,
                    max_available_ctx=max_available_ctx,
                ),
                status_code=422,
            )

        # Swappable candidate pool: ready lanes + suspended lanes with no suspension_reason.
        # Suspended lanes with a suspension_reason were explicitly disabled (e.g. sibling exclusion)
        # and must not be demand-started.
        swappable_rows = [
            row for row in rows
            if _status(row) == "ready"
            or (_status(row) == "suspended" and not row.get("suspension_reason"))
        ]

        # Fallback: find a lane that can serve this model after a swap (has it in viable_models)
        swappable = [
            row for row in swappable_rows
            if any(
                _model_matches_request(model, item.get("model_name"), item.get("tags") or [])
                and _context_is_sufficient(request_context_tokens, item.get("max_ctx"))
                for item in (row.get("local_viable_models") or [])
            )
            or any(
                _model_matches_request(model, item.get("model_name"), item.get("tags") or [])
                and _context_is_sufficient(request_context_tokens, item.get("max_ctx"))
                for item in (row.get("remote_viable_models") or [])
            )
        ]
        context_limited = [
            row for row in swappable_rows
            if (
                any(
                    _model_matches_request(model, item.get("model_name"), item.get("tags") or [])
                    for item in (row.get("local_viable_models") or [])
                )
                or any(
                    _model_matches_request(model, item.get("model_name"), item.get("tags") or [])
                    for item in (row.get("remote_viable_models") or [])
                )
            )
            and row not in swappable
        ]
        if swappable:
            # Pick the GPU lane first, then by host_name
            swappable.sort(key=lambda r: (
                0 if "gpu" in r.get("lane_type","").lower() else 1,
                r.get("host_name") or ""
            ))
            r0 = swappable[0]
            return LaneChoice(
                lane_id=str(r0["lane_id"]),
                worker_id=str(r0["host_name"]),
                base_url=str(r0["base_url"]),
                lane_type=str(r0["lane_type"]),
                backend_type=str(r0.get("backend_type") or "llama"),
                current_model_name=r0.get("current_model_name"),
                current_model_max_ctx=int(r0["current_model_max_ctx"]) if r0.get("current_model_max_ctx") is not None else None,
            )
        if context_limited and request_context_tokens:
            max_available_ctx = 0
            for row in context_limited:
                for group in ("local_viable_models", "remote_viable_models"):
                    for item in row.get(group) or []:
                        if _model_matches_request(model, item.get("model_name"), item.get("tags") or []):
                            item_max_ctx = item.get("max_ctx")
                            if item_max_ctx is not None:
                                max_available_ctx = max(max_available_ctx, int(item_max_ctx))
            raise LanePlacementError(
                _context_limit_message(
                    model=model,
                    required_tokens=request_context_tokens,
                    max_available_ctx=max_available_ctx or None,
                ),
                status_code=422,
            )
            
        return None

    chosen = _pick()
    if not chosen:
        raise RuntimeError(f"no READY lanes available serving requested model: {model}")
    return chosen

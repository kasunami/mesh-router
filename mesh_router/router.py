from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import uuid
from typing import Any

from .db import db, mw_state_db, q
from .config import settings
from .mw_overlay import apply_mw_effective_status

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
    # Concrete host-local model artifact chosen for this request (may differ from request tag).
    resolved_model_name: str | None = None


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


def _family_size_tags_from_keys(keys: set[str]) -> set[str]:
    tags: set[str] = set()
    for key in keys:
        for family, pattern in (
            ("qwen3.5", r"qwen3\.5[-_:]?(\d+(?:\.\d+)?)(b|m)\b"),
            ("falcon3", r"falcon3[-_:]?(\d+(?:\.\d+)?)(b|m)\b"),
            ("lfm2.5", r"lfm2\.5[-_:]?(\d+(?:\.\d+)?)(b|m)\b"),
            ("gemma4", r"gemma4[-_:]?(\d+(?:\.\d+)?)(b|m)\b"),
        ):
            match = re.search(pattern, key)
            if match:
                size = f"{match.group(1)}{match.group(2).lower()}"
                tags.add(f"{family}:{size}")
                tags.add(f"{family}-{size}")
    return tags


def _inferred_model_tags(model_name: str | None) -> set[str]:
    """Return generic selection tags derived from a concrete model artifact."""
    return _family_size_tags_from_keys(_model_lookup_keys(model_name))


def _normalize_backend_type(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"llama.cpp", "llama"}:
        return "llama"
    if raw in {"bitnet.cpp", "bitnet"}:
        return "bitnet"
    if raw in {"stable-diffusion.cpp", "stable-diffusion", "sd"}:
        return "sd"
    if raw == "mlx":
        return "mlx"
    return raw


def _backend_matches_request(row: dict[str, Any], backend_type: str | None) -> bool:
    requested = _normalize_backend_type(backend_type)
    if not requested:
        return True
    actual = _normalize_backend_type(str(row.get("backend_type") or ""))
    if actual:
        return actual == requested
    return requested == "llama"



def _pick_viable_model_name(*, requested_model: str, lane_row: dict[str, Any], request_context_tokens: int | None) -> str | None:
    """Choose a concrete model artifact for a generic request tag.

    Requests often use family:size tags (e.g. qwen3.5:9b). MW/host backends need a
    concrete host-local model identifier (file path, artifact name, etc.).
    Prefer local artifacts; fall back to remote if present.
    """
    for group in ("local_viable_models", "remote_viable_models"):
        for item in (lane_row.get(group) or []):
            if not _model_item_allowed(item):
                continue
            if not _model_matches_request(requested_model, item.get("model_name"), item.get("tags") or []):
                continue
            if not _context_is_sufficient(request_context_tokens, item.get("max_ctx")):
                continue
            name = str(item.get("model_name") or "").strip()
            if name:
                return name
    return None

def _model_item_allowed(item: dict[str, Any]) -> bool:
    return item.get("allowed") is not False


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

    out = {key for key in keys if key}
    out |= _family_size_tags_from_keys(out)
    return out


def _candidate_tags_with_inferred(candidate_model: str | None, candidate_tags: list[str] | None) -> set[str]:
    return _normalized_model_tags(candidate_tags) | _inferred_model_tags(candidate_model)


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
        if (
            candidate == requested_model
            or candidate_stem == requested_model
            or candidate_parent == requested_model
        ):
            return True
    request_keys = _model_lookup_keys(requested_model)
    if request_keys & _model_lookup_keys(candidate):
        return True
    return bool(request_keys & _candidate_tags_with_inferred(candidate, candidate_tags))


def pick_lane_for_model(
    *,
    model: str,
    backend_type: str | None = None,
    request_context_tokens: int | None = None,
    requires_multimodal: bool = False,
    pin_worker: str | None = None,
    pin_base_url: str | None = None,
    pin_lane_type: str | None = None,
    pin_lane_id: str | None = None,
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

    def _augment_declared_models(row: dict[str, Any]) -> None:
        meta = row.get("proxy_auth_metadata") or {}
        if not isinstance(meta, dict):
            return
        declared = meta.get("declared_models") or meta.get("supported_models") or []
        if not isinstance(declared, list):
            return
        tags_by_model = meta.get("declared_model_tags") if isinstance(meta.get("declared_model_tags"), dict) else {}
        max_ctx_by_model = meta.get("declared_max_ctx") if isinstance(meta.get("declared_max_ctx"), dict) else {}
        out: list[dict[str, Any]] = list(row.get("local_viable_models") or [])
        for name in declared:
            model_name = str(name or "").strip()
            if not model_name:
                continue
            out.append(
                {
                    "model_name": model_name,
                    "tags": list(tags_by_model.get(model_name) or []),
                    "max_ctx": max_ctx_by_model.get(model_name),
                    "allowed": True,
                }
            )
        row["local_viable_models"] = out

    if pin_lane_id:
        try:
            uuid.UUID(str(pin_lane_id))
        except Exception as exc:
            raise LanePlacementError("pinned lane_id must be a UUID", status_code=400) from exc
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
                    JOIN hosts h ON h.host_id = l.host_id
                    LEFT JOIN models cm ON cm.model_name = l.current_model_name
                    LEFT JOIN lane_model_policy cmp ON cmp.lane_id = l.lane_id AND cmp.model_id = cm.model_id
                    WHERE l.lane_id=%s
                      AND (%s::text IS NULL OR h.host_name=%s::text)
                      AND (%s::text IS NULL OR l.base_url=%s::text)
                      AND (%s::text IS NULL OR l.backend_type = %s::text)
                      AND (%s::text IS NULL OR l.lane_type::text = %s::text)
                      AND (%s::text[] IS NULL OR l.lane_id::text <> ALL(%s::text[]))
                      AND (l.status='ready' OR (l.proxy_auth_metadata->>'control_plane')='mw')
                      AND NOT EXISTS (
                        SELECT 1 FROM router_leases rl
                        WHERE rl.lane_id = l.lane_id
                          AND rl.state = 'active'
                          AND COALESCE(rl.last_heartbeat_at, rl.acquired_at) > now() - (%s * interval '1 second')
                      )
                    LIMIT 1
                    """,
                    (
                        pin_lane_id,
                        pin_worker,
                        pin_worker,
                        pin_base_url,
                        pin_base_url,
                        backend_type,
                        backend_type,
                        pin_lane_type,
                        pin_lane_type,
                        list(excluded) or None,
                        list(excluded) or None,
                        int(settings.default_lease_stale_seconds),
                    ),
                )
        if not rows:
            raise LanePlacementError("pinned lane_id not found or not eligible", status_code=404)
        apply_mw_effective_status(rows, mw_state_db=mw_state_db, stale_seconds=settings.default_lease_stale_seconds)
        r0 = rows[0]
        if pin_lane_type and str(r0.get("lane_type") or "") != str(pin_lane_type):
            raise LanePlacementError("pinned lane does not match requested lane_type", status_code=409)
        if not _backend_matches_request(r0, backend_type):
            raise LanePlacementError("pinned lane does not match requested backend", status_code=409)
        eff = str(r0.get("effective_status") or r0.get("status") or "")
        if eff != "ready":
            raise LanePlacementError("pinned lane is not ready", status_code=409)
        return LaneChoice(
            lane_id=str(r0["lane_id"]),
            worker_id=str(r0["host_name"]),
            base_url=str(r0["base_url"]),
            lane_type=str(r0["lane_type"]),
            backend_type=str(r0.get("backend_type") or "llama"),
            current_model_name=r0.get("current_model_name"),
            current_model_max_ctx=int(r0["current_model_max_ctx"]) if r0.get("current_model_max_ctx") is not None else None,
            resolved_model_name=str(r0.get("current_model_name") or "").strip() or None,
        )

    if pin_worker and pin_base_url:
        with db.connect() as conn:
            with conn.cursor() as cur:
                rows = q(
                    cur,
                    """
                    SELECT l.lane_id, l.lane_name, h.host_name, l.base_url, l.lane_type, l.backend_type,
                           l.status,
                           l.proxy_auth_metadata,
                           l.current_model_name,
                           m.tags AS current_model_tags,
                           cmp.max_ctx AS current_model_max_ctx
                    FROM lanes l
                    JOIN hosts h ON h.host_id=l.host_id
                    LEFT JOIN models cm ON cm.model_name=l.current_model_name
                    LEFT JOIN models m ON m.model_name=l.current_model_name
                    LEFT JOIN lane_model_policy cmp ON cmp.lane_id=l.lane_id AND cmp.model_id=cm.model_id
                    WHERE h.host_name=%s
                      AND (l.status IN ('ready', 'suspended') OR (l.proxy_auth_metadata->>'control_plane')='mw')
                      AND (%s::text IS NULL OR l.backend_type = %s::text)
                      AND (%s::text IS NULL OR l.lane_type::text = %s::text)
                      AND (%s::text[] IS NULL OR l.lane_id::text <> ALL(%s::text[]))
                    ORDER BY
                      CASE l.lane_type WHEN 'gpu' THEN 0 WHEN 'mlx' THEN 1 WHEN 'cpu' THEN 2 ELSE 9 END,
                      l.base_url ASC
                    LIMIT 20
                    """,
                    (
                        pin_worker,
                        backend_type,
                        backend_type,
                        pin_lane_type,
                        pin_lane_type,
                        list(excluded) or None,
                        list(excluded) or None,
                    ),
                )
        apply_mw_effective_status(rows, mw_state_db=mw_state_db, stale_seconds=settings.default_lease_stale_seconds)
        rows = [
            r for r in rows
            if str(r.get("host_name") or "") == str(pin_worker)
            and str(r.get("base_url") or "") == str(pin_base_url)
            and _backend_matches_request(r, backend_type)
        ]
        if not rows:
            raise RuntimeError("pinned lane not found")
        r0 = rows[0]
        return LaneChoice(
            lane_id=str(r0["lane_id"]),
            worker_id=str(r0["host_name"]),
            base_url=str(r0["base_url"]),
            lane_type=str(r0["lane_type"]),
            backend_type=str(r0.get("backend_type") or "llama"),
            current_model_name=r0.get("current_model_name"),
            current_model_max_ctx=int(r0["current_model_max_ctx"]) if r0.get("current_model_max_ctx") is not None else None,
            resolved_model_name=str(r0.get("current_model_name") or "").strip() or None,
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
        apply_mw_effective_status(rows, mw_state_db=mw_state_db, stale_seconds=settings.default_lease_stale_seconds)
        # Defensive: even if the SQL layer changes, pinning must never route to a different host.
        rows = [
            r for r in rows
            if str(r.get("host_name") or "") == str(pin_worker)
            and _backend_matches_request(r, backend_type)
        ]

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
            resolved_model_name=str(r0.get("current_model_name") or "").strip() or None,
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
                            'max_ctx', p.max_ctx,
                            'allowed', COALESCE(p.allowed, true)
                          )
                          ORDER BY m.model_name
                        )
                        FROM lane_model_viability lmv
                        JOIN host_model_artifacts hma ON hma.artifact_id = lmv.artifact_id
                        JOIN models m ON m.model_id=lmv.model_id
                        LEFT JOIN lane_model_policy p ON p.lane_id=l.lane_id AND p.model_id=lmv.model_id
                        WHERE lmv.lane_id=l.lane_id
                          AND lmv.is_viable=true
                          AND lmv.source_locality='local'
                          AND COALESCE(hma.present, false)=true
                          AND (p.allowed IS DISTINCT FROM false)
                      ), '[]'::jsonb) as local_viable_models,
                      COALESCE((
                        SELECT jsonb_agg(
                          jsonb_build_object(
                            'model_name', m.model_name,
                            'tags', COALESCE(m.tags, '{}'::text[]),
                            'max_ctx', p.max_ctx,
                            'allowed', COALESCE(p.allowed, true)
                          )
                          ORDER BY m.model_name
                        )
                        FROM lane_model_viability lmv
                        JOIN host_model_artifacts hma ON hma.artifact_id = lmv.artifact_id
                        JOIN models m ON m.model_id=lmv.model_id
                        LEFT JOIN lane_model_policy p ON p.lane_id=l.lane_id AND p.model_id=lmv.model_id
                        WHERE lmv.lane_id=l.lane_id
                          AND lmv.is_viable=true
                          AND lmv.source_locality='remote'
                          AND COALESCE(hma.present, false)=true
                          AND (p.allowed IS DISTINCT FROM false)
                          AND jsonb_array_length(COALESCE(h.model_store_paths, '[]'::jsonb)) > 0
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
                          ELSE l.backend_type IN ('llama', 'mlx') OR l.backend_type IS NULL
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
        apply_mw_effective_status(rows, mw_state_db=mw_state_db, stale_seconds=settings.default_lease_stale_seconds)
        for row in rows:
            _augment_declared_models(row)
        rows = [row for row in rows if _backend_matches_request(row, backend_type)]

        if requires_multimodal:
            rows = [
                r
                for r in rows
                if isinstance(r.get("proxy_auth_metadata"), dict)
                and (r.get("proxy_auth_metadata") or {}).get("supports_multimodal") is True
            ]
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
                resolved_model_name=str(r0.get("current_model_name") or "").strip() or None,
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
                _model_item_allowed(item)
                and _model_matches_request(model, item.get("model_name"), item.get("tags") or [])
                and _context_is_sufficient(request_context_tokens, item.get("max_ctx"))
                for item in (row.get("local_viable_models") or [])
            )
            or any(
                _model_item_allowed(item)
                and _model_matches_request(model, item.get("model_name"), item.get("tags") or [])
                and _context_is_sufficient(request_context_tokens, item.get("max_ctx"))
                for item in (row.get("remote_viable_models") or [])
            )
        ]
        context_limited = [
            row for row in swappable_rows
            if (
                any(
                    _model_item_allowed(item)
                    and _model_matches_request(model, item.get("model_name"), item.get("tags") or [])
                    for item in (row.get("local_viable_models") or [])
                )
                or any(
                    _model_item_allowed(item)
                    and _model_matches_request(model, item.get("model_name"), item.get("tags") or [])
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
                resolved_model_name=str(r0.get("current_model_name") or "").strip() or None,
            )
        if context_limited and request_context_tokens:
            max_available_ctx = 0
            for row in context_limited:
                for group in ("local_viable_models", "remote_viable_models"):
                    for item in row.get(group) or []:
                        if _model_item_allowed(item) and _model_matches_request(model, item.get("model_name"), item.get("tags") or []):
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

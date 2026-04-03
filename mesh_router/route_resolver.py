from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .config import settings
from .db import db, mw_state_db
from .perf_registry import get_expectation
from .router import pick_lane_for_model


def _normalize_host_id(host_name: str) -> str:
    return (host_name or "").strip().lower().replace(" ", "-")


def _opportunistic_lane_ids(*, cur: Any) -> set[str]:
    opp = {h.strip() for h in (settings.opportunistic_hosts or "").split(",") if h.strip()}
    if not opp:
        return set()
    cur.execute(
        """
        SELECT l.lane_id
        FROM lanes l
        JOIN hosts h ON h.host_id=l.host_id
        WHERE h.host_name = ANY(%s)
        """,
        (sorted(opp),),
    )
    return {str(r["lane_id"]) for r in (cur.fetchall() or [])}


def _tag_model_candidates(tags: list[str], *, modality: str) -> list[str]:
    normalized = [t.strip().lower().replace("_", "-") for t in (tags or []) if t and t.strip()]
    for t in normalized:
        if t.startswith("model:"):
            name = t.split(":", 1)[1].strip()
            if name:
                return [name]

    cap = "text"
    if "embeddings" in normalized:
        cap = "embeddings"
    if "image-gen" in normalized or "images" in normalized:
        cap = "image-gen"
    if modality == "images":
        cap = "image-gen"
    if modality == "embeddings":
        cap = "embeddings"

    behavior = None
    for b in ("fast", "balanced", "smart", "cheap"):
        if b in normalized:
            behavior = b
            break

    key = f"{cap}:{behavior}" if behavior else cap
    # Deterministic built-in defaults. Operators can override by specifying model explicitly.
    mapping: dict[str, list[str]] = {
        "text:fast": ["qwen3.5-9b", "qwen3.5-4b", "qwen3.5-2b"],
        "text:balanced": ["qwen3.5-9b", "qwen3.5-27b"],
        "text:smart": ["qwen3.5-27b", "qwen3.5-9b"],
        "text:cheap": ["qwen3.5-2b", "qwen3.5-0.8b"],
        "text": ["qwen3.5-9b"],
        "embeddings": ["nomic-embed-text"],
        "image-gen": ["flux1-schnell", "flux.1-schnell"],
    }
    return list(mapping.get(key) or mapping.get(cap) or ["qwen3.5-9b"])


def resolve_route(
    *,
    model: str | None,
    modality: str,
    tags: list[str],
    host_name: str | None,
    lane_id: str | None,
    allow_opportunistic: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None, int | None]:
    """
    Returns: (choice_dict, perf_dict, reason, candidates_considered)
    """
    if host_name and lane_id:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT l.lane_id, l.lane_name, l.base_url, l.lane_type, l.backend_type, l.current_model_name,
                           h.host_name
                    FROM lanes l JOIN hosts h ON h.host_id=l.host_id
                    WHERE l.lane_id=%s AND h.host_name=%s
                    """,
                    (lane_id, host_name),
                )
                row = cur.fetchone()
                if not row:
                    return None, None, "explicit lane not found", None
                choice = {
                    "lane_id": str(row["lane_id"]),
                    "worker_id": str(row["host_name"]),
                    "base_url": str(row["base_url"]),
                    "lane_type": str(row.get("lane_type") or ""),
                    "backend_type": str(row.get("backend_type") or "llama"),
                    "current_model_name": row.get("current_model_name"),
                }
        perf = _perf_for_choice(choice, model=model, modality=modality)
        return choice, perf, None, 1

    candidates = [model] if model else _tag_model_candidates(tags, modality=modality)
    excluded_lane_ids: set[str] = set()
    if not allow_opportunistic:
        with db.connect() as conn:
            with conn.cursor() as cur:
                excluded_lane_ids |= _opportunistic_lane_ids(cur=cur)

    for idx, cand_model in enumerate(candidates, start=1):
        try:
            choice_obj = pick_lane_for_model(
                model=cand_model,
                backend_type="sd" if modality == "images" else None,
                pin_worker=host_name,
                exclude_lane_ids=excluded_lane_ids,
            )
        except Exception:
            continue
        choice = {
            "lane_id": choice_obj.lane_id,
            "worker_id": choice_obj.worker_id,
            "base_url": choice_obj.base_url,
            "lane_type": choice_obj.lane_type,
            "backend_type": choice_obj.backend_type,
            "current_model_name": choice_obj.current_model_name,
            "resolved_model": cand_model,
        }
        perf = _perf_for_choice(choice, model=cand_model, modality=modality)
        return choice, perf, None, idx

    return None, None, "no eligible route found for tags/model constraints", len(candidates)


def _perf_for_choice(choice: dict[str, Any], *, model: str | None, modality: str) -> dict[str, Any] | None:
    lane_id = str(choice.get("lane_id") or "")
    host_name = str(choice.get("worker_id") or "")
    host_id = _normalize_host_id(host_name)
    model_name = str(model or choice.get("resolved_model") or choice.get("current_model_name") or "")
    if not lane_id or not host_id or not model_name:
        return None
    try:
        with mw_state_db.connect() as conn:
            with conn.cursor() as cur:
                exp = get_expectation(cur=cur, host_id=host_id, lane_id=lane_id, model_name=model_name, modality=modality)
                if not exp:
                    return None
                now = datetime.now(tz=UTC)
                staleness_s = (now - exp.updated_at).total_seconds()
                return {
                    "host_id": exp.host_id,
                    "lane_id": exp.lane_id,
                    "model_name": exp.model_name,
                    "modality": exp.modality,
                    "updated_at": exp.updated_at.isoformat(),
                    "sample_count": exp.sample_count,
                    "first_token_ms_p50": exp.first_token_ms_p50,
                    "decode_tps_p50": exp.decode_tps_p50,
                    "total_ms_p50": exp.total_ms_p50,
                    "staleness_s": staleness_s,
                    "source": "observations",
                }
    except Exception:
        return None

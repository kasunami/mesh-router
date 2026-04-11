from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .config import settings
from .db import db, mw_state_db
from .perf_registry import get_expectation
from .router import pick_lane_for_model
from .mw_overlay import apply_mw_effective_status


MODEL_SELECTION_TAGS: dict[str, str] = {
    "qwen3.5:0.8b": "qwen3.5:0.8B",
    "qwen3.5:2b": "qwen3.5:2B",
    "qwen3.5:4b": "qwen3.5:4B",
    "qwen3.5:9b": "qwen3.5:9B",
    "qwen3.5:27b": "qwen3.5:27B",
    "falcon3:10b": "falcon3:10B",
    "lfm2.5:350m": "lfm2.5:350M",
    "gemma4:26b": "gemma4:26B",
}


def _normalize_host_id(host_name: str) -> str:
    return (host_name or "").strip().lower().replace(" ", "-").replace("_", "-")


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
        if t in MODEL_SELECTION_TAGS:
            return [MODEL_SELECTION_TAGS[t]]

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
        "text:fast": ["qwen3.5:9B", "qwen3.5:4B", "qwen3.5:2B"],
        "text:balanced": ["qwen3.5:9B", "qwen3.5:27B"],
        "text:smart": ["qwen3.5:27B", "qwen3.5:9B"],
        "text:cheap": ["qwen3.5:2B", "qwen3.5:0.8B"],
        "text": ["qwen3.5:9B"],
        "embeddings": ["nomic-embed-text"],
        "image-gen": ["flux1-schnell", "flux.1-schnell"],
    }
    return list(mapping.get(key) or mapping.get(cap) or ["qwen3.5:9B"])


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
        row: dict[str, Any] | None
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT l.lane_id, l.lane_name, l.base_url, l.lane_type, l.backend_type, l.current_model_name,
                           l.proxy_auth_metadata,
                           h.host_name
                    FROM lanes l JOIN hosts h ON h.host_id=l.host_id
                    WHERE l.lane_id=%s AND h.host_name=%s
                    """,
                    (lane_id, host_name),
                )
                fetched = cur.fetchone()
                row = dict(fetched) if fetched else None
        if not row:
            return None, None, "explicit lane not found", None

        # If lane is MW-managed, overlay the MW-derived actual model/readiness for truthfulness.
        apply_mw_effective_status([row], mw_state_db=mw_state_db, stale_seconds=settings.default_lease_stale_seconds)

        choice = {
            "lane_id": str(row["lane_id"]),
            "worker_id": str(row["host_name"]),
            "base_url": str(row["base_url"]),
            "lane_type": str(row.get("lane_type") or ""),
            "backend_type": str(row.get("backend_type") or "llama"),
            "current_model_name": row.get("current_model_name"),
            "effective_status": row.get("effective_status"),
        }
        perf = _perf_for_choice(choice, model=model, modality=modality)
        return choice, perf, None, 1

    candidates = [model] if model else _tag_model_candidates(tags, modality=modality)
    excluded_lane_ids: set[str] = set()
    if not allow_opportunistic:
        with db.connect() as conn:
            with conn.cursor() as cur:
                excluded_lane_ids |= _opportunistic_lane_ids(cur=cur)

    best_choice: dict[str, Any] | None = None
    best_perf: dict[str, Any] | None = None
    best_key: tuple[int, float, float] | None = None

    def _rank_key(perf: dict[str, Any] | None) -> tuple[int, float, float]:
        """
        Deterministic ranking:
        - Prefer candidates with perf data.
        - For chat/embeddings: maximize decode TPS, then minimize first-token latency.
        - For images: minimize total_ms.
        """

        if not perf:
            return (0, 0.0, 0.0)
        if modality == "images":
            total = perf.get("total_ms_p50")
            return (1, -float(total) if total is not None else 0.0, 0.0)
        tps = perf.get("decode_tps_p50")
        first = perf.get("first_token_ms_p50")
        return (1, float(tps) if tps is not None else 0.0, -float(first) if first is not None else 0.0)

    for cand_model in candidates:
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
        key = _rank_key(perf)
        if best_key is None or key > best_key:
            best_key = key
            best_choice = choice
            best_perf = perf

    if best_choice:
        return best_choice, best_perf, None, len(candidates)

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

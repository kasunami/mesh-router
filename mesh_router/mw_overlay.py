from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from typing import Any
from urllib.parse import urlparse, urlunparse

from .runtime_state import RuntimeStateStore, get_default_runtime_state_store

logger = logging.getLogger(__name__)

_MW_PROGRESS_TIMEOUT_SECONDS = 180


def _parse_iso_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _is_job_hung(active_job: Any) -> bool:
    if not isinstance(active_job, dict):
        return False
    last_progress = active_job.get("last_progress_at")
    if not last_progress:
        return False
    dt = _parse_iso_dt(last_progress)
    if dt is None:
        return False
    return (datetime.now(tz=UTC) - dt).total_seconds() > _MW_PROGRESS_TIMEOUT_SECONDS


def _normalize_router_backend_type(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"llama.cpp", "llama"}:
        return "llama"
    if raw in {"bitnet.cpp", "bitnet"}:
        return "bitnet"
    if raw in {"stable-diffusion.cpp", "sd", "stable-diffusion"}:
        return "sd"
    if raw == "mlx":
        return "mlx"
    return raw


def _normalize_mw_host_id(host_name: str | None) -> str:
    return str(host_name or "").strip().lower().replace(" ", "-")


def is_explicit_mw_managed(row: dict[str, Any]) -> bool:
    meta = row.get("proxy_auth_metadata") or {}
    return isinstance(meta, dict) and str(meta.get("control_plane") or "").strip().lower() == "mw"


def _candidate_mw_binding(row: dict[str, Any]) -> tuple[str, str, bool] | None:
    pam = row.get("proxy_auth_metadata") or {}
    if isinstance(pam, dict) and pam.get("mw_ignore") is True:
        return None
    if is_explicit_mw_managed(row):
        host_id = str(pam.get("mw_host_id") or "").strip() or _normalize_mw_host_id(str(row.get("host_name") or ""))
        lane_id = str(pam.get("mw_lane_id") or "").strip()
        if not lane_id:
            lane_name = str(row.get("lane_name") or "").strip()
            lane_type = str(row.get("lane_type") or "").strip().lower()
            lane_id = lane_name or (lane_type if lane_type in {"cpu", "gpu", "combined", "mlx"} else "")
        if host_id and lane_id:
            return host_id, lane_id, False
        return None

    # Legacy MR rows may not be explicitly tagged MW-managed even though MW publishes state for the host/lane.
    host_id = _normalize_mw_host_id(str(row.get("host_name") or ""))
    lane_name = str(row.get("lane_name") or "").strip()
    lane_type = str(row.get("lane_type") or "").strip().lower()
    inferred_lane_id = lane_name or (lane_type if lane_type in {"cpu", "gpu", "combined", "mlx"} else "")
    # Legacy MR MLX rows often use lane_name="mlx" while MW publishes the live lane as lane_id="gpu"
    # with lane_type="mlx". Prefer the MW lane id when we can infer that relationship.
    if lane_type == "mlx" and inferred_lane_id == "mlx":
        inferred_lane_id = "gpu"
    if host_id and inferred_lane_id:
        return host_id, inferred_lane_id, True
    return None


def _mw_effective_status_and_reason(
    fact: dict[str, Any],
    *,
    stale_cutoff: datetime,
) -> tuple[str, str | None]:
    hb = fact.get("last_heartbeat_at")
    actual_state = str(fact.get("actual_state") or "").strip()
    health_status = str(fact.get("health_status") or "").strip()

    if not fact:
        return "offline", "mw_state_missing"
    if not hb or not isinstance(hb, datetime):
        return "offline", "stale_heartbeat"
    if hb < stale_cutoff:
        return "offline", "stale_heartbeat"
    if actual_state not in {"running", "ready"}:
        return "offline", "not_running"
    if health_status != "healthy":
        return "offline", "unhealthy"
    return "ready", None


def _base_url_with_listen_port(base_url: str | None, *, listen_port: Any) -> str | None:
    raw_url = str(base_url or "").strip()
    if not raw_url:
        return None
    try:
        port = int(listen_port)
    except (TypeError, ValueError):
        return None
    if port <= 0:
        return None
    parsed = urlparse(raw_url)
    hostname = parsed.hostname
    if not hostname or not parsed.scheme:
        return None
    if ":" in hostname and not hostname.startswith("["):
        host = f"[{hostname}]"
    else:
        host = hostname
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        host = f"{auth}@{host}"
    return urlunparse(parsed._replace(netloc=f"{host}:{port}"))


def apply_mw_effective_status(
    rows: list[dict[str, Any]],
    *,
    mw_state_db: Any,
    stale_seconds: int,
    runtime_store: RuntimeStateStore | None = None,
) -> None:
    """
    Enrich lane rows (from the router DB) with MW-derived readiness + current model when lanes are
    MW-managed and MW state lives in a separate DB.

    This is a *best-effort overlay*:
    - If MW state DB is unavailable, rows are marked offline with a reason.
    - Rows are modified in place with:
      - effective_status: "ready" | "offline" (when MW-managed lanes can be interpreted)
      - readiness_reason: machine-readable exclusion reason when not ready
      - current_model_name: set to MW `actual_model` when present (for more truthful inventory)
    """
    mw_pairs: list[tuple[str, str]] = []
    explicit_bindings: set[tuple[str, str]] = set()
    for row in rows:
        binding = _candidate_mw_binding(row)
        if binding is None:
            continue
        mw_pairs.append((binding[0], binding[1]))
        if not binding[2]:
            explicit_bindings.add((binding[0], binding[1]))
    mw_pairs = list(dict.fromkeys(mw_pairs))
    if not mw_pairs:
        return

    facts: dict[tuple[str, str], dict[str, Any]] = {}

    if runtime_store is None:
        runtime_store = get_default_runtime_state_store()
    if runtime_store is not None:
        try:
            facts.update(runtime_store.get_lane_facts(mw_pairs, stale_seconds=stale_seconds))
        except Exception as exc:  # pragma: no cover - defensive guard around cache plugins/fakes
            logger.warning("MW runtime-state cache read failed: %s", exc)

    # When cache is cold, fall back to mw_state_db for both explicit and inferred lanes.
    # If MW is fully unavailable (no cache + no DB), we mark lanes offline with a reason.
    missing_pairs = [pair for pair in mw_pairs if pair not in facts]
    db_unavailable = False
    if missing_pairs:
        try:
            with mw_state_db.connect() as conn:
                with conn.cursor() as cur:
                    values_sql = ",".join(["(%s,%s)"] * len(missing_pairs))
                    params: list[Any] = []
                    for host_id, lane_id in missing_pairs:
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
                          ml.desired_model,
                          ml.actual_model,
                          ml.backend_type,
                          ml.metadata,
                          ml.service_id,
                          ms.listen_port
                        FROM wanted w
                        LEFT JOIN mw_hosts mh ON mh.host_id = w.host_id
                        LEFT JOIN mw_lanes ml ON ml.host_id = w.host_id AND ml.lane_id = w.lane_id
                        LEFT JOIN mw_services ms ON ms.host_id = ml.host_id AND ms.service_id = ml.service_id
                        """,
                        tuple(params),
                    )
                    for r in cur.fetchall():
                        key = (str(r["host_id"]), str(r["lane_id"]))
                        facts[key] = dict(r)
        except Exception:
            db_unavailable = True

    if db_unavailable:
        for row in rows:
            binding = _candidate_mw_binding(row)
            if binding is None or binding[2]:
                continue
            if (binding[0], binding[1]) in facts:
                continue
            row["effective_status"] = "offline"
            row["readiness_reason"] = "mw_state_unavailable"

    # If both cache and DB are missing for an explicit MW lane, mark as offline with cache miss.
    for row in rows:
        binding = _candidate_mw_binding(row)
        if binding is None or binding[2]:
            continue
        if (binding[0], binding[1]) not in facts and not db_unavailable:
            row["effective_status"] = "offline"
            row["readiness_reason"] = "mw_cache_miss"

    now = datetime.now(tz=UTC)
    stale_cutoff = now - timedelta(seconds=int(stale_seconds))
    for row in rows:
        binding = _candidate_mw_binding(row)
        if binding is None:
            continue
        host_id, lane_id, inferred = binding
        # Explicit MW lane with no cache entry: already marked offline above; skip enrichment.
        if not inferred and (host_id, lane_id) not in facts:
            continue
        if db_unavailable and (host_id, lane_id) not in facts:
            continue
        f = facts.get((host_id, lane_id)) or {}
        if inferred and not any(
            f.get(key) is not None
            for key in ("last_heartbeat_at", "actual_state", "health_status", "actual_model", "backend_type")
        ):
            continue
        effective_status, readiness_reason = _mw_effective_status_and_reason(
            f,
            stale_cutoff=stale_cutoff,
        )
        if f.get("last_heartbeat_at") is not None:
            row["mw_last_heartbeat_at"] = f.get("last_heartbeat_at")
        metadata = f.get("metadata") or {}
        row_backend = _normalize_router_backend_type(str(row.get("backend_type") or ""))
        fact_backend = _normalize_router_backend_type(str(f.get("backend_type") or ""))
        metadata_backend = ""
        if isinstance(metadata, dict):
            metadata_backend = _normalize_router_backend_type(str(metadata.get("current_backend_type") or ""))
        shared_explicit_binding = inferred and (host_id, lane_id) in explicit_bindings
        current_backend_type = metadata_backend or fact_backend
        backend_conflicts = bool(
            row_backend
            and current_backend_type
            and row_backend != current_backend_type
            and (row_backend == "sd" or current_backend_type == "sd")
        )
        if shared_explicit_binding and backend_conflicts:
            row["effective_status"] = "offline"
            row["readiness_reason"] = "backend_mismatch"
            continue
        if not inferred and backend_conflicts:
            row["effective_status"] = "offline"
            row["readiness_reason"] = "backend_mismatch"
            continue
        row["effective_status"] = effective_status
        row["readiness_reason"] = readiness_reason

        # Multimodal capability overlay:
        # - A lane is considered multimodal-capable only when MW is running a VLM-configured service.
        # - Keep this as a lane capability (not a model alias) so the same model can be served as
        #   LLM on CPU lanes and as LMM on GPU lanes.
        try:
            service_id = str(f.get("service_id") or "").strip()
            pam = row.get("proxy_auth_metadata") or {}
            if not isinstance(pam, dict):
                pam = {}
            mm = False
            if isinstance(metadata, dict) and metadata.get("supports_multimodal") is True:
                mm = True
            if service_id in {"llama-vlm", "llama-vlm.service"}:
                mm = True
            pam["supports_multimodal"] = bool(mm)
            row["proxy_auth_metadata"] = pam
        except Exception:
            pass

        if f.get("actual_model"):
            row["current_model_name"] = f.get("actual_model")
        if f.get("desired_model"):
            row["desired_model_name"] = f.get("desired_model")
        if current_backend_type:
            row["backend_type"] = current_backend_type
        if isinstance(metadata, dict):
            try:
                if metadata.get("actual_model_max_ctx") is not None:
                    row["current_model_max_ctx"] = int(metadata["actual_model_max_ctx"])
            except (TypeError, ValueError):
                pass
            if metadata.get("source") is not None:
                row["mw_state_source"] = metadata.get("source")
            for key in (
                "current_backend_type",
                "desired_backend_type",
                "backend_swap_required",
                "model_swap_required",
                "backend_swap_eta_ms",
                "model_swap_eta_ms",
                "total_swap_eta_ms",
                "eta_source",
                "eta_complete",
            ):
                if key in metadata:
                    row[key] = metadata.get(key)
        effective_base_url = _base_url_with_listen_port(row.get("base_url"), listen_port=f.get("listen_port"))
        if effective_base_url:
            row["base_url"] = effective_base_url

        validated_candidates = f.get("validated_candidates")
        if validated_candidates is not None:
            row["validated_candidates"] = validated_candidates

        active_job = f.get("active_job")
        if isinstance(active_job, dict):
            row["active_job"] = active_job
            row["job_hung"] = _is_job_hung(active_job)
        elif "active_job" not in row:
            row.setdefault("active_job", None)
            row.setdefault("job_hung", False)

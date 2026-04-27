from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import median
from typing import Any


_TABLE = "mw_perf_observations"


def normalize_host_id(host_id: str) -> str:
    """
    Canonicalize host ids to match MW state tables (e.g. mw_hosts.host_id / mw_lanes.host_id).

    Operators often use friendly host names like "Worker A"; MW state typically uses
    normalized ids like "worker-a". We normalize at ingest to avoid silent mismatches
    when looking up perf expectations during routing.
    """

    return (host_id or "").strip().lower().replace(" ", "-").replace("_", "-")


@dataclass(frozen=True)
class PerfExpectation:
    host_id: str
    lane_id: str
    model_name: str
    modality: str
    updated_at: datetime
    sample_count: int
    first_token_ms_p50: float | None
    decode_tps_p50: float | None
    total_ms_p50: float | None


def _table_exists(*, cur: Any) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema='public' AND table_name=%s
        """,
        (_TABLE,),
    )
    return bool(cur.fetchone())


def insert_observation(*, cur: Any, obs: dict[str, Any]) -> None:
    if not _table_exists(cur=cur):
        raise RuntimeError(f"missing table: {_TABLE}")
    try:
        from psycopg.types.json import Jsonb  # type: ignore
    except Exception:  # pragma: no cover
        Jsonb = None  # type: ignore

    payload = dict(obs)
    host_id_in = str(payload.get("host_id") or "")
    host_id_norm = normalize_host_id(host_id_in)
    if host_id_norm:
        payload["host_id"] = host_id_norm

    md = payload.get("metadata")
    if md is None:
        md = {}
    if host_id_in and host_id_norm and host_id_in != host_id_norm:
        md = dict(md)
        md.setdefault("host_id_input", host_id_in)
    if Jsonb is not None:
        payload["metadata"] = Jsonb(md)
    else:
        payload["metadata"] = md
    cur.execute(
        f"""
        INSERT INTO {_TABLE}(
          observed_at,
          host_id,
          lane_id,
          model_name,
          backend_type,
          lane_type,
          modality,
          prompt_tokens,
          generated_tokens,
          first_token_ms,
          decode_tps,
          total_ms,
          was_cold,
          ok,
          error_kind,
          error_message,
          metadata
        ) VALUES (
          NOW(),
          %(host_id)s,
          %(lane_id)s,
          %(model_name)s,
          %(backend_type)s,
          %(lane_type)s,
          %(modality)s,
          %(prompt_tokens)s,
          %(generated_tokens)s,
          %(first_token_ms)s,
          %(decode_tps)s,
          %(total_ms)s,
          %(was_cold)s,
          %(ok)s,
          %(error_kind)s,
          %(error_message)s,
          %(metadata)s
        )
        """,
        payload,
    )


def get_expectation(
    *,
    cur: Any,
    host_id: str,
    lane_id: str,
    model_name: str,
    modality: str,
    lookback_n: int = 20,
) -> PerfExpectation | None:
    if not _table_exists(cur=cur):
        return None
    rows: list[dict[str, Any]] = []
    host_id_try = [host_id]
    host_id_norm = normalize_host_id(host_id)
    if host_id_norm and host_id_norm != host_id:
        host_id_try.append(host_id_norm)
    for hid in host_id_try:
        cur.execute(
            f"""
            SELECT
              observed_at,
              first_token_ms,
              decode_tps,
              total_ms
            FROM {_TABLE}
            WHERE host_id=%s AND lane_id=%s AND model_name=%s AND modality=%s
              AND ok=true
            ORDER BY observed_at DESC
            LIMIT %s
            """,
            (hid, lane_id, model_name, modality, int(lookback_n)),
        )
        rows = cur.fetchall() or []
        if rows:
            host_id = hid
            break
    if not rows:
        return None
    first = [float(r["first_token_ms"]) for r in rows if r.get("first_token_ms") is not None]
    tps = [float(r["decode_tps"]) for r in rows if r.get("decode_tps") is not None]
    total = [float(r["total_ms"]) for r in rows if r.get("total_ms") is not None]
    updated_at = rows[0]["observed_at"]
    if updated_at and updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return PerfExpectation(
        host_id=host_id,
        lane_id=lane_id,
        model_name=model_name,
        modality=modality,
        updated_at=updated_at or datetime.now(tz=UTC),
        sample_count=len(rows),
        first_token_ms_p50=median(first) if first else None,
        decode_tps_p50=median(tps) if tps else None,
        total_ms_p50=median(total) if total else None,
    )

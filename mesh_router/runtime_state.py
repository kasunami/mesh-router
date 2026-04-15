from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Iterable

from redis import Redis, RedisError

from .config import settings

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _parse_datetime(value: Any) -> datetime | None:
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


class RuntimeStateStore:
    """Redis-backed cache for fast-changing MW host/lane runtime state.

    Postgres remains the audit/fallback store. This cache is intentionally TTL-bound so stale
    host-local runtime truth expires if MW stops publishing.
    """

    def __init__(self, client: Redis, *, prefix: str = "mr:mw") -> None:
        self.client = client
        self.prefix = prefix.rstrip(":")

    def host_key(self, host_id: str) -> str:
        return f"{self.prefix}:host:{host_id}:snapshot"

    def lane_key(self, host_id: str, lane_id: str) -> str:
        return f"{self.prefix}:lane:{host_id}:{lane_id}"

    def service_key(self, host_id: str, service_id: str) -> str:
        return f"{self.prefix}:service:{host_id}:{service_id}"

    def write_host_snapshot(
        self,
        *,
        host_id: str,
        snapshot: dict[str, Any],
        observed_at: datetime,
        ttl_seconds: int,
    ) -> None:
        ttl = max(1, int(ttl_seconds))
        observed_iso = observed_at.astimezone(UTC).isoformat()
        service_states = [dict(s) for s in (snapshot.get("service_states") or []) if isinstance(s, dict)]
        lane_states = [dict(l) for l in (snapshot.get("lane_states") or []) if isinstance(l, dict)]
        service_by_id = {str(s.get("service_id") or ""): s for s in service_states if str(s.get("service_id") or "")}

        host_payload = {
            "host_id": host_id,
            "observed_at": observed_iso,
            "actual_profile": snapshot.get("actual_profile"),
            "service_count": len(service_states),
            "lane_count": len(lane_states),
        }
        self._set_json(self.host_key(host_id), host_payload, ttl)

        for service in service_states:
            service_id = str(service.get("service_id") or "")
            if not service_id:
                continue
            payload = {"host_id": host_id, "observed_at": observed_iso, **service}
            self._set_json(self.service_key(host_id, service_id), payload, ttl)

        for lane in lane_states:
            lane_id = str(lane.get("lane_id") or "")
            if not lane_id:
                continue
            service_id = str(lane.get("service_id") or "")
            service = service_by_id.get(service_id) or {}
            metadata = {
                "source": "mw_state_snapshot",
                "active_mode": lane.get("active_mode"),
                "actual_model_max_ctx": lane.get("actual_model_max_ctx"),
                "current_backend_type": lane.get("current_backend_type"),
                "desired_backend_type": lane.get("desired_backend_type"),
                "backend_swap_required": lane.get("backend_swap_required"),
                "model_swap_required": lane.get("model_swap_required"),
                "backend_swap_eta_ms": lane.get("backend_swap_eta_ms"),
                "model_swap_eta_ms": lane.get("model_swap_eta_ms"),
                "total_swap_eta_ms": lane.get("total_swap_eta_ms"),
                "eta_source": lane.get("eta_source"),
                "eta_complete": lane.get("eta_complete"),
            }
            payload = {
                "host_id": host_id,
                "lane_id": lane_id,
                "last_heartbeat_at": observed_iso,
                "actual_state": lane.get("actual_state"),
                "health_status": lane.get("health_status"),
                "desired_model": lane.get("desired_model"),
                "actual_model": lane.get("actual_model"),
                "backend_type": lane.get("backend_type"),
                "metadata": metadata,
                "service_id": service_id or None,
                "listen_port": service.get("listen_port"),
            }
            self._set_json(self.lane_key(host_id, lane_id), payload, ttl)

    def get_lane_facts(
        self,
        pairs: Iterable[tuple[str, str]],
        *,
        stale_seconds: int | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        del stale_seconds  # TTL and overlay stale cutoff both enforce freshness; keep signature future-proof.
        unique_pairs = list(dict.fromkeys((str(h), str(l)) for h, l in pairs))
        if not unique_pairs:
            return {}
        try:
            values = self.client.mget([self.lane_key(host_id, lane_id) for host_id, lane_id in unique_pairs])
        except RedisError as exc:
            logger.warning("MW runtime-state cache read failed: %s", exc)
            return {}

        facts: dict[tuple[str, str], dict[str, Any]] = {}
        for pair, raw in zip(unique_pairs, values, strict=False):
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            fact = dict(payload)
            fact["last_heartbeat_at"] = _parse_datetime(fact.get("last_heartbeat_at"))
            facts[pair] = fact
        return facts

    def _set_json(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        try:
            self.client.setex(key, ttl_seconds, json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")))
        except RedisError as exc:
            logger.warning("MW runtime-state cache write failed: %s", exc)


def create_runtime_state_store(redis_url: str | None) -> RuntimeStateStore | None:
    if not redis_url:
        return None
    try:
        client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.5,
        )
    except RedisError as exc:
        logger.warning("MW runtime-state cache init failed: %s", exc)
        return None
    return RuntimeStateStore(client)


@lru_cache(maxsize=1)
def get_default_runtime_state_store() -> RuntimeStateStore | None:
    return create_runtime_state_store(settings.runtime_state_redis_url)

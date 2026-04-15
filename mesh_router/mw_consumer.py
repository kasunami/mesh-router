from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Iterable

from confluent_kafka import Consumer, KafkaError
import psycopg
from psycopg.types.json import Jsonb
from psycopg.rows import dict_row

from .config import settings
from .runtime_state import RuntimeStateStore, get_default_runtime_state_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MwConsumerSettings:
    bootstrap_servers: str
    client_id: str
    consumer_group: str
    state_topic: str
    heartbeats_topic: str
    responses_topic: str


def _upsert_mw_host(
    cur: Any,
    *,
    host_id: str,
    actual_profile: str | None,
    last_heartbeat_at: datetime | None,
    last_state_at: datetime | None,
    notes_patch: dict[str, Any] | None,
) -> None:
    cur.execute(
        """
        INSERT INTO mw_hosts (
          host_id, host_name, platform, service_manager, config_version,
          desired_profile, actual_profile,
          last_heartbeat_at, last_state_at,
          status, notes, updated_at
        )
        VALUES (%s, %s, %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s, now())
        ON CONFLICT (host_id) DO UPDATE SET
          actual_profile = COALESCE(EXCLUDED.actual_profile, mw_hosts.actual_profile),
          last_heartbeat_at = COALESCE(EXCLUDED.last_heartbeat_at, mw_hosts.last_heartbeat_at),
          last_state_at = COALESCE(EXCLUDED.last_state_at, mw_hosts.last_state_at),
          status = EXCLUDED.status,
          notes = mw_hosts.notes || EXCLUDED.notes,
          updated_at = now()
        """,
        (
            host_id,
            host_id,
            "unknown",
            "unknown",
            1,
            None,
            actual_profile,
            last_heartbeat_at,
            last_state_at,
            "online",
            Jsonb(notes_patch or {}),
        ),
    )


def _insert_mw_heartbeat(cur: Any, *, host_id: str, heartbeat_at: datetime, details: dict[str, Any]) -> None:
    cur.execute(
        """
        INSERT INTO mw_heartbeats (host_id, heartbeat_at, agent_version, grpc_listening, kafka_connected, details)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            host_id,
            heartbeat_at,
            str(details.get("agent_version") or ""),
            bool(details.get("grpc_listening")) if details.get("grpc_listening") is not None else None,
            bool(details.get("kafka_connected")) if details.get("kafka_connected") is not None else None,
            Jsonb(details),
        ),
    )


def _upsert_services(cur: Any, *, host_id: str, service_states: Iterable[dict[str, Any]], observed_at: datetime) -> None:
    for s in service_states:
        cur.execute(
            """
            INSERT INTO mw_services (
              host_id, service_id, manager_name, backend_type, kind,
              desired_state, actual_state, listen_host, listen_port,
              health_status, last_health_at, metadata, updated_at
            )
            VALUES (%s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, now())
            ON CONFLICT (host_id, service_id) DO UPDATE SET
              desired_state = EXCLUDED.desired_state,
              actual_state = EXCLUDED.actual_state,
              listen_host = EXCLUDED.listen_host,
              listen_port = EXCLUDED.listen_port,
              health_status = EXCLUDED.health_status,
              last_health_at = EXCLUDED.last_health_at,
              metadata = mw_services.metadata || EXCLUDED.metadata,
              updated_at = now()
            """,
            (
                host_id,
                str(s.get("service_id") or ""),
                str(s.get("manager_name") or ""),
                str(s.get("backend_type") or ""),
                str(s.get("kind") or ""),
                str(s.get("desired_state") or "stopped"),
                str(s.get("actual_state") or "unknown"),
                str(s.get("listen_host") or ""),
                int(s.get("listen_port") or 0) or None,
                str(s.get("health_status") or "unknown"),
                observed_at,
                Jsonb({"source": "mw_state_snapshot"}),
            ),
        )


def _upsert_lanes(cur: Any, *, host_id: str, lane_states: Iterable[dict[str, Any]], observed_at: datetime) -> None:
    for l in lane_states:
        service_id = str(l.get("service_id") or "")
        if not service_id:
            # `mw_lanes.service_id` is FK'd to mw_services; skip malformed lane rows rather than breaking ingestion.
            continue
        actual_state = str(l.get("actual_state") or "unknown")
        health_status = str(l.get("health_status") or "unknown")
        desired_state = "online" if actual_state in {"running", "ready"} else "offline"
        last_healthy_at = observed_at if health_status == "healthy" else None
        cur.execute(
            """
            INSERT INTO mw_lanes (
              host_id, lane_id, lane_type, backend_type, service_id, resource_class,
              desired_model, actual_model, last_loaded_model,
              desired_state, actual_state, health_status, last_healthy_at,
              metadata, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s,
                    %s, %s,
                    %s,
                    %s, %s, %s, %s,
                    %s, now())
            ON CONFLICT (host_id, lane_id) DO UPDATE SET
              backend_type = EXCLUDED.backend_type,
              service_id = EXCLUDED.service_id,
              resource_class = EXCLUDED.resource_class,
              desired_model = EXCLUDED.desired_model,
              actual_model = EXCLUDED.actual_model,
              last_loaded_model = COALESCE(EXCLUDED.last_loaded_model, mw_lanes.last_loaded_model),
              desired_state = EXCLUDED.desired_state,
              actual_state = EXCLUDED.actual_state,
              health_status = EXCLUDED.health_status,
              last_healthy_at = COALESCE(EXCLUDED.last_healthy_at, mw_lanes.last_healthy_at),
              metadata = mw_lanes.metadata || EXCLUDED.metadata,
              updated_at = now()
            """,
            (
                host_id,
                str(l.get("lane_id") or ""),
                str(l.get("lane_type") or "unknown"),
                str(l.get("backend_type") or "unknown"),
                service_id,
                str(l.get("resource_class") or ""),
                str(l.get("desired_model") or "") or None,
                str(l.get("actual_model") or "") or None,
                str(l.get("last_loaded_model") or "") or None,
                desired_state,
                actual_state,
                health_status,
                last_healthy_at,
                Jsonb(
                    {
                        "source": "mw_state_snapshot",
                        "active_mode": l.get("active_mode"),
                        "actual_model_max_ctx": l.get("actual_model_max_ctx"),
                        "current_backend_type": l.get("current_backend_type"),
                        "desired_backend_type": l.get("desired_backend_type"),
                        "backend_swap_required": l.get("backend_swap_required"),
                        "model_swap_required": l.get("model_swap_required"),
                        "backend_swap_eta_ms": l.get("backend_swap_eta_ms"),
                        "model_swap_eta_ms": l.get("model_swap_eta_ms"),
                        "total_swap_eta_ms": l.get("total_swap_eta_ms"),
                        "eta_source": l.get("eta_source"),
                        "eta_complete": l.get("eta_complete"),
                    }
                ),
            ),
        )


def _upsert_transition(cur: Any, *, request_id: str, host_id: str, payload: dict[str, Any], observed_at: datetime) -> None:
    response_type = str(payload.get("response_type") or "")
    command_type = str(payload.get("command_type") or "")
    ok = payload.get("ok")
    error = payload.get("error") or {}
    error_message = error.get("message") if isinstance(error, dict) else str(error)
    cur.execute(
        """
        INSERT INTO mw_transitions (
          request_id, host_id, transition_type, status, source,
          started_at, completed_at, error_message, details, updated_at
        )
        VALUES (%s, %s, %s, %s, %s,
                %s, %s, %s, %s, now())
        ON CONFLICT (request_id) DO UPDATE SET
          status = EXCLUDED.status,
          error_message = COALESCE(EXCLUDED.error_message, mw_transitions.error_message),
          completed_at = COALESCE(EXCLUDED.completed_at, mw_transitions.completed_at),
          details = mw_transitions.details || EXCLUDED.details,
          updated_at = now()
        """,
        (
            request_id,
            host_id,
            command_type or "unknown",
            response_type or "unknown",
            "mw-kafka",
            observed_at if response_type in {"accepted", "started"} else None,
            observed_at if response_type in {"completed", "failed", "cancelled", "rejected"} else None,
            None if (ok is True or ok is None) else str(error_message or ""),
            Jsonb({"ok": ok, "payload": payload}),
        ),
    )


def _insert_transition_event(cur: Any, *, request_id: str, host_id: str, payload: dict[str, Any], observed_at: datetime) -> None:
    cur.execute(
        """
        INSERT INTO mw_transition_events (request_id, host_id, event_type, phase, status, sequence_no, message, error_message, details, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            request_id,
            host_id,
            str(payload.get("response_type") or "event"),
            str(payload.get("phase") or "") or None,
            str(payload.get("status") or "") or None,
            int(payload.get("sequence_no") or 0) or None,
            None,
            (payload.get("error") or {}).get("message") if isinstance(payload.get("error"), dict) else None,
            Jsonb(payload),
            observed_at,
        ),
    )



def _response_host_state_snapshot(body: dict[str, Any]) -> dict[str, Any] | None:
    result = body.get("result")
    if not isinstance(result, dict):
        return None
    host_state = result.get("host_state")
    if not isinstance(host_state, dict):
        return None
    if not isinstance(host_state.get("lane_states"), list) and not isinstance(host_state.get("service_states"), list):
        return None
    return host_state


def process_message(
    *,
    payload: dict[str, Any],
    observed_at: datetime,
    db_connect: Callable[[], Any],
    runtime_store: RuntimeStateStore | None = None,
) -> None:
    message_type = str(payload.get("message_type") or "")
    host_id = str(payload.get("host_id") or "")
    if not host_id:
        return

    with db_connect() as conn:
        with conn.cursor() as cur:
            if message_type == "heartbeat":
                details = dict(payload.get("payload") or {})
                actual_profile = str(details.get("actual_profile") or "") or None
                _upsert_mw_host(
                    cur,
                    host_id=host_id,
                    actual_profile=actual_profile,
                    last_heartbeat_at=observed_at,
                    last_state_at=None,
                    notes_patch={"mw": {"last_heartbeat": payload}},
                )
                _insert_mw_heartbeat(cur, host_id=host_id, heartbeat_at=observed_at, details={"actual_profile": actual_profile, **details})
            elif message_type == "state":
                snapshot = dict(payload.get("payload") or {})
                actual_profile = str(snapshot.get("actual_profile") or "") or None
                _upsert_mw_host(
                    cur,
                    host_id=host_id,
                    actual_profile=actual_profile,
                    last_heartbeat_at=None,
                    last_state_at=observed_at,
                    notes_patch={"mw": {"last_state": payload}},
                )
                service_states = list(snapshot.get("service_states") or [])
                lane_states = list(snapshot.get("lane_states") or [])
                # Some MW versions may omit `service_states` or not include every service_id referenced by lanes.
                # `mw_lanes` has an FK to `mw_services`, so ensure at least stub service rows exist.
                known_services: set[str] = {
                    str(s.get("service_id") or "") for s in service_states if str(s.get("service_id") or "")
                }
                for lane in lane_states:
                    service_id = str(lane.get("service_id") or "")
                    if not service_id or service_id in known_services:
                        continue
                    service_states.append(
                        {
                            "service_id": service_id,
                            "manager_name": "unknown",
                            "backend_type": str(lane.get("backend_type") or "unknown"),
                            "kind": "lane_backend",
                            "desired_state": "unknown",
                            "actual_state": "unknown",
                            "health_status": str(lane.get("health_status") or "unknown"),
                        }
                    )
                    known_services.add(service_id)
                _upsert_services(cur, host_id=host_id, service_states=service_states, observed_at=observed_at)
                _upsert_lanes(cur, host_id=host_id, lane_states=lane_states, observed_at=observed_at)
                if runtime_store is not None:
                    runtime_store.write_host_snapshot(
                        host_id=host_id,
                        snapshot={**snapshot, "service_states": service_states, "lane_states": lane_states},
                        observed_at=observed_at,
                        ttl_seconds=settings.runtime_state_ttl_seconds,
                    )
            elif message_type == "response":
                request_id = str(payload.get("request_id") or "")
                body = dict(payload.get("payload") or {})
                if request_id:
                    _upsert_transition(cur, request_id=request_id, host_id=host_id, payload=body, observed_at=observed_at)
                    _insert_transition_event(cur, request_id=request_id, host_id=host_id, payload=body, observed_at=observed_at)
                snapshot = _response_host_state_snapshot(body)
                if runtime_store is not None and snapshot is not None:
                    runtime_store.write_host_snapshot(
                        host_id=host_id,
                        snapshot=snapshot,
                        observed_at=observed_at,
                        ttl_seconds=settings.runtime_state_ttl_seconds,
                        source="mw_response_snapshot",
                    )
            else:
                return
        conn.commit()


def run_forever() -> None:
    if not settings.mw_control_enabled:
        logger.info("MW consumer disabled (mw_control_enabled=false)")
        return

    cfg = MwConsumerSettings(
        bootstrap_servers=settings.mw_kafka_bootstrap_servers,
        client_id=f"{settings.mw_kafka_client_id}-consumer",
        consumer_group=settings.mw_kafka_consumer_group,
        state_topic=settings.mw_kafka_state_topic,
        heartbeats_topic=settings.mw_kafka_heartbeats_topic,
        responses_topic=settings.mw_kafka_responses_topic,
    )
    consumer = Consumer(
        {
            "bootstrap.servers": cfg.bootstrap_servers,
            "group.id": cfg.consumer_group,
            "client.id": cfg.client_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "session.timeout.ms": 10000,
        }
    )
    consumer.subscribe([cfg.state_topic, cfg.heartbeats_topic, cfg.responses_topic])
    runtime_store = get_default_runtime_state_store()

    logger.info("MW consumer started", extra={"topics": [cfg.state_topic, cfg.heartbeats_topic]})
    try:
        while True:
            msg = consumer.poll(0.5)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.warning("MW consumer kafka error: %s", msg.error())
                continue
            try:
                payload = json.loads(msg.value().decode("utf-8"))
            except Exception as exc:
                logger.warning("MW consumer bad json: %s", exc)
                consumer.commit(message=msg, asynchronous=False)
                continue

            try:
                process_message(
                    payload=payload,
                    observed_at=datetime.now(UTC),
                    db_connect=_mw_state_db_connect,
                    runtime_store=runtime_store,
                )
            except Exception as exc:
                # Tables may not exist yet; keep the consumer alive and retry later.
                logger.warning("MW consumer DB write failed: %s", exc)
                time.sleep(1.0)

            consumer.commit(message=msg, asynchronous=False)
    finally:
        consumer.close()


def _mw_state_db_connect():
    dsn = settings.mw_state_database_url or settings.database_url
    return psycopg.connect(dsn, row_factory=dict_row)

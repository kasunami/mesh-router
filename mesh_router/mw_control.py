from __future__ import annotations

import json
import time
import uuid
from typing import Any

from confluent_kafka import Consumer, Producer

from .config import settings


class MWControlError(RuntimeError):
    pass


class MWControlTimeout(MWControlError):
    def __init__(self, message: str, *, request_id: str, timeout_seconds: int) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.timeout_seconds = timeout_seconds


class MeshWorkerCommandClient:
    def __init__(
        self,
        *,
        bootstrap_servers: str,
        commands_topic: str,
        responses_topic: str,
        client_id: str,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.commands_topic = commands_topic
        self.responses_topic = responses_topic
        self.client_id = client_id
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "client.id": client_id,
                "message.timeout.ms": 5000,
            }
        )

    @classmethod
    def from_settings(cls) -> "MeshWorkerCommandClient":
        return cls(
            bootstrap_servers=settings.mw_kafka_bootstrap_servers,
            commands_topic=settings.mw_kafka_commands_topic,
            responses_topic=settings.mw_kafka_responses_topic,
            client_id=settings.mw_kafka_client_id,
        )

    def send_command(
        self,
        *,
        host_id: str,
        message_type: str,
        payload: dict[str, Any],
        request_id: str | None = None,
        wait: bool = True,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())
        # Public API uses `message_type` as the command type (activate_profile, load_model, ...).
        # Kafka contract uses message_type='command' with payload.command_type/arguments.
        command_type = message_type
        envelope = {
            "schema_version": 1,
            "message_type": "command",
            "message_id": str(uuid.uuid4()),
            "request_id": request_id,
            "host_id": host_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": {
                "command_type": command_type,
                "issued_by": "mr",
                "target": {},
                "arguments": payload,
                "idempotency_key": str(uuid.uuid4()),
            },
        }
        delivery: dict[str, Any] = {"ok": False}

        def on_delivery(err: Any, msg: Any) -> None:
            if err is not None:
                delivery["error"] = str(err)
            else:
                delivery.update(
                    {
                        "ok": True,
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset(),
                    }
                )

        consumer = None
        if wait:
            consumer = Consumer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "group.id": f"{self.client_id}-{request_id}",
                    "client.id": f"{self.client_id}-waiter",
                    "auto.offset.reset": "earliest",
                    "enable.auto.commit": False,
                    "session.timeout.ms": 6000,
                }
            )
            consumer.subscribe([self.responses_topic])

        try:
            self._producer.produce(
                self.commands_topic,
                key=host_id.encode("utf-8"),
                value=json.dumps(envelope, sort_keys=True).encode("utf-8"),
                on_delivery=on_delivery,
            )
            self._producer.poll(0)
            pending = self._producer.flush(5)
            if pending and not delivery.get("ok"):
                raise MWControlError(f"{pending} command message(s) still pending after flush timeout")
            if delivery.get("error"):
                raise MWControlError(str(delivery["error"]))

            result = {
                "ok": True,
                "request_id": request_id,
                "host_id": host_id,
                "message_type": command_type,
                "delivery": delivery,
            }
            if not wait:
                return result
            try:
                result["response"] = self._wait_for_response(consumer, request_id, timeout_seconds)
            except MWControlTimeout as exc:
                # Avoid false-negative operational behavior: MW may still complete after a slow swap.
                # Return a "pending" result and allow callers to poll `mw_transitions` (via MR endpoints).
                return {
                    **result,
                    "ok": True,
                    "pending": True,
                    "warning": str(exc),
                    "timeout_seconds": exc.timeout_seconds,
                }
            payload_obj = (result["response"] or {}).get("payload", {}) if result.get("response") else {}
            result["ok"] = bool(payload_obj.get("ok", False))
            error_obj = payload_obj.get("error")
            if isinstance(error_obj, dict):
                result["error"] = error_obj.get("message") or str(error_obj)
            else:
                result["error"] = error_obj
            result["result"] = payload_obj.get("result") or {}
            return result
        finally:
            if consumer is not None:
                consumer.close()

    def _wait_for_response(
        self,
        consumer: Consumer | None,
        request_id: str,
        timeout_seconds: int | None,
    ) -> dict[str, Any]:
        if consumer is None:
            raise MWControlError("response wait requested without consumer")
        timeout = int(timeout_seconds or settings.mw_command_timeout_seconds)
        deadline = time.time() + timeout
        terminal = {"ready", "completed", "failed", "cancelled", "rejected"}
        while time.time() < deadline:
            msg = consumer.poll(0.5)
            if msg is None:
                continue
            if msg.error():
                continue
            try:
                payload = json.loads(msg.value().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if str(payload.get("request_id") or "") != request_id:
                continue
            payload_body = payload.get("payload") or {}
            if isinstance(payload_body, dict):
                response_type = str(payload_body.get("response_type") or "")
                if response_type and response_type not in terminal:
                    continue
            consumer.commit(asynchronous=False)
            return payload
        raise MWControlTimeout(
            f"timed out waiting for MeshWorker response for request_id={request_id}",
            request_id=request_id,
            timeout_seconds=timeout,
        )

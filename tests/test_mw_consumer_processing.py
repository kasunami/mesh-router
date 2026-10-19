from __future__ import annotations

import contextlib
import unittest
from datetime import UTC, datetime, timedelta

from mesh_router.mw_consumer import process_message


class CapturingCursor:
    def __init__(self, *, heartbeat_event_at=None) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.heartbeat_event_at = heartbeat_event_at

    def execute(self, sql: str, params: tuple) -> None:
        self.executed.append((sql, params))

    def fetchone(self):
        return {"last_heartbeat_event_at": self.heartbeat_event_at}

    def __enter__(self) -> "CapturingCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


class CapturingConn:
    def __init__(self, cursor: CapturingCursor) -> None:
        self._cursor = cursor
        self.committed = False

    def cursor(self) -> CapturingCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def __enter__(self) -> "CapturingConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None




class CapturingRuntimeStore:
    def __init__(self) -> None:
        self.snapshots: list[dict] = []

    def write_host_snapshot(self, **kwargs) -> None:  # noqa: ANN003
        self.snapshots.append(kwargs)


def make_db(cursor: CapturingCursor):
    @contextlib.contextmanager
    def _connect():
        yield CapturingConn(cursor)

    return _connect


class MwConsumerProcessingTests(unittest.TestCase):
    def _capture_heartbeat(
        self,
        *,
        payload_timestamp=None,
        received_at=None,
        message_timestamp=None,
    ):
        received = received_at or datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
        payload = {"message_type": "heartbeat", "host_id": "test-host", "payload": {}}
        if payload_timestamp is not None:
            payload["timestamp"] = payload_timestamp
        cursor = CapturingCursor()
        process_message(
            payload=payload,
            observed_at=received,
            db_connect=make_db(cursor),
            message_timestamp=message_timestamp,
        )
        return cursor

    def test_delayed_heartbeat_uses_source_event_time_not_processing_time(self) -> None:
        received = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
        event = received - timedelta(hours=28)
        cursor = self._capture_heartbeat(payload_timestamp=event.isoformat(), received_at=received)
        host_params = next(params for sql, params in cursor.executed if "INSERT INTO mw_hosts" in sql)
        heartbeat_params = next(params for sql, params in cursor.executed if "INSERT INTO mw_heartbeats" in sql)
        self.assertEqual(host_params[7], event)
        self.assertEqual(heartbeat_params[1], event)

    def test_reordered_heartbeats_keep_database_timestamp_monotonic(self) -> None:
        received = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
        newer = received - timedelta(seconds=15)
        older = received - timedelta(minutes=2)
        newer_cursor = self._capture_heartbeat(payload_timestamp=newer.isoformat(), received_at=received)
        older_cursor = self._capture_heartbeat(payload_timestamp=older.isoformat(), received_at=received)
        self.assertEqual(
            next(params for sql, params in newer_cursor.executed if "INSERT INTO mw_hosts" in sql)[7],
            newer,
        )
        older_sql = next(sql for sql, _ in older_cursor.executed if "INSERT INTO mw_hosts" in sql)
        older_params = next(params for sql, params in older_cursor.executed if "INSERT INTO mw_hosts" in sql)
        self.assertIn("GREATEST(mw_hosts.last_heartbeat_at, EXCLUDED.last_heartbeat_at)", older_sql)
        self.assertEqual(older_params[7], older)
        self.assertIn("last_heartbeat_event_at", older_sql)
        self.assertEqual(older_params[8], older)

    def test_missing_event_timestamp_falls_back_to_valid_kafka_timestamp(self) -> None:
        received = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
        broker_time = received - timedelta(seconds=20)
        cursor = self._capture_heartbeat(received_at=received, message_timestamp=broker_time)
        params = next(params for sql, params in cursor.executed if "INSERT INTO mw_hosts" in sql)
        self.assertEqual(params[7], broker_time)

    def test_invalid_or_timezone_naive_event_timestamp_falls_back_to_kafka_timestamp(self) -> None:
        received = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
        broker_time = received - timedelta(seconds=25)
        for invalid in ("not-an-iso-time", "2026-10-19T09:59:00"):
            with self.subTest(invalid=invalid):
                cursor = self._capture_heartbeat(
                    payload_timestamp=invalid,
                    received_at=received,
                    message_timestamp=broker_time,
                )
                params = next(params for sql, params in cursor.executed if "INSERT INTO mw_hosts" in sql)
                self.assertEqual(params[7], broker_time)

    def test_missing_or_invalid_timestamps_without_valid_fallback_do_not_refresh(self) -> None:
        received = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
        for timestamp in (None, "invalid", "2026-10-19T09:59:00"):
            with self.subTest(timestamp=timestamp):
                cursor = self._capture_heartbeat(payload_timestamp=timestamp, received_at=received)
                self.assertEqual(cursor.executed, [])

    def test_small_future_clock_skew_is_clamped_to_receive_time(self) -> None:
        received = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
        future = received + timedelta(seconds=10)
        cursor = self._capture_heartbeat(payload_timestamp=future.isoformat(), received_at=received)
        params = next(params for sql, params in cursor.executed if "INSERT INTO mw_hosts" in sql)
        self.assertEqual(params[7], received)

    def test_excessive_future_timestamp_uses_valid_broker_time_or_is_rejected(self) -> None:
        received = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
        future = received + timedelta(minutes=5)
        broker_time = received - timedelta(seconds=10)
        fallback_cursor = self._capture_heartbeat(
            payload_timestamp=future.isoformat(),
            received_at=received,
            message_timestamp=broker_time,
        )
        fallback_params = next(params for sql, params in fallback_cursor.executed if "INSERT INTO mw_hosts" in sql)
        self.assertEqual(fallback_params[7], broker_time)
        rejected_cursor = self._capture_heartbeat(payload_timestamp=future.isoformat(), received_at=received)
        self.assertEqual(rejected_cursor.executed, [])

    def test_process_state_upserts_services_and_lanes(self) -> None:
        cursor = CapturingCursor()
        runtime_store = CapturingRuntimeStore()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)
        process_message(
            payload={
                "message_type": "state",
                "host_id": "worker-a",
                "payload": {
                    "actual_profile": "split_default",
                    "service_states": [
                        {
                            "service_id": "llama-gpu",
                            "backend_type": "llama.cpp",
                            "kind": "text",
                            "manager_name": "llama-gpu.service",
                            "listen_host": "127.0.0.1",
                            "listen_port": 21434,
                            "desired_state": "running",
                            "actual_state": "running",
                            "health_status": "healthy",
                        }
                    ],
                    "lane_states": [
                        {
                            "lane_id": "gpu",
                            "lane_type": "gpu",
                            "backend_type": "llama.cpp",
                            "service_id": "llama-gpu",
                            "resource_class": "gpu-primary",
                            "desired_model": "qwen3.5-4b",
                            "actual_model": "qwen3.5-4b",
                            "actual_state": "running",
                            "health_status": "healthy",
                            "active_mode": "gpu-llama-text",
                        }
                    ],
                    "validated_candidates": [
                        {"canonical_id": "qwen3.5-4b", "lane_ids": ["gpu"]},
                        {"canonical_id": "falcon3-10b", "lane_ids": ["cpu"]},
                    ],
                },
            },
            observed_at=now,
            db_connect=db_connect,
            runtime_store=runtime_store,
        )
        sql = "\n".join(s for (s, _p) in cursor.executed)
        self.assertIn("INSERT INTO mw_hosts", sql)
        self.assertIn("INSERT INTO mw_services", sql)
        self.assertIn("INSERT INTO mw_lanes", sql)
        self.assertEqual(len(runtime_store.snapshots), 1)
        self.assertEqual(runtime_store.snapshots[0]["host_id"], "worker-a")
        self.assertEqual(runtime_store.snapshots[0]["snapshot"]["lane_states"][0]["actual_model"], "qwen3.5-4b")
        self.assertEqual(
            runtime_store.snapshots[0]["snapshot"]["lane_states"][0]["validated_candidates"],
            [{"canonical_id": "qwen3.5-4b", "lane_ids": ["gpu"]}],
        )
        self.assertIn("SELECT last_heartbeat_event_at FROM mw_hosts", sql)
        self.assertIsNone(runtime_store.snapshots[0]["heartbeat_event_at"])

    def test_state_snapshot_cache_receives_event_time_from_dedicated_db_column(self) -> None:
        event = datetime(2026, 10, 19, 9, 59, tzinfo=UTC)
        cursor = CapturingCursor(heartbeat_event_at=event)
        runtime_store = CapturingRuntimeStore()
        process_message(
            payload={"message_type": "state", "host_id": "worker-a", "payload": {"lane_states": []}},
            observed_at=datetime(2026, 10, 19, 10, 0, tzinfo=UTC),
            db_connect=make_db(cursor),
            runtime_store=runtime_store,
        )
        self.assertEqual(runtime_store.snapshots[0]["heartbeat_event_at"], event)

    def test_response_snapshot_cache_receives_event_time_from_dedicated_db_column(self) -> None:
        event = datetime(2026, 10, 19, 9, 59, tzinfo=UTC)
        cursor = CapturingCursor(heartbeat_event_at=event)
        runtime_store = CapturingRuntimeStore()
        process_message(
            payload={
                "message_type": "response",
                "host_id": "worker-a",
                "payload": {"result": {"host_state": {"lane_states": [{"lane_id": "gpu"}]}}},
            },
            observed_at=datetime(2026, 10, 19, 10, 0, tzinfo=UTC),
            db_connect=make_db(cursor),
            runtime_store=runtime_store,
        )
        self.assertEqual(runtime_store.snapshots[0]["heartbeat_event_at"], event)


    def test_process_response_refreshes_runtime_cache_from_host_state(self) -> None:
        cursor = CapturingCursor()
        runtime_store = CapturingRuntimeStore()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)

        process_message(
            payload={
                "message_type": "response",
                "host_id": "worker-a",
                "request_id": "00000000-0000-0000-0000-000000000002",
                "payload": {
                    "response_type": "failed",
                    "command_type": "load_model",
                    "ok": False,
                    "result": {
                        "host_state": {
                            "actual_profile": "split_default",
                            "service_states": [
                                {
                                    "service_id": "llama-gpu",
                                    "backend_type": "llama.cpp",
                                    "listen_port": 21434,
                                    "actual_state": "running",
                                    "health_status": "healthy",
                                }
                            ],
                            "lane_states": [
                                {
                                    "lane_id": "gpu",
                                    "lane_type": "gpu",
                                    "backend_type": "llama.cpp",
                                    "service_id": "llama-gpu",
                                    "actual_model": "qwen3.5-9b",
                                    "actual_state": "running",
                                    "health_status": "healthy",
                                }
                            ],
                        }
                    },
                    "error": {"code": "FAILED", "message": "swap failed after convergence check"},
                },
            },
            observed_at=now,
            db_connect=db_connect,
            runtime_store=runtime_store,
        )

        sql = "\n".join(s for (s, _p) in cursor.executed)
        self.assertIn("INSERT INTO mw_transitions", sql)
        self.assertEqual(len(runtime_store.snapshots), 1)
        self.assertEqual(runtime_store.snapshots[0]["host_id"], "worker-a")
        self.assertEqual(runtime_store.snapshots[0]["source"], "mw_response_snapshot")
        self.assertEqual(runtime_store.snapshots[0]["snapshot"]["lane_states"][0]["actual_model"], "qwen3.5-9b")

    def test_process_response_upserts_transition_and_event(self) -> None:
        cursor = CapturingCursor()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)
        process_message(
            payload={
                "message_type": "response",
                "host_id": "worker-a",
                "request_id": "00000000-0000-0000-0000-000000000001",
                "payload": {"response_type": "completed", "command_type": "load_model", "ok": True, "result": {}},
            },
            observed_at=now,
            db_connect=db_connect,
        )
        sql = "\n".join(s for (s, _p) in cursor.executed)
        self.assertIn("INSERT INTO mw_transitions", sql)
        self.assertIn("INSERT INTO mw_transition_events", sql)

    def test_process_response_with_invalid_request_id_is_skipped(self) -> None:
        cursor = CapturingCursor()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)
        process_message(
            payload={
                "message_type": "response",
                "host_id": "worker-a",
                "request_id": "not-a-uuid",
                "payload": {"response_type": "completed", "command_type": "load_model", "ok": True, "result": {}},
            },
            observed_at=now,
            db_connect=db_connect,
        )
        sql = "\n".join(s for (s, _p) in cursor.executed)
        self.assertNotIn("INSERT INTO mw_transitions", sql)
        self.assertNotIn("INSERT INTO mw_transition_events", sql)

    def test_completed_response_with_false_ok_is_stored_as_failed(self) -> None:
        cursor = CapturingCursor()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)
        process_message(
            payload={
                "message_type": "response",
                "host_id": "worker-a",
                "request_id": "00000000-0000-0000-0000-000000000003",
                "payload": {
                    "response_type": "completed",
                    "command_type": "load_model",
                    "ok": False,
                    "error": {"message": "target service health check failed"},
                },
            },
            observed_at=now,
            db_connect=db_connect,
        )

        transition_params = next(params for (sql, params) in cursor.executed if "INSERT INTO mw_transitions" in sql)
        self.assertEqual(transition_params[3], "failed")
        self.assertEqual(transition_params[7], "target service health check failed")

    def test_ready_response_is_stored_as_terminal_success(self) -> None:
        cursor = CapturingCursor()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)
        process_message(
            payload={
                "message_type": "response",
                "host_id": "worker-a",
                "request_id": "00000000-0000-0000-0000-000000000004",
                "payload": {"response_type": "ready", "command_type": "load_model", "ok": True, "result": {}},
            },
            observed_at=now,
            db_connect=db_connect,
        )

        transition_params = next(params for (sql, params) in cursor.executed if "INSERT INTO mw_transitions" in sql)
        self.assertEqual(transition_params[3], "ready")
        self.assertEqual(transition_params[6], now)


if __name__ == "__main__":
    unittest.main()

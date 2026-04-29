from __future__ import annotations

import contextlib
import unittest
from datetime import UTC, datetime

from mesh_router.mw_consumer import process_message


class CapturingCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple) -> None:
        self.executed.append((sql, params))

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
    def test_process_state_upserts_services_and_lanes(self) -> None:
        cursor = CapturingCursor()
        runtime_store = CapturingRuntimeStore()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)
        process_message(
            payload={
                "message_type": "state",
                "host_id": "static-deskix",
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
        self.assertEqual(runtime_store.snapshots[0]["host_id"], "static-deskix")
        self.assertEqual(runtime_store.snapshots[0]["snapshot"]["lane_states"][0]["actual_model"], "qwen3.5-4b")
        self.assertEqual(
            runtime_store.snapshots[0]["snapshot"]["lane_states"][0]["validated_candidates"],
            [{"canonical_id": "qwen3.5-4b", "lane_ids": ["gpu"]}],
        )


    def test_process_response_refreshes_runtime_cache_from_host_state(self) -> None:
        cursor = CapturingCursor()
        runtime_store = CapturingRuntimeStore()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)

        process_message(
            payload={
                "message_type": "response",
                "host_id": "static-deskix",
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
        self.assertEqual(runtime_store.snapshots[0]["host_id"], "static-deskix")
        self.assertEqual(runtime_store.snapshots[0]["source"], "mw_response_snapshot")
        self.assertEqual(runtime_store.snapshots[0]["snapshot"]["lane_states"][0]["actual_model"], "qwen3.5-9b")

    def test_process_response_upserts_transition_and_event(self) -> None:
        cursor = CapturingCursor()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)
        process_message(
            payload={
                "message_type": "response",
                "host_id": "static-deskix",
                "request_id": "00000000-0000-0000-0000-000000000001",
                "payload": {"response_type": "completed", "command_type": "load_model", "ok": True, "result": {}},
            },
            observed_at=now,
            db_connect=db_connect,
        )
        sql = "\n".join(s for (s, _p) in cursor.executed)
        self.assertIn("INSERT INTO mw_transitions", sql)
        self.assertIn("INSERT INTO mw_transition_events", sql)

    def test_completed_response_with_false_ok_is_stored_as_failed(self) -> None:
        cursor = CapturingCursor()
        db_connect = make_db(cursor)
        now = datetime.now(UTC)
        process_message(
            payload={
                "message_type": "response",
                "host_id": "static-deskix",
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
                "host_id": "static-deskix",
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

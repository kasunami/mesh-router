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


def make_db(cursor: CapturingCursor):
    @contextlib.contextmanager
    def _connect():
        yield CapturingConn(cursor)

    return _connect


class MwConsumerProcessingTests(unittest.TestCase):
    def test_process_state_upserts_services_and_lanes(self) -> None:
        cursor = CapturingCursor()
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
                },
            },
            observed_at=now,
            db_connect=db_connect,
        )
        sql = "\n".join(s for (s, _p) in cursor.executed)
        self.assertIn("INSERT INTO mw_hosts", sql)
        self.assertIn("INSERT INTO mw_services", sql)
        self.assertIn("INSERT INTO mw_lanes", sql)

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


if __name__ == "__main__":
    unittest.main()


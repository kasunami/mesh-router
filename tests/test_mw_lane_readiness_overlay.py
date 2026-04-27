from __future__ import annotations

from datetime import datetime, timezone
import unittest

from fastapi.testclient import TestClient

from mesh_router import app as app_module
class _FakeCursor:
    def __init__(self, *, execute_rows: list[list[dict]]) -> None:
        self._execute_rows = execute_rows
        self._idx = 0
        self.last_sql: str | None = None
        self.last_params: object | None = None

    def execute(self, sql, params=None):  # noqa: ANN001
        self.last_sql = str(sql)
        self.last_params = params

    def fetchall(self):  # noqa: ANN001
        if self._idx >= len(self._execute_rows):
            return []
        rows = self._execute_rows[self._idx]
        self._idx += 1
        return rows

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self):  # noqa: ANN001
        return self._cursor

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakeDb:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def connect(self):  # noqa: ANN001
        return _FakeConn(self._cursor)


class MWLaneReadinessOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_db = app_module.db
        self.original_mw_state_db = app_module.mw_state_db

    def tearDown(self) -> None:
        app_module.db = self.original_db  # type: ignore[assignment]
        app_module.mw_state_db = self.original_mw_state_db  # type: ignore[assignment]

    def test_api_lanes_overlays_mw_readiness_and_model(self) -> None:
        # Base lanes listing: lane is offline in MR DB but MW control plane should override.
        base_rows = [
            {
                "lane_id": "lane-1",
                "host_id": "host-1",
                "host_name": "Static-Deskix",
                "lane_name": "gpu",
                "lane_type": "gpu",
                "backend_type": "llama",
                "base_url": "http://10.0.0.99:11434",
                "status": "offline",
                "current_model_name": None,
                "ram_budget_bytes": None,
                "vram_budget_bytes": None,
                "proxy_auth_mode": None,
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "suspension_reason": None,
                "last_probe_at": None,
                "last_ok_at": None,
                "created_at": None,
                "updated_at": None,
            }
        ]
        now = datetime.now(tz=timezone.utc)
        mw_execute_rows = [
            [
                {
                    "host_id": "static-deskix",
                    "lane_id": "gpu",
                    "last_heartbeat_at": now,
                    "actual_model": "Qwen3.5-9B-Q4_K_M.gguf",
                    "actual_state": "running",
                    "health_status": "healthy",
                }
            ],
        ]

        app_module.db = _FakeDb(_FakeCursor(execute_rows=[base_rows]))  # type: ignore[assignment]
        app_module.mw_state_db = _FakeDb(_FakeCursor(execute_rows=mw_execute_rows))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.get("/api/lanes")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["items"]), 1)
        item = body["items"][0]
        self.assertEqual(item["host_name"], "Static-Deskix")
        self.assertEqual(item["lane_name"], "gpu")
        self.assertEqual(item["status"], "ready")
        self.assertIsNone(item["readiness_reason"])
        self.assertEqual(item["current_model_name"], "Qwen3.5-9B-Q4_K_M.gguf")

    def test_api_lanes_reports_machine_readable_reason_when_mw_lane_not_running(self) -> None:
        base_rows = [
            {
                "lane_id": "lane-1",
                "host_id": "host-1",
                "host_name": "Static-Deskix",
                "lane_name": "gpu",
                "lane_type": "gpu",
                "backend_type": "llama",
                "base_url": "http://10.0.0.99:11434",
                "status": "ready",
                "current_model_name": "Qwen3.5-9B-Q4_K_M.gguf",
                "ram_budget_bytes": None,
                "vram_budget_bytes": None,
                "proxy_auth_mode": None,
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "suspension_reason": None,
                "last_probe_at": None,
                "last_ok_at": None,
                "created_at": None,
                "updated_at": None,
            }
        ]
        now = datetime.now(tz=timezone.utc)
        mw_execute_rows = [
            [
                {
                    "host_id": "static-deskix",
                    "lane_id": "gpu",
                    "last_heartbeat_at": now,
                    "actual_model": "Qwen3.5-9B-Q4_K_M.gguf",
                    "actual_state": "starting",
                    "health_status": "healthy",
                }
            ],
        ]

        app_module.db = _FakeDb(_FakeCursor(execute_rows=[base_rows]))  # type: ignore[assignment]
        app_module.mw_state_db = _FakeDb(_FakeCursor(execute_rows=mw_execute_rows))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.get("/api/lanes")
        self.assertEqual(resp.status_code, 200)
        item = resp.json()["items"][0]
        self.assertEqual(item["status"], "offline")
        self.assertEqual(item["readiness_reason"], "not_running")

    def test_api_lanes_infers_mw_for_legacy_cpu_lane_without_metadata(self) -> None:
        base_rows = [
            {
                "lane_id": "lane-cpu",
                "host_id": "host-1",
                "host_name": "pupix1",
                "lane_name": "cpu",
                "lane_type": "cpu",
                "backend_type": "llama",
                "base_url": "http://10.0.0.95:11435",
                "status": "ready",
                "current_model_name": "Falcon3-10B-Instruct-1.58bit",
                "ram_budget_bytes": None,
                "vram_budget_bytes": None,
                "proxy_auth_mode": None,
                "proxy_auth_metadata": {},
                "suspension_reason": None,
                "last_probe_at": None,
                "last_ok_at": None,
                "created_at": None,
                "updated_at": None,
            }
        ]
        now = datetime.now(tz=timezone.utc)
        mw_execute_rows = [
            [
                {
                    "host_id": "pupix1",
                    "lane_id": "cpu",
                    "last_heartbeat_at": now,
                    "actual_model": "qwen3.5-4b",
                    "actual_state": "running",
                    "health_status": "healthy",
                    "backend_type": "llama.cpp",
                }
            ],
        ]

        app_module.db = _FakeDb(_FakeCursor(execute_rows=[base_rows]))  # type: ignore[assignment]
        app_module.mw_state_db = _FakeDb(_FakeCursor(execute_rows=mw_execute_rows))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.get("/api/lanes")
        self.assertEqual(resp.status_code, 200)
        item = resp.json()["items"][0]
        self.assertEqual(item["status"], "ready")
        self.assertEqual(item["current_model_name"], "qwen3.5-4b")
        self.assertEqual(item["backend_type"], "llama")

    def test_api_lanes_infers_mw_gpu_lane_for_legacy_mlx_row(self) -> None:
        base_rows = [
            {
                "lane_id": "lane-mlx",
                "host_id": "host-1",
                "host_name": "tiffs-macbook",
                "lane_name": "mlx",
                "lane_type": "mlx",
                "backend_type": "llama",
                "base_url": "http://10.0.0.97:11435",
                "status": "offline",
                "current_model_name": "mlx-community/Llama-3.1-8B-Instruct-4bit",
                "ram_budget_bytes": None,
                "vram_budget_bytes": None,
                "proxy_auth_mode": "static_bearer_env",
                "proxy_auth_metadata": {"proxy_auth_mode": "static_bearer_env"},
                "suspension_reason": None,
                "last_probe_at": None,
                "last_ok_at": None,
                "created_at": None,
                "updated_at": None,
            }
        ]
        now = datetime.now(tz=timezone.utc)
        mw_execute_rows = [
            [
                {
                    "host_id": "tiffs-macbook",
                    "lane_id": "gpu",
                    "last_heartbeat_at": now,
                    "actual_model": "/Users/kasunami/models/Falcon3-10B-Instruct-1.58bit",
                    "actual_state": "running",
                    "health_status": "healthy",
                    "backend_type": "mlx",
                }
            ],
        ]

        app_module.db = _FakeDb(_FakeCursor(execute_rows=[base_rows]))  # type: ignore[assignment]
        app_module.mw_state_db = _FakeDb(_FakeCursor(execute_rows=mw_execute_rows))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.get("/api/lanes")
        self.assertEqual(resp.status_code, 200)
        item = resp.json()["items"][0]
        self.assertEqual(item["status"], "ready")
        self.assertIsNone(item["readiness_reason"])
        self.assertEqual(item["current_model_name"], "/Users/kasunami/models/Falcon3-10B-Instruct-1.58bit")
        self.assertEqual(item["backend_type"], "mlx")


if __name__ == "__main__":
    unittest.main()

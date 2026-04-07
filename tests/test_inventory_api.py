from __future__ import annotations

from datetime import datetime, timezone
import unittest

from fastapi.testclient import TestClient

from mesh_router import app as app_module
from mesh_router import inventory as inventory_module


class _FakeCursor:
    def __init__(self, *, fetchall_rows: list[list[dict]], fetchone_rows: list[object] | None = None) -> None:
        self._fetchall_rows = fetchall_rows
        self._fetchone_rows = fetchone_rows or []
        self._fa_idx = 0
        self._fo_idx = 0

    def execute(self, sql, params=None):  # noqa: ANN001
        return None

    def fetchall(self):  # noqa: ANN001
        if self._fa_idx >= len(self._fetchall_rows):
            return []
        rows = self._fetchall_rows[self._fa_idx]
        self._fa_idx += 1
        return rows

    def fetchone(self):  # noqa: ANN001
        if self._fo_idx >= len(self._fetchone_rows):
            return None
        row = self._fetchone_rows[self._fo_idx]
        self._fo_idx += 1
        return row

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


class InventoryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_db = app_module.db
        self.original_inventory_mw_state_db = inventory_module.mw_state_db

    def tearDown(self) -> None:
        app_module.db = self.original_db  # type: ignore[assignment]
        inventory_module.mw_state_db = self.original_inventory_mw_state_db  # type: ignore[assignment]

    def test_api_inventory_overlays_mw_effective_status(self) -> None:
        base_rows = [
            {
                "lane_id": "lane-1",
                "lane_name": "gpu",
                "lane_type": "gpu",
                "backend_type": "llama",
                "base_url": "http://10.0.0.99:11434",
                "status": "offline",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "current_model_name": None,
                "host_id": "host-1",
                "host_name": "Static-Deskix",
                "viable_models": [
                    {"model_name": "qwen3.5-9b", "tags": [], "max_ctx": 8192, "locality": "local"},
                    {"model_name": "qwen3.5-27b", "tags": [], "max_ctx": 8192, "locality": "remote"},
                ],
            }
        ]
        now = datetime.now(tz=timezone.utc)
        state_rows = [
            {
                "host_id": "static-deskix",
                "lane_id": "gpu",
                "last_heartbeat_at": now,
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "Qwen3.5-9B-Q4_K_M.gguf",
            }
        ]

        app_module.db = _FakeDb(_FakeCursor(fetchall_rows=[base_rows]))  # type: ignore[assignment]
        inventory_module.mw_state_db = _FakeDb(_FakeCursor(fetchall_rows=[state_rows]))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.get("/api/inventory")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["items"]), 1)
        host = body["items"][0]
        self.assertEqual(host["host_name"], "Static-Deskix")
        self.assertIn("stable", host.get("tags") or [])
        self.assertEqual(len(host["lanes"]), 1)
        lane = host["lanes"][0]
        self.assertEqual(lane["lane_name"], "gpu")
        self.assertEqual(lane["status"], "ready")
        self.assertIsNone(lane["readiness_reason"])
        self.assertEqual(lane["current_model_name"], "Qwen3.5-9B-Q4_K_M.gguf")
        self.assertEqual(len(lane["local_viable_models"]), 1)
        self.assertEqual(len(lane["remote_viable_models"]), 1)


if __name__ == "__main__":
    unittest.main()

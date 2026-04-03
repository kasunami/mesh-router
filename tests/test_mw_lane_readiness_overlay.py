from __future__ import annotations

from datetime import datetime, timezone
import unittest

from fastapi.testclient import TestClient

from mesh_router import app as app_module
from mesh_router import db as db_module


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
        self.original_mw_state_db = db_module.mw_state_db

    def tearDown(self) -> None:
        app_module.db = self.original_db  # type: ignore[assignment]
        db_module.mw_state_db = self.original_mw_state_db  # type: ignore[assignment]

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
        # mw_hosts query + mw_lanes query
        now = datetime.now(tz=timezone.utc)
        mw_execute_rows = [
            [{"host_id": "static-deskix", "last_heartbeat_at": now}],
            [
                {
                    "host_id": "static-deskix",
                    "lane_id": "gpu",
                    "actual_model": "Qwen3.5-9B-Q4_K_M.gguf",
                    "actual_state": "running",
                    "health_status": "healthy",
                }
            ],
        ]

        app_module.db = _FakeDb(_FakeCursor(execute_rows=[base_rows]))  # type: ignore[assignment]
        db_module.mw_state_db = _FakeDb(_FakeCursor(execute_rows=mw_execute_rows))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.get("/api/lanes")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["items"]), 1)
        item = body["items"][0]
        self.assertEqual(item["host_name"], "Static-Deskix")
        self.assertEqual(item["lane_name"], "gpu")
        self.assertEqual(item["status"], "ready")
        self.assertEqual(item["current_model_name"], "Qwen3.5-9B-Q4_K_M.gguf")


if __name__ == "__main__":
    unittest.main()


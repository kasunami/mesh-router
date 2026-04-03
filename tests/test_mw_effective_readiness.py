from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from mesh_router import router as router_module


class _FakeCursor:
    def __init__(self, *, fetchall_rows: list[dict], fetchone_row: dict | None = None) -> None:
        self._fetchall_rows = fetchall_rows
        self._fetchone_row = fetchone_row
        self.last_sql: str | None = None
        self.last_params: tuple | None = None

    def execute(self, sql: str, params: tuple = ()) -> None:  # noqa: ANN001
        self.last_sql = sql
        self.last_params = params

    def fetchall(self):  # noqa: ANN001
        return list(self._fetchall_rows)

    def fetchone(self):  # noqa: ANN001
        return self._fetchone_row

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


class MwEffectiveReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_db = router_module.db
        self.orig_mw_state_db = router_module.mw_state_db

    def tearDown(self) -> None:
        router_module.db = self.orig_db  # type: ignore[assignment]
        router_module.mw_state_db = self.orig_mw_state_db  # type: ignore[assignment]

    def test_pick_lane_for_model_accepts_mw_effective_ready(self) -> None:
        base_rows = [
            {
                "lane_id": "lane-1",
                "host_name": "Static-Deskix",
                "base_url": "http://10.0.0.99:11434",
                "lane_type": "gpu",
                "backend_type": "llama",
                "status": "offline",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "current_model_name": "Qwen3.5-9B-Q4_K_M.gguf",
                "current_model_tags": [],
                "current_model_max_ctx": None,
            }
        ]
        now = datetime.now(tz=timezone.utc)
        mw_rows = [
            {
                "host_id": "static-deskix",
                "lane_id": "gpu",
                "last_heartbeat_at": now,
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "Qwen3.5-9B-Q4_K_M.gguf",
            }
        ]
        router_module.db = _FakeDb(_FakeCursor(fetchall_rows=base_rows))  # type: ignore[assignment]
        router_module.mw_state_db = _FakeDb(_FakeCursor(fetchall_rows=mw_rows))  # type: ignore[assignment]

        choice = router_module.pick_lane_for_model(
            model="Qwen3.5-9B-Q4_K_M.gguf",
            pin_worker="Static-Deskix",
        )
        self.assertEqual(choice.worker_id, "Static-Deskix")
        self.assertEqual(choice.lane_id, "lane-1")

    def test_pick_lane_for_model_rejects_mw_stale_heartbeat(self) -> None:
        base_rows = [
            {
                "lane_id": "lane-1",
                "host_name": "Static-Deskix",
                "base_url": "http://10.0.0.99:11434",
                "lane_type": "gpu",
                "backend_type": "llama",
                "status": "offline",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "current_model_name": "Qwen3.5-9B-Q4_K_M.gguf",
                "current_model_tags": [],
                "current_model_max_ctx": None,
            }
        ]
        stale = datetime.now(tz=timezone.utc) - timedelta(days=1)
        mw_rows = [
            {
                "host_id": "static-deskix",
                "lane_id": "gpu",
                "last_heartbeat_at": stale,
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "Qwen3.5-9B-Q4_K_M.gguf",
            }
        ]
        router_module.db = _FakeDb(_FakeCursor(fetchall_rows=base_rows))  # type: ignore[assignment]
        router_module.mw_state_db = _FakeDb(_FakeCursor(fetchall_rows=mw_rows))  # type: ignore[assignment]

        with self.assertRaisesRegex(RuntimeError, "no READY lanes for pinned worker"):
            router_module.pick_lane_for_model(
                model="Qwen3.5-9B-Q4_K_M.gguf",
                pin_worker="Static-Deskix",
            )


if __name__ == "__main__":
    unittest.main()


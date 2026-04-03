from __future__ import annotations

import unittest
from unittest import mock

from mesh_router import router as router_module
from datetime import datetime, timezone


class _Cur:
    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _Conn:
    def cursor(self):  # noqa: ANN001
        return _Cur()

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _Db:
    def connect(self):  # noqa: ANN001
        return _Conn()


class PinWorkerPlacementTests(unittest.TestCase):
    def test_pin_worker_falls_back_to_any_ready_lane_on_host(self) -> None:
        # When pinning by worker only, placement should still succeed even if the requested
        # model isn't already loaded. The MW pre-stream `load_model` step handles alignment.
        rows = [
            {
                "lane_id": "lane-other",
                "host_name": "Other-Host",
                "base_url": "http://10.0.0.88:11434",
                "lane_type": "gpu",
                "backend_type": "llama",
                "current_model_name": "Qwen3.5-2B-Q4_K_M.gguf",
                "current_model_tags": [],
                "current_model_max_ctx": 8192,
            },
            {
                "lane_id": "lane-1",
                "host_name": "Static-Deskix",
                "base_url": "http://10.0.0.99:11434",
                "lane_type": "gpu",
                "backend_type": "llama",
                "current_model_name": "Qwen3.5-2B-Q4_K_M.gguf",
                "current_model_tags": [],
                "current_model_max_ctx": 8192,
            }
        ]

        with mock.patch.object(router_module, "db", _Db()), mock.patch.object(router_module, "q", return_value=rows):
            choice = router_module.pick_lane_for_model(
                model="qwen3.5-9b",
                pin_worker="Static-Deskix",
            )

        self.assertEqual(choice.worker_id, "Static-Deskix")
        self.assertEqual(choice.lane_id, "lane-1")

    def test_pin_worker_allows_mw_lane_when_state_db_reports_ready(self) -> None:
        rows = [
            {
                "lane_id": "lane-mw",
                "host_name": "Static-Deskix",
                "base_url": "http://10.0.0.99:11434",
                "lane_type": "gpu",
                "backend_type": "llama",
                "status": "offline",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "current_model_name": "Qwen3.5-9B-Q4_K_M.gguf",
                "current_model_tags": [],
                "current_model_max_ctx": 8192,
            }
        ]

        state_rows = [
            {
                "host_id": "static-deskix",
                "lane_id": "gpu",
                "last_heartbeat_at": datetime.now(tz=timezone.utc),
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "Qwen3.5-9B-Q4_K_M.gguf",
            }
        ]

        class _StateCur:
            def execute(self, *args, **kwargs):  # noqa: ANN001
                return None

            def fetchall(self):  # noqa: ANN001
                return state_rows

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _StateConn:
            def cursor(self):  # noqa: ANN001
                return _StateCur()

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _StateDb:
            def connect(self):  # noqa: ANN001
                return _StateConn()

        with (
            mock.patch.object(router_module, "db", _Db()),
            mock.patch.object(router_module, "mw_state_db", _StateDb()),
            mock.patch.object(router_module, "q", return_value=rows),
        ):
            choice = router_module.pick_lane_for_model(
                model="qwen3.5-9b",
                pin_worker="Static-Deskix",
            )
        self.assertEqual(choice.worker_id, "Static-Deskix")
        self.assertEqual(choice.lane_id, "lane-mw")

    def test_pin_worker_rejects_mw_lane_when_state_db_unavailable(self) -> None:
        rows = [
            {
                "lane_id": "lane-mw",
                "host_name": "Static-Deskix",
                "base_url": "http://10.0.0.99:11434",
                "lane_type": "gpu",
                "backend_type": "llama",
                "status": "offline",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "current_model_name": "Qwen3.5-9B-Q4_K_M.gguf",
                "current_model_tags": [],
                "current_model_max_ctx": 8192,
            }
        ]

        class _StateDbBroken:
            def connect(self):  # noqa: ANN001
                raise RuntimeError("db down")

        with (
            mock.patch.object(router_module, "db", _Db()),
            mock.patch.object(router_module, "mw_state_db", _StateDbBroken()),
            mock.patch.object(router_module, "q", return_value=rows),
        ):
            with self.assertRaisesRegex(RuntimeError, "no READY lanes"):
                router_module.pick_lane_for_model(
                    model="qwen3.5-9b",
                    pin_worker="Static-Deskix",
                )


if __name__ == "__main__":
    unittest.main()

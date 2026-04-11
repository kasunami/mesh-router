from __future__ import annotations

from datetime import datetime, timezone
import unittest
from unittest import mock

from mesh_router import router as router_module


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


class PreferMwLanePlacementTests(unittest.TestCase):
    def test_prefer_mw_lanes_frontloads_mw_candidates(self) -> None:
        rows = [
            {
                "lane_id": "lane-non-mw",
                "host_name": "Other-Host",
                "base_url": "http://10.0.0.88:11434",
                "lane_type": "gpu",
                "backend_type": "llama",
                "status": "ready",
                "proxy_auth_metadata": {},
                "current_model_name": "qwen3.5-9b",
                "current_model_tags": [],
                "current_model_max_ctx": 8192,
            },
            {
                "lane_id": "lane-mw",
                "host_name": "Static-Deskix",
                "base_url": "http://10.0.0.99:11434",
                "lane_type": "gpu",
                "backend_type": "llama",
                "status": "ready",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "current_model_name": "qwen3.5-9b",
                "current_model_tags": [],
                "current_model_max_ctx": 8192,
            },
        ]

        with (
            mock.patch.object(router_module, "db", _Db()),
            mock.patch.object(router_module, "q", return_value=rows),
            mock.patch.object(router_module.settings, "placement_prefer_mw_lanes", True),
        ):
            choice = router_module.pick_lane_for_model(model="qwen3.5-9b")

        self.assertEqual(choice.lane_id, "lane-mw")
        self.assertEqual(choice.worker_id, "Static-Deskix")

    def test_mw_overlay_backend_change_filters_image_lane_from_chat_swap_candidates(self) -> None:
        rows = [
            {
                "lane_id": "image-lane",
                "lane_name": "gpu",
                "host_name": "Static-Deskix",
                "base_url": "http://10.0.0.99:21440",
                "lane_type": "gpu",
                "backend_type": "llama",
                "status": "ready",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "current_model_name": "flux1-schnell-Q4_K_S",
                "current_model_tags": [],
                "current_model_max_ctx": None,
                "local_viable_models": [{"model_name": "Qwen3.5-0.8B-Q4_K_M.gguf", "tags": [], "max_ctx": 8192}],
                "remote_viable_models": [],
            },
            {
                "lane_id": "chat-lane",
                "lane_name": "cpu",
                "host_name": "packhub02",
                "base_url": "http://10.0.0.4:11434",
                "lane_type": "cpu",
                "backend_type": "llama",
                "status": "ready",
                "proxy_auth_metadata": {},
                "current_model_name": "Qwen3.5-4B-Q4_K_M.gguf",
                "current_model_tags": [],
                "current_model_max_ctx": 8192,
                "local_viable_models": [{"model_name": "Qwen3.5-0.8B-Q4_K_M.gguf", "tags": [], "max_ctx": 8192}],
                "remote_viable_models": [],
            },
        ]
        state_rows = [
            {
                "host_id": "static-deskix",
                "lane_id": "gpu",
                "last_heartbeat_at": datetime.now(tz=timezone.utc),
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "flux1-schnell-Q4_K_S",
                "backend_type": "stable-diffusion.cpp",
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
            choice = router_module.pick_lane_for_model(model="qwen3.5-0.8b")

        self.assertEqual(choice.lane_id, "chat-lane")
        self.assertEqual(choice.worker_id, "packhub02")

    def test_policy_disallowed_viability_is_not_swappable(self) -> None:
        rows = [
            {
                "lane_id": "disallowed-gpu",
                "host_name": "Static-Mobile-2",
                "base_url": "http://10.0.0.132:21436",
                "lane_type": "gpu",
                "backend_type": "llama",
                "status": "ready",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-mobile-2", "mw_lane_id": "lfm"},
                "current_model_name": "LFM2.5-350M-Q4_K_M.gguf",
                "current_model_tags": [],
                "current_model_max_ctx": 8192,
                "local_viable_models": [
                    {
                        "model_name": "Qwen3.5-0.8B-Q4_K_M.gguf",
                        "tags": [],
                        "max_ctx": 8192,
                        "allowed": False,
                    }
                ],
                "remote_viable_models": [],
            },
            {
                "lane_id": "allowed-cpu",
                "host_name": "packhub02",
                "base_url": "http://10.0.0.4:11434",
                "lane_type": "cpu",
                "backend_type": "llama",
                "status": "ready",
                "proxy_auth_metadata": {},
                "current_model_name": "Qwen3.5-4B-Q4_K_M.gguf",
                "current_model_tags": [],
                "current_model_max_ctx": 8192,
                "local_viable_models": [
                    {
                        "model_name": "Qwen3.5-0.8B-Q4_K_M.gguf",
                        "tags": [],
                        "max_ctx": 8192,
                        "allowed": True,
                    }
                ],
                "remote_viable_models": [],
            },
        ]

        with (
            mock.patch.object(router_module, "db", _Db()),
            mock.patch.object(router_module, "q", return_value=rows),
        ):
            choice = router_module.pick_lane_for_model(model="qwen3.5-0.8b")

        self.assertEqual(choice.lane_id, "allowed-cpu")
        self.assertEqual(choice.worker_id, "packhub02")


if __name__ == "__main__":
    unittest.main()

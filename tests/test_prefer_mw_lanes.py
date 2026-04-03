from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()


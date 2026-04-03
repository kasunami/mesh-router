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


class PinWorkerPlacementTests(unittest.TestCase):
    def test_pin_worker_falls_back_to_any_ready_lane_on_host(self) -> None:
        # When pinning by worker only, placement should still succeed even if the requested
        # model isn't already loaded. The MW pre-stream `load_model` step handles alignment.
        rows = [
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


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import unittest

from mesh_router import app as app_module
from mesh_router import router as router_module


class NormalizePinLaneIdTests(unittest.TestCase):
    def test_chat_normalization_includes_pin_lane_id(self) -> None:
        normalized = app_module._normalize_route_request(
            route="chat",
            raw_payload={
                "model": "qwen3.5-2b",
                "messages": [{"role": "user", "content": "hi"}],
                "mesh_pin_lane_id": "lane-123",
            },
        )
        self.assertEqual(normalized.get("pin_lane_id"), "lane-123")

    def test_embeddings_normalization_includes_pin_lane_id(self) -> None:
        normalized = app_module._normalize_route_request(
            route="embeddings",
            raw_payload={"model": "nomic-embed-text", "input": "hi", "mesh_pin_lane_id": "lane-456"},
        )
        self.assertEqual(normalized.get("pin_lane_id"), "lane-456")

    def test_images_normalization_includes_pin_lane_id(self) -> None:
        normalized = app_module._normalize_route_request(
            route="images",
            raw_payload={"model": "flux.1-schnell", "prompt": "cat", "mesh_pin_lane_id": "lane-789"},
        )
        self.assertEqual(normalized.get("pin_lane_id"), "lane-789")


class PinLaneIdPlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_db = router_module.db
        self.orig_q = router_module.q
        self.orig_apply = router_module.apply_mw_effective_status

    def tearDown(self) -> None:
        router_module.db = self.orig_db  # type: ignore[assignment]
        router_module.q = self.orig_q  # type: ignore[assignment]
        router_module.apply_mw_effective_status = self.orig_apply  # type: ignore[assignment]

    def test_pin_lane_id_not_found_is_404(self) -> None:
        router_module.q = lambda cur, sql, params: []  # type: ignore[assignment]

        class _Conn:
            def cursor(self):  # noqa: ANN001
                return self

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Db:
            def connect(self):  # noqa: ANN001
                return _Conn()

        router_module.db = _Db()  # type: ignore[assignment]
        router_module.apply_mw_effective_status = lambda *a, **k: None  # type: ignore[assignment]

        with self.assertRaises(router_module.LanePlacementError) as ctx:
            router_module.pick_lane_for_model(model="qwen", pin_lane_id="missing")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_pin_lane_id_offline_is_409(self) -> None:
        row = {
            "lane_id": "lane-1",
            "host_name": "Static-Deskix",
            "base_url": "http://10.0.0.99:11434",
            "lane_type": "gpu",
            "backend_type": "llama",
            "status": "offline",
            "proxy_auth_metadata": {},
            "current_model_name": None,
            "current_model_max_ctx": None,
        }
        router_module.q = lambda cur, sql, params: [row]  # type: ignore[assignment]

        class _Conn:
            def cursor(self):  # noqa: ANN001
                return self

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Db:
            def connect(self):  # noqa: ANN001
                return _Conn()

        router_module.db = _Db()  # type: ignore[assignment]
        router_module.apply_mw_effective_status = lambda *a, **k: None  # type: ignore[assignment]

        with self.assertRaises(router_module.LanePlacementError) as ctx:
            router_module.pick_lane_for_model(model="qwen", pin_lane_id="lane-1")
        self.assertEqual(ctx.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()

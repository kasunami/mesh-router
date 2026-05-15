from __future__ import annotations

import unittest
from unittest import mock

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

    def test_chat_normalization_strips_openai_provider_prefix_for_router(self) -> None:
        normalized = app_module._normalize_route_request(
            route="chat",
            raw_payload={
                "model": "openai/qwen3.5-9b",
                "messages": [{"role": "user", "content": "hi"}],
                "mesh_pin_lane_id": "lane-123",
            },
        )
        self.assertEqual(normalized.get("requested_model_name"), "qwen3.5-9b")
        self.assertEqual(normalized.get("request_payload", {}).get("model"), "qwen3.5-9b")

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
    def test_pin_lane_id_not_found_is_404(self) -> None:
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

        with (
            mock.patch.object(router_module, "db", _Db()),
            mock.patch.object(router_module, "q", return_value=[]),
            mock.patch.object(router_module, "apply_mw_effective_status", lambda *a, **k: None),
        ):
            with self.assertRaises(router_module.LanePlacementError) as ctx:
                router_module.pick_lane_for_model(model="qwen", pin_lane_id="11111111-1111-1111-1111-111111111111")
        self.assertEqual(getattr(ctx.exception, "status_code", None), 404)

    def test_pin_lane_id_invalid_format_is_400(self) -> None:
        with self.assertRaises(router_module.LanePlacementError) as ctx:
            router_module.pick_lane_for_model(model="qwen", pin_lane_id="lane-does-not-exist")
        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)

    def test_pin_lane_id_offline_is_409(self) -> None:
        row = {
            "lane_id": "22222222-2222-2222-2222-222222222222",
            "host_name": "Static-Deskix",
            "base_url": "http://10.0.0.99:11434",
            "lane_type": "gpu",
            "backend_type": "llama",
            "status": "offline",
            "proxy_auth_metadata": {},
            "current_model_name": None,
            "current_model_max_ctx": None,
        }

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

        with (
            mock.patch.object(router_module, "db", _Db()),
            mock.patch.object(router_module, "q", return_value=[row]),
            mock.patch.object(router_module, "apply_mw_effective_status", lambda *a, **k: None),
        ):
            with self.assertRaises(router_module.LanePlacementError) as ctx:
                router_module.pick_lane_for_model(model="qwen", pin_lane_id="22222222-2222-2222-2222-222222222222")
        self.assertEqual(getattr(ctx.exception, "status_code", None), 409)

    def test_pin_lane_id_with_matching_worker_and_base_url_is_accepted(self) -> None:
        row = {
            "lane_id": "44444444-4444-4444-4444-444444444444",
            "host_name": "Static-Deskix",
            "base_url": "http://10.0.0.99:21434",
            "lane_type": "gpu",
            "backend_type": "llama",
            "status": "ready",
            "proxy_auth_metadata": {},
            "current_model_name": "qwen3.5-9b",
            "current_model_max_ctx": 131072,
        }

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

        with (
            mock.patch.object(router_module, "db", _Db()),
            mock.patch.object(router_module, "q", return_value=[row]),
            mock.patch.object(router_module, "apply_mw_effective_status", lambda *a, **k: None),
        ):
            choice = router_module.pick_lane_for_model(
                model="openai/qwen3.5-9b",
                pin_lane_id="44444444-4444-4444-4444-444444444444",
                pin_worker="Static-Deskix",
                pin_base_url="http://10.0.0.99:21434/",
                pin_lane_type="gpu",
            )
        self.assertEqual(choice.lane_id, "44444444-4444-4444-4444-444444444444")

    def test_pin_lane_id_operator_suspended_overlay_is_409(self) -> None:
        row = {
            "lane_id": "33333333-3333-3333-3333-333333333333",
            "host_name": "pupix1",
            "base_url": "http://10.0.0.95:11436",
            "lane_type": "other",
            "backend_type": "llama",
            "status": "suspended",
            "proxy_auth_metadata": {"control_plane": "mw"},
            "current_model_name": "gemma-4-26B-A4B-it-Q4_K_M",
            "current_model_max_ctx": None,
        }

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

        def _overlay(rows, **_kwargs):  # noqa: ANN001
            rows[0]["effective_status"] = "suspended"
            rows[0]["readiness_reason"] = "operator_suspended"

        with (
            mock.patch.object(router_module, "db", _Db()),
            mock.patch.object(router_module, "q", return_value=[row]),
            mock.patch.object(router_module, "apply_mw_effective_status", _overlay),
        ):
            with self.assertRaises(router_module.LanePlacementError) as ctx:
                router_module.pick_lane_for_model(
                    model="gemma-4-26B-A4B-it-Q4_K_M",
                    pin_lane_id="33333333-3333-3333-3333-333333333333",
                )
        self.assertEqual(getattr(ctx.exception, "status_code", None), 409)


if __name__ == "__main__":
    unittest.main()

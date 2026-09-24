from __future__ import annotations

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from mesh_router import app as app_module
from mesh_router import route_resolver as resolver_module


class _Choice:
    def __init__(self) -> None:
        self.lane_id = "lane-1"
        self.worker_id = "Worker-A"
        self.base_url = "http://worker-a.example:11434"
        self.lane_type = "gpu"
        self.backend_type = "llama"
        self.current_model_name = "Qwen3.5-9B-Q4_K_M.gguf"


class RouteResolveApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_pick = resolver_module.pick_lane_for_model
        self.orig_perf = resolver_module._perf_for_choice

    def tearDown(self) -> None:
        resolver_module.pick_lane_for_model = self.orig_pick  # type: ignore[assignment]
        resolver_module._perf_for_choice = self.orig_perf  # type: ignore[assignment]

    def test_route_resolve_by_tags_returns_choice(self) -> None:
        resolver_module.pick_lane_for_model = lambda **kwargs: _Choice()  # type: ignore[assignment]
        client = TestClient(app_module.app)
        resp = client.post(
            "/api/routes/resolve",
            json={"tags": ["text", "fast"], "modality": "chat", "allow_opportunistic": True},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["choice"]["worker_id"], "Worker-A")

    def test_qwen_selection_tag_resolves_as_model_candidate(self) -> None:
        self.assertEqual(
            resolver_module._tag_model_candidates(["qwen3.5:0.8B"], modality="chat"),
            ["qwen3.5:0.8B"],
        )

    def test_firecalc_visual_capability_resolves_as_vlm_tag(self) -> None:
        self.assertEqual(
            resolver_module._tag_model_candidates(["firecalc-pdf-visual"], modality="chat"),
            ["firecalc.pdf.visual"],
        )

    def test_firecalc_visual_capability_requires_multimodal_lane(self) -> None:
        seen: list[dict] = []

        def _pick(**kwargs):  # noqa: ANN001
            seen.append(kwargs)
            choice = _Choice()
            choice.current_model_name = "Qwen3.5-9B-VLM-Q4_K_M.gguf"
            choice.resolved_model_name = "Qwen3.5-9B-VLM-Q4_K_M.gguf"
            return choice

        resolver_module.pick_lane_for_model = _pick  # type: ignore[assignment]
        client = TestClient(app_module.app)
        resp = client.post(
            "/api/routes/resolve",
            json={"tags": ["firecalc.pdf.visual"], "modality": "chat", "allow_opportunistic": True},
        )

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        self.assertTrue(seen[0]["requires_multimodal"])
        self.assertEqual(resp.json()["choice"]["resolved_model"], "Qwen3.5-9B-VLM-Q4_K_M.gguf")

    def test_firecalc_visual_capability_rejects_explicit_text_only_lane(self) -> None:
        class _Cursor:
            def execute(self, sql, params):  # noqa: ANN001, ARG002
                return None

            def fetchone(self):
                return {
                    "lane_id": "lane-text",
                    "lane_name": "gpu",
                    "base_url": "http://worker-a.example:11434",
                    "lane_type": "gpu",
                    "backend_type": "llama",
                    "current_model_name": "qwen3.5-9b",
                    "proxy_auth_metadata": {"control_plane": "mw"},
                    "host_name": "worker-a",
                    "status": "ready",
                }

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Conn:
            def cursor(self):
                return _Cursor()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _Db:
            def connect(self):
                return _Conn()

        with mock.patch.object(resolver_module, "db", _Db()), mock.patch.object(
            resolver_module, "apply_mw_effective_status", lambda *args, **kwargs: None
        ):
            choice, _perf, reason, _count = resolver_module.resolve_route(
                model=None,
                modality="chat",
                tags=["firecalc.pdf.visual"],
                host_name="worker-a",
                lane_id="lane-text",
                allow_opportunistic=True,
            )

        self.assertIsNone(choice)
        self.assertEqual(reason, "explicit lane does not support required multimodal capability")

    def test_route_resolve_prefers_best_perf_candidate(self) -> None:
        # Ensure resolve_route ranks among model candidates deterministically when perf expectations exist.
        def _pick(**kwargs):  # noqa: ANN001
            c = _Choice()
            c.current_model_name = kwargs.get("model", c.current_model_name)
            return c

        def _perf(choice, *, model, modality):  # noqa: ANN001
            # Favor the middle candidate.
            tps = {"qwen3.5:9B": 50.0, "qwen3.5:4B": 120.0, "qwen3.5:2B": 80.0}.get(str(model), 0.0)
            return {
                "host_id": "worker-a",
                "lane_id": "lane-1",
                "model_name": str(model),
                "modality": str(modality),
                "updated_at": "2026-04-03T00:00:00Z",
                "sample_count": 3,
                "decode_tps_p50": tps,
                "first_token_ms_p50": 10.0,
            }

        resolver_module.pick_lane_for_model = _pick  # type: ignore[assignment]
        resolver_module._perf_for_choice = _perf  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.post(
            "/api/routes/resolve",
            json={"tags": ["text", "fast"], "modality": "chat", "allow_opportunistic": True},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["choice"]["resolved_model"], "qwen3.5:4B")

    def test_route_resolve_passes_lane_id_pin_to_picker(self) -> None:
        seen: list[dict] = []

        def _pick(**kwargs):  # noqa: ANN001
            seen.append(kwargs)
            return _Choice()

        resolver_module.pick_lane_for_model = _pick  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.post(
            "/api/routes/resolve",
            json={
                "modality": "chat",
                "model": "qwen3.5-4b",
                "lane_id": "8a37c3e3-eefc-43b0-90b7-737c57198287",
                "allow_opportunistic": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        self.assertEqual(seen[0]["pin_lane_id"], "8a37c3e3-eefc-43b0-90b7-737c57198287")

    def test_capability_route_filters_live_lane_caps_context_and_pins_exact_choice(self) -> None:
        lanes = [
            {"lane_id": "11111111-1111-1111-1111-111111111111", "host_name": "small", "effective_status": "ready", "current_model_name": "model-small", "current_model_max_ctx": 8192, "capabilities": ["chat", "inference"]},
            {"lane_id": "22222222-2222-2222-2222-222222222222", "host_name": "large", "effective_status": "ready", "current_model_name": "model-large", "current_model_max_ctx": 16384, "capabilities": ["chat", "inference"]},
            {"lane_id": "33333333-3333-3333-3333-333333333333", "host_name": "unhealthy", "effective_status": "suspended", "current_model_name": "model-down", "current_model_max_ctx": 32768, "capabilities": ["chat", "inference"]},
        ]
        seen: list[dict] = []

        class _Inventory:
            def model_dump(self, *, mode=None):  # noqa: ANN001, ARG002
                return {"items": [{"host_name": lane["host_name"], "lanes": [lane]} for lane in lanes]}

        def _pick(**kwargs):  # noqa: ANN001
            seen.append(kwargs)
            choice = _Choice()
            choice.lane_id = kwargs["pin_lane_id"]
            choice.worker_id = kwargs["pin_worker"]
            choice.current_model_name = kwargs["model"]
            return choice

        with mock.patch.object(app_module, "api_inventory", return_value=_Inventory()), mock.patch.object(
            app_module, "pick_lane_for_model", side_effect=_pick
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/api/routes/resolve",
                json={"required_capabilities": ["chat", "inference"], "min_context_tokens": 12000},
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["choice"]["worker_id"], "large")
        self.assertEqual(body["choice"]["current_model_name"], "model-large")
        self.assertEqual(body["choice"]["max_context_tokens"], 16384)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["pin_lane_id"], "22222222-2222-2222-2222-222222222222")
        self.assertEqual(seen[0]["request_context_tokens"], 12000)

    def test_capability_route_fails_when_no_live_lane_advertises_required_capability(self) -> None:
        class _Inventory:
            def model_dump(self, *, mode=None):  # noqa: ANN001, ARG002
                return {"items": [{"host_name": "chat-only", "lanes": [{
                    "lane_id": "11111111-1111-1111-1111-111111111111",
                    "effective_status": "ready",
                    "current_model_name": "model-chat",
                    "current_model_max_ctx": 16384,
                    "capabilities": ["chat", "inference"],
                }]}]}

        with mock.patch.object(app_module, "api_inventory", return_value=_Inventory()):
            client = TestClient(app_module.app)
            resp = client.post(
                "/api/routes/resolve",
                json={"required_capabilities": ["fim", "completion"], "min_context_tokens": 4096},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["ok"])
        self.assertIn("no healthy lane satisfies", resp.json()["reason"])

    def test_fim_capability_route_uses_completion_modality(self) -> None:
        class _Inventory:
            def model_dump(self, *, mode=None):  # noqa: ANN001, ARG002
                return {"items": [{"host_name": "fim-worker", "lanes": [{
                    "lane_id": "33333333-3333-3333-3333-333333333333",
                    "effective_status": "ready",
                    "current_model_name": "fim-model-from-worker",
                    "current_model_max_ctx": 8192,
                    "capabilities": ["fim", "completion"],
                }]}]}

        seen: list[dict] = []
        def _pick(**kwargs):  # noqa: ANN001
            seen.append(kwargs)
            return _Choice()

        with mock.patch.object(app_module, "api_inventory", return_value=_Inventory()), mock.patch.object(
            app_module, "pick_lane_for_model", _pick
        ):
            client = TestClient(app_module.app)
            resp = client.post("/api/routes/resolve", json={
                "modality": "completion",
                "required_capabilities": ["fim", "completion"],
                "min_context_tokens": 4096,
            })

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        self.assertEqual(resp.json()["choice"]["current_model_name"], "fim-model-from-worker")
        self.assertIsNone(seen[0]["backend_type"])

    def test_explicit_lane_resolve_rejects_not_ready_overlay(self) -> None:
        class _Cursor:
            def execute(self, sql, params):  # noqa: ANN001, ARG002
                return None

            def fetchone(self):  # noqa: ANN001
                return {
                    "lane_id": "85557f61-07bd-43af-ae00-1f5c566c8b48",
                    "lane_name": "mlx",
                    "base_url": "http://worker-d.example:11434",
                    "lane_type": "mlx",
                    "backend_type": "mlx",
                    "current_model_name": "/models/Qwen3.5-9B-MLX-4bit",
                    "proxy_auth_metadata": {"control_plane": "mw"},
                    "host_name": "worker-d",
                    "status": "ready",
                }

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Conn:
            def cursor(self):  # noqa: ANN001
                return _Cursor()

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
            mock.patch.object(resolver_module, "db", _Db()),
            mock.patch.object(resolver_module, "apply_mw_effective_status", _overlay),
        ):
            choice, perf, reason, count = resolver_module.resolve_route(
                model="/models/Qwen3.5-9B-MLX-4bit",
                modality="chat",
                tags=[],
                host_name="worker-d",
                lane_id="85557f61-07bd-43af-ae00-1f5c566c8b48",
                allow_opportunistic=False,
            )

        self.assertIsNone(choice)
        self.assertIsNone(perf)
        self.assertEqual(reason, "explicit lane is not ready")
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()

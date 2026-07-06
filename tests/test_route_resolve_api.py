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

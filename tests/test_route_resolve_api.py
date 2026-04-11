from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from mesh_router import app as app_module
from mesh_router import route_resolver as resolver_module


class _Choice:
    def __init__(self) -> None:
        self.lane_id = "lane-1"
        self.worker_id = "Static-Deskix"
        self.base_url = "http://10.0.0.99:11434"
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
        self.assertEqual(body["choice"]["worker_id"], "Static-Deskix")

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
                "host_id": "static-deskix",
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


if __name__ == "__main__":
    unittest.main()

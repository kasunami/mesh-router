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

    def tearDown(self) -> None:
        resolver_module.pick_lane_for_model = self.orig_pick  # type: ignore[assignment]

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


if __name__ == "__main__":
    unittest.main()

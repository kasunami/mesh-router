from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from mesh_router import app as app_module


class RouterRequestsApiTests(unittest.TestCase):
    def test_active_router_requests_uses_collection_route(self) -> None:
        with patch.object(
            app_module,
            "_list_router_requests",
            return_value=[
                {
                    "request_id": "11111111-1111-1111-1111-111111111111",
                    "route": "chat",
                    "state": "running",
                    "owner": "test",
                    "job_type": "chat",
                    "requested_model_name": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                }
            ],
        ) as mocked_list:
            client = TestClient(app_module.app)
            resp = client.get("/api/router-requests/active")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["count"], 1)
            self.assertEqual(body["items"][0]["request_id"], "11111111-1111-1111-1111-111111111111")
            mocked_list.assert_called_once()

    def test_invalid_router_request_id_returns_404(self) -> None:
        with patch.object(app_module, "_fetch_router_request", return_value=None):
            client = TestClient(app_module.app)
            resp = client.get("/api/router-requests/not-a-uuid")
            self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()

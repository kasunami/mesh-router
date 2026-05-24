from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from mesh_router import app as app_module


class ClientRequestIdHeaderTests(unittest.TestCase):
    def test_chat_records_client_request_id_header(self) -> None:
        with (
            patch.object(app_module, "_create_router_request", return_value="req-1") as create_router_request,
            patch.object(app_module, "_execute_router_request", return_value={"ok": True}),
            patch.object(app_module, "_fetch_router_request", return_value=None),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/chat/completions",
                headers={"x-mesh-client-request-id": "mc-tool-worker-run-1"},
                json={"model": "qwen3.5-2b", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(create_router_request.call_args.kwargs["client_request_id"], "mc-tool-worker-run-1")

    def test_embeddings_records_client_request_id_header(self) -> None:
        with (
            patch.object(app_module, "_create_router_request", return_value="req-1") as create_router_request,
            patch.object(app_module, "_execute_router_request", return_value={"data": []}),
            patch.object(app_module, "_fetch_router_request", return_value=None),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/embeddings",
                headers={"x-mesh-client-request-id": "mc-embedding-1"},
                json={"model": "embed-model", "input": "hello"},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(create_router_request.call_args.kwargs["client_request_id"], "mc-embedding-1")

    def test_images_records_client_request_id_header(self) -> None:
        with (
            patch.object(app_module, "_create_router_request", return_value="req-1") as create_router_request,
            patch.object(app_module, "_execute_router_request", return_value={"data": []}),
            patch.object(app_module, "_fetch_router_request", return_value=None),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/images/generations",
                headers={"x-mesh-client-request-id": "mc-image-1"},
                json={"model": "image-model", "prompt": "test"},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(create_router_request.call_args.kwargs["client_request_id"], "mc-image-1")


if __name__ == "__main__":
    unittest.main()

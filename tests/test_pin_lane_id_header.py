from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from mesh_router import app as app_module


class PinLaneIdHeaderTests(unittest.TestCase):
    def test_chat_accepts_pin_lane_id_header(self) -> None:
        captured: dict[str, object] = {}

        def _fake_normalize_route_request(*, route: str, raw_payload: dict):  # noqa: ANN001
            self.assertEqual(route, "chat")
            self.assertEqual(raw_payload.get("mesh_pin_lane_id"), "lane-123")
            captured["raw_payload"] = dict(raw_payload)
            return {
                "request_payload": {"model": raw_payload.get("model"), "stream": False, "messages": raw_payload.get("messages", [])},
                "requested_model_name": raw_payload.get("model") or "qwen3.5-2b",
                "pin_worker": None,
                "pin_base_url": None,
                "pin_lane_type": None,
                "pin_lane_id": "lane-123",
            }

        with (
            patch.object(app_module, "_normalize_route_request", side_effect=_fake_normalize_route_request),
            patch.object(app_module, "resolve_route", return_value=({"lane_id": "lane-123"}, None, None, 1)) as resolve_route,
            patch.object(app_module, "_create_router_request", return_value="req-1"),
            patch.object(app_module, "_execute_router_request", return_value={"ok": True}),
            patch.object(app_module, "_fetch_router_request", return_value=None),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/chat/completions",
                headers={"x-mesh-pin-lane-id": "lane-123"},
                json={"model": "qwen3.5-2b", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json().get("ok"))
            self.assertIn("raw_payload", captured)
            resolve_route.assert_called_once()
            self.assertFalse(resolve_route.call_args.kwargs["allow_opportunistic"])

    def test_chat_propagates_http_exception_for_explicit_route_failures(self) -> None:
        # Requestor-grade explicit routing should preserve actionable status codes.
        with (
            patch.object(app_module, "_normalize_route_request", return_value={"request_payload": {"stream": False}, "requested_model_name": "qwen3.5-2b"}),
            patch.object(app_module, "_create_router_request", return_value="req-1"),
            patch.object(app_module, "_execute_router_request", side_effect=HTTPException(status_code=404, detail="pinned lane not found")),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/chat/completions",
                headers={"x-mesh-pin-lane-id": "missing"},
                json={"model": "qwen3.5-2b", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(resp.status_code, 404)
            self.assertIn("pinned lane not found", resp.text)

    def test_chat_rejects_pin_lane_id_when_route_resolver_finds_no_choice(self) -> None:
        with (
            patch.object(
                app_module,
                "_normalize_route_request",
                return_value={
                    "request_payload": {"stream": False},
                    "requested_model_name": "gemma-4-26B-A4B-it-Q4_K_M",
                    "pin_lane_id": "79c17e79-052b-48b5-9781-acbb199f81f7",
                },
            ),
            patch.object(
                app_module,
                "resolve_route",
                return_value=(None, None, "no eligible route found for tags/model constraints", 1),
            ),
            patch.object(app_module, "_create_router_request", return_value="req-1"),
            patch.object(app_module, "_execute_router_request", return_value={"ok": True}),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gemma-4-26B-A4B-it-Q4_K_M", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(resp.status_code, 409)
            self.assertIn("no eligible route", resp.text)


if __name__ == "__main__":
    unittest.main()

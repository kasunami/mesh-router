from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from fastapi.testclient import TestClient

from mesh_router import app as app_module
from mesh_router.perf_registry import PerfExpectation


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


class PerfExpectationHeadersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_enabled = app_module.settings.route_debug_headers_enabled
        app_module.settings.route_debug_headers_enabled = True

    def tearDown(self) -> None:
        app_module.settings.route_debug_headers_enabled = self.orig_enabled

    def test_chat_includes_perf_headers_when_enabled_and_available(self) -> None:
        exp = PerfExpectation(
            host_id="static-deskix",
            lane_id="11111111-1111-1111-1111-111111111111",
            model_name="qwen3.5-2b",
            modality="chat",
            updated_at=datetime(2026, 4, 3, 18, 0, 0, tzinfo=UTC),
            sample_count=7,
            first_token_ms_p50=12.5,
            decode_tps_p50=45.0,
            total_ms_p50=400.0,
        )

        with (
            patch.object(app_module, "mw_state_db", _Db()),
            patch.object(app_module, "get_expectation", return_value=exp),
            patch.object(app_module, "_normalize_route_request", return_value={"request_payload": {"stream": False}, "requested_model_name": "qwen3.5-2b"}),
            patch.object(app_module, "_create_router_request", return_value="req-1"),
            patch.object(app_module, "_execute_router_request", return_value={"ok": True}),
            patch.object(
                app_module,
                "_fetch_router_request",
                return_value={
                    "lane_id": "11111111-1111-1111-1111-111111111111",
                    "worker_id": "Static-Deskix",
                    "downstream_model_name": "qwen3.5-2b",
                },
            ),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "qwen3.5-2b", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers.get("X-Mesh-Perf-Sample-Count"), "7")
            self.assertEqual(resp.headers.get("X-Mesh-Perf-Updated-At"), exp.updated_at.isoformat())
            self.assertEqual(resp.headers.get("X-Mesh-Perf-FirstTokenMs-P50"), "12.5")
            self.assertEqual(resp.headers.get("X-Mesh-Perf-DecodeTps-P50"), "45.0")
            self.assertEqual(resp.headers.get("X-Mesh-Perf-TotalMs-P50"), "400.0")

    def test_perf_headers_disabled_by_default(self) -> None:
        app_module.settings.route_debug_headers_enabled = False
        with (
            patch.object(app_module, "mw_state_db", _Db()),
            patch.object(app_module, "get_expectation") as mocked_get,
            patch.object(app_module, "_normalize_route_request", return_value={"request_payload": {"stream": False}, "requested_model_name": "qwen3.5-2b"}),
            patch.object(app_module, "_create_router_request", return_value="req-1"),
            patch.object(app_module, "_execute_router_request", return_value={"ok": True}),
            patch.object(app_module, "_fetch_router_request", return_value={"lane_id": "lane-1", "worker_id": "Static-Deskix", "downstream_model_name": "qwen3.5-2b"}),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "qwen3.5-2b", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIsNone(resp.headers.get("X-Mesh-Perf-Sample-Count"))
            mocked_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()


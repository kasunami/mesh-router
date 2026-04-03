from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from mesh_router import app as app_module


class FakeMWClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.next_result: dict[str, object] | None = None

    def send_command(
        self,
        *,
        host_id: str,
        message_type: str,
        payload: dict[str, object],
        request_id: str | None = None,
        wait: bool = True,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "host_id": host_id,
                "message_type": message_type,
                "payload": payload,
                "request_id": request_id,
                "wait": wait,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.next_result is not None:
            return self.next_result
        return {
            "ok": True,
            "host_id": host_id,
            "request_id": request_id or "req-123",
            "message_type": message_type,
            "result": {"echo": payload},
            "response": {"payload": {"ok": True, "result": {"echo": payload}}},
        }


class MWControlApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_getter = app_module._mw_client
        self.fake = FakeMWClient()
        app_module._mw_client.cache_clear()
        app_module._mw_client = lambda: self.fake  # type: ignore[method-assign]
        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        app_module._mw_client = self.original_getter
        app_module._mw_client.cache_clear()

    def test_mw_command_status_not_found(self) -> None:
        original_db = app_module.db

        class _Cur:
            def execute(self, *args, **kwargs):  # noqa: ANN001
                return None

            def fetchone(self):  # noqa: ANN001
                return None

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

        app_module.db = _Db()  # type: ignore[assignment]
        try:
            response = self.client.get("/api/mw/commands/00000000-0000-0000-0000-000000000000")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertFalse(body["found"])
        finally:
            app_module.db = original_db  # type: ignore[assignment]

    def test_generic_mw_command_endpoint(self) -> None:
        response = self.client.post(
            "/api/mw/commands",
            json={
                "host_id": "static-deskix",
                "message_type": "load_model",
                "payload": {"lane_id": "gpu", "model_name": "qwen3.5-4b"},
                "wait": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["message_type"], "load_model")
        self.assertEqual(body["result"]["echo"]["model_name"], "qwen3.5-4b")

    def test_load_model_accepts_model_id_alias(self) -> None:
        response = self.client.post(
            "/api/mw/commands",
            json={
                "host_id": "static-deskix",
                "message_type": "load_model",
                "payload": {"lane_id": "gpu", "model_id": "qwen3.5-2b"},
                "wait": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        # Verify MR normalized the payload before dispatch.
        self.assertEqual(self.fake.calls[-1]["payload"]["model_name"], "qwen3.5-2b")

    def test_load_model_rejects_missing_model_name(self) -> None:
        response = self.client.post(
            "/api/mw/commands",
            json={
                "host_id": "static-deskix",
                "message_type": "load_model",
                "payload": {"lane_id": "gpu"},
                "wait": True,
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_mw_command_timeout_returns_pending_202(self) -> None:
        self.fake.next_result = {
            "ok": True,
            "pending": True,
            "host_id": "static-deskix",
            "request_id": "req-timeout-1",
            "message_type": "activate_profile",
            "warning": "timed out waiting for MeshWorker response",
            "timeout_seconds": 60,
        }
        response = self.client.post(
            "/api/mw/commands",
            json={
                "host_id": "static-deskix",
                "message_type": "activate_profile",
                "payload": {"profile_id": "split_default"},
                "wait": True,
            },
        )
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["pending"])
        self.assertEqual(body["request_id"], "req-timeout-1")

    def test_health_probe_shortcut_endpoint(self) -> None:
        response = self.client.post("/api/mw/hosts/static-deskix/health-probe?service_id=llama-gpu")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(self.fake.calls[-1]["message_type"], "health_probe")
        self.assertEqual(self.fake.calls[-1]["payload"], {"service_id": "llama-gpu"})

    def test_lane_gateway_health_prefers_mw(self) -> None:
        ok = app_module._lane_gateway_healthy(  # type: ignore[attr-defined]
            "http://127.0.0.1:21434",
            host_id="static-deskix",
            lane_id="gpu",
        )
        self.assertTrue(ok)
        self.assertEqual(self.fake.calls[-1]["message_type"], "health_probe")
        self.assertEqual(self.fake.calls[-1]["payload"], {"lane_id": "gpu"})

    def test_lane_service_action_prefers_mw(self) -> None:
        result = app_module._call_lane_service_action(  # type: ignore[attr-defined]
            base_url="http://127.0.0.1:21434",
            action="stop",
            host_id="static-deskix",
            lane_id="gpu",
        )
        self.assertEqual(result["echo"], {"lane_id": "gpu"})
        self.assertEqual(self.fake.calls[-1]["message_type"], "stop_service")


if __name__ == "__main__":
    unittest.main()

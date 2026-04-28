from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from mesh_router import app as app_module
from mesh_router import mw_commands as mw_commands_module


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
        original_mw_state_db = app_module.mw_state_db

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

        app_module.mw_state_db = _Db()  # type: ignore[assignment]
        try:
            response = self.client.get("/api/mw/commands/00000000-0000-0000-0000-000000000000")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertFalse(body["found"])
        finally:
            app_module.mw_state_db = original_mw_state_db  # type: ignore[assignment]

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
        self.assertEqual(response.headers.get("location"), "/api/mw/commands/req-timeout-1")
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

    def test_unload_lane_command_endpoint(self) -> None:
        response = self.client.post(
            "/api/mw/commands",
            json={
                "host_id": "static-deskix",
                "message_type": "unload_lane",
                "payload": {"lane_id": "gpu"},
                "wait": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(self.fake.calls[-1]["message_type"], "unload_lane")
        self.assertEqual(self.fake.calls[-1]["payload"], {"lane_id": "gpu"})

    def test_unload_service_command_endpoint(self) -> None:
        response = self.client.post(
            "/api/mw/commands",
            json={
                "host_id": "static-deskix",
                "message_type": "unload_service",
                "payload": {"service_id": "llama-gpu"},
                "wait": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(self.fake.calls[-1]["message_type"], "unload_service")
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

    def test_activate_profile_reconciles_mw_managed_lane_rows(self) -> None:
        original_db = app_module.db
        updates: list[tuple[object, ...]] = []

        class _Cur:
            def execute(self, sql, params=None):  # noqa: ANN001
                if "SELECT l.lane_id" in sql:
                    return None
                if "UPDATE lanes" in sql:
                    updates.append(tuple(params or ()))

            def fetchall(self):  # noqa: ANN001
                return [
                    {
                        "lane_id": "gpu-row",
                        "backend_type": "llama",
                        "proxy_auth_metadata": {
                            "control_plane": "mw",
                            "mw_host_id": "static-deskix",
                            "mw_lane_id": "gpu",
                        },
                    },
                    {
                        "lane_id": "image-row",
                        "backend_type": "sd",
                        "proxy_auth_metadata": {
                            "control_plane": "mw",
                            "mw_host_id": "static-deskix",
                            "mw_lane_id": "gpu",
                        },
                    },
                ]

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Conn:
            def cursor(self):  # noqa: ANN001
                return _Cur()

            def commit(self):  # noqa: ANN001
                return None

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Db:
            def connect(self):  # noqa: ANN001
                return _Conn()

        self.fake.next_result = {
            "ok": True,
            "host_id": "static-deskix",
            "request_id": "req-profile-1",
            "message_type": "activate_profile",
            "result": {
                "host_state": {
                    "lane_states": [
                        {
                            "lane_id": "gpu",
                            "actual_state": "running",
                            "health_status": "healthy",
                            "actual_model": "qwen3.5-9b",
                            "backend_type": "llama.cpp",
                        }
                    ]
                }
            },
        }

        try:
            app_module.db = _Db()  # type: ignore[assignment]
            response = self.client.post(
                "/api/mw/commands",
                json={
                    "host_id": "static-deskix",
                    "message_type": "activate_profile",
                    "payload": {"profile_id": "split_default"},
                    "wait": True,
                },
            )
        finally:
            app_module.db = original_db

        self.assertEqual(response.status_code, 200)
        self.assertIn(("ready", "qwen3.5-9b", "gpu-row"), updates)
        self.assertIn(("suspended", "qwen3.5-9b", "image-row"), updates)

    def test_pending_mw_load_waits_for_terminal_transition(self) -> None:
        self.fake.next_result = {
            "ok": True,
            "pending": True,
            "host_id": "static-deskix",
            "request_id": "req-pending-1",
            "message_type": "load_model",
        }
        with patch.object(  # type: ignore[name-defined]
            app_module,
            "_mw_client",
            lambda: self.fake,
        ), patch.object(
            mw_commands_module,
            "wait_for_mw_transition_terminal",
            return_value={"request_id": "req-pending-1", "status": "completed"},
        ) as waiter:
            result = app_module._send_mw_command_require_ready(  # type: ignore[attr-defined]
                host_id="static-deskix",
                message_type="load_model",
                payload={"lane_id": "gpu", "model_name": "qwen3.5-4b"},
                timeout_seconds=60,
            )
        self.assertTrue(result["ok"])
        self.assertFalse(result["pending"])
        waiter.assert_called_once_with(request_id="req-pending-1", timeout_seconds=60)

    def test_ready_transition_is_terminal_success(self) -> None:
        with patch.object(
            mw_commands_module,
            "fetch_mw_transition_status",
            return_value={"request_id": "req-ready-1", "status": "ready"},
        ):
            result = mw_commands_module.wait_for_mw_transition_terminal(
                request_id="req-ready-1",
                timeout_seconds=1,
            )

        self.assertEqual(result["status"], "ready")


if __name__ == "__main__":
    unittest.main()

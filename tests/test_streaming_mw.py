from __future__ import annotations

import contextlib
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from mesh_router import app as app_module
from mesh_router import request_store as request_store_module
from mesh_router.router import LaneChoice
from mesh_router.mw_grpc import MwGrpcTarget


class FakeCursor:
    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def execute(self, sql: str, params: tuple | None = None) -> None:  # noqa: ARG002
        return

    def fetchone(self) -> dict:
        return {"model_id": "model-1"}


class FakeConn:
    def __enter__(self) -> "FakeConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor()

    def commit(self) -> None:
        return


@contextlib.contextmanager
def fake_db_connect():
    yield FakeConn()


async def fake_grpc_stream_chat(self, *, target, request_id, model, messages, temperature, max_tokens, deadline_unix_ms):  # noqa: ARG001
    yield SimpleNamespace(event_type="delta", raw_backend_payload=b'{"choices":[{"delta":{"content":"hi"}}]}')
    yield SimpleNamespace(event_type="completed", raw_backend_payload=b"")


async def fake_reasoning_only_stream_chat(self, *, target, request_id, model, messages, temperature, max_tokens, deadline_unix_ms):  # noqa: ARG001
    yield SimpleNamespace(
        event_type="delta",
        raw_backend_payload=b'{"choices":[{"finish_reason":null,"delta":{"reasoning_content":"thinking"}}]}',
    )
    yield SimpleNamespace(
        event_type="delta",
        raw_backend_payload=b'{"choices":[{"finish_reason":"length","delta":{}}]}',
    )
    yield SimpleNamespace(event_type="completed", raw_backend_payload=b"")


class StreamingMwTests(unittest.TestCase):
    def test_reasoning_budget_inflates_backend_max_tokens(self) -> None:
        payload = {"model": "qwen3.5-9b", "messages": [], "max_tokens": 16}
        adjusted = app_module._apply_reasoning_token_budget(model_name="qwen3.5-9b", payload=payload)  # type: ignore[attr-defined]
        self.assertEqual(adjusted["max_tokens"], 1280)
        self.assertEqual(payload["max_tokens"], 16)

    def test_reasoning_chunk_filter_hides_hidden_reasoning(self) -> None:
        raw = b'{"choices":[{"finish_reason":null,"delta":{"reasoning_content":"thinking"}}]}'
        self.assertIsNone(app_module._sanitize_stream_chat_chunk(raw))  # type: ignore[attr-defined]

    def test_chat_streaming_uses_mw_grpc_when_enabled(self) -> None:
        app_module._mw_client.cache_clear()
        fake_mw_client = SimpleNamespace(send_command=lambda **kwargs: {"ok": True})
        choice = LaneChoice(
            lane_id="lane-1",
            worker_id="Static-Deskix",
            base_url="http://10.0.1.99:21434",
            lane_type="gpu",
            backend_type="llama",
            current_model_name="qwen3.5-4b",
        )
        target = MwGrpcTarget(endpoint="127.0.0.1:50061", host_id="static-deskix", lane_id="gpu")
        fake_db = SimpleNamespace(connect=fake_db_connect)

        with (
            patch.object(app_module, "_create_router_request", return_value="req-1"),
            patch.object(app_module, "_touch_router_request", return_value=None),
            patch.object(app_module, "_request_cancel_requested", return_value=False),
            patch.object(app_module, "pick_lane_for_model", return_value=choice),
            patch.object(app_module, "_resolve_downstream_model_for_lane", return_value="qwen3.5-4b"),
            patch.object(app_module, "_acquire_router_lease", return_value=("lease-1", datetime.now(UTC) + timedelta(minutes=5))),
            patch.object(app_module, "_heartbeat_router_lease", return_value=None),
            patch.object(app_module, "_release_router_lease", return_value=None),
            patch.object(app_module, "sign_token", return_value="token"),
            patch.object(app_module, "_mw_target_for_lane", return_value=target),
            patch.object(app_module, "_mw_client", lambda: fake_mw_client),
            patch.object(app_module, "db", fake_db),
            patch.object(request_store_module, "db", fake_db),
            patch.object(app_module.MwGrpcClient, "stream_chat", fake_grpc_stream_chat),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.5-4b",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
            lines = [line for line in resp.iter_lines() if line]
            self.assertTrue(any("data:" in line for line in lines))
            self.assertTrue(any("[DONE]" in line for line in lines))

    def test_reasoning_only_stream_returns_error_instead_of_empty_success(self) -> None:
        app_module._mw_client.cache_clear()
        fake_mw_client = SimpleNamespace(send_command=lambda **kwargs: {"ok": True})
        choice = LaneChoice(
            lane_id="lane-1",
            worker_id="worker-gpu-a",
            base_url="http://192.0.2.10:21434",
            lane_type="gpu",
            backend_type="llama",
            current_model_name="qwen3.5-9b",
        )
        target = MwGrpcTarget(endpoint="127.0.0.1:50061", host_id="worker-gpu-a", lane_id="gpu")
        fake_db = SimpleNamespace(connect=fake_db_connect)

        with (
            patch.object(app_module, "_create_router_request", return_value="req-reasoning"),
            patch.object(app_module, "_touch_router_request", return_value=None),
            patch.object(app_module, "_request_cancel_requested", return_value=False),
            patch.object(app_module, "pick_lane_for_model", return_value=choice),
            patch.object(app_module, "_resolve_downstream_model_for_lane", return_value="qwen3.5-9b"),
            patch.object(app_module, "_acquire_router_lease", return_value=("lease-1", datetime.now(UTC) + timedelta(minutes=5))),
            patch.object(app_module, "_heartbeat_router_lease", return_value=None),
            patch.object(app_module, "_release_router_lease", return_value=None),
            patch.object(app_module, "sign_token", return_value="token"),
            patch.object(app_module, "_mw_target_for_lane", return_value=target),
            patch.object(app_module, "_mw_client", lambda: fake_mw_client),
            patch.object(app_module, "db", fake_db),
            patch.object(request_store_module, "db", fake_db),
            patch.object(app_module.MwGrpcClient, "stream_chat", fake_reasoning_only_stream_chat),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.5-9b",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
            )
            self.assertEqual(resp.status_code, 200)
            body = "\n".join(line for line in resp.iter_lines() if line)
            self.assertIn("reasoning_budget_exhausted", body)
            self.assertNotIn("thinking", body)

    def test_chat_non_streaming_uses_mw_grpc_when_enabled(self) -> None:
        app_module._mw_client.cache_clear()
        sent_commands = []
        fake_mw_client = SimpleNamespace(send_command=lambda **kwargs: sent_commands.append(kwargs) or {"ok": True})
        choice = LaneChoice(
            lane_id="lane-1",
            worker_id="Static-Mobile-2",
            base_url="http://10.0.0.132:21434",
            lane_type="cpu",
            backend_type="llama",
            current_model_name="Qwen3.5-0.8B-Q4_K_M.gguf",
        )
        target = MwGrpcTarget(endpoint="127.0.0.1:50061", host_id="static-mobile-2", lane_id="qwen")
        fake_db = SimpleNamespace(connect=fake_db_connect)

        with (
            patch.object(app_module, "_create_router_request", return_value="req-2"),
            patch.object(app_module, "_touch_router_request", return_value=None),
            patch.object(app_module, "_request_cancel_requested", return_value=False),
            patch.object(app_module, "pick_lane_for_model", return_value=choice),
            patch.object(app_module, "_resolve_downstream_model_for_lane", return_value="Qwen3.5-0.8B-Q4_K_M.gguf"),
            patch.object(app_module, "_acquire_router_lease", return_value=("lease-2", datetime.now(UTC) + timedelta(minutes=5))),
            patch.object(app_module, "_heartbeat_router_lease", return_value=None),
            patch.object(app_module, "_release_router_lease", return_value=None),
            patch.object(app_module, "sign_token", return_value="token"),
            patch.object(app_module, "_mw_target_for_lane", return_value=target),
            patch.object(app_module, "_mw_client", lambda: fake_mw_client),
            patch.object(app_module, "db", fake_db),
            patch.object(request_store_module, "db", fake_db),
            patch.object(app_module.MwGrpcClient, "stream_chat", fake_grpc_stream_chat),
        ):
            client = TestClient(app_module.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.5:0.8B",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["choices"][0]["message"]["content"], "hi")
            self.assertEqual(body["model"], "Qwen3.5-0.8B-Q4_K_M.gguf")
            self.assertEqual(sent_commands, [])


if __name__ == "__main__":
    unittest.main()

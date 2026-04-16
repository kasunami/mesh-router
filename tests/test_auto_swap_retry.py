from __future__ import annotations

import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from mesh_router import app as app_module


class _Cur:
    def __init__(self) -> None:
        self._last_select_model_id = False

    def execute(self, query: str, params=None):  # noqa: ANN001
        q = str(query)
        self._last_select_model_id = "SELECT model_id FROM models" in q
        return None

    def fetchone(self):  # noqa: ANN001
        if self._last_select_model_id:
            return {"model_id": "00000000-0000-0000-0000-000000000001"}
        return {"model_id": "00000000-0000-0000-0000-000000000001"}

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


class _FakeHttpxResp:
    status_code = 200

    def json(self):  # noqa: ANN001
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class _FakeHttpxClient:
    def __init__(self, *args, **kwargs):  # noqa: ANN001
        self.posts: list[str] = []

    def post(self, url: str, json=None, headers=None):  # noqa: ANN001
        self.posts.append(url)
        return _FakeHttpxResp()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class AutoSwapRetryTests(unittest.TestCase):
    def test_auto_swap_failure_retries_other_lane(self) -> None:
        # Lane 1: doesn't currently serve the requested model; swap will fail.
        lane1 = SimpleNamespace(
            lane_id="lane-1",
            worker_id="host-a",
            base_url="http://10.0.0.1:11434",
            current_model_name="some-other-model",
            backend_type="llama",
            lane_type="cpu",
            lane_max_ctx=8192,
            context_default=8192,
        )
        # Lane 2: already serves the requested model.
        lane2 = SimpleNamespace(
            lane_id="lane-2",
            worker_id="host-b",
            base_url="http://10.0.0.2:11434",
            current_model_name="qwen3.5:0.8B",
            backend_type="llama",
            lane_type="cpu",
            lane_max_ctx=8192,
            context_default=8192,
        )

        picks = [lane1, lane2]

        def _fake_pick_lane_for_model(*args, **kwargs):  # noqa: ANN001
            return picks.pop(0)

        swap_calls: list[str] = []

        def _fake_swap(lane_id: str, req):  # noqa: ANN001
            swap_calls.append(lane_id)
            raise RuntimeError("409: model is not viable for this lane")

        def _fake_acquire_router_lease(*, lane_id: str, model_id: str, owner: str, job_type: str, ttl_seconds: int, details: dict):  # noqa: ANN001
            return "lease-1", datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

        with (
            mock.patch.object(app_module, "db", _Db()),
            mock.patch.object(app_module, "pick_lane_for_model", side_effect=_fake_pick_lane_for_model),
            mock.patch.object(app_module, "api_lane_swap_model", side_effect=_fake_swap),
            mock.patch.object(app_module, "_acquire_router_lease", side_effect=_fake_acquire_router_lease),
            mock.patch.object(app_module, "_heartbeat_router_lease", return_value=None),
            mock.patch.object(app_module, "_release_router_lease", return_value=None),
            mock.patch.object(app_module, "_request_cancel_requested", return_value=False),
            mock.patch.object(app_module, "_touch_router_request", return_value=None),
            mock.patch.object(app_module, "_resolve_downstream_model_for_lane", side_effect=lambda *a, **k: k.get("requested_model_name") or "qwen3.5:0.8B"),
            mock.patch.object(app_module, "_mw_target_for_lane", return_value=None),
            mock.patch.object(app_module.httpx, "Client", _FakeHttpxClient),
            mock.patch.object(app_module, "_maybe_record_perf_observation", return_value=None),
        ):
            out = app_module._execute_router_request(
                request_id="req-1",
                route="chat",
                raw_payload={
                    "model": "qwen3.5:0.8B",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                owner="test",
                job_type="test",
            )

        self.assertEqual(out["choices"][0]["message"]["content"], "ok")
        # We attempted a swap only for the first lane, then retried and used lane2.
        self.assertEqual(swap_calls, ["lane-1"])
        self.assertEqual(len(picks), 0)


if __name__ == "__main__":
    unittest.main()

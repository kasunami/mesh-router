from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from mesh_router import app as app_module


class FakeCursor:
    def __init__(self) -> None:
        self._last = ""

    def execute(self, query: str, params=None):  # noqa: ANN001
        self._last = str(query)
        return None

    def fetchone(self):  # noqa: ANN001
        q = self._last
        if "FROM lanes l" in q:
            # _mw_target_for_lane SELECT ... FROM lanes l JOIN hosts h ...
            return {
                "lane_id": "lane-1",
                "lane_name": "gpu",
                "lane_type": "gpu",
                "backend_type": "llama",
                "base_url": "http://10.0.0.99:21434",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "host_name": "Static-Deskix",
            }
        if "SELECT proxy_auth_metadata FROM lanes" in q:
            return {"proxy_auth_metadata": {"control_plane": "mw"}}
        if "SELECT model_id FROM models" in q:
            return {"model_id": "00000000-0000-0000-0000-000000000001"}
        return {"model_id": "00000000-0000-0000-0000-000000000001"}

    def fetchall(self):  # noqa: ANN001
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class FakeConn:
    def cursor(self):  # noqa: ANN001
        return FakeCursor()

    def commit(self):  # noqa: ANN001
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class FakeDb:
    def connect(self):  # noqa: ANN001
        return FakeConn()


class StrictMwFallbackTests(unittest.TestCase):
    def test_explicit_mw_lane_without_target_refuses_base_url_fallback(self) -> None:
        lane = SimpleNamespace(
            lane_id="lane-1",
            worker_id="Static-Deskix",
            base_url="http://10.0.0.99:21434",
            current_model_name="some-other",
            backend_type="llama",
            lane_type="gpu",
            lane_max_ctx=8192,
            context_default=8192,
        )

        def _fake_pick_lane_for_model(*args, **kwargs):  # noqa: ANN001
            return lane

        def _fake_acquire_router_lease(*, lane_id: str, model_id: str, owner: str, job_type: str, ttl_seconds: int, details: dict):  # noqa: ANN001
            return "lease-1", datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

        with (
            mock.patch.object(app_module, "db", FakeDb()),
            mock.patch.object(app_module, "pick_lane_for_model", side_effect=_fake_pick_lane_for_model),
            mock.patch.object(app_module, "api_lane_swap_model", return_value={"ok": True}),
            mock.patch.object(app_module, "_acquire_router_lease", side_effect=_fake_acquire_router_lease),
            mock.patch.object(app_module, "_heartbeat_router_lease", return_value=None),
            mock.patch.object(app_module, "_release_router_lease", return_value=None),
            mock.patch.object(app_module, "_request_cancel_requested", return_value=False),
            mock.patch.object(app_module, "_touch_router_request", return_value=None),
            mock.patch.object(app_module, "_resolve_downstream_model_for_lane", side_effect=lambda *a, **k: k.get("requested_model_name") or "qwen3.5:0.8B"),
            # Force MW target resolution failure.
            mock.patch.object(app_module, "_mw_target_for_lane", return_value=None),
            mock.patch.object(app_module, "_maybe_record_perf_observation", return_value=None),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                app_module._execute_router_request(
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

        self.assertIn("refusing base_url fallback", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()

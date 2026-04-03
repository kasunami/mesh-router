from __future__ import annotations

import unittest

from mesh_router import app as app_module


class _Cur:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _Conn:
    def __init__(self, cur: _Cur) -> None:
        self._cur = cur

    def cursor(self):  # noqa: ANN001
        return self._cur

    def commit(self) -> None:
        return None

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _Db:
    def __init__(self, cur: _Cur) -> None:
        self._cur = cur

    def connect(self):  # noqa: ANN001
        return _Conn(self._cur)


class PerfAutoObserveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_db = app_module.mw_state_db
        self.orig_insert = app_module.insert_observation
        # Ensure observe is on and unfiltered for tests.
        self.orig_enabled = app_module.settings.perf_auto_observe_enabled
        self.orig_sample = app_module.settings.perf_auto_observe_sample_rate
        self.orig_min = app_module.settings.perf_auto_observe_min_elapsed_ms
        self.orig_max = app_module.settings.perf_auto_observe_max_total_ms
        app_module.settings.perf_auto_observe_enabled = True
        app_module.settings.perf_auto_observe_sample_rate = 1.0
        app_module.settings.perf_auto_observe_min_elapsed_ms = 0
        app_module.settings.perf_auto_observe_max_total_ms = 10_000_000

    def tearDown(self) -> None:
        app_module.mw_state_db = self.orig_db  # type: ignore[assignment]
        app_module.insert_observation = self.orig_insert  # type: ignore[assignment]
        app_module.settings.perf_auto_observe_enabled = self.orig_enabled
        app_module.settings.perf_auto_observe_sample_rate = self.orig_sample
        app_module.settings.perf_auto_observe_min_elapsed_ms = self.orig_min
        app_module.settings.perf_auto_observe_max_total_ms = self.orig_max

    def test_auto_observe_ingests_ok(self) -> None:
        cur = _Cur()
        app_module.mw_state_db = _Db(cur)  # type: ignore[assignment]

        captured: dict = {}

        def _insert(*, cur, obs):  # noqa: ANN001
            captured["obs"] = obs

        app_module.insert_observation = _insert  # type: ignore[assignment]

        app_module._maybe_record_perf_observation(
            host_name="Static-Deskix",
            lane_id="lane-1",
            model_name="qwen3.5-2b",
            modality="chat",
            elapsed_ms=123,
            first_token_ms=10.0,
            prompt_tokens=5,
            completion_tokens=7,
            decode_tps=70.0,
            ok=True,
            error_kind=None,
            error_message=None,
            metadata={"request_id": "req-1"},
        )

        self.assertIn("obs", captured)
        obs = captured["obs"]
        self.assertEqual(obs["host_id"], "Static-Deskix")
        self.assertEqual(obs["lane_id"], "lane-1")
        self.assertEqual(obs["model_name"], "qwen3.5-2b")
        self.assertEqual(obs["ok"], True)
        self.assertEqual(obs["generated_tokens"], 7)
        self.assertEqual(obs["metadata"]["source"], "auto_observe")
        self.assertEqual(obs["metadata"]["request_id"], "req-1")

    def test_auto_observe_filters_user_cancel(self) -> None:
        cur = _Cur()
        app_module.mw_state_db = _Db(cur)  # type: ignore[assignment]

        called = {"n": 0}

        def _insert(*, cur, obs):  # noqa: ANN001
            called["n"] += 1

        app_module.insert_observation = _insert  # type: ignore[assignment]

        app_module._maybe_record_perf_observation(
            host_name="Static-Deskix",
            lane_id="lane-1",
            model_name="qwen3.5-2b",
            modality="chat",
            elapsed_ms=123,
            first_token_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            decode_tps=None,
            ok=False,
            error_kind="canceled",
            error_message="user canceled",
            metadata={"request_id": "req-2"},
        )

        self.assertEqual(called["n"], 0)


if __name__ == "__main__":
    unittest.main()


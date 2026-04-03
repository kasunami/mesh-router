from __future__ import annotations

import unittest

from mesh_router import app as app_module


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

    def commit(self) -> None:
        return None


class _Db:
    def connect(self):  # noqa: ANN001
        return _Conn()


class PerfAutoObserveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_mw_state_db = app_module.mw_state_db
        self.orig_insert = app_module.insert_observation
        self.orig_enabled = app_module.settings.perf_auto_observe_enabled
        self.orig_rate = app_module.settings.perf_auto_observe_sample_rate
        self.orig_min = app_module.settings.perf_auto_observe_min_elapsed_ms
        self.orig_max = app_module.settings.perf_auto_observe_max_total_ms

    def tearDown(self) -> None:
        app_module.mw_state_db = self.orig_mw_state_db  # type: ignore[assignment]
        app_module.insert_observation = self.orig_insert  # type: ignore[assignment]
        app_module.settings.perf_auto_observe_enabled = self.orig_enabled
        app_module.settings.perf_auto_observe_sample_rate = self.orig_rate
        app_module.settings.perf_auto_observe_min_elapsed_ms = self.orig_min
        app_module.settings.perf_auto_observe_max_total_ms = self.orig_max

    def test_maybe_record_perf_observation_honors_guards_and_calls_insert(self) -> None:
        app_module.mw_state_db = _Db()  # type: ignore[assignment]
        app_module.settings.perf_auto_observe_enabled = True
        app_module.settings.perf_auto_observe_sample_rate = 1.0
        app_module.settings.perf_auto_observe_min_elapsed_ms = 1
        app_module.settings.perf_auto_observe_max_total_ms = 1000

        seen: list[dict] = []

        def _insert(*, cur, obs):  # noqa: ANN001
            seen.append(dict(obs))

        app_module.insert_observation = _insert  # type: ignore[assignment]

        app_module._maybe_record_perf_observation(
            host_name="Static-Deskix",
            lane_id="lane-1",
            model_name="qwen3.5-4b",
            modality="chat",
            backend_type="llama",
            lane_type="gpu",
            elapsed_ms=100,
            first_token_ms=12.0,
            prompt_tokens=5,
            completion_tokens=10,
            decode_tps=99.0,
            ok=True,
            error_kind=None,
            error_message=None,
            metadata={"request_id": "req-1"},
        )

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["host_id"], "Static-Deskix")
        self.assertEqual(seen[0]["lane_type"], "gpu")
        self.assertEqual(seen[0]["backend_type"], "llama")
        self.assertEqual(seen[0]["metadata"]["source"], "auto_observe")

        # Over max_total_ms -> drop.
        app_module._maybe_record_perf_observation(
            host_name="Static-Deskix",
            lane_id="lane-1",
            model_name="qwen3.5-4b",
            modality="chat",
            elapsed_ms=999999,
            first_token_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            decode_tps=None,
            ok=True,
            error_kind=None,
            error_message=None,
        )
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import unittest
from unittest import mock

from mesh_router import app as app_module


class BackendCompatibilityTests(unittest.TestCase):
    def test_bitnet_requires_bitnet_cpu_lane(self) -> None:
        reason = app_module._backend_compatibility_reason(  # type: ignore[attr-defined]
            model_name="Falcon3-10B-Instruct-1.58bit",
            tags=["bitnet", "cpu"],
            backend_type="llama",
            lane_type="cpu",
        )
        self.assertEqual(reason, "model requires bitnet backend")

    def test_flux_requires_sd_backend(self) -> None:
        reason = app_module._backend_compatibility_reason(  # type: ignore[attr-defined]
            model_name="flux1-schnell-Q4_K_S",
            tags=[],
            backend_type="llama",
            lane_type="gpu",
        )
        self.assertEqual(reason, "model requires stable-diffusion backend")

    def test_standard_chat_model_allows_llama_backend(self) -> None:
        reason = app_module._backend_compatibility_reason(  # type: ignore[attr-defined]
            model_name="gemma-4-e4b-it-Q4_K_M",
            tags=[],
            backend_type="llama",
            lane_type="cpu",
        )
        self.assertIsNone(reason)

    def test_standard_quantized_mlx_model_is_not_misclassified_as_bitnet(self) -> None:
        reason = app_module._backend_compatibility_reason(  # type: ignore[attr-defined]
            model_name="/Users/kasunami/models/Qwen3.5-9B-6bit",
            tags=["qwen3.5:9b"],
            backend_type="llama",
            lane_type="mlx",
        )
        self.assertIsNone(reason)

    def test_mw_authoritative_capabilities_hide_remote_candidates(self) -> None:
        self.assertFalse(
            app_module._should_include_candidate_for_capabilities(  # type: ignore[attr-defined]
                mw_authoritative=True,
                source_locality="remote",
            )
        )
        self.assertTrue(
            app_module._should_include_candidate_for_capabilities(  # type: ignore[attr-defined]
                mw_authoritative=True,
                source_locality="local",
            )
        )
        self.assertTrue(
            app_module._should_include_candidate_for_capabilities(  # type: ignore[attr-defined]
                mw_authoritative=False,
                source_locality="remote",
            )
        )

    def test_path_matches_local_model_root_accepts_model_dir_and_children(self) -> None:
        self.assertTrue(
            app_module._path_matches_local_model_root(  # type: ignore[attr-defined]
                artifact_path="/Users/kasunami/models/Qwen3.5-9B-6bit",
                local_model_root="/Users/kasunami/models",
            )
        )
        self.assertTrue(
            app_module._path_matches_local_model_root(  # type: ignore[attr-defined]
                artifact_path="/Users/kasunami/models/Qwen3.5-9B-6bit/config.json",
                local_model_root="/Users/kasunami/models",
            )
        )

    def test_path_matches_local_model_root_rejects_other_local_roots(self) -> None:
        self.assertFalse(
            app_module._path_matches_local_model_root(  # type: ignore[attr-defined]
                artifact_path="/Users/kasunami/mlx-model-bank/blobs/abcdef",
                local_model_root="/Users/kasunami/models",
            )
        )

    def test_prune_lane_model_viability_outside_local_root_uses_root_filter(self) -> None:
        calls: list[tuple[str, tuple[object, ...] | None]] = []

        class _FakeCursor:
            def execute(self, sql, params=None):  # noqa: ANN001
                calls.append((sql, params))

        app_module._prune_lane_model_viability_outside_local_root(  # type: ignore[attr-defined]
            _FakeCursor(),
            lane_id="lane-mlx",
            local_model_root="/Users/kasunami/models",
        )
        self.assertEqual(len(calls), 1)
        _sql, params = calls[0]
        self.assertEqual(params, ("lane-mlx", "/Users/kasunami/models", "/Users/kasunami/models/%"))

    def test_resolve_downstream_model_for_lane_prefers_alias_for_current_model(self) -> None:
        class _FakeCursor:
            def __init__(self) -> None:
                self._last_sql = ""

            def execute(self, sql, params=None):  # noqa: ANN001
                self._last_sql = sql

            def fetchone(self):  # noqa: ANN001
                if "FROM lanes l" in self._last_sql:
                    return {"current_model_name": "Qwen3.5-9B-6bit", "current_model_tags": []}
                if "SELECT model_id FROM models" in self._last_sql:
                    return {"model_id": "model-1"}
                return None

        with mock.patch.object(
            app_module,
            "_resolve_lane_downstream_alias",
            side_effect=lambda cur, *, lane_id, model_id: "/Users/kasunami/models/Qwen3.5-9B-6bit",
        ):
            result = app_module._resolve_downstream_model_for_lane(  # type: ignore[attr-defined]
                _FakeCursor(),
                lane_id="lane-mlx",
                requested_model_name="Qwen3.5-9B-6bit",
                model_id="model-1",
            )
        self.assertEqual(result, "/Users/kasunami/models/Qwen3.5-9B-6bit")

    def test_resolve_downstream_model_for_lane_uses_alias_for_non_current_model(self) -> None:
        class _FakeCursor:
            def __init__(self) -> None:
                self._last_sql = ""

            def execute(self, sql, params=None):  # noqa: ANN001
                self._last_sql = sql

            def fetchone(self):  # noqa: ANN001
                if "FROM lanes l" in self._last_sql:
                    return {"current_model_name": "Falcon3-10B-Instruct-1.58bit", "current_model_tags": []}
                return None

        with mock.patch.object(
            app_module,
            "_resolve_lane_downstream_alias",
            side_effect=lambda cur, *, lane_id, model_id: "/Users/kasunami/models/Qwen3.5-4B-MLX-4bit",
        ):
            result = app_module._resolve_downstream_model_for_lane(  # type: ignore[attr-defined]
                _FakeCursor(),
                lane_id="lane-mlx",
                requested_model_name="Qwen3.5-4B-MLX-4bit",
                model_id="model-2",
            )
        self.assertEqual(result, "/Users/kasunami/models/Qwen3.5-4B-MLX-4bit")


if __name__ == "__main__":
    unittest.main()

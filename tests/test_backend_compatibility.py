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

    def test_mw_runtime_candidate_tags_include_backend_and_lane_hints(self) -> None:
        self.assertEqual(
            app_module._mw_runtime_candidate_tags(  # type: ignore[attr-defined]
                lane_row={"backend_type": "bitnet", "lane_type": "cpu"}
            ),
            ["bitnet", "cpu", "bitnet"],
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

    def test_mw_effective_lane_row_for_capabilities_prefers_mw_backend_and_model_truth(self) -> None:
        lane_row = {
            "lane_id": "lane-cpu",
            "lane_name": "cpu",
            "lane_type": "cpu",
            "backend_type": "llama",
            "current_model_name": "LFM2.5-350M-Q4_K_M.gguf",
            "desired_model_name": None,
            "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "cpu"},
        }
        host_row = {"host_name": "Static-Deskix"}
        with mock.patch.object(
            app_module,
            "apply_mw_effective_status",
            side_effect=lambda rows, **kwargs: rows[0].update(
                {
                    "backend_type": "bitnet",
                    "current_model_name": "falcon3-10b",
                    "desired_model_name": "falcon3-10b",
                    "current_backend_type": "bitnet",
                    "desired_backend_type": "bitnet",
                }
            ),
        ):
            result = app_module._mw_effective_lane_row_for_capabilities(  # type: ignore[attr-defined]
                lane_row=lane_row,
                host_row=host_row,
            )
        self.assertEqual(result["host_name"], "Static-Deskix")
        self.assertEqual(result["backend_type"], "bitnet")
        self.assertEqual(result["current_model_name"], "falcon3-10b")
        self.assertEqual(result["desired_model_name"], "falcon3-10b")

    def test_build_lane_capability_payload_uses_mw_effective_row_for_active_falcon_candidate(self) -> None:
        class _FakeCursor:
            def __init__(self) -> None:
                self._last_sql = ""
                self._params = None

            def execute(self, sql, params=None):  # noqa: ANN001
                self._last_sql = sql
                self._params = params

            def fetchone(self):  # noqa: ANN001
                if "FROM hosts WHERE host_id" in self._last_sql:
                    return {"host_id": "host-1", "host_name": "Static-Deskix", "local_model_root": "/home/kasunami/models"}
                if "SELECT p.max_ctx" in self._last_sql:
                    return {"max_ctx": 32768}
                return None

            def fetchall(self):  # noqa: ANN001
                if "FROM host_model_artifacts hma" in self._last_sql:
                    return [
                        {
                            "artifact_id": "artifact-1",
                            "host_id": "host-1",
                            "host_name": "Static-Deskix",
                            "storage_scope": "host",
                            "storage_provider": "local",
                            "local_path": "/home/kasunami/models/LFM2.5-350M-Q4_K_M.gguf",
                            "size_bytes": 267060512,
                            "present": True,
                            "model_id": "model-1",
                            "model_name": "LFM2.5-350M-Q4_K_M.gguf",
                            "tags": [],
                            "required_ram_bytes": None,
                            "required_vram_bytes": None,
                            "allowed": True,
                            "max_ctx": None,
                        }
                    ]
                return []

        with mock.patch.object(
            app_module,
            "_resolve_lane",
            return_value={
                "lane_id": "lane-cpu",
                "host_id": "host-1",
                "lane_name": "cpu",
                "lane_type": "cpu",
                "backend_type": "llama",
                "base_url": "http://10.0.0.99:11435",
                "current_model_name": "LFM2.5-350M-Q4_K_M.gguf",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "cpu"},
                "usable_memory_bytes": 12884901888,
                "ram_budget_bytes": 12884901888,
                "runtime_overhead_bytes": 0,
                "reserved_headroom_bytes": 1073741824,
            },
        ), mock.patch.object(
            app_module,
            "_mw_target_for_lane",
            return_value=object(),
        ), mock.patch.object(
            app_module,
            "_resolve_host",
            return_value={"host_id": "host-1", "host_name": "Static-Deskix", "ram_ai_budget_bytes": 12884901888, "local_model_root": "/home/kasunami/models"},
        ), mock.patch.object(
            app_module,
            "_mw_effective_lane_row_for_capabilities",
            return_value={
                "lane_id": "lane-cpu",
                "host_id": "host-1",
                "lane_name": "cpu",
                "lane_type": "cpu",
                "backend_type": "bitnet",
                "current_model_name": "falcon3-10b",
                "desired_model_name": "falcon3-10b",
                "current_backend_type": "bitnet",
                "desired_backend_type": "bitnet",
                "usable_memory_bytes": 12884901888,
                "ram_budget_bytes": 12884901888,
                "runtime_overhead_bytes": 0,
                "reserved_headroom_bytes": 1073741824,
                "current_model_max_ctx": 32768,
            },
        ), mock.patch.object(
            app_module,
            "_local_model_root",
            return_value="/home/kasunami/models",
        ), mock.patch.object(
            app_module,
            "_prune_lane_model_viability_outside_local_root",
            return_value=None,
        ), mock.patch.object(
            app_module,
            "_historical_swap_ms",
            return_value=0,
        ):
            _state, payload = app_module._build_lane_capability_payload(_FakeCursor(), "lane-cpu")  # type: ignore[attr-defined]
        self.assertIn("falcon3-10b", payload.supported_models)
        self.assertEqual(payload.current_model, "falcon3-10b")
        falcon = next(item for item in payload.local_viable_models if item.model_name == "falcon3-10b")
        self.assertEqual(falcon.artifact_provider, "mw_runtime")
        self.assertEqual(falcon.estimated_swap_ms, 0)


if __name__ == "__main__":
    unittest.main()

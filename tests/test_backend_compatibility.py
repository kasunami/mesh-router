from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()

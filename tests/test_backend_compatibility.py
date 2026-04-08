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


if __name__ == "__main__":
    unittest.main()
